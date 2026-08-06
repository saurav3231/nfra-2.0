"""O(1)-per-token stateful decode for the lean Cortex model.

Retention (decayed causal QK^T, no softmax) has an exact recurrent dual:

    y_h[i] = (q_i / sqrt(Hd)) . R_h,   R_h <- gamma_h R_h + k_i^T v_i,
    gamma_h = exp(-exp(log_decay_h))

so after a one-time prefill the whole model can decode ONE token with O(1)
work per token (a handful of small GEMMs) instead of re-evaluating the entire
sequence every step. This module implements that dual for the default LEAN
Cortex block (value gate / phase / neuromodulator / exit all pruned) with the
PER-TOKEN GroupNorm (cortex_per_token_gn) enabled: per-token GN normalizes
each head over its own channels for a single token, so the ONLY cross-token
state is each mixer's retention R (and the model-level global brain's causal
running prefix mean). With full-sequence GroupNorm the dual is impossible --
that norm couples all positions, so a re-eval at longer context silently
rewrites every prior token's hidden state.

Guarding: `supported(model)` refuses models with non-lean features (adaptive
exit, neuromodulator, value gate, phase). `stateful_equivalence()` checks the
decoded logits against repeated full re-evals on the growing context and the
benchmark only reports stateful gen speed when the match passes — a wrong dual
can never be silently trusted.
"""

from __future__ import annotations

import math
import time

import torch

from .cortex import NFRA_Cortex_Block, PerTokenGN


def supported(model) -> bool:
    """True iff every block is a lean Cortex block the stateful dual covers.
    Requires the per-token GroupNorm: nn.GroupNorm couples positions (it
    normalizes over the sequence), so a recurrent dual could never match it;
    per-token GN leaves retention as the only cross-token state."""
    if not getattr(model, "config", None) or not model.config.use_cortex:
        return False
    if not model.config.cortex_per_token_gn:
        return False
    if model.depth_passes != 1:  # depth-time FiLM / pass_lora / global_brain
        return False
    if model.pass_lora is not None:
        return False
    for layer in model.layers:
        if not isinstance(layer, NFRA_Cortex_Block):
            return False
        m = layer.mixer
        if not isinstance(m.gn, PerTokenGN):
            return False
        if layer.neuromodulator is not None or layer.exit_gate is not None:
            return False  # adaptive exit / neuromodulation not covered
        if not m.iso_vgate or not m.iso_phase:
            return False  # value gate / phase modulate the state
    return True


def make_states(model, batch=1, device="cpu"):
    st = []
    for layer in model.layers:
        m = layer.mixer
        entry = {
            "R": torch.zeros(
                batch, m.n_heads, m.head_dim, m.head_dim,
                device=device, dtype=torch.float32,
            ),
        }
        if m.stm is not None:  # STM working-tag ring: cached recent mixer inputs
            entry["stm_ctx"] = []
        st.append(entry)
    if model.global_brain is not None:
        st.append(
            {
                "gb_sum": torch.zeros(
                    batch, model.config.hidden_size, device=device, dtype=torch.float32
                ),
                "count": 0.0,
            }
        )
    return st


def _mixer_step(mixer, x1, st):
    B, _S1, D = x1.shape
    H, Hd = mixer.n_heads, mixer.head_dim
    qkvr = mixer.qkvr(x1)
    t = qkvr.view(B, 1, 3 + mixer.n_gates, H, Hd).permute(2, 0, 3, 1, 4)
    q, k, v = t[0], t[1], t[2]
    r = t[-1] if not mixer.iso_rgate else None

    l = mixer._eff_log_decay().float()
    gamma = torch.exp(-torch.exp(l)).view(1, H, 1, 1)  # [1,H,1,1]
    R = st["R"].float()
    R_new = R * gamma + torch.matmul(
        k.transpose(-2, -1).float(), v.float()
    )  # [B,H,Hd,Hd]
    st["R"] = R_new
    y = torch.matmul(q.float() * (Hd**-0.5), R_new)  # [B,H,1,Hd]
    y = y.permute(0, 2, 1, 3).reshape(B, 1, D)

    # Per-token GroupNorm: normalizes each head over its own channels for THIS
    # token only (no cross-token state), so it is trivially exact per step.
    y = mixer.gn(y.permute(0, 2, 1)).permute(0, 2, 1)

    if r is not None:  # receptance gate (per-token, no state)
        rg = torch.sigmoid(r).permute(0, 2, 1, 3).reshape(B, 1, D)
        y = y * rg

    if mixer.stm is not None:  # STM working-tag ring read + cache update
        ctx = st.get("stm_ctx")
        if ctx is None:
            ctx = st["stm_ctx"] = []
        y = y + mixer.stm.read_step(x1, ctx)
        ctx.append(x1)
        if len(ctx) > mixer.stm.window:
            del ctx[0]
    return mixer.proj_out(y), st


def _block_step(layer, x1, st):
    residual = x1
    n = layer.ln1(x1)
    mix_out, st = _mixer_step(layer.mixer, n, st)
    x = residual + layer.dropout(mix_out)
    residual = x
    n = layer.ln2(x)
    n = layer.mlp(n)
    x = residual + layer.dropout(n)
    return x, st


@torch.no_grad()
def decode_step(model, x1, states):
    """One-token decode. x1 [B,1] token ids; states updated in place (list).
    The final element of `states` is the model-level global-brain running mean
    (only present when the model has a global_brain)."""
    if not supported(model):
        raise NotImplementedError("stateful decode needs the lean Cortex model")
    hidden = model.embed_tokens(x1)
    for layer, st in zip(model.layers, states[:-1] if model.global_brain is not None else states):
        hidden, st = _block_step(layer, hidden, st)
    if model.global_brain is not None and len(states) == len(model.layers) + 1:
        gb = states[-1]
        gb_sum = gb["gb_sum"].float() + hidden[:, 0]
        count = gb["count"] + 1.0
        gb["gb_sum"] = gb_sum
        gb["count"] = float(count)
        mean = gb_sum / count  # causal prefix mean [B, D]
        h = torch.tanh(model.global_brain.pool_proj(mean))
        proj = model.global_brain.inject_proj(h)
        gain = torch.sigmoid(model.global_brain.gate(h))
        hidden = hidden + gain * proj
    return model.lm_head(hidden), states


@torch.no_grad()
def prefill(model, input_ids, states=None):
    """Run the prompt through the stateful dual, one token at a time."""
    if states is None:
        states = make_states(model, input_ids.shape[0], input_ids.device)
    for t in range(input_ids.shape[1]):
        decode_step(model, input_ids[:, t : t + 1], states)
    return states


@torch.no_grad()
def generate(model, prompt_ids, n_new, states=None, greedy=True):
    """Greedy stateful generation after (optional) prefill of prompt_ids."""
    model.eval()
    if states is None:
        states = make_states(model, prompt_ids.shape[0], prompt_ids.device)
        prefill(model, prompt_ids, states)
    out = [prompt_ids]
    cur = prompt_ids[:, -1:]
    for _ in range(n_new):
        logits, states = decode_step(model, cur, states)
        if greedy:
            cur = logits[:, -1, :].argmax(-1, keepdim=True)
        else:
            cur = torch.multinomial(
                torch.softmax(logits[:, -1, :] / 0.8, dim=-1), 1
            )
        out.append(cur)
    return torch.cat(out, dim=1), states


@torch.no_grad()
def stateful_equivalence(model, prompt_ids, n_new):
    """Compare stateful-decode logits vs full re-eval on the growing context.
    Returns (max_abs, max_rel, ok). Guards the fast path: if the dual diverges
    from the true model the benchmark reports the slow path instead."""
    model.eval()
    states = make_states(model, prompt_ids.shape[0], prompt_ids.device)
    prefill(model, prompt_ids, states)
    ctx = prompt_ids
    cur = prompt_ids[:, -1:]
    max_abs, max_rel = 0.0, 0.0
    for _ in range(n_new):
        logits, states = decode_step(model, cur, states)
        ctx = torch.cat([ctx, cur], dim=1)
        ref = model(ctx)["logits"][:, -1, :]  # full re-eval at current length
        d = (logits[:, -1, :].float() - ref.float()).abs()
        scale = ref.float().abs().max().clamp_min(1e-6)
        max_abs = max(max_abs, float(d.max()))
        max_rel = max(max_rel, float(d.max() / scale))
        cur = logits[:, -1, :].argmax(-1, keepdim=True)
    return max_abs, max_rel, max_rel < 1e-2


@torch.no_grad()
def stateful_generate_metrics(model, vocab, prompt_len=64, gen_len=32, device="cuda"):
    """Stateful decode throughput + equivalence. Returns a dict (None fields if
    the model is unsupported). Decode tok/s is measured after prefill only, so
    it is comparable to the slow generate_metrics."""
    if not supported(model):
        return {
            "gen_sf": None,
            "sf_abs": None,
            "sf_rel": None,
            "sf_ok": None,
        }
    model.eval()
    x = torch.randint(0, vocab, (1, prompt_len), device=device)
    states = make_states(model, 1, device)
    prefill(model, x, states)
    cur = x[:, -1:]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(gen_len):
        logits, states = decode_step(model, cur, states)
        cur = logits[:, -1, :].argmax(-1, keepdim=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = max(time.perf_counter() - t0, 1e-6)
    max_abs, max_rel, ok = stateful_equivalence(model, x, gen_len)
    return {
        "gen_sf": gen_len / dt,
        "sf_abs": max_abs,
        "sf_rel": max_rel,
        "sf_ok": ok,
    }
