"""
Smoke tests for NFRA model construction, forward/backward, and IO.

Runs on CPU with tiny configs. Use: pytest tests/
"""

import torch

from nfra import NFRAConfig, NFRAForCausalLM
from nfra.models.nfra_lite import NFRALiteForCausalLM
from nfra.models.model_io import save_pretrained, from_pretrained
from nfra.core.energy import DynamicEnergyBudgetAllocator


def _brain(vocab=96, dim=128, layers=6, unique=2, ckpt=False):
    cfg = NFRAConfig(mode="brain", vocab_size=vocab, hidden_size=dim,
                     num_layers=layers, unique_blocks=unique,
                     gradient_checkpointing=ckpt)
    return NFRAForCausalLM(cfg)


def test_brain_forward_backward():
    torch.manual_seed(0)
    model = _brain()
    x = torch.randint(0, 96, (2, 32))
    out = model(x)
    assert out["logits"].shape == (2, 32, 96)
    out["logits"].float().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_unique_blocks_scale_params():
    counts = [
        sum(p.numel() for p in _brain(dim=128, layers=12, unique=u).parameters())
        for u in (1, 2, 4)
    ]
    assert counts[1] > counts[0]
    assert counts[2] > counts[1]


def test_brain_config_respects_band_knob():
    # The NFRA_BANDS ablation knob must reach the config: an explicit n_bands
    # is respected (was silently force-overridden to 16); only the dataclass
    # default (4) is promoted to the legacy 16-head hierarchy.
    assert NFRAConfig(mode="brain", n_bands=8).n_bands == 8
    assert NFRAConfig(mode="brain", n_bands=2).n_bands == 2
    assert NFRAConfig(mode="brain").n_bands == 16


def test_band_knob_reaches_brainmixer():
    cfg = NFRAConfig(mode="brain", vocab_size=96, hidden_size=128,
                     num_layers=2, n_bands=8)
    model = NFRAForCausalLM(cfg)
    assert model.layers[0].mixer.n_heads == 8
    default = NFRAForCausalLM(NFRAConfig(mode="brain", vocab_size=96,
                                         hidden_size=128, num_layers=2))
    assert default.layers[0].mixer.n_heads == 16


def test_lite_build_and_forward():
    torch.manual_seed(0)
    cfg = NFRAConfig(mode="lite", vocab_size=96)
    model = NFRALiteForCausalLM(cfg)
    assert model.get_model_info()["mode"] == "lite"
    x = torch.randint(0, 96, (2, 16))
    out = model(x)
    assert out["logits"].shape == (2, 16, 96)


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = _brain()
    save_pretrained(model, str(tmp_path))
    loaded = from_pretrained(str(tmp_path))
    model.eval()
    x = torch.randint(0, 96, (1, 16))
    with torch.no_grad():
        a = model(x)["logits"]
        b = loaded(x)["logits"]
    assert torch.allclose(a, b, atol=1e-5)


def test_brain_learns_short_recall():
    """Smoke test: NFRA Brain must LEARN a 2-step recall task (drop below the
    ln(16) floor) on a tiny CPU config, with OBSERVABLE keys. Guards against
    the block wiring starving the recurrence entirely. (The old probe hid the
    keys, so no model could learn the span — see recall_probe docstring.)"""
    import math

    V, k, seq = 16, 2, 32
    torch.manual_seed(0)
    cfg = NFRAConfig(mode="brain", vocab_size=V, hidden_size=64,
                     num_layers=6, unique_blocks=2, dropout=0.0,
                     gradient_checkpointing=False)
    model = NFRAForCausalLM(cfg)
    from nfra.benchmark.compare import rescale_embed
    from nfra.benchmark.recall_probe import make_loader
    rescale_embed(model)

    x, y = next(iter(make_loader(k, seq, 8, seed=42)))

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            model(x)["logits"].reshape(-1, V), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    floor = math.log(V)
    assert losses[-1] < floor - 0.03, f"stuck at floor: {losses[-1]:.3f}"
    assert losses[-1] < losses[0] - 0.1, f"no learning: {losses[0]:.3f}->{losses[-1]:.3f}"


def test_brain_no_future_leak():
    """NFRA Brain must be a proper autoregressive model: changing a FUTURE
    token must not change logits at EARLIER positions. Regression guard for
    the whole-sequence global-pool leak (NeuroModulator / GlobalBrainState /
    astro / router pool all used x.mean(dim=1), leaking the future)."""
    torch.manual_seed(0)
    cfg = NFRAConfig(mode="brain", vocab_size=16, hidden_size=64,
                     num_layers=4, unique_blocks=2, dropout=0.0,
                     gradient_checkpointing=False)
    model = NFRAForCausalLM(cfg)
    model.eval()
    x = torch.randint(0, 16, (2, 24))
    with torch.no_grad():
        a = model(x)["logits"]
        xp = x.clone()
        xp[:, 17] = (xp[:, 17] + 3) % 16          # perturb a FUTURE token
        b = model(xp)["logits"]
    earlier = (a[:, :8] - b[:, :8]).abs().max().item()
    assert earlier < 1e-5, \
        f"future token leaked into earlier logits: {earlier:.6f}"


def test_recall_dataset_keys_observable():
    """The recall probe must expose the keys (the old version hid them, making
    the span unlearnable). y[t] must equal (key[t-k]+1) % V for t>=k, and the
    key at t-k is always visible in the input prefix x = keys[:, :-1]."""
    from nfra.benchmark.recall_probe import RecallDataset, V

    k, seq = 4, 40
    ds = RecallDataset(64, seq, k, seed=0)
    keys, targets = ds.toks, ds.targets
    for i in range(64):
        for t in range(k, seq - 1):
            assert targets[i, t] == (keys[i, t - k] + 1) % V, \
                "span target must be the value of the key k back"
            assert 0 <= t - k < seq - 1, "key must be visible in the prefix"
    # padding positions (t < k) are random noise, not the key->value map
    assert any(targets[0, t] != (keys[0, t - k] + 1) % V for t in range(k))


def test_local_pool_matches_sliding_window():
    """Causal local pooling (cortical microcircuit routing context) must equal
    a manual sliding-window mean, including the sequence-start correction and
    the window edge (seq > local_win=64)."""
    from nfra.core.neuro import BrainMLP
    mlp = BrainMLP(dim=8, hidden_mult=2.0)
    torch.manual_seed(0)
    x = torch.randn(2, 70, 8)
    got = mlp._local_pool(x)
    w = 64
    ref = torch.zeros_like(x)
    for t in range(70):
        lo = max(0, t - w + 1)
        ref[:, t] = x[:, lo:t + 1].mean(1)
    assert torch.allclose(got, ref, atol=1e-6)


def test_brain_feature_toggles_forward_backward():
    """Each 'small-but-powerful' brain lever must build, forward, and produce
    finite gradients (CPU smoke)."""
    cfg = NFRAConfig(mode="brain", vocab_size=16, hidden_size=64,
                     num_layers=4, unique_blocks=2, dropout=0.0,
                     local_route=True, div_norm=True, astro=True,
                     theta=True, ach_retain=True, gain_nov=True, lora_rank=8)
    model = NFRAForCausalLM(cfg)
    assert model.layers[0].mlp.local_route is True
    assert model.layers[0].mlp.div_norm is True
    assert model.layers[0].mixer.astro_proj is not None
    assert model.layers[0].mixer.theta_amp is not None
    assert model.layers[0].mixer.ach_retain is True
    assert model.layers[0].mixer.gain_nov is True
    assert model.pass_lora is not None and len(model.pass_lora) == model.depth_passes
    torch.manual_seed(0)
    x = torch.randint(0, 16, (2, 32))
    out = model(x)
    assert out["logits"].shape == (2, 32, 16)
    out["logits"].float().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_brain_feature_toggles_off_by_default():
    cfg = NFRAConfig(mode="brain", vocab_size=16, hidden_size=64,
                     num_layers=2, unique_blocks=2)
    model = NFRAForCausalLM(cfg)
    assert model.layers[0].mlp.local_route is False
    assert model.layers[0].mlp.div_norm is False
    assert model.layers[0].mixer.astro_proj is None
    assert model.layers[0].mixer.theta_amp is None
    assert model.layers[0].mixer.ach_retain is False
    assert model.layers[0].mixer.gain_nov is False
    assert model.pass_lora is None


def test_brain_levers_identity_init():
    """Theta / novelty-gain / LoRA are identity at init (amp=0, gain=0, B=0):
    with shared weights copied across models, enabling them must produce
    logits IDENTICAL to the baseline. (New params consume RNG mid-construction,
    so models can't share init via the same seed — copy weights by name.)"""
    base = NFRAForCausalLM(NFRAConfig(mode="brain", vocab_size=16,
                                      hidden_size=64, num_layers=4,
                                      unique_blocks=2, dropout=0.0))
    on = NFRAForCausalLM(NFRAConfig(mode="brain", vocab_size=16,
                                    hidden_size=64, num_layers=4,
                                    unique_blocks=2, dropout=0.0,
                                    theta=True, gain_nov=True, lora_rank=8))
    on_state = dict(on.named_parameters())
    with torch.no_grad():
        for n, p in base.named_parameters():
            on_state[n].copy_(p)
    # sanity: identity-init params really are at identity
    assert on.layers[0].mixer.theta_amp.abs().max().item() == 0.0
    assert on.layers[0].mixer.gain_nov_w.item() == 0.0
    assert all(l.B.abs().max().item() == 0.0 for l in on.pass_lora)
    base.eval(); on.eval()
    x = torch.randint(0, 16, (2, 32))
    with torch.no_grad():
        a = base(x)["logits"]
        b = on(x)["logits"]
    assert (a - b).abs().max().item() < 1e-5, \
        "theta/gain_nov/lora must be identity at init"
    assert sum(p.numel() for p in on.parameters()) > \
        sum(p.numel() for p in base.parameters()), "lora must add params"


def test_brain_no_future_leak_all_levers():
    """Strict causality must hold even with every future-safe lever enabled
    (theta is t-only, gain_nov is a causal prefix variance, lora is per-token,
    ach_retain is per-token)."""
    torch.manual_seed(0)
    cfg = NFRAConfig(mode="brain", vocab_size=16, hidden_size=64,
                     num_layers=4, unique_blocks=2, dropout=0.0,
                     theta=True, ach_retain=True, gain_nov=True, lora_rank=8)
    model = NFRAForCausalLM(cfg)
    model.eval()
    x = torch.randint(0, 16, (2, 24))
    with torch.no_grad():
        a = model(x)["logits"]
        xp = x.clone()
        xp[:, 17] = (xp[:, 17] + 3) % 16
        b = model(xp)["logits"]
    earlier = (a[:, :8] - b[:, :8]).abs().max().item()
    assert earlier < 1e-5, \
        f"future token leaked into earlier logits: {earlier:.6f}"


def test_energy_allocator_no_graph_leak():
    a = DynamicEnergyBudgetAllocator(4)
    b1 = a()
    b2 = a()
    assert not a.current_budget.requires_grad
    assert b2.requires_grad
    assert torch.isfinite(a.current_budget).all()


def test_cortex_chunked_retention_equals_parallel():
    """The chunked retention mixer (cortex_chunk_size>0) must be EXACTLY
    equivalent to the O(S^2) parallel form — same decayed-QK^T operator, only a
    different summation order — including on sequence lengths that need padding
    to a chunk multiple. This is the load-bearing speed/memory lever of Tier 1:
    it must not move the verified loss even by a hair."""
    from nfra.core.cortex import CortexMixer

    torch.manual_seed(0)
    kwargs = dict(dim=64, n_heads=8, iso_vgate=True, iso_rgate=False, iso_phase=True)
    par = CortexMixer(chunk_size=0, **kwargs)
    chu = CortexMixer(chunk_size=16, **kwargs)
    with torch.no_grad():
        for p, q in zip(par.parameters(), chu.parameters()):
            q.copy_(p)
    par.eval()
    chu.eval()
    for S in (50, 64, 256):
        x = torch.randn(2, S, 64)
        with torch.no_grad():
            a = par(x)
            b = chu(x)
        err = (a - b).abs().max().item()
        assert torch.allclose(a, b, atol=1e-4), f"S={S} max err {err:.2e}"


def test_cortex_chunked_retention_grads_finite():
    """Chunked retention must backprop (the cross-chunk state is a sequential
    recurrence — a common source of vanishing/NaN gradients) and produce the
    same gradient as the parallel form."""
    from nfra.core.cortex import CortexMixer

    torch.manual_seed(0)
    kwargs = dict(dim=64, n_heads=8, iso_vgate=True, iso_rgate=False, iso_phase=True)
    par = CortexMixer(chunk_size=0, **kwargs)
    chu = CortexMixer(chunk_size=16, **kwargs)
    with torch.no_grad():
        for p, q in zip(par.parameters(), chu.parameters()):
            q.copy_(p)
    x = torch.randn(2, 80, 64)
    a = par(x).pow(2).mean()
    b = chu(x).pow(2).mean()
    a.backward()
    b.backward()
    for p, q in zip(par.parameters(), chu.parameters()):
        assert q.grad is not None and torch.isfinite(q.grad).all()
        assert torch.allclose(p.grad, q.grad, atol=1e-3), \
            f"{p.shape}: {p.grad.abs().max().item()} vs {q.grad.abs().max().item()}"


def test_cortex_chunked_model_train_smoke():
    """Full NFRAForCausalLM with the chunked cortex mixer: builds, forwards,
    backward, finite grads (CPU smoke — guards config plumbing)."""
    torch.manual_seed(0)
    cfg = NFRAConfig(
        mode="brain", vocab_size=32, hidden_size=64, num_layers=4,
        use_cortex=True, cortex_chunk_size=16, dropout=0.0,
        iso_gland=True, iso_vgate=True, iso_rgate=False,
        iso_phase=True, iso_exit=True,
    )
    model = NFRAForCausalLM(cfg)
    x = torch.randint(0, 32, (2, 40))
    out = model(x)
    assert out["logits"].shape == (2, 40, 32)
    out["logits"].float().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
