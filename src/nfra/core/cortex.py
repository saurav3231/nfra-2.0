"""
NFRA 3.3b Cortex-Resonance Block.

Diagnosis that motivated this rewrite (see docs/OVERNIGHT_VERIFIED_RESULTS.md sec 9):

  * The 3.3 matrix-state LINEAR-RECURRENCE mixer lost to RetNet by ~0.25 nats
    (nfra 2.296 vs retnet 2.040 @ 5M). Root cause: a linear recurrence
    (accumulate gate*value (x) B, read with C) has NO query-key interaction --
    linear attention is fundamentally weaker per layer on language than
    QK-based mixing, regardless of depth or decay mechanics.
  * It was also SLOWER (4323 tok/s vs retnet 23338). The recurrence ran as a
    [B,H,S,Hd,N] 5-D elementwise grid (cumsum + einsum + permutes) -- dozens of
    tiny memory-bound kernels per block, plus 8 components/block. RetNet is
    fast because each block is ~3 big GEMMs.

The 3.3b block keeps NFRA's identity (multi-scale resonance, neuromodulation,
adaptive exit) but swaps the weak/slow recurrence for a RETENTION mixer: the
multi-scale decayed QK^T attention of RetNet, computed as two O(S^2) matmuls
(parallel training) with an O(1)-per-step recurrent dual for generation.

Preserved identity:
  * RESONANCE   : per-head exponential decays are init across a multi-scale
                  grid (log_decay -5..3 -> long-range 0.99+ down to local) --
                  the "multiple timescales" of the brain/cortex framing.
  * SELECTIVITY : a neuromodulated VALUE gate (ACh -> HOLD) + an output
                  receptance gate (RWKV-style) keep input-dependent routing
                  without breaking the constant-decay parallel form.
  * NEUROMOD    : lean causal hormone gland threads ACh/NE/etc across blocks.
  * ADAPTIVE    : the per-token exit gate (Gumbel ST) stays.
  * RECURRENCE  : retention = parallel-trainable AND recurrent-inferable
                  (O(S^2) train, O(1) per token gen), so length generalization
                  and fast generation survive.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class CortexMixer(nn.Module):
    """Resonance retention mixer (the 3.3b core).

    RetNet-style softmax-free QK retention in NFRA's multi-scale framing.
    Each head owns a learned exponential decay; the mixing operator is the
    decayed causal attention

        y_h = ( (Q_h . K_h^T / sqrt(Hd)) . D_h ) @ V_h,
        D_h[i, j] = gamma_h^(i - j)   for j <= i, else 0
        gamma_h = exp(-exp(log_decay_h))

    computed as two O(S^2) matmuls (GPU-efficient, no per-token scan kernel).
    Selectivity (the linear-attention advantage) is kept cheaply: a learned
    VALUE gate per token (ACh/phase-modulated, like 3.1's write gate) and an
    output receptance gate (RWKV-style) -- both input-dependent but applied as
    plain elementwise multiplies, so the parallel form stays intact.
    """

    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        if dim != self.n_heads * self.head_dim:
            raise ValueError('dim (%d) must be divisible by %d heads'
                             % (dim, self.n_heads))

        # Fused QKV (one GEMM) + value gate + output receptance gate + out.
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.r_proj = nn.Linear(dim, dim, bias=False)
        self.gn = nn.GroupNorm(n_heads, dim)
        self.proj_out = nn.Linear(dim, dim, bias=False)

        # Multi-scale resonance decay: heads spread across timescales. Init
        # exactly like the proven RetNet grid (log_decay -5 -> gamma ~0.993,
        # 0.993^255 ~0.17 survives; log_decay +3 -> gamma ~2e-9, local only)
        # and left trainable.
        self.log_decay = nn.Parameter(torch.linspace(-5.0, 3.0, n_heads))

        # Resonance phase (oscillatory identity, cheap [S, H] gate modulation).
        self.freqs = nn.Parameter(torch.randn(self.n_heads) * 0.5 + 2.0)
        self.phases = nn.Parameter(torch.randn(self.n_heads) * math.pi)

    def decay_mask(self, S: int) -> torch.Tensor:
        """Causal decayed mask D_h[i,j] = gamma_h^(i-j), j <= i. [1,H,S,S].

        The relative-position index and the causal triu are cached per
        (S, device) -- they are constants -- so each forward only pays the
        [H,S,S] exp. Called once per block per step; this was the per-forward
        arange-diff + triu construction that at depth 33 is pure overhead."""
        cache = getattr(self, '_mask_cache', None)
        if cache is None:
            cache = self._mask_cache = {}
        key = (S, self.log_decay.device)
        if key not in cache:
            idx = torch.arange(S, device=self.log_decay.device).float()
            rel = (idx.view(S, 1) - idx.view(1, S)).clamp(min=0.0)   # [S,S]
            causal = torch.triu(torch.ones(S, S, device=idx.device,
                                           dtype=torch.bool), 1)
            cache[key] = (rel, causal)
        rel, causal = cache[key]
        decay = torch.exp(-torch.exp(self.log_decay.view(1, self.n_heads, 1, 1))
                          * rel.view(1, 1, S, S))                # [1,H,S,S]
        return decay.masked_fill(causal, 0.0)

    def forward(self, x: torch.Tensor,
                hormones: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = x.shape
        H, Hd = self.n_heads, self.head_dim

        qkv = self.qkv(x).view(B, S, 3, H, Hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                         # [B,H,S,Hd]

        # Selectivity: input-dependent value gate (ACh -> HOLD, phase-modulated).
        gate = torch.sigmoid(self.gate_proj(x)).view(B, S, H, Hd)
        if hormones is not None:
            ach = hormones[:, :, 0:1]
            gate = gate / (1.0 + 0.5 * ach.unsqueeze(-1))        # high ACh -> HOLD
        pos = torch.arange(S, device=x.device).float()
        phase = torch.sin(2.0 * math.pi * self.freqs[:, None] * pos[None, :] / S
                          + self.phases[:, None])                # [H,S]
        gate = gate * (1.0 + 0.05 * phase.t())[None, :, :, None]
        v = v * gate.permute(0, 2, 1, 3)

        # Retention (parallel form): decayed QK^T attention, no softmax.
        scores = torch.matmul(q * (Hd ** -0.5), k.transpose(-2, -1))
        y = torch.matmul(scores * self.decay_mask(S), v)         # [B,H,S,Hd]
        y = y.permute(0, 2, 1, 3).reshape(B, S, D)
        y = self.gn(y.permute(0, 2, 1)).permute(0, 2, 1)

        # Output receptance gate (RWKV-style selectivity on the read).
        r = torch.sigmoid(self.r_proj(x))
        return self.proj_out(y * r)


class CortexExit(nn.Module):
    """Adaptive resonance exit: a per-token stop gate for depth passes.

    Thinking depth becomes real adaptive compute. During training a Gumbel
    straight-through mask gates whether the token keeps computing; a small
    compute regularizer (weight `reg`) pulls the expected pass count down so
    easy tokens exit early. At inference the hard mask skips further passes.
    """

    def __init__(self, dim: int, reg: float = 1e-3, tau: float = 1.0):
        super().__init__()
        self.gate = nn.Linear(dim, 1, bias=True)
        self.reg = reg
        self.tau = tau
        nn.init.constant_(self.gate.bias, -1.0)             # start "keep going"

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(x))                  # [B, S, 1]

    def sample_mask(self, x: torch.Tensor, hard: bool = False):
        """Return (continue_mask, exit_prob). continue=1 means keep computing."""
        p = self.prob(x)
        if not self.training or hard:
            cont = (p < 0.5).float()
            return cont, p
        # Gumbel straight-through: forward is 0/1, backward flows through p.
        u = torch.rand_like(p)
        g = -torch.log(-torch.log(u + 1e-8) + 1e-8)
        logit = (p + 1e-8).log() - (1.0 - p + 1e-8).log()   # logit(p)
        y = torch.sigmoid((logit + g) / self.tau)
        cont = (y > 0.5).float()
        return cont + (p - p.detach()), p


class CortexMLP(nn.Module):
    """Lean gated MLP (silu(gate) * up -> down) with a cheap NE gain.

    Replaces 3.3's router-based BrainMLP in the Cortex path: the router,
    cortisol pruning and k-WTA all cost extra kernels that (at small width)
    make the block launch-bound. A plain SwiGLU gives the same FLOPs at a
    fraction of the kernels -- this is what makes RetNet's block fast.

    hidden_mult=2.0 deliberately rebalances capacity toward the retention
    mixer: with the 3-proj gated MLP the block becomes (6 mixer + 6 MLP) D^2 --
    the SAME total as RetNet's (4 + 8) D^2 -- so at matched params nfra builds
    the identical dim/depth geometry as retnet (e.g. 112/33 @ 5M) and the
    mixer's extra selectivity carries the differentiator.
    """

    def __init__(self, dim: int, hidden_mult: float = 2.0,
                 k_wta_frac: float = 0.0):
        super().__init__()
        self.dim = dim
        self.k_wta_frac = k_wta_frac
        hidden = int(dim * hidden_mult)
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor,
                hormones: Optional[torch.Tensor] = None) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        hidden = gate * self.up_proj(x)
        if self.k_wta_frac > 0.0:
            k = max(1, int(math.ceil(self.k_wta_frac * hidden.shape[-1])))
            thr = hidden.topk(k, dim=-1).values[..., -1:]
            hidden = hidden * (hidden >= thr).float()
        if hormones is not None:
            ne = hormones[:, :, 1:2]
            hidden = hidden * (1.0 + 0.5 * ne)
        return self.down_proj(hidden)


class CortexNeuromodulator(nn.Module):
    """Lean neuromodulator: causal prefix gland (ACh/NE/DA/5HT/CORT/OX).

    3.3's NeuroModulator also computed a prefix-variance novelty term (a
    second cumsum pair per block). For the launch-bound small-width regime
    that overhead is pure cost -- the prefix-mean gland carries the same
    "slow whole-state mood" identity at half the kernels.
    """

    def __init__(self, dim: int, n_hormones: int = 6, smoothing: float = 0.9):
        super().__init__()
        self.n_hormones = n_hormones
        self.smoothing = smoothing
        self.context_gland = nn.Linear(dim, n_hormones, bias=False)
        self.baseline = nn.Parameter(torch.zeros(n_hormones))

    def forward(self, x: torch.Tensor,
                prev_hormones: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Prefix-pool the LOW-DIM readout, not the full hidden state. Since
        # context_gland is linear, cumsum(gland(x))/cnt == gland(cumsum(x)/cnt)
        # exactly, so this is bit-equivalent to the old prefix_pool(x) -> gland
        # path but scans [B,S,6] instead of [B,S,D] -- the per-block D-wide
        # cumsum was pure overhead at depth 33 and dominated the scan cost.
        proj = self.context_gland(x)                         # [B,S,6]
        cnt = torch.arange(1, x.shape[1] + 1, device=x.device,
                           dtype=proj.dtype).view(1, x.shape[1], 1)
        pooled = proj.cumsum(1) / cnt                        # causal prefix
        raw = torch.sigmoid(pooled + self.baseline.unsqueeze(0))
        if prev_hormones is not None:
            return self.smoothing * prev_hormones + (1.0 - self.smoothing) * raw
        return raw


class NFRA_Cortex_Block(nn.Module):
    """NFRA 3.3b Cortex block: lean neuromodulated retention mixer + gated MLP.

    The whole block is ~3 matmuls for the mixer, 3 for the MLP, plus norms,
    gates and the cheap hormone gland -- RetNet-shaped, so it trains at
    RetNet speed instead of the 5-D-scan 3.3 block (4323 vs 23338 tok/s).
    """

    def __init__(self, dim: int, n_bands: int = 16, dropout: float = 0.1,
                 d_state: int = 8, exit_reg: float = 1e-3,
                 k_wta_frac: float = 0.0, local_route: bool = False,
                 div_norm: bool = False):
        super().__init__()
        self.dim = dim

        self.neuromodulator = CortexNeuromodulator(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.mixer = CortexMixer(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = CortexMLP(dim, k_wta_frac=k_wta_frac)

        self.dropout = nn.Dropout(dropout)
        self.exit_gate = CortexExit(dim, reg=exit_reg)

    def forward(
        self,
        x: torch.Tensor,
        hormones: Optional[torch.Tensor] = None,
        energy_budget: Optional[float] = None,
        exit_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (out, hormones, exit_logit). exit_state unused (kept for a
        compatible signature with depth-pass threading)."""
        B, S, D = x.shape

        hormones = self.neuromodulator(x, prev_hormones=hormones)

        residual = x
        n = self.ln1(x)
        mix_out = self.mixer(n, hormones=hormones)
        x = residual + self.dropout(mix_out)

        residual = x
        n = self.ln2(x)
        n = self.mlp(n, hormones=hormones)
        x = residual + self.dropout(n)

        exit_logit = self.exit_gate.gate(x)
        return x, hormones.detach(), exit_logit

    def exit_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.exit_gate.prob(x)

    def get_sparsity(self) -> float:
        return 0.0

    def reset_stats(self):
        pass
