"""
NFRA 3.3 Cortex-Resonance Block.

Addresses the verified overnight gaps (see docs/OVERNIGHT_VERIFIED_RESULTS.md)
while keeping NFRA's core identity (multi-scale resonance, neuromodulation,
depth-sharing memory, length generalization, graceful recall):

  LOSS  : state width was 16 heads x 24 = 384 vs mamba's d_state x d_inner
          ~5632. CortexMixer carries a MATRIX state  [Hd x N] per head
          (write/read vectors B,C like SSD) -> ~8x the effective state at
          the same parameter cost (the projections were already dim-sized;
          the state was just a cheap view).
  SPEED : the old block ran ~20 small kernels per forward (predictor, gist,
          thalamus, depth_refine, O(H^2) topk+scatter+einsum, per-forward
          attention-mask rebuild). The Cortex block removes the redundant
          linear streams, caches the sliding-window mask, and fuses QKV.
  ADAPT : dopamine "thinking depth" was decorative (depth_f scaling). It is
          now a REAL learnable per-pass EXIT GATE (Gumbel straight-through,
          compute-regularized): easy tokens stop early, hard tokens spend all
          depth_passes. Inference skips passes for exited tokens.

Preserved: depth-sharing (sub-GB memory), recurrence core (length
generalization), local attention (graceful recall), neuromodulation identity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .resonance import parallel_scan_time_varying, ResonanceGuidedLocalAttention
from .neuro import prefix_pool


class CortexMixer(nn.Module):
    """Matrix-state multi-scale resonance mixer (the 3.3 core).

    Each head carries a matrix state  H_t in R^{Hd x N}  instead of a single
    vector. The recurrence per (channel, state-index):

        H_t[hd, n] = alpha_t[hd, n] * H_{t-1}[hd, n] + B_t[n] * gate_t[hd] * value_t[hd]

    alpha_t is the multi-scale resonance decay (per-scale targets like 3.1,
    modulated per token by ACh -> dt selectivity), B/C are learned write/read
    vectors (SSD-style) giving ~N times the effective memory per parameter.

    Scan runs on the flattened [B, H, S, Hd*N] grid so it reuses the existing
    parallel_scan_time_varying (closed-form, torch.compile-friendly).
    """

    def __init__(self, dim: int, n_scales: Tuple[int, ...] = (8, 4, 2, 1),
                 d_state: int = 8):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.head_counts = list(n_scales)
        self.n_heads = sum(self.head_counts) + 1          # +1 router head
        self.head_dim = dim // self.n_heads
        if dim != self.n_heads * self.head_dim:
            raise ValueError('dim (%d) must be divisible by %d heads'
                             % (dim, self.n_heads))

        # Fused gate+value (one GEMM, like 3.1) + write/read vectors + out.
        self.proj_gate_value = nn.Linear(dim, 2 * dim, bias=False)
        self.B_proj = nn.Linear(dim, self.n_heads * d_state, bias=False)
        self.C_proj = nn.Linear(dim, self.n_heads * d_state, bias=False)
        self.proj_out = nn.Linear(dim, dim, bias=False)

        # Input-dependent (selective) per-token, per-head decay rate.
        self.dt_proj = nn.Linear(dim, self.n_heads, bias=True)
        nn.init.zeros_(self.dt_proj.bias)

        # Multi-scale decay targets (like 3.1's [0.90, 0.95, 0.98, 0.995]),
        # one per scale group, over the N state indices.
        targets = [0.90, 0.95, 0.98, 0.995]
        log_a = []
        for n, d in zip(self.head_counts, targets[:len(self.head_counts)]):
            s = (d - 0.85) / 0.15
            log_a.append(torch.full((n, d_state), math.log(s / (1.0 - s))))
        s_router = (0.9 - 0.85) / 0.15
        log_a.append(torch.full((1, d_state),
                                math.log(s_router / (1.0 - s_router))))
        self.log_alpha = nn.Parameter(torch.cat(log_a, dim=0))   # [H, N]

        # Resonance readout: fixed per-head frequencies/phase (no extra
        # state), applied elementwise after the scan.
        self.freqs = nn.Parameter(torch.randn(self.n_heads) * 0.5 + 2.0)
        self.phases = nn.Parameter(torch.randn(self.n_heads) * math.pi)

    def forward(
        self,
        x: torch.Tensor,
        hormones: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        H, Hd, N = self.n_heads, self.head_dim, self.d_state

        gv = self.proj_gate_value(x)                        # [B, S, 2D]
        gate = torch.sigmoid(gv[..., :D]).view(B, S, H, Hd)
        value = gv[..., D:].view(B, S, H, Hd)

        # Selective per-token, per-head decay (ACh modulates retention).
        dt = torch.sigmoid(self.dt_proj(x))                 # [B, S, H] in (0,1)
        if hormones is not None:
            ach = hormones[:, :, 0:1]
            dt = dt / (1.0 + 0.5 * ach)                     # high ACh -> HOLD

        # alpha[b,h,t,hd,n] = exp(log(base_alpha) * dt)  (const over hd)
        # base_alpha = 0.85 + 0.15*sigmoid(log_alpha) — log_alpha is a LOGIT
        # (init: sigmoid(log_alpha)=s so base_alpha lands on the targets).
        # Using the logit directly in exp() would give alpha > 1 for the
        # 0.995 group and clamp every head to the scan ceiling — no multi-scale.
        base_alpha = 0.85 + 0.15 * torch.sigmoid(self.log_alpha)   # [H, N]
        base_log = torch.log(base_alpha.clamp(min=1e-4))           # [H, N] (<0)
        a = torch.exp(base_log.unsqueeze(1) * dt.permute(0, 2, 1).unsqueeze(-1))
        a = a.unsqueeze(3).expand(B, H, S, Hd, N)           # [B,H,S,Hd,N]
        a_flat = a.reshape(B, H, S, Hd * N)                 # (hd,n) inner

        # Write: u = gate * value (outer) B write vectors -> [B,H,S,Hd,N]
        gv_c = (gate * value).unsqueeze(-1)                 # [B,S,H,Hd,1]
        Bt = self.B_proj(x).view(B, S, H, N)                # [B,S,H,N]
        u5 = gv_c * Bt.unsqueeze(2)                         # [B,S,H,Hd,N]
        u_flat = u5.permute(0, 2, 1, 3, 4).reshape(B, H, S, Hd * N)

        h = parallel_scan_time_varying(None, u_flat, a_flat,
                                       alpha_min=0.75, alpha_max=0.9995)
        h = h.reshape(B, H, S, Hd, N).permute(0, 2, 1, 3, 4)  # [B,S,H,Hd,N]

        # Resonance phase modulation (multi-scale identity), elementwise.
        pos = torch.arange(S, device=x.device).float()
        phase = torch.sin(
            2.0 * math.pi * self.freqs[:, None] * pos[None, :] / S
            + self.phases[:, None])                        # [H, S]
        phase_mod = (1.0 + 0.05 * phase.t().view(1, S, H, 1, 1))
        h = h * phase_mod

        # Read: y[hd] = sum_n h[hd,n] * C_t[n]
        Ct = self.C_proj(x).view(B, S, H, N)                # [B,S,H,N]
        y = torch.einsum('bshdn,bshn->bshd', h, Ct)         # [B,S,H,Hd]
        y = y.reshape(B, S, D)

        router_state = y[..., -self.head_dim:]
        router_score = torch.sigmoid(router_state.mean(dim=-1, keepdim=True))

        return self.proj_out(y), router_score


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


class NFRA_Cortex_Block(nn.Module):
    """NFRA 3.3 Cortex block: neuromodulator + matrix-state mixer +
    cached-window local attention + fractal MLP + adaptive exit gate."""

    def __init__(self, dim: int, n_bands: int = 16, dropout: float = 0.1,
                 d_state: int = 8, exit_reg: float = 1e-3,
                 k_wta_frac: float = 0.0, local_route: bool = False,
                 div_norm: bool = False):
        super().__init__()
        self.dim = dim

        from .neuro import NeuroModulator, BrainMLP

        self.neuromodulator = NeuroModulator(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.mixer = CortexMixer(dim, d_state=d_state)
        self.local_attn = ResonanceGuidedLocalAttention(
            dim, n_heads=2, head_dim=max(4, dim // 8), window=64)
        self.attn_gate = nn.Linear(dim, 1, bias=False)

        self.ln2 = nn.LayerNorm(dim)
        self.mlp = BrainMLP(dim, hidden_mult=4.0,
                            k_wta_frac=k_wta_frac,
                            local_route=local_route, div_norm=div_norm)

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
        mix_out, router_score = self.mixer(n, hormones=hormones)
        attn = self.local_attn(n, router_score)
        attn_gate = torch.sigmoid(self.attn_gate(n)).clamp(0.0, 0.5)
        mix_out = mix_out + attn_gate * attn
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
