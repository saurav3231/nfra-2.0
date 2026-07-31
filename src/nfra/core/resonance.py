"""
Resonance-based sparse activation mechanisms for NFRA 3.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


def parallel_gated_scan(
    gate: Optional[torch.Tensor],
    value: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    Closed-form parallel scan for h_t = alpha * h_{t-1} + gate_t * value_t.

    This replaces the Python-level sequential loop (O(S) iterations of small
    GPU ops) with a single vectorized cumsum in log-space:

        h_t = alpha^t * sum_{k<=t} alpha^{-k} * gate_k * value_k

    Args:
        gate:   [B, H, S, Hd] input gates in (0, 1), or None for no gating
        value:  [B, H, S, Hd] values
        alpha:  [1, H, 1, Hd] per-head decay in (0, 1)

    Returns:
        [B, H, S, Hd] hidden states for every timestep (parallel, O(S) memory)
    """
    u = value if gate is None else gate * value
    w = torch.log(alpha.clamp(min=0.85, max=0.99))   # w <= -0.01 for numerical safety
    S = u.shape[2]
    pos = torch.arange(1, S + 1, device=u.device).float().view(1, 1, S, 1)
    decay_inv = torch.exp(-w * pos)                   # alpha^{-t}  (<= 1e18 at S=256)
    decay_fwd = torch.exp(w * pos)                    # alpha^{t}
    return decay_fwd * torch.cumsum(decay_inv * u, dim=2)


class ResonanceSignature:
    def __init__(self, frequency: float, phase: float):
        self.frequency = frequency
        self.phase = phase

    def similarity(self, other: 'ResonanceSignature') -> float:
        freq_diff = abs(self.frequency - other.frequency)
        phase_diff = min(abs(self.phase - other.phase), math.pi)
        return math.exp(-freq_diff) * (1.0 + math.cos(phase_diff)) / 2.0


class ResonanceRouter(nn.Module):
    def __init__(self, input_dim: int, num_paths: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.num_paths = num_paths
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_paths),
            nn.Sigmoid()
        )
        self.register_parameter(
            'frequencies',
            nn.Parameter(torch.randn(num_paths) * 0.1 + 1.0)
        )

    def forward(self, x: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
        routing_scores = self.router(x)
        mask = (routing_scores > threshold).float()
        mask = mask + (routing_scores - routing_scores.detach())
        return mask


class SpikeResonanceLayer(nn.Module):
    def __init__(self, dim: int, threshold: float = 0.5):
        super().__init__()
        self.dim = dim
        self.threshold = threshold
        self.weight = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(dim))
        self.freq = nn.Parameter(torch.ones(dim) * 0.5)
        self.phase = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, time: float = 0.0) -> torch.Tensor:
        resonance = torch.sin(2 * math.pi * self.freq * time + self.phase)
        resonance_mask = (resonance.abs() > self.threshold).float()
        output = torch.matmul(x, self.weight * resonance_mask.unsqueeze(0)) + self.bias
        return output * resonance_mask.unsqueeze(0)


class CausalResonanceMixer(nn.Module):
    """
    Multi-band gated linear recurrence with position-modulated resonance.

    Each band is a first-order recurrence: h_t = decay * h_{t-1} + content_t
    with learned frequency modulation and per-band output gating.

    Uses a preallocated output buffer to avoid Python list.append + cat overhead,
    making torch.compile fusion more effective.
    """

    def __init__(self, dim: int, n_bands: int = 4, max_positions: int = 2048):
        super().__init__()
        self.dim = dim
        self.n_bands = n_bands
        self.band_dim = dim // n_bands
        if dim % n_bands != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_bands ({n_bands})")

        self.proj_in = nn.Linear(dim, dim * 2)
        self.proj_out = nn.Linear(dim, dim)

        raw_decays = torch.empty(n_bands)
        nn.init.normal_(raw_decays, mean=0.0, std=0.5)
        self.log_decay = nn.Parameter(raw_decays)

        pos_freqs = torch.empty(n_bands)
        nn.init.uniform_(pos_freqs, 0.5, 8.0)
        self.position_freqs = nn.Parameter(pos_freqs)

        self.position_phases = nn.Parameter(torch.zeros(n_bands))

        self.band_gate_logits = nn.Parameter(torch.zeros(n_bands))

        self.register_buffer('freq_scale', torch.tensor(2.0 * math.pi), persistent=False)

    def forward(self, x: torch.Tensor, energy_budget: Optional[float] = None) -> torch.Tensor:
        B, S, D = x.shape

        gated = self.proj_in(x)
        input_gate, content = gated.chunk(2, dim=-1)
        input_gate = torch.sigmoid(input_gate)

        content = content.view(B, S, self.n_bands, self.band_dim)
        content = content.permute(0, 2, 1, 3)

        decay = torch.sigmoid(self.log_decay)
        decay = decay.view(1, self.n_bands, 1, 1)

        if energy_budget is not None:
            scale = 1.5 - 0.5 * energy_budget
            decay = decay ** scale

        positions = torch.arange(S, device=x.device).float()
        pos_mod = torch.sin(
            self.freq_scale * self.position_freqs[:, None] * positions[None, :] / S
            + self.position_phases[:, None]
        )
        pos_mod = pos_mod.view(1, self.n_bands, S, 1)
        content = content * (1.0 + 0.1 * pos_mod)

        # Parallel closed-form scan (no Python loop)
        out = parallel_gated_scan(None, content, decay)

        band_weights = torch.sigmoid(self.band_gate_logits)
        out = out * band_weights.view(1, self.n_bands, 1, 1)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, D)
        out = out * input_gate

        return self.proj_out(out)


class ParallelGatedRecurrence(nn.Module):
    """
    Multi-head gated recurrence with preallocated buffer for torch.compile fusion.
    """

    def __init__(self, dim: int, n_heads: int = 16):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")

        self.proj_gate = nn.Linear(dim, dim, bias=False)
        self.proj_value = nn.Linear(dim, dim, bias=False)
        self.proj_out = nn.Linear(dim, dim, bias=False)

        self.log_alpha = nn.Parameter(torch.full((n_heads, self.head_dim), 1.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H, Hd = self.n_heads, self.head_dim

        gate = torch.sigmoid(self.proj_gate(x))
        value = self.proj_value(x)

        gate = gate.view(B, S, H, Hd).permute(0, 2, 1, 3)
        value = value.view(B, S, H, Hd).permute(0, 2, 1, 3)

        alpha = torch.sigmoid(self.log_alpha)
        alpha = 0.8 + 0.19 * alpha
        alpha = alpha.view(1, H, 1, Hd)

        out = parallel_gated_scan(gate, value, alpha)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, D)
        return self.proj_out(out)


class MultiScaleGatedRecurrence(nn.Module):
    """
    Multi-scale gated recurrence with hierarchical decay rates.

    Head counts auto-computed to evenly divide dim.
    Uses preallocated buffer for torch.compile fusion.
    """

    def __init__(self, dim: int, n_heads: Optional[int] = 16):
        super().__init__()
        target = n_heads - 1 if (n_heads is not None and n_heads > 1) else 15
        ratios = [8, 4, 2, 1]
        ratio_sum = sum(ratios)
        # Find head counts that divide dim and approximate target
        best_hc, best_total = None, 0
        for total_recur in range(max(target, 4), 0, -1):
            hc = [max(1, total_recur * r // ratio_sum) for r in ratios]
            total = sum(hc) + 1
            if dim % total == 0:
                best_hc, best_total = hc, total
                break
        if best_hc is None:
            # Fallback: use largest divisor of dim ≤ target+1
            for total in range(min(dim // 32, target + 1), 3, -1):
                if dim % total == 0:
                    n_rec = total - 1
                    hc = [max(1, n_rec * r // ratio_sum) for r in ratios]
                    diff = n_rec - sum(hc)
                    if diff > 0:
                        hc[0] += diff
                    best_hc, best_total = hc, total
                    break
        if best_hc is None:
            best_hc, best_total = [1, 1, 1, 0], 4
        self.head_counts = best_hc
        self.n_heads = best_total
        self.head_dim = dim // self.n_heads

        self.proj_gate = nn.Linear(dim, dim, bias=False)
        self.proj_value = nn.Linear(dim, dim, bias=False)
        self.proj_out = nn.Linear(dim, dim, bias=False)

        log_alphas = []
        target_decays = [0.90, 0.95, 0.98, 0.995]
        for lvl, (n, d) in enumerate(zip(self.head_counts, target_decays)):
            log_a = torch.full((n, self.head_dim), math.log(d / (1.0 - d)))
            log_alphas.append(log_a)
        log_alphas.append(torch.full((1, self.head_dim), math.log(0.9 / 0.1)))
        self.log_alpha = nn.Parameter(torch.cat(log_alphas, dim=0))

        self._dim = dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        H, Hd = self.n_heads, self.head_dim

        gate = torch.sigmoid(self.proj_gate(x))
        value = self.proj_value(x)

        gate = gate.view(B, S, H, Hd).permute(0, 2, 1, 3)
        value = value.view(B, S, H, Hd).permute(0, 2, 1, 3)

        alpha = torch.sigmoid(self.log_alpha)
        alpha = alpha.view(1, H, 1, Hd)

        out = parallel_gated_scan(gate, value, alpha)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, D)

        router_state = out[..., -self.head_dim:]
        router_score = torch.sigmoid(router_state.mean(dim=-1, keepdim=True))

        return self.proj_out(out), router_score


class ResonanceGuidedLocalAttention(nn.Module):
    """
    Fixed-window local attention gated by resonance scores.

    Uses F.scaled_dot_product_attention (FlashAttention on compatible GPUs)
    and caches the sliding-window mask to avoid recomputation.
    """

    def __init__(self, dim: int, n_heads: int = 2, head_dim: int = 64, window: int = 64):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.window = window

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.router_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self, x: torch.Tensor, router_score: torch.Tensor
    ) -> torch.Tensor:
        B, S, D = x.shape
        H, Hd = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, S, H, Hd).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, Hd).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, Hd).transpose(1, 2)

        # Build causal sliding-window mask (recomputed each forward — O(S²) but small)
        positions = torch.arange(S, device=x.device)
        rel_dist = positions[:, None] - positions[None, :]
        local_mask = (rel_dist.abs() <= self.window // 2) & (rel_dist >= 0)
        attn_mask = local_mask.float()
        attn_mask = attn_mask.masked_fill(attn_mask == 0.0, float('-inf'))
        attn_mask = attn_mask.masked_fill(attn_mask == 1.0, 0.0)

        # FlashAttention via scaled_dot_product_attention
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / (Hd ** 0.5)
            scores = scores + attn_mask[None, None, :, :]
            attn_weights = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, S, H * Hd)
        out = self.o_proj(out)

        gate = torch.sigmoid(router_score + self.router_bias)
        return out * gate
