"""
Resonance-based sparse activation mechanisms for NFRA 3.1
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class ResonanceSignature:
    def __init__(self, frequency: float, phase: float):
        self.frequency = frequency
        self.phase = phase

    def similarity(self, other: 'ResonanceSignature') -> float:
        freq_diff = abs(self.frequency - other.frequency)
        phase_diff = abs(self.phase - other.phase)
        return math.exp(-freq_diff) * math.cos(phase_diff)


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

    Each band is a first-order recurrence: h_t = alpha * h_{t-1} + content_t
    with learned frequency modulation and per-band output gating.

    O(S * D) complexity. Energy budget scales the recurrence decay (lower budget
    = faster decay = less context retained).
    """

    def __init__(self, dim: int, n_bands: int = 4, max_positions: int = 2048):
        super().__init__()
        self.dim = dim
        self.n_bands = n_bands
        self.band_dim = dim // n_bands
        assert dim % n_bands == 0, f"dim ({dim}) must be divisible by n_bands ({n_bands})"

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

        h = torch.zeros_like(content[:, :, 0:1])
        outputs = []
        for t in range(S):
            c = content[:, :, t:t+1]
            h = h * decay + c
            outputs.append(h)

        out = torch.cat(outputs, dim=2)
        band_weights = torch.sigmoid(self.band_gate_logits)
        out = out * band_weights.view(1, self.n_bands, 1, 1)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, D)
        out = out * input_gate

        return self.proj_out(out)


class ParallelGatedRecurrence(nn.Module):
    """
    Multi-head gated recurrence for high GPU utilization.

    Each head: h_t = α * h_{t-1} + gate_t ⊙ v_t
    α is learned per head (position-independent).
    gate_t = sigmoid(W_g @ x_t) is data-dependent.

    The scan loop is O(S) but each step is just a saxpy on (B, H, head_dim).
    At B=8, H=16, head_dim=48 this is ~6K elements/step — negligible vs matmuls.
    """

    def __init__(self, dim: int, n_heads: int = 16):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0

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

        h = torch.zeros(B, H, 1, Hd, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(S):
            c = value[:, :, t:t+1]
            g = gate[:, :, t:t+1]
            h = h * alpha + g * c
            outputs.append(h)

        out = torch.cat(outputs, dim=2)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, D)
        return self.proj_out(out)


class MultiScaleGatedRecurrence(nn.Module):
    """
    Multi-scale gated recurrence with hierarchical decay rates.

    Heads organized as [8, 4, 2, 1] + 1 router = 16 total.
    Each level has a different decay rate (temporal resolution):
      Level 0: 8 heads,  α ≈ 0.90  — fast, high temporal resolution
      Level 1: 4 heads,  α ≈ 0.95  — medium
      Level 2: 2 heads,  α ≈ 0.98  — slow
      Level 3: 1 head,   α ≈ 0.995 — long-term memory
      Router:  1 head,   α ≈ 0.90  — produces per-position gating scores

    Total heads is self-computed from head_counts. The caller must ensure
    dim is divisible by total_heads (default 16 for dim=768).
    """

    def __init__(self, dim: int, n_heads: Optional[int] = 16):
        super().__init__()
        self.head_counts = [8, 4, 2, 1]
        self.n_heads = sum(self.head_counts) + 1
        self.head_dim = dim // self.n_heads
        assert dim == self.n_heads * self.head_dim, (
            f"dim ({dim}) must be divisible by {self.n_heads} "
            f"(sum(head_counts) + 1 router)"
        )

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

        h = torch.zeros(B, H, 1, Hd, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(S):
            c = value[:, :, t:t+1]
            g = gate[:, :, t:t+1]
            h = h * alpha + g * c
            outputs.append(h)

        out = torch.cat(outputs, dim=2)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, D)

        router_state = out[..., -self.head_dim:]
        router_score = torch.sigmoid(router_state.mean(dim=-1, keepdim=True))

        return self.proj_out(out), router_score


class ResonanceGuidedLocalAttention(nn.Module):
    """
    Fixed-window local attention gated by resonance scores — GPU-efficient.

    Uses a pre-built causal sliding window mask (window=64) and learns a
    per-position gate from the router score. When the gate ≈ 0, attention
    contributes nothing — the model learns when to attend.

    No per-position loops, no GPU-CPU syncs. Single batched matmul.
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

        positions = torch.arange(S, device=x.device)
        rel_dist = positions[:, None] - positions[None, :]
        local_mask = (rel_dist.abs() <= self.window // 2) & (rel_dist >= 0)
        mask = local_mask.float()
        mask = mask.masked_fill(mask == 0.0, float('-inf'))
        mask = mask.masked_fill(mask == 1.0, 0.0)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (Hd ** 0.5)
        scores = scores + mask[None, None, :, :]
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, S, H * Hd)
        out = self.o_proj(out)

        gate = torch.sigmoid(router_score + self.router_bias)
        return out * gate
