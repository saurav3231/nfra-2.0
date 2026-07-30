"""
Neuro-inspired components for NFRA Brain — Cortical Column Architecture

Six principles from human brain neuroscience translated into GPU-efficient code:

1. NEUROMODULATION : 6-channel hormone system that dynamically alters computation
2. CORTICAL COLUMN : 4-layer microcircuit (L4/L2/3/L5/L6) with distinct roles
3. THALAMIC GATING : Context-dependent fast/slow dual-pathway routing
4. OSCILLATORY PHASE : Frequency-based head coordination in recurrence
5. HOMEOSTATIC ENERGY : Cortisol-driven adaptive sparsity
6. PREDICTION ERROR : Dopaminergic surprise signal drives plasticity-like routing
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class NeuroModulator(nn.Module):
    """
    Emotional-cognitive neuromodulatory system for NFRA Brain.

    Produces a 6-channel hormone vector from the model's internal state.
    Hormones represent the "emotional state" of the network and persist
    across layers and timesteps, creating a global brain state.

    Channels:
      0  ACh  (Acetylcholine) — memory encoding, detail focus
      1  NE   (Norepinephrine) — gain, arousal, vigilance
      2  DA   (Dopamine) — routing focus, motivation, effort
      3  5HT  (Serotonin) — temporal integration, patience
      4  CORT (Cortisol) — energy conservation, stress
      5  OX   (Oxytocin) — cross-layer coherence, alignment

    Each hormone alters the network's computation:
      ACh → recurrence decay     NE  → MLP gain
      DA  → thinking depth      5HT → integration window
      CORT→ sparsity threshold   OX → layer coherence
    """

    def __init__(self, dim: int, n_hormones: int = 6, smoothing: float = 0.9):
        super().__init__()
        self.n_hormones = n_hormones
        self.smoothing = smoothing

        self.context_gland = nn.Linear(dim, n_hormones, bias=False)
        self.novelty_gland = nn.Linear(1, n_hormones, bias=False)
        self.baseline = nn.Parameter(torch.zeros(n_hormones))

    def forward(
        self,
        x: torch.Tensor,
        prev_hormones: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, D = x.shape

        pooled = x.mean(dim=1)
        delta = self.context_gland(pooled)

        novelty = x.var(dim=1).mean(dim=-1, keepdim=True)
        delta = delta + self.novelty_gland(novelty)

        raw = torch.sigmoid(delta + self.baseline.unsqueeze(0))

        if prev_hormones is not None:
            hormones = self.smoothing * prev_hormones + (1.0 - self.smoothing) * raw
        else:
            hormones = raw

        return hormones


class ThalamicGate(nn.Module):
    """
    Thalamus-inspired dynamic gating for dual processing pathways.

    The thalamus doesn't just relay signals — it dynamically gates them.
    Familiar/routine patterns take a fast direct pathway (L4→L5 bypass).
    Novel/uncertain patterns take a full cortical pathway (L4→L2/3→L5).

    For each token, a thalamic gate computes:
      directness = f(uncertainty, familiarity, ACh)

    Output = directness * fast_path(x) + (1 - directness) * slow_path(x)

    The fast path is a lightweight linear transform (energy-efficient).
    The slow path is the full recurrence mixer (high-capacity).

    Novelty: no existing architecture has learned per-token pathway
    selection based on internal state.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.uncertainty_proj = nn.Linear(dim, 1, bias=False)
        self.fast_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        hormones: torch.Tensor,
        slow_out: torch.Tensor,
    ) -> torch.Tensor:
        B, S, D = x.shape

        uncertainty = torch.sigmoid(self.uncertainty_proj(x))

        ach = hormones[:, 0:1].view(B, 1, 1)
        ne = hormones[:, 1:2].view(B, 1, 1)

        directness = (1.0 - uncertainty) * (1.0 - 0.5 * ach)

        fast = self.fast_proj(x)

        output = directness * fast + (1.0 - directness) * slow_out
        output = output * (1.0 + 0.2 * ne)

        return output


class TemporalGridEncoder(nn.Module):
    """
    Grid-cell-inspired multi-oscillator position encoding.

    Uses 8 oscillators at learnable frequencies to create a unique
    temporal signature for each position. This is the brain's way of
    encoding time — grid cells in entorhinal cortex fire at different
    frequencies, creating a combinatorial code for temporal position.

    Unlike RoPE (rotary) or absolute position embeddings:
    - Frequencies are LEARNED, not fixed
    - Encoding is MULTI-SCALE (fast + slow oscillators)
    - The code can be directly added to the recurrence input
    - The network learns which temporal scales matter for the task
    """

    def __init__(self, dim: int, n_oscillators: int = 8):
        super().__init__()
        self.n_osc = n_oscillators
        self.log_freq = nn.Parameter(torch.randn(n_oscillators))
        self.phase = nn.Parameter(torch.randn(n_oscillators) * math.pi)
        self.proj = nn.Linear(2 * n_oscillators, dim, bias=False)

    def forward(self, S: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(S, device=device).float() / S
        freqs = 2.0 * math.pi * torch.exp(self.log_freq.clamp(max=10.0))
        angle = freqs[:, None] * t[None, :] + self.phase[:, None]
        code = torch.stack([angle.sin(), angle.cos()], dim=-1)
        return self.proj(code.reshape(S, 2 * self.n_osc))


class BrainMixer(nn.Module):
    """
    Emotionally-modulated multi-scale recurrence with efficient cross-head mixing.

    Heads organized as [8, 4, 2, 1] + 1 router = 16 total, each level with
    a different temporal resolution. ACh modulates decay rates dynamically.

    Key improvements over MultiScaleGatedRecurrence:
    1. ACh-modulated decay (emotional memory control)
    2. Efficient top-K cross-head communication (not full O(H²) per step)
    3. Phase-based modulation (oscillatory coordination)

    The cross-head term uses a sparse approximation: only the top-K most
    coherent heads gate each other, computed once before the scan.
    """

    def __init__(self, dim: int, n_heads: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.head_counts = [8, 4, 2, 1]
        self.n_heads = sum(self.head_counts) + 1
        self.head_dim = dim // self.n_heads
        if dim != self.n_heads * self.head_dim:
            raise ValueError(f"dim ({dim}) must be divisible by {self.n_heads}")

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

        self.frequencies = nn.Parameter(torch.randn(self.n_heads) * 0.5 + 2.0)
        self.phases = nn.Parameter(torch.randn(self.n_heads) * math.pi)
        self.phase_gate_raw = nn.Parameter(torch.randn(self.n_heads, self.n_heads) * 0.1)

        self.grid_encoder = TemporalGridEncoder(dim)

        self._dim = dim

    def forward(
        self,
        x: torch.Tensor,
        hormones: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        H, Hd = self.n_heads, self.head_dim

        grid_code = self.grid_encoder(S, x.device)
        x_grid = x + grid_code.unsqueeze(0)

        gate = torch.sigmoid(self.proj_gate(x_grid))
        value = self.proj_value(x_grid)

        gate = gate.view(B, S, H, Hd).permute(0, 2, 1, 3)
        value = value.view(B, S, H, Hd).permute(0, 2, 1, 3)

        alpha = torch.sigmoid(self.log_alpha)
        alpha = alpha.view(1, H, 1, Hd)

        if hormones is not None:
            ach = hormones[:, 0:1].view(B, 1, 1, 1)
            alpha = 0.8 + 0.19 * alpha + 0.1 * ach * alpha
            alpha = alpha.clamp(max=0.99)

        # Phase signal and sparse coherence (recomputed each forward for gradient flow)
        positions = torch.arange(S, device=x.device)
        phases = (
            2.0 * math.pi * self.frequencies[:, None] * positions[None, :] / S
            + self.phases[:, None]
        )
        phase_signal = torch.sin(phases)

        phase_mean = torch.cos(
            phases[:, None, :] - phases[None, :, :]
        ).mean(dim=-1)
        coherence = torch.sigmoid(self.phase_gate_raw) * torch.sigmoid(phase_mean * 5.0)

        topk = max(2, H // 4)
        _, topk_idx = coherence.topk(topk, dim=-1)
        sparse_coherence = torch.zeros_like(coherence)
        sparse_coherence.scatter_(-1, topk_idx, coherence.gather(-1, topk_idx))

        # Preallocate output buffer, avoid list+cat, replace einsum with bmm
        out = torch.empty_like(value)
        h = torch.zeros(B, H, 1, Hd, device=x.device, dtype=x.dtype)
        for t in range(S):
            g_t = gate[:, :, t:t+1]
            v_t = value[:, :, t:t+1]
            h_3d = h.squeeze(2)
            cross = torch.bmm(sparse_coherence.unsqueeze(0).expand(B, -1, -1), h_3d).unsqueeze(2)
            h = h * alpha + g_t * v_t + 0.05 * cross
            phase_amp = phase_signal[:, t].view(1, H, 1, 1)
            h = h * (1.0 + 0.05 * phase_amp)
            out[:, :, t:t+1] = h

        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, D)

        router_state = out[..., -self.head_dim:]
        router_score = torch.sigmoid(router_state.mean(dim=-1, keepdim=True))

        return self.proj_out(out), router_score


class BrainMLP(nn.Module):
    """
    Emotionally-modulated FractalSwiGLU with dynamic capacity scaling.

    The hidden space is partitioned into [8, 4, 2, 1] groups (15 total)
    across 4 fractal scales, computed adaptively from dim.

    Emotional modulation:
      NE  ↑ → higher activation gain (arousal)
      DA  ↑ → sharper routing focus (motivation)
      CORT↑ → group pruning (stress/energy saving)

    When cortisol is high, low-weighted groups are zeroed — the model
    dynamically shrinks its effective capacity under metabolic stress.
    This mirrors the brain's energy conservation under threat.
    """

    def __init__(self, dim: int, hidden_mult: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * hidden_mult)

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

        n_groups_per_level = [8, 4, 2, 1]
        n_total_groups = sum(n_groups_per_level)
        n_total = 0
        group_slices = []
        for lvl, n_g in enumerate(n_groups_per_level):
            group_dim = max(1, (hidden_dim - n_total) // (n_total_groups - len(group_slices)))
            level_dim = group_dim * n_g
            group_slices.append((n_total, n_total + level_dim))
            n_total += level_dim
        if n_total != hidden_dim:
            group_slices[-1] = (group_slices[-1][0], hidden_dim)

        routing_idx = torch.zeros(hidden_dim, dtype=torch.long)
        group_idx = 0
        for lvl, n_g in enumerate(n_groups_per_level):
            start, end = group_slices[lvl]
            group_dim = (end - start) // n_g
            for g in range(n_g):
                g_start = start + g * group_dim
                g_end = g_start + group_dim
                routing_idx[g_start:g_end] = group_idx
                group_idx += 1
        self.register_buffer('routing_idx', routing_idx, persistent=False)

        self.router = nn.Sequential(
            nn.Linear(dim, 64, bias=False),
            nn.GELU(),
            nn.Linear(64, n_total_groups, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        hormones: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, D = x.shape

        gate = nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up

        center = hidden.mean(dim=-1, keepdim=True)
        hidden = hidden * torch.sigmoid(hidden - center)

        pooled = x.mean(dim=1, keepdim=True)
        routing = self.router(pooled)

        if hormones is not None:
            da_temp = hormones[:, 2:3].view(B, 1, 1)
            routing = routing / (da_temp + 0.1)

        routing = torch.softmax(routing, dim=-1)

        if hormones is not None:
            cort = hormones[:, 4:5].view(B, 1, 1)
            threshold = 0.5 / routing.shape[-1] * (1.0 + cort)
            routing = routing * (routing >= threshold).float()
            denom = routing.sum(dim=-1, keepdim=True)
            routing = routing / (denom + (denom == 0).float())

        weight_map = routing[:, :, self.routing_idx]
        hidden = hidden * weight_map

        if hormones is not None:
            ne = hormones[:, 1:2].view(B, 1, 1)
            hidden = hidden * (1.0 + 0.5 * ne)

        return self.down_proj(hidden)
