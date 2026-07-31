"""
Fractal Resonance Block (FRB) - Core computational unit of NFRA 3.1

Pre-LN with CausalResonanceMixer + FractalGatedMLP.
Energy budget controls both mixer decay and MLP sparsity.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import math


class FractalGatedMLP(nn.Module):
    """
    SwiGLU-style gated MLP with hierarchical fractal routing.

    Gate + Up → SwiGLU → hierarchical sub-expert routing → Down.
    Energy budget controls number of active sub-experts and threshold.
    """

    def __init__(self, dim: int, hidden_mult: float = 8.0 / 3.0,
                 scales: Optional[List[int]] = None):
        super().__init__()
        self.dim = dim
        self.scales = [1, 2] if scales is None else scales
        hidden_dim = int(dim * hidden_mult)

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

        self.sub_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // s, bias=False),
                nn.GELU(),
                nn.Linear(hidden_dim // s, hidden_dim, bias=False),
            )
            for s in scales
        ])

        self.coarse_router = nn.Linear(dim, 1, bias=False)
        self.fine_router = nn.Linear(dim, len(scales), bias=False)

    def forward(
        self, x: torch.Tensor, energy_budget: Optional[float] = None
    ) -> torch.Tensor:
        B, S, D = x.shape

        gate = self.gate_proj(x)
        gate = nn.functional.silu(gate)
        up = self.up_proj(x)
        hidden = gate * up

        pooled = x.mean(dim=1, keepdim=True)

        coarse = torch.sigmoid(self.coarse_router(pooled))
        fine = torch.softmax(self.fine_router(pooled), dim=-1)

        if energy_budget is not None:
            n_active = max(1, int(energy_budget * len(self.scales)))
            keep_mask = torch.zeros(len(self.scales), device=x.device)
            keep_mask[:n_active] = 1.0

            threshold = (1.0 - energy_budget) * 0.5
            fine = fine * keep_mask.unsqueeze(0).unsqueeze(0)
            fine = fine * (fine > threshold).float()
            denom = fine.sum(dim=-1, keepdim=True)
            fine = fine / (denom + (denom == 0).float())
            coarse = coarse * energy_budget
        else:
            coarse = coarse * 1.0

        output = torch.zeros_like(hidden)
        coarse_w = coarse.squeeze(1).squeeze(-1)
        fine_w = fine.squeeze(1)
        # Record which experts are meaningfully active as a tensor of flags
        # instead of a Python int: `if w.mean() > 0.01` on a CUDA tensor is an
        # implicit .item() sync (one GPU->CPU stall per expert per forward).
        # The flags are summed lazily in get_sparsity, keeping the forward loop
        # free of device syncs.
        active = []
        for i, expert in enumerate(self.sub_experts):
            w = coarse_w * fine_w[:, i]
            w = w.view(-1, 1, 1)
            active.append((w.mean() > 0.01).reshape(-1))
            output = output + w * expert(hidden)

        self._n_active_flags = torch.cat(active) if active else None
        return self.down_proj(output)


class FractalResonanceBlock(nn.Module):
    """
    NFRA 3.1 Core Block: Pre-LN with CausalResonanceMixer + FractalGatedMLP.

    Architecture:
        x → LN → Mixer → dropout → + residual
          → LN → MLP → dropout → + residual

    Energy budget controls both mixer decay rate and MLP sparsity.
    """

    def __init__(
        self,
        dim: int,
        scales: Optional[List[int]] = None,
        n_bands: int = 4,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.scales = [1, 2] if scales is None else scales
        self.n_bands = n_bands
        self.use_residual = use_residual

        from .resonance import CausalResonanceMixer

        self.ln1 = nn.LayerNorm(dim)
        self.mixer = CausalResonanceMixer(dim, n_bands)

        self.ln2 = nn.LayerNorm(dim)
        self.mlp = FractalGatedMLP(dim, scales=self.scales)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('activation_count', torch.zeros(1))
        self.register_buffer('total_count', torch.zeros(1))
        self._current_budget = 1.0

    def forward(
        self,
        x: torch.Tensor,
        energy_budget: Optional[float] = None,
        return_stats: bool = False
    ) -> torch.Tensor:
        budget = energy_budget if energy_budget is not None else 1.0
        self._current_budget = budget

        residual = x
        x = self.ln1(x)
        x = self.mixer(x, energy_budget=budget)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.ln2(x)
        x = self.mlp(x, energy_budget=budget)
        x = self.dropout(x)
        x = residual + x

        flags = getattr(self.mlp, '_n_active_flags', None)
        if flags is not None:
            self.activation_count += flags.float().sum()
        total_experts = len(self.mlp.scales)
        self.total_count += total_experts

        if return_stats:
            return x, {
                'sparsity': self.get_sparsity(),
                'budget': budget,
            }

        return x

    def get_sparsity(self) -> float:
        tc = self.total_count.item()
        if tc == 0:
            return 0.0
        return 1.0 - (self.activation_count.item() / max(tc, 1))

    def reset_stats(self):
        self.activation_count.zero_()
        self.total_count.zero_()


class SwiGLU_MLP(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * hidden_mult)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class FractalSwiGLU(nn.Module):
    """
    Fractal-structured SwiGLU with hierarchical group gating.

    Standard SwiGLU matmul shapes: gate_proj(768→3072), up_proj(768→3072),
    down_proj(3072→768). Hidden dim (3072) is partitioned into 15 groups
    across 4 fractal scales:

      Level 0: 8 groups × 128 = 1024  (finest, most independent channels)
      Level 1: 4 groups × 128 = 512
      Level 2: 2 groups × 256 = 512
      Level 3: 1 group  × 1024 = 1024 (coarsest, single broad channel)
      Total: 1024+512+512+1024 = 3072

    A tiny router produces per-token gating weights for each of the 15 groups.
    The routing is applied as element-wise multiplication after SwiGLU:

      hidden = SiLU(gate(x)) * up(x)
      hidden = hidden * expand(routing_weights)
      output = down_proj(hidden)

    Key novelty: the fractal hierarchy creates a learned prior that the network
    can use — fine-grained groups can specialize, coarse groups provide
    broad transformations. The router dynamically emphasizes different scales
    per token. No other architecture structures the MLP hidden space this way.

    All compute in 3 large matmuls + 1 tiny router. No sequential loops.
    """

    def __init__(self, dim: int, hidden_mult: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * hidden_mult)

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

        self.n_groups_per_level = [8, 4, 2, 1]
        n_total_groups = sum(self.n_groups_per_level)

        # Dynamic group slicing to support any hidden_dim
        n_total = 0
        group_slices = []
        remaining_groups = n_total_groups
        for lvl, n_g in enumerate(self.n_groups_per_level):
            group_dim = max(1, (hidden_dim - n_total) // remaining_groups)
            level_dim = group_dim * n_g
            group_slices.append((n_total, n_total + level_dim))
            n_total += level_dim
            remaining_groups -= n_g
        last_end = group_slices[-1][1]
        if last_end < hidden_dim:
            group_slices[-1] = (group_slices[-1][0], hidden_dim)
        elif last_end > hidden_dim:
            group_slices[-1] = (group_slices[-1][0], hidden_dim)

        routing_idx = torch.zeros(hidden_dim, dtype=torch.long)
        group_idx = 0
        for lvl, n_groups in enumerate(self.n_groups_per_level):
            start, end = group_slices[lvl]
            group_dim = max(1, (end - start) // n_groups)
            for g in range(n_groups):
                g_start = start + g * group_dim
                g_end = min(g_start + group_dim, hidden_dim)
                if g_start >= hidden_dim:
                    break
                routing_idx[g_start:g_end] = group_idx
                group_idx += 1
        self.register_buffer('routing_idx', routing_idx, persistent=False)

        self.router = nn.Sequential(
            nn.Linear(dim, 64, bias=False),
            nn.GELU(),
            nn.Linear(64, n_total_groups, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        gate = nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up

        pooled = x.mean(dim=1, keepdim=True)
        routing = torch.softmax(self.router(pooled), dim=-1)
        weight_map = routing[:, :, self.routing_idx]
        hidden = hidden * weight_map

        return self.down_proj(hidden)


class NFRA_Max_Block(nn.Module):
    """
    NFRA Max Block v2 — three novel features working together.

    Architecture:
      x → LN
        → MultiScaleGatedRecurrence  [15-level hierarchical recurrence + router]
        → ResonanceGuidedLocalAttention [attention bypass, gated by router]
        → dropout → + residual
      → LN
        → FractalSwiGLU  [fractal-structured MLP with hierarchical group gating]
        → dropout → + residual

    Three architectural innovations (all GPU-efficient):

    1. MULTI-SCALE RECURRENCE: 15 heads at 4 temporal resolutions (α∈[0.90,0.995])
       creating a fractal decomposition of the sequence. Each level captures
       different timescales, from fast local patterns to slow global trends.

    2. FRACTAL-STRUCTURED MLP: SwiGLU hidden space partitioned into 15 groups
       across 4 fractal scales with input-dependent per-group gating. The weight
       matrices are dense (tensor-core optimal), but the latent structure is
       hierarchical. First architecture to organize MLP internals this way.

    3. DATA-DEPENDENT ATTENTION BYPASS: local window attention gated by the
       recurrence router head. The model decides per-token whether to use
       attention (long-range mixing) or pure recurrence. Window size is also
       dynamically chosen. First true hybrid with per-position gating.

    All three use dense matmuls + element-wise ops. No sequential loops,
    no block-sparse ops, no custom CUDA. Pure PyTorch, GPU-optimal.
    """

    def __init__(self, dim: int, n_bands: int = 16, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_bands = n_bands

        from .resonance import MultiScaleGatedRecurrence, ResonanceGuidedLocalAttention

        self.ln1 = nn.LayerNorm(dim)
        self.mixer = MultiScaleGatedRecurrence(dim, n_heads=n_bands)
        self.local_attn = ResonanceGuidedLocalAttention(dim)

        self.ln2 = nn.LayerNorm(dim)
        self.mlp = FractalSwiGLU(dim, hidden_mult=4.0)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, energy_budget: Optional[float] = None, **kwargs) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        recurrence_out, router_score = self.mixer(x)
        # Energy-gated attention bypass: at low budgets skip the local attention
        # entirely (pure recurrence path) to save compute on low-end GPUs.
        if energy_budget is not None and energy_budget < 0.4:
            attn_out = torch.zeros_like(recurrence_out)
        else:
            attn_out = self.local_attn(x, router_score)
        attn_scale = energy_budget if energy_budget is not None else 1.0
        x = recurrence_out + attn_out * min(attn_scale, 1.0)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = residual + x
        return x

    def get_sparsity(self) -> float:
        return 0.0

    def reset_stats(self):
        pass


class NFRA_Brain_Block(nn.Module):
    """
    NFRA Brain Block — cortical column with emotional state, thinking depth,
    gist intuition, and meta-cognitive awareness.

    Architecture:
      x → LN → [Gist Pathway] → fast intuition stream
        → [ThalamicGate + BrainMixer] → slow deep processing
        → gate combine → dropout → + residual
      [Thinking Loop]: repeat above thinking_depth times with state accumulation
      → LN → BrainMLP (neuromodulated) → dropout → + residual
      → return updated emotional state for next layer

    Seven brain-inspired innovations:

    1. EMOTIONAL STATE: 6-channel hormone vector that persists across layers.
       The network has a coherent "mood" that colors all processing.

    2. GIST PATHWAY: A fast intuition stream that produces a quick estimate
       before full processing. Familiar patterns use gist; novel patterns
       use full depth. This is the brain's dual-processing theory.

    3. THINKING DEPTH: Controlled by dopamine (motivation). Higher DA →
       more iterative refinement steps per token. The model "tries harder"
       when it senses important or uncertain input.

    4. THALAMIC GATING: Per-token fast/slow routing based on uncertainty.

    5. MULTI-SCALE RECURRENCE: [8,4,2,1] heads at 4 temporal resolutions.

    6. FRACTAL MLP WITH HOMEOSTATIC SPARSITY: Cortisol-gated capacity scaling.

    7. META-COGNITION: Uncertainty is explicitly tracked and fed back into
       the routing and thinking depth decisions.

    8. PREDICTIVE FORWARD: Each block generates a top-down prediction of its
       own input and uses the prediction error (surprise) as a gating signal:
       surprising tokens take the deep recurrence/attention stream, predictable
       tokens take the cheap gist stream. The residual is always the input x
       (never the prediction), so the recurrence can never be starved by a
       predictor collapsing toward identity. This is the brain's free-energy
       principle in action — the network minimises surprise by learning to
       predict its own representations.

    Two further mechanisms live inside the block's sub-modules:
    9. TEMPORAL GRID CODING (in BrainMixer): multi-oscillator position
       encoding inspired by grid cells in entorhinal cortex. Creates a
       combinatorial temporal signature for each position using learnable
       frequencies.
    10. LATERAL INHIBITION (optional, in BrainMLP): k-WTA winner-take-all
        sparsity. When config.k_wta_frac > 0 only the top-k fraction of
        hidden units survive per token (input-dependent sparsity, no extra
        parameters). Off by default.
    """

    def __init__(self, dim: int, n_bands: int = 16, dropout: float = 0.1,
                 max_thinking_depth: int = 3, k_wta_frac: float = 0.0,
                 local_route: bool = False, div_norm: bool = False,
                 astro: bool = False, theta: bool = False,
                 ach_retain: bool = False, gain_nov: bool = False):
        super().__init__()
        self.dim = dim
        self.max_thinking_depth = max_thinking_depth

        from .neuro import NeuroModulator, ThalamicGate, BrainMixer, BrainMLP
        from .resonance import ResonanceGuidedLocalAttention

        self.neuromodulator = NeuroModulator(dim)

        self.predictor = nn.Linear(dim, dim, bias=False)

        self.ln1 = nn.LayerNorm(dim)
        self.mixer = BrainMixer(dim, n_heads=n_bands, astro=astro,
                                theta=theta, ach_retain=ach_retain,
                                gain_nov=gain_nov)
        self.thalamus = ThalamicGate(dim)

        # Lightweight sliding-window attention for content-addressable retrieval
        # (exact recall of recent tokens — spelling/character context that pure
        # recurrence handles poorly). Gated by the mixer's router score.
        self.local_attn = ResonanceGuidedLocalAttention(
            dim, n_heads=2, head_dim=dim // 8, window=64)
        self.attn_gate = nn.Linear(dim, 1, bias=False)

        self.gist_proj = nn.Linear(dim, dim, bias=False)
        self.gist_gate = nn.Linear(dim, 1, bias=False)

        # Learned depth-refinement operator (tensorized — replaces the 3x loop)
        self.depth_refine = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.GELU(),
            nn.Linear(dim, dim, bias=False),
        )

        self.ln2 = nn.LayerNorm(dim)
        self.mlp = BrainMLP(dim, hidden_mult=4.0, k_wta_frac=k_wta_frac,
                            local_route=local_route, div_norm=div_norm)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        hormones: Optional[torch.Tensor] = None,
        energy_budget: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape

        hormones = self.neuromodulator(x, prev_hormones=hormones)

        if energy_budget is not None:
            budget_factor = 1.0 - energy_budget
            cort = hormones[:, :, 4:5]
            hormones = torch.cat([
                hormones[:, :, :4],
                cort + budget_factor * (1.0 - cort),
                hormones[:, :, 5:],
            ], dim=-1)

        # Per-token thinking depth: dopamine → per-token depth factor [B,S,1]
        da = hormones[:, :, 2:3]
        depth_f = 1.0 + da * (self.max_thinking_depth - 1)
        depth_f = depth_f.clamp(1.0, self.max_thinking_depth)

        # Predictive surprise (free-energy): predict the input, then route
        # processing by how surprising the token is. The prediction error must
        # NOT be the block's main input or residual: `residual = prediction`
        # with `n = ln1(error)` made identity a degenerate attractor — once
        # predictor -> I (x -> x), error -> 0, ln1(0) = 0 with ZERO gradient,
        # so the recurrence/attention streams died and the whole block reduced
        # to a per-token pass-through with no memory (recall probes never even
        # fit the training set). The residual is `x`; the error only gates
        # gist-vs-deep, preserving the free-energy intent without the collapse.
        prediction = self.predictor(x)
        error = x - prediction

        residual = x
        n = self.ln1(x)

        # Surprise-gated routing: surprising tokens take the deep stream,
        # well-predicted tokens take the cheap gist stream (one matmul).
        surprise = error.norm(dim=-1, keepdim=True) / math.sqrt(D)
        gate = (0.7 * torch.sigmoid(self.gist_gate(error))
                + 0.3 * torch.sigmoid(surprise)).clamp(0.05, 0.95)
        if energy_budget is not None:
            gate = gate * (0.3 + 0.7 * energy_budget)  # low budget → prefer gist

        gist = self.gist_proj(n)                       # cheap intuition stream

        # Deep stream: single parallel-scan mixer pass (no sequential loop)
        recurrence_out, router_score = self.mixer(n, hormones=hormones)
        deep = self.thalamus(n, hormones, recurrence_out)

        # Content retrieval: sliding-window attention on the same stream, gated
        # per-token by the recurrence router (familiar tokens skip it).
        attn_out = self.local_attn(n, router_score)
        attn_gate = torch.sigmoid(self.attn_gate(n)).clamp(0.0, 0.5)
        deep = deep + attn_gate * attn_out

        # Tensorized depth refinement applied ONCE (dopamine scales its strength)
        deep = deep + depth_f * self.depth_refine(deep)

        combined = gate * deep + (1.0 - gate) * gist
        x = residual + self.dropout(combined)

        residual = x
        n = self.ln2(x)
        n = self.mlp(n, hormones=hormones)
        n = self.dropout(n)
        x = residual + n

        return x, hormones.detach()

    def get_sparsity(self) -> float:
        return 0.0

    def reset_stats(self):
        pass
