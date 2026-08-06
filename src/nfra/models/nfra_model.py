"""
NFRA Model with Mode Support (Lite / Mid / Max)

Created by Saurav Bhandari
"""

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn

ModeType = Literal["lite", "mid", "max", "brain"]


@dataclass
class NFRAConfig:
    """Configuration for NFRA models with mode support"""

    mode: ModeType = "max"

    # Base parameters
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    dropout: float = 0.1
    gradient_checkpointing: bool = False

    # Fractal settings
    fractal_scales: list[int] = field(default_factory=lambda: [1, 2, 4])
    n_bands: int = 4

    # Advanced features (controlled by mode)
    use_mixture_of_fractals: bool = True
    use_selective_scanning: bool = True
    use_dynamic_precision: bool = True
    num_fractal_experts: int = 8
    top_k_experts: int = 3

    # Energy settings
    energy_aware: bool = True
    aggressive_sparsity: bool = False

    # Depth sharing (universal-transformer style): fewer unique blocks
    # reused over multiple passes → far fewer params at equal depth.
    depth_shared: bool = False
    unique_blocks: int = 4

    # k-WTA lateral inhibition in the Brain MLP: keep only the top-k fraction
    # of hidden units per token (0.0 = off). Input-dependent sparsity, no
    # extra params or skipped compute.
    k_wta_frac: float = 0.0

    # "Small but powerful" brain-inspired levers (all near-zero cost; see
    # neuro.py / fractal_block.py for the mechanisms). Off by default so
    # baselines stay untouched; each is A/B-testable via env toggles.
    #
    # local_route : cortical-microcircuit routing — the BrainMLP router reads
    #               a per-token LOCAL context (sliding causal window) blended
    #               with the global pool, instead of one decision per sequence.
    #               Gives a small model input-dependent capacity for free.
    # div_norm    : divisive (contrast) normalization of MLP hidden units by
    #               pooled intensity — the cortex's gain-control mechanism.
    # astro       : astrocytic timescale homeostat — a slow per-sequence signal
    #               that shifts the recurrence's overall memory horizon.
    local_route: bool = False
    div_norm: bool = False
    astro: bool = False

    # More brain levers (same contract: off by default, one axis, near-zero
    # cost, identity-init so the baseline is untouched until trained).
    # theta      : TIME axis — per-band theta-gamma rhythmic decay (memory
    #              windows rhythmically open/close). Learnable rhythm, amp
    #              starts at 0.
    # ach_retain : TIME axis — ACh-retention polarity: high ACh → HOLD memory
    #              (encoding hypothesis) instead of the legacy "forget" prior.
    # gain_nov   : GAIN axis — causal prefix-variance (novelty/contrast) scales
    #              the recurrence write value; learnable scalar, starts at 0.
    # lora_rank  : SPACE axis — per-pass low-rank adapters on the depth-shared
    #              block (0 = off). Breaks depth-sharing symmetry cheaply.
    theta: bool = False
    ach_retain: bool = False
    gain_nov: bool = False
    lora_rank: int = 0

    # NFRA 3.3 Cortex (opt-in; NFRA_Brain_Block 3.2 stays intact for A/B).
    # use_cortex : build NFRA_Cortex_Block (matrix-state mixer + cached-window
    #              attention + real adaptive-compute exit gate) instead of the
    #              3.2 Brain block. Set via NFRA_CORTEX=1 in the arena.
    # cortex_state : matrix-state width N per head (CortexMixer d_state).
    # exit_reg    : compute regularizer on the exit gate (pulls expected pass
    #               count down; easy tokens exit early, hard tokens spend all).
    use_cortex: bool = False
    cortex_state: int = 8
    exit_reg: float = 1e-3

    # Isolation-ablation switches (FUTURE_PLAN Part 11): each turns ONE 3.3b
    # mechanism OFF to attribute the verified quality win. All default False =
    # the exact verified architecture. Set via NFRA_ISO=<list> in the arena.
    iso_gland: bool = False  # OFF -> drop neuromodulator (no ACh/NE)
    iso_vgate: bool = False  # OFF -> no input-dependent value gate
    iso_rgate: bool = False  # OFF -> no output receptance gate
    iso_phase: bool = False  # OFF -> no resonance phase modulation
    iso_exit: bool = False  # OFF -> no adaptive-compute exit gate

    # cortex_chunk_size : >0 computes the CortexMixer retention as EXACT
    #                     chunked retention (within-chunk quadratic attention +
    #                     cross-chunk linear state) — the same decayed-QK^T
    #                     operator with ~C/S of the FLOPs and a fraction of the
    #                     O(S^2) parallel form's memory. 0 = parallel (the
    #                     verified board path). Tier-1 speed/memory lever.
    cortex_chunk_size: int = 0
    # cortex_triton : >0 runs the chunked CortexMixer retention forward as ONE
    #                 fused Triton kernel per (batch, head) instead of the eager
    #                 per-chunk loop (the Tier-1 speed lever; backward stays a
    #                 checkpoint-recompute through the eager reference so grads
    #                 are unchanged). Falls back to eager on any machine without
    #                 Triton/CUDA. Requires cortex_chunk_size > 0 to matter.
    cortex_triton: bool = False
    # ── Tier-1 experiment flags (ALL default off → every one preserves the exact
    #    verified 1.7 baseline unless explicitly enabled; those marked * change
    #    the loss by design — they are A/B experiments, not exact-math kernels).
    # cortex_lsr*       : per-head learned long/short route (a bias added to
    #                     log_decay so heads specialize local vs global).
    # cortex_int8_state*: keep the chunked retention's long-range linear state
    #                     in int8 (asymmetric precision → cheaper memory).
    # cortex_depth_time*: continuous function of the depth-pass index instead of
    #                     free per-pass FiLM scalars (fractional-depth capacity).
    cortex_lsr: bool = False
    cortex_int8_state: bool = False
    cortex_depth_time: bool = False
    # cortex_per_token_gn*: normalize the mixer GroupNorm over each head's
    #                       channels PER TOKEN (no cross-token coupling) so the
    #                       retention dual runs O(1)-exact (stateful.py).
    cortex_per_token_gn: bool = False
    # cortex_stm_ring : STM working-tag ring (RSM short-term store): >0 = window
    #                   size k of a tiny windowed causal read per mixer block.
    #                   Zero-init -> adds 0 at init (no regression); O(1) decode
    #                   exact via cached per-layer window (stateful.py).
    cortex_stm_ring: int = 0
    cortex_stm_dim: int = 32
    # ckpt_gems : recompute the two biggest GEMM activations (qkvr, MLP gate_up)
    #             in backward instead of storing them. Trade compute for ~8 MB/
    #             layer of memory. Do NOT combine with torch.compile.
    ckpt_gems: bool = False

    def __post_init__(self):
        valid_modes = ["lite", "mid", "max", "brain"]
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{self.mode}'")

        if self.mode == "lite":
            self.hidden_size = 384
            self.num_layers = 8
            self.fractal_scales = [1, 2]
            self.n_bands = 4
            self.use_mixture_of_fractals = False
            self.use_selective_scanning = False
            self.use_dynamic_precision = False
            self.aggressive_sparsity = True
            self.num_fractal_experts = 0
            self.top_k_experts = 0

        elif self.mode == "mid":
            self.hidden_size = 512
            self.num_layers = 10
            self.fractal_scales = [1, 2, 4]
            self.n_bands = 4
            self.use_mixture_of_fractals = True
            self.use_selective_scanning = False
            self.num_fractal_experts = 4
            self.top_k_experts = 2

        elif self.mode == "max":
            # Respect user num_layers/hidden_size — defaults (768/12) already
            # come from the dataclass fields. Pass num_layers=24 for full Max.
            self.depth_shared = False

        elif self.mode == "brain":
            # Don't override hidden_size/num_layers/unique_blocks — respect
            # user values (unique_blocks default is 4 in the dataclass).
            # n_bands: promote the dataclass default (4) to the legacy 16-head
            # hierarchy, but RESPECT an explicit value so the NFRA_BANDS
            # band-count ablation knob actually reaches BrainMixer.
            self.fractal_scales = [1]
            if self.n_bands == 4:
                self.n_bands = 16
            self.use_mixture_of_fractals = False
            self.use_selective_scanning = False
            self.use_dynamic_precision = False
            self.aggressive_sparsity = False
            self.num_fractal_experts = 0
            self.top_k_experts = 0
            self.depth_shared = True


class PassLoRA(nn.Module):
    """Per-depth-pass low-rank adapter (AFC-LoRA, Space axis).

    y = x + (x @ A) @ B   with A: [dim, r], B: [r, dim], and B initialized to
    0 so the adapter is the exact identity at init (safe to enable, changes
    nothing until trained). At rank r: two tiny matmuls (2·dim·r per token),
    2·r·dim params per pass — far cheaper than duplicating the block. Applied
    per depth pass it breaks depth-sharing symmetry, so the SAME shared weights
    compute a different function at each depth (the depth-dilution fix)."""

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(dim, rank) * 0.02)
        self.B = nn.Parameter(torch.zeros(rank, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (x @ self.A) @ self.B


class NFRAForCausalLM(nn.Module):
    """Base NFRA Model (used by all modes)"""

    def __init__(self, config: NFRAConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Import here to avoid circular imports
        from ..core import (
            DynamicEnergyBudgetAllocator,
            FractalResonanceBlock,
            NFRA_Brain_Block,
            NFRA_Cortex_Block,
            NFRA_Max_Block,
        )

        if config.mode == "max":
            block = NFRA_Max_Block
        elif config.mode == "brain":
            block = NFRA_Brain_Block
        else:
            block = FractalResonanceBlock
        if config.use_cortex:
            block = NFRA_Cortex_Block

        # Depth sharing: n_unique unique blocks reused depth_passes times.
        # Effective depth ≈ num_layers, but params only scale with n_unique.
        if config.depth_shared:
            self.n_unique = max(1, min(config.unique_blocks, config.num_layers))
            self.depth_passes = max(1, config.num_layers // self.n_unique)
        else:
            self.n_unique = config.num_layers
            self.depth_passes = 1

        block_kwargs = {
            "dim": config.hidden_size,
            "n_bands": config.n_bands,
            "dropout": config.dropout,
        }
        if config.use_cortex:
            block_kwargs["d_state"] = config.cortex_state
            block_kwargs["exit_reg"] = config.exit_reg
            block_kwargs["k_wta_frac"] = config.k_wta_frac
            block_kwargs["local_route"] = config.local_route
            block_kwargs["div_norm"] = config.div_norm
            block_kwargs["iso_gland"] = config.iso_gland
            block_kwargs["iso_vgate"] = config.iso_vgate
            block_kwargs["iso_rgate"] = config.iso_rgate
            block_kwargs["iso_phase"] = config.iso_phase
            block_kwargs["iso_exit"] = config.iso_exit
            block_kwargs["chunk_size"] = config.cortex_chunk_size
            block_kwargs["ckpt_gems"] = config.ckpt_gems
            block_kwargs["triton"] = config.cortex_triton
            block_kwargs["lsr"] = config.cortex_lsr
            block_kwargs["int8_state"] = config.cortex_int8_state
            block_kwargs["per_token_gn"] = config.cortex_per_token_gn
            block_kwargs["stm_ring"] = config.cortex_stm_ring
            block_kwargs["stm_dim"] = config.cortex_stm_dim
        elif config.mode == "brain":
            block_kwargs["k_wta_frac"] = config.k_wta_frac
            block_kwargs["local_route"] = config.local_route
            block_kwargs["div_norm"] = config.div_norm
            block_kwargs["astro"] = config.astro
            block_kwargs["theta"] = config.theta
            block_kwargs["ach_retain"] = config.ach_retain
            block_kwargs["gain_nov"] = config.gain_nov
        self.layers = nn.ModuleList(
            [block(**block_kwargs) for _ in range(self.n_unique)]
        )

        # Per-pass adapters (FiLM): tiny per-pass scale/shift applied at the
        # start of each depth pass. Breaks depth-sharing symmetry so the SAME
        # weights don't compute identically at every depth — each pass learns
        # a cheap "depth position" specialization (~2*depth_passes*dim params).
        # idea 4 (cortex_depth_time): replace the free per-pass scalars with a
        # continuous function of the depth index (fewer params, fractional-depth
        # capacity). Mutually exclusive with the per-pass FiLM scalars.
        self.depth_time = None
        if config.depth_shared and self.depth_passes > 1 and config.cortex_depth_time:
            from ..core.experiments import DepthTimeAdapter

            self.depth_time = DepthTimeAdapter(
                self.depth_passes, config.hidden_size
            )
        if (
            config.depth_shared
            and self.depth_passes > 1
            and not config.cortex_depth_time
        ):
            self.pass_scale = nn.Parameter(
                torch.ones(self.depth_passes, config.hidden_size)
            )
            self.pass_bias = nn.Parameter(
                torch.zeros(self.depth_passes, config.hidden_size)
            )
        else:
            self.register_parameter("pass_scale", None)
            self.register_parameter("pass_bias", None)

        # Per-pass LoRA (SPACE axis): one low-rank adapter per depth pass on
        # the shared block. B starts at 0 → exact identity at init.
        if config.depth_shared and self.depth_passes > 1 and config.lora_rank > 0:
            self.pass_lora = nn.ModuleList(
                [
                    PassLoRA(config.hidden_size, config.lora_rank)
                    for _ in range(self.depth_passes)
                ]
            )
        else:
            self.pass_lora = None

        if config.energy_aware:
            self.energy_allocator = DynamicEnergyBudgetAllocator(
                num_blocks=self.n_unique
            )
        else:
            self.energy_allocator = None

        # Static check: all layers have neuromodulator (brain mode) or none
        self._has_neuromodulator = (
            hasattr(self.layers[0], "neuromodulator") if len(self.layers) > 0 else False
        )

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

        # Global brain state: a slow global-state aggregator that summarizes a
        # whole depth pass and injects it top-down into subsequent passes. Gives
        # the network a persistent "global state" (like slow neuromodulatory
        # loops) instead of only local per-token signals. Threaded at the model
        # level. (The 0.5x prev-pass carry makes it a leaky integrator, not a
        # GRU.)
        self.global_brain = None
        if self._has_neuromodulator:
            from ..core.neuro import GlobalBrainState

            self.global_brain = GlobalBrainState(
                config.hidden_size, state_dim=max(32, config.hidden_size // 8)
            )

    def forward(self, input_ids, energy_budget=None, return_dict=True):
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D tensor [batch, seq_len]")

        if energy_budget is not None and not (0.0 <= energy_budget <= 1.0):
            raise ValueError("energy_budget must be between 0.0 and 1.0")

        hidden_states = self.embed_tokens(input_ids)

        # Convert budgets to floats ONCE before the loop to avoid per-layer GPU sync
        if self.energy_allocator is not None and energy_budget is not None:
            budgets_t = self.energy_allocator(hardware_factor=energy_budget)
            budgets = budgets_t.detach().cpu().tolist()
        elif energy_budget is not None:
            budgets = [energy_budget] * self.n_unique
        else:
            budgets = [1.0] * self.n_unique

        hormones = None
        global_state = None
        exit_aux = None
        use_ckpt = (
            self.config.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )
        if use_ckpt:
            checkpoint = torch.utils.checkpoint.checkpoint

        if self.config.use_cortex:
            if self.config.iso_exit:
                # Isolation ablation: no adaptive-compute exit gate -- plain
                # full-depth forward, no freezing, no compute regularizer.
                for p in range(self.depth_passes):
                    if self.depth_time is not None:
                        _sc, _bs = self.depth_time.film(p)
                        hidden_states = hidden_states * _sc.view(1, 1, -1) + _bs.view(
                            1, 1, -1
                        )
                    elif self.pass_scale is not None:
                        hidden_states = hidden_states * self.pass_scale[p].view(
                            1, 1, -1
                        ) + self.pass_bias[p].view(1, 1, -1)
                    for i, layer in enumerate(self.layers):
                        if use_ckpt:
                            hidden_states, hormones, _el = checkpoint(
                                self._run_cortex_layer,
                                layer,
                                hidden_states,
                                budgets[i],
                                hormones,
                                use_reentrant=False,
                            )
                        else:
                            hidden_states, hormones, _el = layer(
                                hidden_states,
                                hormones=hormones,
                                energy_budget=budgets[i],
                            )
                    if self.pass_lora is not None:
                        hidden_states = self.pass_lora[p](hidden_states)
                    if self.global_brain is not None:
                        global_state = self.global_brain(hidden_states, global_state)
                        hidden_states = self.global_brain.inject(
                            global_state, hidden_states
                        )
            else:
                # NFRA 3.3 Cortex: each block returns (out, hormones, exit_logit).
                # The exit gate gives per-token, per-pass adaptive compute: easy
                # tokens freeze early (their state is preserved), hard tokens spend
                # all depth passes. Training uses Gumbel straight-through + a small
                # compute regularizer; inference hard-masks and skips the remaining
                # passes when the whole batch has exited.
                B, S, _D = hidden_states.shape
                active = torch.ones(
                    B, S, 1, dtype=hidden_states.dtype, device=hidden_states.device
                )
                final_states = hidden_states.clone()
                exit_aux = torch.zeros(
                    (), device=hidden_states.device, dtype=hidden_states.dtype
                )
                for p in range(self.depth_passes):
                    if not self.training and bool((active == 0).all()):
                        break
                    if self.depth_time is not None:
                        _sc, _bs = self.depth_time.film(p)
                        hidden_states = hidden_states * _sc.view(1, 1, -1) + _bs.view(
                            1, 1, -1
                        )
                    elif self.pass_scale is not None:
                        hidden_states = hidden_states * self.pass_scale[p].view(
                            1, 1, -1
                        ) + self.pass_bias[p].view(1, 1, -1)
                    for i, layer in enumerate(self.layers):
                        if use_ckpt:
                            hidden_states, hormones, _el = checkpoint(
                                self._run_cortex_layer,
                                layer,
                                hidden_states,
                                budgets[i],
                                hormones,
                                use_reentrant=False,
                            )
                        else:
                            hidden_states, hormones, _el = layer(
                                hidden_states,
                                hormones=hormones,
                                energy_budget=budgets[i],
                            )
                    if self.pass_lora is not None:
                        hidden_states = self.pass_lora[p](hidden_states)
                    if self.global_brain is not None:
                        global_state = self.global_brain(hidden_states, global_state)
                        hidden_states = self.global_brain.inject(
                            global_state, hidden_states
                        )
                    # Exit decision (last block's gate): continue = keep going.
                    p_exit = self.layers[-1].exit_gate.prob(hidden_states)  # [B,S,1]
                    if self.training:
                        cont, _ = self.layers[-1].exit_gate.sample_mask(
                            hidden_states, hard=False
                        )  # ST 0/1
                    else:
                        cont = (p_exit < 0.5).float()
                    active_new = active * cont
                    newly_done = active - active_new
                    # Freeze tokens that just exited at their current state.
                    final_states = (
                        final_states * (1 - newly_done) + hidden_states * newly_done
                    )
                    hidden_states = hidden_states * active_new + final_states * (
                        1 - active_new
                    )
                    # Compute regularizer: penalize CONTINUING (expected pass count).
                    n_active = active.sum().clamp(min=1.0)
                    exit_aux = exit_aux + self.config.exit_reg * (
                        ((1.0 - p_exit) * active).sum() / n_active
                    )
                    active = active_new.detach()
                hidden_states = hidden_states * active + final_states * (1 - active)
        elif self._has_neuromodulator:
            for p in range(self.depth_passes):
                # Per-pass adapter: depth-shared blocks compute a DIFFERENT
                # function at each pass (breaks symmetry → more capacity).
                if self.pass_scale is not None:
                    hidden_states = hidden_states * self.pass_scale[p].view(
                        1, 1, -1
                    ) + self.pass_bias[p].view(1, 1, -1)
                for i, layer in enumerate(self.layers):
                    if use_ckpt:
                        hidden_states, hormones = checkpoint(
                            self._run_layer,
                            layer,
                            hidden_states,
                            budgets[i],
                            hormones,
                            use_reentrant=False,
                        )
                    else:
                        hidden_states, hormones = layer(
                            hidden_states, hormones=hormones, energy_budget=budgets[i]
                        )
                # Per-pass LoRA: adapt this pass's output (identity at init).
                if self.pass_lora is not None:
                    hidden_states = self.pass_lora[p](hidden_states)
                # Top-down global brain state: aggregate the current pass, then
                # feed it back into the NEXT pass (slow neuromodulatory loop).
                if self.global_brain is not None:
                    global_state = self.global_brain(hidden_states, global_state)
                    hidden_states = self.global_brain.inject(
                        global_state, hidden_states
                    )
        else:
            for p in range(self.depth_passes):
                if self.pass_scale is not None:
                    hidden_states = hidden_states * self.pass_scale[p].view(
                        1, 1, -1
                    ) + self.pass_bias[p].view(1, 1, -1)
                for i, layer in enumerate(self.layers):
                    if use_ckpt:
                        hidden_states = checkpoint(
                            self._run_layer,
                            layer,
                            hidden_states,
                            budgets[i],
                            None,
                            use_reentrant=False,
                        )
                    else:
                        hidden_states = layer(hidden_states, energy_budget=budgets[i])
                if self.pass_lora is not None:
                    hidden_states = self.pass_lora[p](hidden_states)

        logits = self.lm_head(hidden_states)

        if return_dict:
            out = {"logits": logits}
            # Exit-gate compute regularizer is a TRAINING cost only — eval loss
            # stays pure CE so the 3.2-vs-3.3 head-to-head is not biased by a
            # fixed additive constant.
            if exit_aux is not None and self.training:
                out["exit_aux"] = exit_aux
            return out
        return logits

    @staticmethod
    def _run_layer(layer, hidden_states, budget, hormones):
        """Single layer pass used by gradient checkpointing (recomputed in backward)."""
        if hasattr(layer, "neuromodulator"):
            hidden_states, hormones = layer(
                hidden_states, hormones=hormones, energy_budget=budget
            )
            return hidden_states, hormones
        return layer(hidden_states, energy_budget=budget)

    @staticmethod
    def _run_cortex_layer(layer, hidden_states, budget, hormones):
        """Checkpoint wrapper for NFRA_Cortex_Block (returns the exit logit too)."""
        hidden_states, hormones, _el = layer(
            hidden_states, hormones=hormones, energy_budget=budget
        )
        return hidden_states, hormones, _el


class NFRAForSequenceClassification(nn.Module):
    """NFRA Model for sequence classification tasks."""

    def __init__(self, config: NFRAConfig, num_labels: int = 2):
        super().__init__()
        self.config = config
        self.num_labels = num_labels

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        from ..core import FractalResonanceBlock

        self.layers = nn.ModuleList(
            [
                FractalResonanceBlock(
                    dim=config.hidden_size,
                    scales=config.fractal_scales,
                    n_bands=config.n_bands,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, energy_budget=None, return_dict=True):
        _batch_size, _seq_len = input_ids.shape

        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, energy_budget=energy_budget)

        # Mean pooling over sequence
        pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)

        if return_dict:
            return {"logits": logits}
        return logits


def create_nfra_model(mode: ModeType = "brain"):
    """Factory function to create NFRA model based on mode"""
    if mode == "lite":
        from .nfra_lite import NFRALiteForCausalLM

        return NFRALiteForCausalLM()
    else:
        config = NFRAConfig(mode=mode)
        return NFRAForCausalLM(config)
