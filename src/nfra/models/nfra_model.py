"""
NFRA Model with Mode Support (Lite / Mid / Max)

Created by Saurav Bhandari
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List, Literal


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
    fractal_scales: List[int] = field(default_factory=lambda: [1, 2, 4])
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
            self.fractal_scales = [1]
            self.n_bands = 16
            self.use_mixture_of_fractals = False
            self.use_selective_scanning = False
            self.use_dynamic_precision = False
            self.aggressive_sparsity = False
            self.num_fractal_experts = 0
            self.top_k_experts = 0
            self.depth_shared = True


class NFRAForCausalLM(nn.Module):
    """Base NFRA Model (used by all modes)"""
    
    def __init__(self, config: NFRAConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Import here to avoid circular imports
        from ..core import (
            FractalResonanceBlock, DynamicEnergyBudgetAllocator,
            SwiGLU_MLP, ParallelGatedRecurrence, NFRA_Max_Block,
            NFRA_Brain_Block,
        )
        
        if config.mode == "max":
            block = NFRA_Max_Block
        elif config.mode == "brain":
            block = NFRA_Brain_Block
        else:
            block = FractalResonanceBlock

        # Depth sharing: n_unique unique blocks reused depth_passes times.
        # Effective depth ≈ num_layers, but params only scale with n_unique.
        if config.depth_shared:
            self.n_unique = max(1, min(config.unique_blocks, config.num_layers))
            self.depth_passes = max(1, config.num_layers // self.n_unique)
        else:
            self.n_unique = config.num_layers
            self.depth_passes = 1

        block_kwargs = dict(dim=config.hidden_size, n_bands=config.n_bands,
                            dropout=config.dropout)
        if config.mode == "brain":
            block_kwargs['k_wta_frac'] = config.k_wta_frac
        self.layers = nn.ModuleList([
            block(**block_kwargs)
            for _ in range(self.n_unique)
        ])

        # Per-pass adapters (FiLM): tiny per-pass scale/shift applied at the
        # start of each depth pass. Breaks depth-sharing symmetry so the SAME
        # weights don't compute identically at every depth — each pass learns
        # a cheap "depth position" specialization (~2*depth_passes*dim params).
        if config.depth_shared and self.depth_passes > 1:
            self.pass_scale = nn.Parameter(
                torch.ones(self.depth_passes, config.hidden_size))
            self.pass_bias = nn.Parameter(
                torch.zeros(self.depth_passes, config.hidden_size))
        else:
            self.register_parameter('pass_scale', None)
            self.register_parameter('pass_bias', None)
        
        if config.energy_aware:
            self.energy_allocator = DynamicEnergyBudgetAllocator(
                num_blocks=self.n_unique
            )
        else:
            self.energy_allocator = None
            
        # Static check: all layers have neuromodulator (brain mode) or none
        self._has_neuromodulator = (
            hasattr(self.layers[0], 'neuromodulator') if len(self.layers) > 0 else False
        )
            
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

        # Global brain state: a GRU that aggregates a whole-depth summary and
        # injects it top-down into subsequent passes. Gives the network a
        # persistent "global state" (like slow neuromodulatory loops) instead of
        # only local per-token signals. Threaded at the model level.
        self.global_brain = None
        if self._has_neuromodulator:
            from ..core.neuro import GlobalBrainState
            self.global_brain = GlobalBrainState(
                config.hidden_size, state_dim=max(32, config.hidden_size // 8))

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
        use_ckpt = (
            self.config.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )
        if use_ckpt:
            checkpoint = torch.utils.checkpoint.checkpoint

        if self._has_neuromodulator:
            for p in range(self.depth_passes):
                # Per-pass adapter: depth-shared blocks compute a DIFFERENT
                # function at each pass (breaks symmetry → more capacity).
                if self.pass_scale is not None:
                    hidden_states = (hidden_states * self.pass_scale[p].view(1, 1, -1)
                                     + self.pass_bias[p].view(1, 1, -1))
                for i, layer in enumerate(self.layers):
                    if use_ckpt:
                        hidden_states, hormones = checkpoint(
                            self._run_layer, layer, hidden_states,
                            budgets[i], hormones, use_reentrant=False,
                        )
                    else:
                        hidden_states, hormones = layer(
                            hidden_states, hormones=hormones, energy_budget=budgets[i]
                        )
                # Top-down global brain state: aggregate the current pass, then
                # feed it back into the NEXT pass (slow neuromodulatory loop).
                if self.global_brain is not None:
                    global_state = self.global_brain(hidden_states, global_state)
                    hidden_states = self.global_brain.inject(global_state, hidden_states)
        else:
            for p in range(self.depth_passes):
                if self.pass_scale is not None:
                    hidden_states = (hidden_states * self.pass_scale[p].view(1, 1, -1)
                                     + self.pass_bias[p].view(1, 1, -1))
                for i, layer in enumerate(self.layers):
                    if use_ckpt:
                        hidden_states = checkpoint(
                            self._run_layer, layer, hidden_states,
                            budgets[i], None, use_reentrant=False,
                        )
                    else:
                        hidden_states = layer(hidden_states, energy_budget=budgets[i])
        
        logits = self.lm_head(hidden_states)
        
        if return_dict:
            return {"logits": logits}
        return logits

    @staticmethod
    def _run_layer(layer, hidden_states, budget, hormones):
        """Single layer pass used by gradient checkpointing (recomputed in backward)."""
        if hasattr(layer, 'neuromodulator'):
            hidden_states, hormones = layer(
                hidden_states, hormones=hormones, energy_budget=budget
            )
            return hidden_states, hormones
        return layer(hidden_states, energy_budget=budget)


class NFRAForSequenceClassification(nn.Module):
    """NFRA Model for sequence classification tasks."""

    def __init__(self, config: NFRAConfig, num_labels: int = 2):
        super().__init__()
        self.config = config
        self.num_labels = num_labels

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        from ..core import FractalResonanceBlock

        self.layers = nn.ModuleList([
            FractalResonanceBlock(
                dim=config.hidden_size,
                scales=config.fractal_scales,
                n_bands=config.n_bands,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])

        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, energy_budget=None, return_dict=True):
        batch_size, seq_len = input_ids.shape

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