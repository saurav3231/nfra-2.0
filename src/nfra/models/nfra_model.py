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
            self.hidden_size = 768
            self.num_layers = 24

        elif self.mode == "brain":
            self.hidden_size = 768
            self.num_layers = 24
            self.fractal_scales = [1]
            self.n_bands = 16
            self.use_mixture_of_fractals = False
            self.use_selective_scanning = False
            self.use_dynamic_precision = False
            self.aggressive_sparsity = False
            self.num_fractal_experts = 0
            self.top_k_experts = 0


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
        
        self.layers = nn.ModuleList([
            block(
                dim=config.hidden_size,
                n_bands=config.n_bands,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        if config.energy_aware:
            self.energy_allocator = DynamicEnergyBudgetAllocator(
                num_blocks=config.num_layers
            )
        else:
            self.energy_allocator = None
            
        # Static check: all layers have neuromodulator (brain mode) or none
        self._has_neuromodulator = (
            hasattr(self.layers[0], 'neuromodulator') if self.layers else False
        )
            
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

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
            budgets = [energy_budget] * self.config.num_layers
        else:
            budgets = [1.0] * self.config.num_layers
        
        hormones = None
        if self._has_neuromodulator:
            for i, layer in enumerate(self.layers):
                hidden_states, hormones = layer(
                    hidden_states, hormones=hormones, energy_budget=budgets[i]
                )
        else:
            for i, layer in enumerate(self.layers):
                hidden_states = layer(hidden_states, energy_budget=budgets[i])
        
        logits = self.lm_head(hidden_states)
        
        if return_dict:
            return {"logits": logits}
        return logits


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