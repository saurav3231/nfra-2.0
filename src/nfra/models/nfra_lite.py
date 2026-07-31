"""
NFRA Lite Model

Optimized for very old and low-power hardware (Intel Core i5-337U and similar).

This is the most lightweight and fastest variant of NFRA.

Created by Saurav Bhandari
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .nfra_model import NFRAConfig, NFRAForCausalLM


class NFRALiteForCausalLM(NFRAForCausalLM):
    """
    NFRA Lite - Highly optimized for low-power and legacy CPUs.
    
    Key characteristics:
    - Small hidden size (384)
    - Only 2 fractal scales
    - No advanced features (Mixture of Fractals, Selective Scanner)
    - Aggressive sparsity
    - INT8 friendly
    """
    
    def __init__(self, config: Optional[NFRAConfig] = None):
        if config is None:
            config = NFRAConfig(mode="lite")
        else:
            # Ensure Lite constraints are enforced even if custom config is passed
            config.mode = "lite"
            config.hidden_size = 384
            config.num_layers = 8
            config.fractal_scales = [1, 2]
            config.n_bands = 2
            config.use_mixture_of_fractals = False
            config.use_selective_scanning = False
            config.use_dynamic_precision = False
            config.aggressive_sparsity = True
        
        super().__init__(config)
        
        # Lite-specific optimizations
        self._lite_mode = True

    def forward(
        self, 
        input_ids: torch.Tensor, 
        energy_budget: Optional[float] = None,
        return_dict: bool = True,
    ) -> Dict:
        """
        Optimized forward pass for Lite mode.
        Forces high sparsity and energy efficiency.
        """
        if energy_budget is None:
            energy_budget = 0.5  # Very aggressive for Lite
            
        return super().forward(
            input_ids=input_ids,
            energy_budget=energy_budget,
            return_dict=return_dict
        )
    
    def get_model_info(self) -> Dict:
        """Return information about the Lite model."""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "mode": "lite",
            "parameters": total_params,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "fractal_scales": self.config.fractal_scales,
            "target_hardware": "Intel Core i5-337U and similar",
            "note": "Advanced features (Mixture of Fractals, Selective Scanner) are disabled for performance."
        }


def create_nfra_lite() -> NFRALiteForCausalLM:
    """Convenience function to create NFRA Lite model."""
    return NFRALiteForCausalLM()