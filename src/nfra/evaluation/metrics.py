"""
Evaluation Metrics for NFRA 2.0

Created by Saurav Bhandari
"""

import torch
import torch.nn.functional as F
from typing import Dict


def compute_perplexity(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute perplexity from logits and targets."""
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return torch.exp(loss).item()


def compute_sparsity(model) -> Dict[str, float]:
    """Compute average sparsity across all FractalResonanceBlocks."""
    sparsity_values = []
    
    for module in model.modules():
        if hasattr(module, 'get_sparsity'):
            sparsity = module.get_sparsity()
            if sparsity > 0:
                sparsity_values.append(sparsity)
    
    if not sparsity_values:
        return {"avg_sparsity": 0.0, "min_sparsity": 0.0, "max_sparsity": 0.0}
    
    return {
        "avg_sparsity": sum(sparsity_values) / len(sparsity_values),
        "min_sparsity": min(sparsity_values),
        "max_sparsity": max(sparsity_values)
    }


def estimate_energy(model: torch.nn.Module, input_shape: tuple, device: str = "cpu") -> float:
    """
    Estimate energy consumption (very rough approximation).
    Returns estimated Joules per forward pass.
    """
    param_count = sum(p.numel() for p in model.parameters())
    
    # Very rough estimation based on device
    if device == "cuda":
        energy = param_count * 2e-10  # More efficient on GPU
    else:
        energy = param_count * 5e-10  # Higher on CPU
    
    return energy