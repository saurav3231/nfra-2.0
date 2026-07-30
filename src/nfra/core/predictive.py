"""
Predictive Resonance Coding components
"""

import torch
import torch.nn as nn
from typing import Optional


class PredictiveGenerator(nn.Module):
    """
    Generates predictions at multiple scales for predictive coding.
    """
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        
        self.predictor = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        
        self.error_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate prediction for next state."""
        return self.predictor(x)
    
    def compute_prediction_error(
        self, 
        actual: torch.Tensor, 
        prediction: torch.Tensor
    ) -> torch.Tensor:
        """Compute prediction error (core of predictive coding)."""
        error = actual - prediction
        return error * self.error_scale


class MultiScalePredictor(nn.Module):
    """
    Predictive generator that operates at multiple fractal scales.
    """
    
    def __init__(self, dim: int, scales: list = [1, 2, 4]):
        super().__init__()
        self.predictors = nn.ModuleDict({
            f"scale_{s}": PredictiveGenerator(dim // s if s > 1 else dim)
            for s in scales
        })
        
    def forward(self, x: torch.Tensor, scale: int = 1) -> torch.Tensor:
        key = f"scale_{scale}"
        if key in self.predictors:
            return self.predictors[key](x)
        return x  # Identity if scale not available