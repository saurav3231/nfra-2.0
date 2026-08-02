"""
Predictive Resonance Coding components
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PredictiveGenerator(nn.Module):
    """
    Generates predictions at multiple scales for predictive coding.
    """

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2

        self.predictor = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
        )

        self.error_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate prediction for next state."""
        return self.predictor(x)

    def compute_prediction_error(
        self, actual: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        """Compute prediction error (core of predictive coding)."""
        error = actual - prediction
        return error * self.error_scale


class MultiScalePredictor(nn.Module):
    """
    Predictive generator that operates at multiple fractal scales.

    Note: for scale s > 1 the sub-predictor operates on a dim//s
    representation. Forward adaptively pools the input to that width, so
    the returned tensor has dim//s channels (coarser prediction), not the
    full `dim`. Scale-1 predictors keep the full width.
    """

    def __init__(self, dim: int, scales: list[int] | None = None):
        super().__init__()
        scales = [1, 2, 4] if scales is None else scales
        self.dim = dim
        self.scales = [s for s in scales if s >= 1 and dim % s == 0]
        self.predictors = nn.ModuleDict(
            {
                f"scale_{s}": PredictiveGenerator(dim // s if s > 1 else dim)
                for s in self.scales
            }
        )

    def forward(self, x: torch.Tensor, scale: int = 1) -> torch.Tensor:
        key = f"scale_{scale}"
        pred = self.predictors.get(key)
        if pred is None:
            return x  # Identity if scale not available
        in_dim = pred.predictor[0].in_features
        if in_dim != x.shape[-1]:
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), in_dim).transpose(1, 2)
        return pred(x)
