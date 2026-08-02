"""
Loss functions for NFRA.

Created by Saurav Bhandari
"""

from __future__ import annotations

import torch
from torch import nn


class NFRACombinedLoss(nn.Module):
    """
    Combined loss for NFRA including:
    - Task loss (CrossEntropy)
    - Resonance sparsity loss
    - Prediction error loss
    - Energy regularization
    """

    def __init__(
        self,
        task_weight: float = 1.0,
        resonance_weight: float = 0.1,
        prediction_weight: float = 0.05,
        energy_weight: float = 0.01,
    ):
        super().__init__()
        self.task_weight = task_weight
        self.resonance_weight = resonance_weight
        self.prediction_weight = prediction_weight
        self.energy_weight = energy_weight

        self.task_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        resonance_stats: dict | None = None,
        prediction_error: torch.Tensor = None,
        energy_used: float | None = None,
    ):
        """
        Compute combined NFRA loss.
        """
        # Task loss
        task_loss = self.task_loss(logits.view(-1, logits.size(-1)), targets.view(-1))

        total_loss = self.task_weight * task_loss
        loss_dict = {"task_loss": task_loss.item()}

        # Resonance sparsity / energy terms are reported but NOT added to the
        # differentiable loss: they come from detached floats, so they have no
        # gradient path — folding them in silently inflated the loss value
        # without training the model (a pure no-op constant). The prediction
        # error term below is a tensor and does contribute gradients.
        if resonance_stats is not None:
            sparsity = resonance_stats.get("sparsity", 0.0)
            resonance_loss = max(0, 0.7 - sparsity) + max(0, sparsity - 0.95)
            loss_dict["resonance_loss"] = resonance_loss

        # Prediction error loss
        if prediction_error is not None:
            pred_loss = prediction_error.mean()
            total_loss = total_loss + self.prediction_weight * pred_loss
            loss_dict["prediction_loss"] = pred_loss.item()

        # Energy regularization
        if energy_used is not None:
            energy_loss = max(0, energy_used - 0.6)  # Penalize using too much energy
            loss_dict["energy_loss"] = energy_loss

        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict
