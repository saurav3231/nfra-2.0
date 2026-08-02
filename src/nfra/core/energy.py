"""
Dynamic Energy Budget Allocator (DEBA)
"""

from __future__ import annotations

import torch
from torch import nn


class DynamicEnergyBudgetAllocator(nn.Module):
    """
    Dynamically allocates computational energy budget across fractal blocks
    based on task importance, hardware constraints, and remaining power.
    """

    def __init__(
        self,
        num_blocks: int,
        default_budget: float = 1.0,
        min_budget: float = 0.1,
        adaptation_rate: float = 0.1,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.default_budget = default_budget
        self.min_budget = min_budget
        self.adaptation_rate = adaptation_rate

        # Learnable importance weights per block
        self.importance = nn.Parameter(torch.ones(num_blocks))

        # Current budget state
        self.register_buffer("current_budget", torch.ones(num_blocks) * default_budget)

    def forward(
        self,
        task_importance: torch.Tensor | None = None,
        hardware_factor: float = 1.0,
        power_remaining: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute per-block energy budgets.

        Args:
            task_importance: Importance scores for current task
            hardware_factor: Hardware capability multiplier (0.0-1.0)
            power_remaining: Remaining battery/power (0.0-1.0)
        """
        # Base budget from importance
        if task_importance is not None:
            base = task_importance * self.importance
        else:
            base = self.importance

        # Scale weights so their MEAN is 1: with uniform importance every block
        # gets exactly `hardware_factor` (so an energy_budget of 0.5 really
        # means "half compute per block", matching the model-layer contract
        # where 1.0 = full). The old sum-normalization made per-block budgets
        # ~hardware_factor / num_blocks, so e.g. Lite's 0.5 silently collapsed
        # to the 0.1 floor for every block.
        base = base / (base.mean() + 1e-8)

        # Apply hardware and power constraints
        budget = base * hardware_factor * power_remaining

        # Ensure minimum budget
        budget = torch.clamp(budget, min=self.min_budget)

        # Update current budget with smoothing (detached)
        self.current_budget = (
            1 - self.adaptation_rate
        ) * self.current_budget.detach() + self.adaptation_rate * budget.detach()

        return budget

    def get_total_energy_used(self) -> float:
        """Returns sum of current energy allocation."""
        return self.current_budget.sum().item()

    def reset(self):
        """Reset to default budgets."""
        self.current_budget.fill_(self.default_budget)
