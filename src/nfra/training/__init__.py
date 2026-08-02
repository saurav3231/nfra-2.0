"""Training components for NFRA (Nonlinear Factorized Recurrent Attention)."""

from .losses import NFRACombinedLoss
from .trainer import NFRATrainer

__all__ = ["NFRACombinedLoss", "NFRATrainer"]
