"""Training components for NFRA 2.0"""

from .losses import NFRACombinedLoss
from .trainer import NFRATrainer

__all__ = ["NFRACombinedLoss", "NFRATrainer"]