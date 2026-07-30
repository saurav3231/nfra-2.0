"""
NFRA 3.0 — NeuroFractal Resonance Architecture

A brain-inspired neural network with efficient multi-band sequence mixing
and fractal gated MLPs for ultra-low-power and legacy hardware.
"""

__version__ = "3.1.0"
__author__ = "NFRA Research Team"
__license__ = "MIT"

from .core import (
    FractalResonanceBlock,
    FractalGatedMLP,
    CausalResonanceMixer,
    ParallelGatedRecurrence,
    MultiScaleGatedRecurrence,
    ResonanceGuidedLocalAttention,
    SwiGLU_MLP,
    FractalSwiGLU,
    NFRA_Max_Block,
    NFRA_Brain_Block,
    ResonanceRouter,
    PredictiveGenerator,
    DynamicEnergyBudgetAllocator,
    NeuroModulator,
    ThalamicGate,
    BrainMixer,
    BrainMLP,
    TemporalGridEncoder,
)
from .models import NFRAForCausalLM, NFRAForSequenceClassification, NFRAConfig

__all__ = [
    "FractalResonanceBlock",
    "FractalGatedMLP",
    "CausalResonanceMixer",
    "ParallelGatedRecurrence",
    "MultiScaleGatedRecurrence",
    "ResonanceGuidedLocalAttention",
    "SwiGLU_MLP",
    "FractalSwiGLU",
    "NFRA_Max_Block",
    "NFRA_Brain_Block",
    "ResonanceRouter",
    "PredictiveGenerator",
    "DynamicEnergyBudgetAllocator",
    "NeuroModulator",
    "ThalamicGate",
    "BrainMixer",
    "BrainMLP",
    "TemporalGridEncoder",
    "NFRAForCausalLM",
    "NFRAForSequenceClassification",
    "NFRAConfig",
]