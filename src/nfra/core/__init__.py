"""Core building blocks of NFRA 3.1 Brain"""

from .fractal_block import (
    FractalResonanceBlock,
    FractalGatedMLP,
    SwiGLU_MLP,
    FractalSwiGLU,
    NFRA_Max_Block,
    NFRA_Brain_Block,
)
from .resonance import (
    ResonanceRouter,
    ResonanceSignature,
    SpikeResonanceLayer,
    CausalResonanceMixer,
    ParallelGatedRecurrence,
    MultiScaleGatedRecurrence,
    ResonanceGuidedLocalAttention,
)
from .neuro import (
    NeuroModulator,
    ThalamicGate,
    BrainMixer,
    BrainMLP,
    TemporalGridEncoder,
    GlobalBrainState,
)
from .predictive import PredictiveGenerator, MultiScalePredictor
from .energy import DynamicEnergyBudgetAllocator

__all__ = [
    "FractalResonanceBlock",
    "FractalGatedMLP",
    "SwiGLU_MLP",
    "FractalSwiGLU",
    "NFRA_Max_Block",
    "NFRA_Brain_Block",
    "ResonanceRouter",
    "ResonanceSignature",
    "SpikeResonanceLayer",
    "CausalResonanceMixer",
    "ParallelGatedRecurrence",
    "MultiScaleGatedRecurrence",
    "ResonanceGuidedLocalAttention",
    "NeuroModulator",
    "ThalamicGate",
    "BrainMixer",
    "BrainMLP",
    "TemporalGridEncoder",
    "GlobalBrainState",
    "PredictiveGenerator",
    "MultiScalePredictor",
    "DynamicEnergyBudgetAllocator",
]
