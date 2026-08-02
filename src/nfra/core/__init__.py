"""Core building blocks of NFRA (Nonlinear Factorized Recurrent Attention)."""

from .cortex import (
    CortexExit,
    CortexMixer,
    NFRA_Cortex_Block,
)
from .energy import DynamicEnergyBudgetAllocator
from .fractal_block import (
    FractalGatedMLP,
    FractalResonanceBlock,
    FractalSwiGLU,
    NFRA_Brain_Block,
    NFRA_Max_Block,
    SwiGLU_MLP,
)
from .neuro import (
    BrainMixer,
    BrainMLP,
    GlobalBrainState,
    NeuroModulator,
    TemporalGridEncoder,
    ThalamicGate,
)
from .predictive import MultiScalePredictor, PredictiveGenerator
from .resonance import (
    CausalResonanceMixer,
    MultiScaleGatedRecurrence,
    ParallelGatedRecurrence,
    ResonanceGuidedLocalAttention,
    ResonanceRouter,
    ResonanceSignature,
    SpikeResonanceLayer,
)

__all__ = [
    "BrainMLP",
    "BrainMixer",
    "CausalResonanceMixer",
    "CortexExit",
    "CortexMixer",
    "DynamicEnergyBudgetAllocator",
    "FractalGatedMLP",
    "FractalResonanceBlock",
    "FractalSwiGLU",
    "GlobalBrainState",
    "MultiScaleGatedRecurrence",
    "MultiScalePredictor",
    "NFRA_Brain_Block",
    "NFRA_Cortex_Block",
    "NFRA_Max_Block",
    "NeuroModulator",
    "ParallelGatedRecurrence",
    "PredictiveGenerator",
    "ResonanceGuidedLocalAttention",
    "ResonanceRouter",
    "ResonanceSignature",
    "SpikeResonanceLayer",
    "SwiGLU_MLP",
    "TemporalGridEncoder",
    "ThalamicGate",
]
