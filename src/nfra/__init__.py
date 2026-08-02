"""
NFRA — Nonlinear Factorized Recurrent Attention

An efficient recurrent language-model block: a depth-shared, decayed
query-key retention mixer (RetNet-style) gated by a token-wise receptance
gate (RWKV-style) and a SwiGLU feed-forward. Designed to deliver strong
quality-per-parameter on modest hardware, verified head-to-head against
RetNet, RWKV, Mamba, and GPT-2 baselines.

Author: SAURAV BHANDARI
"""

__version__ = "3.3.0"
__author__ = "SAURAV BHANDARI"
__license__ = "MIT"

from .core import (
    BrainMixer,
    BrainMLP,
    CausalResonanceMixer,
    DynamicEnergyBudgetAllocator,
    FractalGatedMLP,
    FractalResonanceBlock,
    FractalSwiGLU,
    MultiScaleGatedRecurrence,
    NeuroModulator,
    NFRA_Brain_Block,
    NFRA_Cortex_Block,
    NFRA_Max_Block,
    ParallelGatedRecurrence,
    PredictiveGenerator,
    ResonanceGuidedLocalAttention,
    ResonanceRouter,
    SwiGLU_MLP,
    TemporalGridEncoder,
    ThalamicGate,
)
from .models import NFRAConfig, NFRAForCausalLM, NFRAForSequenceClassification

__all__ = [
    "BrainMLP",
    "BrainMixer",
    "CausalResonanceMixer",
    "DynamicEnergyBudgetAllocator",
    "FractalGatedMLP",
    "FractalResonanceBlock",
    "FractalSwiGLU",
    "MultiScaleGatedRecurrence",
    "NFRAConfig",
    "NFRAForCausalLM",
    "NFRAForSequenceClassification",
    "NFRA_Brain_Block",
    "NFRA_Cortex_Block",
    "NFRA_Max_Block",
    "NeuroModulator",
    "ParallelGatedRecurrence",
    "PredictiveGenerator",
    "ResonanceGuidedLocalAttention",
    "ResonanceRouter",
    "SwiGLU_MLP",
    "TemporalGridEncoder",
    "ThalamicGate",
]
