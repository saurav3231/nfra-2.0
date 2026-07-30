"""Evaluation and benchmarking module for NFRA 2.0"""

from .metrics import compute_perplexity, compute_sparsity, estimate_energy
from .benchmark import NFRABenchmark

__all__ = [
    "compute_perplexity",
    "compute_sparsity", 
    "estimate_energy",
    "NFRABenchmark"
]