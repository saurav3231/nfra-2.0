"""
Custom CUDA kernels for NFRA.

Each kernel is a Triton kernel with a pure-torch fallback so the code runs
everywhere (CPU / no-triton) and only accelerates when CUDA + Triton exist
(Kaggle T4, local NVIDIA GPU).

Toggles:
  NFRA_SCAN_KERNEL   0 = always torch, 1 = auto (default), 2 = force triton
"""

from .scan import selective_scan, parallel_scan_time_varying  # noqa: F401

__all__ = ['selective_scan', 'parallel_scan_time_varying']
