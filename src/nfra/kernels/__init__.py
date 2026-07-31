"""
Custom CUDA kernels for NFRA.

Each kernel is a Triton kernel with a pure-torch fallback so the code runs
everywhere (CPU / no-triton) and only accelerates when CUDA + Triton exist
(Kaggle T4, local NVIDIA GPU).

Toggles:
  NFRA_SCAN_KERNEL   0 = always torch, 1 = auto (default), 2 = force triton

Lazy __getattr__ (PEP 562) so `python -m nfra.kernels.scan` does not warn:
the submodule is only imported on attribute access.
"""

__all__ = ['selective_scan', 'parallel_scan_time_varying']


def __getattr__(name):
    if name == 'selective_scan':
        from .scan import selective_scan
        return selective_scan
    if name == 'parallel_scan_time_varying':
        from .scan import parallel_scan_time_varying
        return parallel_scan_time_varying
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
