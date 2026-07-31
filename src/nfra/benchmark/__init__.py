"""
NFRA benchmark suite.

Head-to-head comparison of NFRA Brain vs Mamba-SSM vs GPT-2,
matched on params, identical data + optimizer.
"""

from .compare import main

__all__ = ["main"]
