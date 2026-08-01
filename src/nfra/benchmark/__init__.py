"""
NFRA benchmark suite.

Head-to-head comparison of NFRA Brain vs RWKV vs RetNet vs GPT-2 (Mamba-SSM
optional), matched on params, identical data + optimizer.
"""

from .compare import main
from .arena import main as arena_main

__all__ = ["main", "arena_main"]
