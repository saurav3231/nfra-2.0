"""
NFRA benchmark suite.

Head-to-head comparison of NFRA vs RWKV vs RetNet vs GPT-2 (Mamba-SSM
optional), matched on params, identical data + optimizer.
"""

from .arena import main as arena_main
from .compare import main

__all__ = ["arena_main", "main"]
