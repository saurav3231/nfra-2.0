"""Shim: run the benchmark from the repo checkout without installing.
Delegates to the in-package `nfra.benchmark.compare` (single source of truth).

    python benchmarks/nfra_vs_mamba_vs_gpt2.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from nfra.benchmark.compare import main

if __name__ == '__main__':
    main()
