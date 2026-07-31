#!/usr/bin/env python
"""Thin wrapper: NFRA 3.1 vs 3.2 A/B comparison.

Real logic lives in the package so it is also runnable after
`pip install git+https://github.com/saurav3231/nfra-2.0.git`:

    python -m nfra.benchmark.compare_versions

Usage:
  python scripts/compare_nfra_versions.py
  NFRA_STEPS=600 NFRA_DATA=wikitext2 python -m nfra.benchmark.compare_versions
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, 'src'))

from nfra.benchmark.compare_versions import main

if __name__ == '__main__':
    main()
