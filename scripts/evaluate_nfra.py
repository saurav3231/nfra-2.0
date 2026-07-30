#!/usr/bin/env python3
"""
NFRA 2.0 Evaluation & Benchmarking Script

Created by Saurav Bhandari
"""

import argparse
import torch
from nfra.models import NFRAConfig, NFRAForCausalLM
from nfra.evaluation import NFRABenchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=[128, 256])
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    # In real usage, load checkpoint here
    config = NFRAConfig()
    model = NFRAForCausalLM(config)
    
    benchmark = NFRABenchmark(model, device=args.device)
    results = benchmark.run_full_benchmark()
    
    print("\n=== NFRA 2.0 Benchmark Results ===")
    print(results)


if __name__ == "__main__":
    main()