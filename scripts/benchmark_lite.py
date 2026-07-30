#!/usr/bin/env python3
"""
NFRA Lite Benchmark Script

Measures performance on CPU (including old hardware like i5-337U).

Created by Saurav Bhandari
"""

import torch
import time
from nfra.models import create_nfra_lite


def benchmark_nfra_lite(seq_lengths=[128, 256, 512], batch_size=2, repeats=10):
    print("Creating NFRA Lite model...")
    model = create_nfra_lite()
    model.eval()
    
    device = "cpu"
    model = model.to(device)
    
    print(f"Model info: {model.get_model_info()}")
    print(f"Running benchmark on: {device}\n")
    
    results = {}
    
    for seq_len in seq_lengths:
        input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids, energy_budget=0.5)
        
        # Benchmark
        start = time.time()
        with torch.no_grad():
            for _ in range(repeats):
                _ = model(input_ids, energy_budget=0.5)
        elapsed = time.time() - start
        
        tokens_per_sec = (batch_size * seq_len * repeats) / elapsed
        
        results[seq_len] = {
            "tokens_per_second": round(tokens_per_sec, 2),
            "ms_per_token": round(1000 / tokens_per_sec, 2)
        }
        
        print(f"Seq Len: {seq_len:4d} | "
              f"Tokens/sec: {tokens_per_sec:6.2f} | "
              f"ms/token: {1000/tokens_per_sec:5.2f}")
    
    return results


if __name__ == "__main__":
    benchmark_nfra_lite()