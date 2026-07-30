#!/usr/bin/env python3
"""
NFRA Lite Validation Script

Validates that NFRA Lite can run on low-power hardware.

Created by Saurav Bhandari
"""

import torch
import time
import psutil
from nfra.models import create_nfra_lite


def validate_nfra_lite():
    print("=" * 50)
    print("NFRA Lite Validation")
    print("=" * 50)
    
    # System info
    print(f"\nSystem Info:")
    print(f"  CPU Cores: {psutil.cpu_count(logical=False)}")
    print(f"  Logical Cores: {psutil.cpu_count(logical=True)}")
    print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # Create model
    print("\nCreating NFRA Lite model...")
    model = create_nfra_lite()
    model.eval()
    
    info = model.get_model_info()
    print(f"Model Parameters: {info['parameters']:,}")
    print(f"Hidden Size: {info['hidden_size']}")
    print(f"Layers: {info['num_layers']}")
    
    # Memory usage before
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)
    
    # Run inference
    print("\nRunning inference test...")
    input_ids = torch.randint(0, 50257, (1, 256))
    
    start = time.time()
    with torch.no_grad():
        for _ in range(5):
            outputs = model(input_ids, energy_budget=0.5)
    elapsed = time.time() - start
    
    mem_after = process.memory_info().rss / (1024 * 1024)
    
    print(f"\nResults:")
    print(f"  Avg Time per forward: {elapsed/5*1000:.1f} ms")
    print(f"  Memory Usage: {mem_after - mem_before:.1f} MB")
    print(f"  Estimated Tokens/sec (256 seq): {256 / (elapsed/5):.2f}")
    
    print("\n" + "=" * 50)
    print("Validation Complete")
    print("=" * 50)
    
    return {
        "parameters": info['parameters'],
        "memory_mb": mem_after - mem_before,
        "tokens_per_sec": 256 / (elapsed/5)
    }


if __name__ == "__main__":
    validate_nfra_lite()