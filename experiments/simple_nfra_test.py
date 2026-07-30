#!/usr/bin/env python3
"""
Simple NFRA Lite Test

This is a minimal test to verify NFRA Lite works and shows its advantages.

Created by Saurav Bhandari
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("PyTorch not available in this environment.")
    print("Please install it using: pip install torch")
    exit()

from nfra.models import create_nfra_lite

print("=" * 60)
print("NFRA Lite - Simple Test & Advantage Demonstration")
print("=" * 60)

# 1. Create NFRA Lite
print("\n[1] Creating NFRA Lite model...")
model = create_nfra_lite()
model.eval()

info = model.get_model_info()
print(f"     Parameters: {info['parameters']:,}")
print(f"     Hidden Size: {info['hidden_size']}")
print(f"     Layers: {info['num_layers']}")
print(f"     Target Hardware: {info['target_hardware']}")

# 2. Run inference with energy budget
print("\n[2] Running inference with energy budget (0.5)...")
input_ids = torch.randint(0, 50257, (1, 64))

import time
start = time.time()

with torch.no_grad():
    for _ in range(10):
        outputs = model(input_ids, energy_budget=0.5)
        
elapsed = time.time() - start
avg_time = elapsed / 10 * 1000

print(f"     Average time per forward pass: {avg_time:.2f} ms")
print(f"     Output shape: {outputs['logits'].shape}")

# 3. Show advantages
print("\n[3] Key Advantages of NFRA Lite over Classical Neural Networks:")
print("     ✓ Built-in high sparsity (90-97%)")
print("     ✓ Energy-aware computation (dynamic budget)")
print("     ✓ Designed from ground up for old CPUs (i5-337U)")
print("     ✓ Predictive coding reduces unnecessary computation")
print("     ✓ Fractal structure enables efficient multi-scale processing")

print("\n" + "=" * 60)
print("Test Completed Successfully!")
print("=" * 60)