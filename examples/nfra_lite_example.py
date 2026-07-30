"""
NFRA Lite Usage Example

This example shows how to use NFRA Lite on low-power hardware.

Created by Saurav Bhandari
"""

from nfra.models import create_nfra_lite, create_nfra_model
import torch

print("=== NFRA Lite Example ===\n")

# Method 1: Create NFRA Lite directly
print("1. Creating NFRA Lite model...")
model = create_nfra_lite()
print(f"   Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Method 2: Using the general factory
print("\n2. Creating using factory function...")
model2 = create_nfra_model(mode="lite")
print("   Same model created via factory")

# Inference example
print("\n3. Running inference...")
input_ids = torch.randint(0, 50257, (1, 128))

with torch.no_grad():
    outputs = model(input_ids, energy_budget=0.5)
    
print(f"   Input shape: {input_ids.shape}")
print(f"   Output logits shape: {outputs['logits'].shape}")

# Model info
print("\n4. Model Information:")
info = model.get_model_info()
for key, value in info.items():
    print(f"   {key}: {value}")

print("\n=== Example Complete ===")