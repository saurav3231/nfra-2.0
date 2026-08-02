"""
Basic inference example for NFRA (verified Cortex architecture).

Created by Saurav Bhandari
"""

import torch
from nfra.models import NFRAConfig, NFRAForCausalLM

# Small NFRA model using the verified lean Cortex block
config = NFRAConfig(
    vocab_size=32000,
    hidden_size=384,
    num_layers=6,
    unique_blocks=2,
    depth_shared=True,
    use_cortex=True,
)

model = NFRAForCausalLM(config)
model.eval()

# Dummy input
input_ids = torch.randint(0, 32000, (1, 32))

with torch.no_grad():
    outputs = model(input_ids)

print("Logits shape:", outputs["logits"].shape)
print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
