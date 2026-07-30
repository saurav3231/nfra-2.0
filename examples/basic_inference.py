"""
Basic Inference Example for NFRA 2.0

Created by Saurav Bhandari
"""

import torch
from nfra.models import NFRAConfig, NFRAForCausalLM

# Load a small NFRA model
config = NFRAConfig(
    vocab_size=50257,
    hidden_size=384,
    num_layers=6,
    fractal_scales=[1, 2, 4],
    use_mixture_of_fractals=True,
    use_selective_scanning=True
)

model = NFRAForCausalLM(config)
model.eval()

# Dummy input
input_ids = torch.randint(0, 50257, (1, 32))

# Run inference with energy budget (simulating low-power device)
with torch.no_grad():
    outputs = model(input_ids, energy_budget=0.5)
    
print("Logits shape:", outputs["logits"].shape)
print("Inference successful with energy budget = 0.5")