"""
Energy Budget Sweep Example

Demonstrates how NFRA 2.0 performs under different energy constraints.
Created by Saurav Bhandari
"""

import torch
from nfra.models import NFRAConfig, NFRAForCausalLM
from nfra.evaluation import NFRABenchmark

config = NFRAConfig(hidden_size=256, num_layers=4)
model = NFRAForCausalLM(config)

benchmark = NFRABenchmark(model, device="cpu")

# Test different energy budgets
results = benchmark.benchmark_energy(
    energy_budgets=[0.2, 0.4, 0.6, 0.8, 1.0],
    seq_len=128,
    batch_size=2
)

print("Energy Budget Sweep Results:")
for budget, metrics in results.items():
    print(f"Budget {budget}: {metrics['tokens_per_second']:.1f} tokens/sec")