"""
Hardware Benchmark Example

Run this on different devices (CPU, Raspberry Pi, old laptop) to measure performance.
Created by Saurav Bhandari
"""

import torch
from nfra.models import NFRAConfig, NFRAForCausalLM
from nfra.evaluation import NFRABenchmark

config = NFRAConfig(hidden_size=512, num_layers=6)
model = NFRAForCausalLM(config)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running benchmark on: {device}")

benchmark = NFRABenchmark(model, device=device)
results = benchmark.benchmark_inference(seq_lengths=[128, 256, 512])

print("\nInference Benchmark Results:")
for seq_len, metrics in results.items():
    print(f"Seq Len {seq_len}: {metrics['tokens_per_second']} tokens/sec")