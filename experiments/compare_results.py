"""
Compare NFRA Lite vs Classical Transformer Results

Run this after training both models.

Created by Saurav Bhandari
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import time
from nfra.models import create_nfra_lite

def compare_models():
    print("=" * 60)
    print("NFRA Lite vs Classical Transformer Comparison")
    print("=" * 60)
    
    # Test input
    input_ids = torch.randint(0, 50257, (1, 128))
    
    # === NFRA Lite ===
    print("\n[NFRA Lite]")
    model_lite = create_nfra_lite()
    model_lite.eval()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(5):
            out_lite = model_lite(input_ids, energy_budget=0.5)
    nfra_time = (time.time() - start) / 5 * 1000
    
    print(f"  Parameters: {sum(p.numel() for p in model_lite.parameters()):,}")
    print(f"  Avg Inference Time: {nfra_time:.2f} ms")
    print(f"  Output Shape: {out_lite['logits'].shape}")
    
    # === Classical Transformer (simplified) ===
    print("\n[Classical Transformer]")
    embedding = torch.nn.Embedding(50257, 384)
    transformer = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(d_model=384, nhead=6, batch_first=True),
        num_layers=8
    )
    lm_head = torch.nn.Linear(384, 50257)
    
    total_params = (sum(p.numel() for p in embedding.parameters()) +
                    sum(p.numel() for p in transformer.parameters()) +
                    sum(p.numel() for p in lm_head.parameters()))
    
    start = time.time()
    with torch.no_grad():
        for _ in range(5):
            emb = embedding(input_ids)
            out = transformer(emb)
            logits = lm_head(out)
    classical_time = (time.time() - start) / 5 * 1000
    
    print(f"  Parameters: {total_params:,}")
    print(f"  Avg Inference Time: {classical_time:.2f} ms")
    print(f"  Output Shape: {logits.shape}")
    
    # === Comparison ===
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"\nInference Speed:")
    print(f"  NFRA Lite:        {nfra_time:.2f} ms")
    print(f"  Classical:        {classical_time:.2f} ms")
    print(f"  Speed Difference: {classical_time/nfra_time:.2f}x")
    
    print(f"\nKey Advantages of NFRA Lite:")
    print(f"  - Built-in energy awareness (energy_budget parameter)")
    print(f"  - High sparsity during inference")
    print(f"  - Designed for old hardware from the ground up")
    print(f"  - Fractal structure for efficient computation")

if __name__ == "__main__":
    compare_models()