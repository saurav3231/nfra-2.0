"""
NFRA Lite - Executable Entry Point

This file is used to build the standalone .exe.

Created by Saurav Bhandari
"""

from nfra.models import create_nfra_lite
import torch
import time

def main():
    print("=" * 50)
    print("NFRA Lite - Running on your system")
    print("=" * 50)
    
    print("\nLoading NFRA Lite model...")
    model = create_nfra_lite()
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Simple inference test
    print("\nRunning test inference...")
    input_ids = torch.randint(0, 50257, (1, 64))
    
    start = time.time()
    with torch.no_grad():
        outputs = model(input_ids, energy_budget=0.5)
    elapsed = time.time() - start
    
    print(f"Inference completed in {elapsed*1000:.2f} ms")
    print(f"Output shape: {outputs['logits'].shape}")
    
    print("\n" + "=" * 50)
    print("NFRA Lite is working correctly!")
    print("=" * 50)
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()