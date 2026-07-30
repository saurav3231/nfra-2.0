"""
Simple NFRA Lite Runner

Run this file to test NFRA Lite.
Any errors will be shown in the terminal.

Usage:
    python run_nfra_lite.py

Created by Saurav Bhandari
"""

from nfra.models import create_nfra_lite
import torch
import time
import traceback

def main():
    try:
        print("=" * 55)
        print("NFRA Lite - Testing on your hardware")
        print("=" * 55)
        
        print("\n[1] Loading NFRA Lite model...")
        model = create_nfra_lite()
        model.eval()
        
        info = model.get_model_info()
        print(f"    ✓ Model loaded successfully!")
        print(f"    Parameters: {info['parameters']:,}")
        print(f"    Hidden Size: {info['hidden_size']}")
        
        print("\n[2] Running inference test...")
        input_ids = torch.randint(0, 50257, (1, 64))
        
        start = time.time()
        with torch.no_grad():
            outputs = model(input_ids, energy_budget=0.5)
        elapsed = time.time() - start
        
        print(f"    ✓ Inference successful!")
        print(f"    Time taken: {elapsed*1000:.2f} ms")
        print(f"    Output shape: {outputs['logits'].shape}")
        
        print("\n" + "=" * 55)
        print("NFRA Lite is working correctly on your system!")
        print("=" * 55)
        
    except Exception as e:
        print("\n" + "=" * 55)
        print("ERROR OCCURRED:")
        print("=" * 55)
        traceback.print_exc()
        print("\nPlease share the error above if you need help.")

if __name__ == "__main__":
    main()