"""
Basic INT8 Quantization Utilities for NFRA Lite

Created by Saurav Bhandari
"""

import torch
import torch.nn as nn
from typing import Dict


def quantize_to_int8(model: nn.Module) -> Dict:
    """
    Simple post-training INT8 quantization.
    This is a basic implementation suitable for CPU inference.
    """
    quantized_state = {}
    
    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            # Simple symmetric quantization
            scale = param.abs().max() / 127.0
            if scale == 0:
                scale = 1.0
            quantized = torch.round(param / scale).clamp(-128, 127).to(torch.int8)
            quantized_state[name] = {
                'quantized': quantized,
                'scale': scale
            }
        else:
            quantized_state[name] = param
    
    return quantized_state


def dequantize(state: Dict) -> Dict:
    """Convert quantized state back to float32 for inference."""
    dequantized = {}
    for name, value in state.items():
        if isinstance(value, dict) and 'quantized' in value:
            dequantized[name] = value['quantized'].float() * value['scale']
        else:
            dequantized[name] = value
    return dequantized


def apply_int8_to_model(model: nn.Module):
    """
    Apply basic INT8 quantization to linear layers.
    Preserves original weights in original_weight attribute for reference.
    Skips layers that have already been quantized.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Skip if already quantized
            if hasattr(module, 'is_quantized') and module.is_quantized:
                continue
            
            # Store original weight for reference (only once)
            if not hasattr(module, 'original_weight'):
                module.original_weight = module.weight.data.clone()
            
            # Quantize weight
            scale = module.weight.data.abs().max() / 127.0
            if scale > 0:
                module.weight.data = torch.round(module.weight.data / scale).clamp(-128, 127).float() * scale
            
            module.is_quantized = True
    
    return model