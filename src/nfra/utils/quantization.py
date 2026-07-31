"""
Basic INT8 Quantization Utilities for NFRA Lite

Created by Saurav Bhandari
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class Int8Linear(nn.Module):
    """Per-tensor symmetric INT8 quantization of an nn.Linear's weight.

    The weight is stored as real int8 (+ one fp32 scale) — genuine 8-bit
    storage — and dequantized on the fly at forward. The output is
    numerically identical to rounding the original weight to the INT8 grid.
    Bias (if any) stays fp32. For CPU inference (no gradient).
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = linear.bias
        scale = linear.weight.data.abs().max() / 127.0
        self.scale = scale if scale > 0 else 1.0
        self.qweight = nn.Parameter(
            torch.round(linear.weight.data / self.scale).clamp(-128, 127).to(torch.int8),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.qweight.float() * self.scale
        return F.linear(x, w, self.bias)


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
    Replace every nn.Linear with Int8Linear (real int8 weight storage).

    Apply AFTER loading: it swaps linear layers in-place for CPU inference
    (weights stored as int8, dequantized per forward). Skips layers that have
    already been quantized. Note: the resulting module's state_dict keys
    change (weight -> qweight + scale), so save the float model first if you
    need to round-trip through checkpoints.
    """
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, Int8Linear):
            continue
        if hasattr(module, 'is_quantized') and module.is_quantized:
            continue
        quantized = Int8Linear(module)
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], quantized)
        quantized.is_quantized = True
    
    return model
