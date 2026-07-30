"""
Hardware detection and energy estimation utilities

Created by Saurav Bhandari
"""

import torch
import platform


def get_hardware_info():
    """Detect available hardware and capabilities."""
    info = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        info["cpu_cores"] = torch.get_num_threads()
    
    return info


def estimate_energy_usage(model, input_shape, device="cpu"):
    """
    Rough estimate of energy usage during inference.
    """
    param_count = sum(p.numel() for p in model.parameters())
    
    # Very rough estimation
    if device == "cpu":
        energy_per_token = param_count * 1e-9 * 0.5  # Joules (very approximate)
    else:
        energy_per_token = param_count * 1e-9 * 0.2
        
    return energy_per_token