"""
Hardware detection and energy estimation utilities

Created by Saurav Bhandari
"""

import platform

import torch


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


def gpu_tier(gpu_memory_gb=None) -> str:
    """
    Classify the available GPU into a VRAM tier for default max-mode presets.

    Returns one of: "low" (<8GB), "mid" (8-16GB), "high" (>=16GB), "cpu".
    """
    if gpu_memory_gb is None:
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            return "cpu"
    if gpu_memory_gb >= 16:
        return "high"
    if gpu_memory_gb >= 8:
        return "mid"
    return "low"


def recommended_max_config(vocab_size: int = 50257, gpu_memory_gb=None) -> dict:
    """
    Max-mode presets tuned per VRAM tier so low-end GPUs can actually train it.

        low  (<8GB):  dim 384, 8 layers
        mid  (8-16GB): dim 512, 12 layers
        high (>=16GB): dim 768, 24 layers
        cpu:           dim 256, 6 layers

    Usage: NFRAConfig(**recommended_max_config(vocab_size=VOCAB))
    """
    tier = gpu_tier(gpu_memory_gb)
    presets = {
        "low": {"hidden_size": 384, "num_layers": 8},
        "mid": {"hidden_size": 512, "num_layers": 12},
        "high": {"hidden_size": 768, "num_layers": 24},
        "cpu": {"hidden_size": 256, "num_layers": 6},
    }
    return {"mode": "max", "vocab_size": vocab_size, **presets[tier]}


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
