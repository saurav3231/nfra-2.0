"""
Model saving and loading utilities for NFRA.

Created by Saurav Bhandari
"""

import json
import os

import torch

from ..models.nfra_model import NFRAConfig, NFRAForCausalLM


def save_pretrained(model: NFRAForCausalLM, save_directory: str):
    """Save NFRA model in a Hugging Face compatible format."""
    os.makedirs(save_directory, exist_ok=True)

    # Save config
    config_path = os.path.join(save_directory, "config.json")
    with open(config_path, "w") as f:
        json.dump(model.config.__dict__, f, indent=2)

    # Save weights
    model_path = os.path.join(save_directory, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to {save_directory}")


def from_pretrained(save_directory: str, device: str = "cpu") -> NFRAForCausalLM:
    """Load NFRA model from directory. Supports Lite/Mid/Max variants."""
    config_path = os.path.join(save_directory, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    config = NFRAConfig(**config_dict)

    # Instantiate the correct class based on mode
    mode = config_dict.get("mode", "max")
    if mode == "lite":
        from .nfra_lite import NFRALiteForCausalLM

        model = NFRALiteForCausalLM(config)
    else:
        model = NFRAForCausalLM(config)

    model_path = os.path.join(save_directory, "pytorch_model.bin")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model
