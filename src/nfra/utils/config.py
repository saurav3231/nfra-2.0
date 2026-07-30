"""
Configuration loader for NFRA 2.0

Created by Saurav Bhandari
"""

import yaml
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    hidden_size: int = 512
    num_layers: int = 8
    fractal_scales: List[int] = None
    dropout: float = 0.1


@dataclass
class AdvancedConfig:
    use_mixture_of_fractals: bool = True
    use_selective_scanning: bool = True
    use_dynamic_precision: bool = True
    num_fractal_experts: int = 6
    top_k_experts: int = 2


@dataclass
class TrainingConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    batch_size: int = 6
    max_seq_length: int = 512
    energy_budget: float = 0.65


@dataclass
class DatasetConfig:
    name: str = "wikitext"
    config: str = "wikitext-2-raw-v1"
    max_samples: Optional[int] = None


@dataclass
class HardwareConfig:
    device: str = "auto"
    precision: str = "auto"


@dataclass
class NFRAFullConfig:
    model: ModelConfig
    advanced: AdvancedConfig
    training: TrainingConfig
    dataset: DatasetConfig
    hardware: HardwareConfig


def load_config(path: str) -> NFRAFullConfig:
    """Load YAML config and return structured config object."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    # Use defaults for any missing sections
    model_data = data.get("model", {})
    advanced_data = data.get("advanced", {})
    training_data = data.get("training", {})
    dataset_data = data.get("dataset", {})
    hardware_data = data.get("hardware", {})
    
    return NFRAFullConfig(
        model=ModelConfig(**model_data),
        advanced=AdvancedConfig(**advanced_data),
        training=TrainingConfig(**training_data),
        dataset=DatasetConfig(**dataset_data),
        hardware=HardwareConfig(**hardware_data)
    )