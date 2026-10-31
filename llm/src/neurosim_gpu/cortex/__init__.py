"""Cortex-LM: a sparse cortical sheet between a pretrained encoder and decoder.

The package is optional at import time: only PyTorch is required. The
Hugging Face backbone path is imported lazily so the core test suite stays
CPU-only and dependency-free.
"""

from .brain import CorticalCore
from .config import (
    BackboneConfig,
    BrainConfig,
    ConfigError,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
    apply_overrides,
    load_config,
    save_config,
)
from .data import CorpusMeta, build_dataloaders, build_datasets
from .model import CortexOutput, CortexSandwichLM, language_model_loss
from .tokenization import ByteTokenizer, build_tokenizer
from .topology import CorticalTopology, build_cortical_topology

__all__ = [
    "BackboneConfig",
    "BrainConfig",
    "ByteTokenizer",
    "ConfigError",
    "CorpusMeta",
    "CortexOutput",
    "CortexSandwichLM",
    "CorticalCore",
    "CorticalTopology",
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "TrainConfig",
    "apply_overrides",
    "build_cortical_topology",
    "build_dataloaders",
    "build_datasets",
    "build_tokenizer",
    "language_model_loss",
    "load_config",
    "save_config",
]
