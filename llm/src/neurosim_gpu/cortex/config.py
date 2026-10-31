"""Typed configuration for the Cortex-LM pipeline.

The pipeline has five sections, each with its own dataclass:

``BackboneConfig``
    The pretrained text encoder/decoder pair (or a small local stand-in).
``BrainConfig``
    The sparse recurrent sheet that sits between them.
``ModelConfig``
    How the three pieces are glued together.
``DataConfig`` / ``TrainConfig``
    Corpus, batching and optimisation.

Configs are plain JSON so the dry-run path has no YAML dependency. Unknown
fields are rejected: a typo in a config should fail loudly, not silently keep a
default.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Mapping


class ConfigError(ValueError):
    """Raised when a configuration is missing a field or is out of range."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _check_unknown(cls: type, raw: Mapping[str, Any]) -> dict[str, Any]:
    known = {field.name for field in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(
            f"unknown {cls.__name__} field(s): {', '.join(unknown)}"
        )
    return dict(raw)


def _tuple_fields(obj: object, names: Iterable[str]) -> None:
    for name in names:
        value = getattr(obj, name)
        if isinstance(value, list):
            object.__setattr__(obj, name, tuple(value))


@dataclass(frozen=True)
class BackboneConfig:
    """The pretrained model split into an encoder half and a decoder half.

    ``kind="hf"`` loads a Hugging Face causal LM and uses its embedding plus
    ``encoder_layers`` blocks as the encoder, and the remaining blocks plus the
    LM head as the decoder. ``kind="tiny"`` builds the same split from scratch
    for tests and offline smoke runs.
    """

    kind: str = "tiny"
    model_id: str = "Qwen/Qwen3-0.6B"
    tokenizer_id: str = ""
    revision: str = ""
    local_files_only: bool = False
    trust_remote_code: bool = False
    dtype: str = "float32"
    encoder_layers: int = 2
    decoder_layers: int = 0  # 0 = all remaining; N>0 = last N, middle dropped
    freeze_encoder: bool = True
    freeze_decoder: bool = True
    lora_rank: int = 0
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj")
    gradient_checkpointing: bool = False

    # Tiny backbone geometry. Ignored when kind == "hf".
    tiny_vocab_size: int = 0  # 0 means "use the tokenizer's vocabulary"
    tiny_hidden_size: int = 128
    tiny_intermediate_size: int = 256
    tiny_num_layers: int = 4
    tiny_num_heads: int = 4
    tiny_num_kv_heads: int = 4
    tiny_max_position_embeddings: int = 1024

    def __post_init__(self) -> None:
        _tuple_fields(self, ("lora_targets",))
        _require(self.kind in {"tiny", "hf"}, "backbone.kind must be tiny or hf")
        _require(
            self.dtype in {"float32", "bfloat16", "float16"},
            "backbone.dtype must be float32, bfloat16 or float16",
        )
        _require(self.encoder_layers >= 0, "backbone.encoder_layers must be >= 0")
        _require(self.decoder_layers >= 0, "backbone.decoder_layers must be >= 0")
        _require(self.lora_rank >= 0, "backbone.lora_rank must be >= 0")
        _require(self.lora_alpha > 0, "backbone.lora_alpha must be > 0")
        _require(
            0.0 <= self.lora_dropout < 1.0,
            "backbone.lora_dropout must be in [0, 1)",
        )
        if self.lora_rank:
            _require(
                bool(self.lora_targets),
                "backbone.lora_targets cannot be empty when lora_rank > 0",
            )
            _require(
                self.kind == "hf",
                "LoRA targets named transformer projections; use kind=hf",
            )
            _require(
                self.freeze_encoder and self.freeze_decoder,
                "LoRA keeps the base weights frozen; set freeze_encoder and "
                "freeze_decoder to true, or set lora_rank to 0 for full tuning",
            )
        if self.kind == "tiny":
            _require(
                self.tiny_hidden_size > 0
                and self.tiny_num_layers > 0
                and self.tiny_num_heads > 0
                and self.tiny_num_kv_heads > 0,
                "tiny backbone dimensions must be positive",
            )
            _require(
                self.tiny_hidden_size % self.tiny_num_heads == 0,
                "tiny_hidden_size must be divisible by tiny_num_heads",
            )
            _require(
                self.tiny_num_heads % self.tiny_num_kv_heads == 0,
                "tiny_num_heads must be divisible by tiny_num_kv_heads",
            )
            total = self.tiny_num_layers
            decoder = self.decoder_layers or total - self.encoder_layers
            _require(
                self.encoder_layers + decoder == total,
                "tiny encoder_layers + decoder_layers must equal tiny_num_layers",
            )
        else:
            _require(bool(self.model_id), "backbone.model_id cannot be empty")


@dataclass(frozen=True)
class BrainConfig:
    """Geometry and dynamics of the sparse cortical sheet."""

    n_neurons: int = 2048
    edges_per_neuron: int = 96
    weight_mode: str = "sparse"
    init_from_backbone: bool = False
    init_radius: float = 0.95
    spatial: bool = False
    spatial_edges_per_neuron: int = 64
    migration_interval: int = 0
    rewire_interval: int = 0
    migration_speed: float = 0.6
    rewire_fraction: float = 0.01
    rewire_loss_driven: bool = True
    memory_decay: float = 0.0
    recurrent_noise: float = 0.0
    intrinsic_noise: float = 0.0
    pattern_k: int = 0
    slow_decay: float = 0.0
    slow_inhibition: float = 0.0
    slow_delay: int = 0
    slow_ei: bool = False
    intrinsic_drive: float = 0.0
    eligibility_decay: float = 0.0
    consolidation_rate: float = 0.0
    consolidation_interval: int = 32
    dopamine_scale: float = 1.0
    homeostatic_strength: float = 0.0
    short_term_tau: float = 0.0
    short_term_utilization: float = 0.0
    regions: int = 8
    local_fraction: float = 0.6
    inhibitory_fraction: float = 0.2
    weight_scale: float = 1.0
    incoming_norm: float = 2.0
    alpha_min: float = 0.90
    alpha_max: float = 0.97
    gain: float = 1.0
    ticks_per_token: int = 2
    readout: str = "mean"
    drive_scale: float = 4.0
    learnable_gain: bool = True
    learnable_region_gain: bool = True
    detach_interval: int = 32
    seed: int = 0

    def __post_init__(self) -> None:
        _require(self.n_neurons > 0, "brain.n_neurons must be > 0")
        _require(self.edges_per_neuron >= 0, "brain.edges_per_neuron must be >= 0")
        _require(
            self.weight_mode in {"sparse", "dense"},
            "brain.weight_mode must be sparse or dense",
        )
        if self.weight_mode == "dense":
            _require(
                self.edges_per_neuron == 0,
                "brain.edges_per_neuron must be 0 in dense mode",
            )
        _require(self.init_radius > 0, "brain.init_radius must be > 0")
        _require(
            self.spatial_edges_per_neuron > 0,
            "brain.spatial_edges_per_neuron must be > 0",
        )
        _require(
            self.migration_interval >= 0,
            "brain.migration_interval must be >= 0",
        )
        _require(
            self.rewire_interval >= 0,
            "brain.rewire_interval must be >= 0",
        )
        _require(
            self.migration_speed >= 0,
            "brain.migration_speed must be >= 0",
        )
        _require(
            0.0 <= self.rewire_fraction <= 1.0,
            "brain.rewire_fraction must be in [0, 1]",
        )
        _require(
            0.0 <= self.memory_decay < 1.0,
            "brain.memory_decay must be in [0, 1)",
        )
        _require(
            self.recurrent_noise >= 0,
            "brain.recurrent_noise must be >= 0",
        )
        _require(
            self.intrinsic_noise >= 0,
            "brain.intrinsic_noise must be >= 0",
        )
        _require(
            self.pattern_k >= 0,
            "brain.pattern_k must be >= 0",
        )
        _require(
            0.0 <= self.slow_decay < 1.0,
            "brain.slow_decay must be in [0, 1)",
        )
        _require(
            self.slow_inhibition >= 0,
            "brain.slow_inhibition must be >= 0",
        )
        _require(
            self.slow_delay >= 0,
            "brain.slow_delay must be >= 0",
        )
        _require(
            self.intrinsic_drive >= 0,
            "brain.intrinsic_drive must be >= 0",
        )
        _require(
            0.0 <= self.eligibility_decay < 1.0,
            "brain.eligibility_decay must be in [0, 1)",
        )
        _require(
            self.consolidation_rate >= 0,
            "brain.consolidation_rate must be >= 0",
        )
        _require(
            self.consolidation_interval > 0,
            "brain.consolidation_interval must be > 0",
        )
        _require(
            self.dopamine_scale > 0,
            "brain.dopamine_scale must be > 0",
        )
        _require(
            0.0 <= self.homeostatic_strength <= 1.0,
            "brain.homeostatic_strength must be in [0, 1]",
        )
        _require(
            self.short_term_tau >= 0,
            "brain.short_term_tau must be >= 0",
        )
        _require(
            0.0 <= self.short_term_utilization <= 1.0,
            "brain.short_term_utilization must be in [0, 1]",
        )
        if self.spatial:
            _require(
                self.weight_mode == "dense",
                "brain.spatial requires dense weights",
            )
        if self.init_from_backbone:
            _require(
                self.weight_mode == "dense",
                "brain.init_from_backbone requires dense weights",
            )
        _require(self.regions > 0, "brain.regions must be > 0")
        _require(
            self.regions <= self.n_neurons,
            "brain.regions cannot exceed brain.n_neurons",
        )
        _require(
            0.0 <= self.local_fraction <= 1.0,
            "brain.local_fraction must be in [0, 1]",
        )
        _require(
            0.0 <= self.inhibitory_fraction <= 1.0,
            "brain.inhibitory_fraction must be in [0, 1]",
        )
        _require(self.weight_scale > 0, "brain.weight_scale must be > 0")
        _require(self.incoming_norm > 0, "brain.incoming_norm must be > 0")
        _require(
            0.0 <= self.alpha_min < self.alpha_max < 1.0,
            "brain alpha range must satisfy 0 <= min < max < 1",
        )
        _require(self.gain > 0, "brain.gain must be > 0")
        _require(self.ticks_per_token >= 1, "brain.ticks_per_token must be >= 1")
        _require(
            self.readout in {"mean", "last"},
            "brain.readout must be mean or last",
        )
        _require(self.drive_scale > 0, "brain.drive_scale must be > 0")
        _require(self.detach_interval >= 0, "brain.detach_interval must be >= 0")
        _require(self.seed >= 0, "brain.seed must be >= 0")

    @property
    def n_edges(self) -> int:
        return self.n_neurons * self.edges_per_neuron


@dataclass(frozen=True)
class ModelConfig:
    use_brain: bool = True
    brain_gate_init: float = 0.05
    dropout: float = 0.0
    label_smoothing: float = 0.0
    z_loss: float = 0.0
    eos_weight: float = 1.0
    bos_weight: float = 1.0
    open_loop_weight: float = 0.0
    open_loop_tokens: int = 8
    open_loop_probability: float = 0.25
    differentiation_weight: float = 0.0
    low_freq_weight: float = 0.0
    low_freq_lag: int = 16
    low_freq_target: float = 0.6

    def __post_init__(self) -> None:
        _require(
            abs(self.brain_gate_init) <= 1.0,
            "model.brain_gate_init must be in [-1, 1]",
        )
        _require(0.0 <= self.dropout < 1.0, "model.dropout must be in [0, 1)")
        _require(
            0.0 <= self.label_smoothing < 1.0,
            "model.label_smoothing must be in [0, 1)",
        )
        _require(self.z_loss >= 0, "model.z_loss must be >= 0")
        _require(self.eos_weight > 0, "model.eos_weight must be > 0")
        _require(self.bos_weight > 0, "model.bos_weight must be > 0")
        _require(
            self.open_loop_weight >= 0,
            "model.open_loop_weight must be >= 0",
        )
        _require(
            self.open_loop_tokens > 0,
            "model.open_loop_tokens must be > 0",
        )
        _require(
            0.0 <= self.open_loop_probability <= 1.0,
            "model.open_loop_probability must be in [0, 1]",
        )
        _require(
            self.differentiation_weight >= 0,
            "model.differentiation_weight must be >= 0",
        )
        _require(
            self.low_freq_weight >= 0,
            "model.low_freq_weight must be >= 0",
        )
        _require(
            self.low_freq_lag > 0,
            "model.low_freq_lag must be > 0",
        )
        _require(
            0.0 <= self.low_freq_target <= 1.0,
            "model.low_freq_target must be in [0, 1]",
        )


@dataclass(frozen=True)
class DataConfig:
    corpus: tuple[str, ...] = ()
    tokenizer: str = "hf"
    block_size: int = 512
    batch_size: int = 1
    eval_batch_size: int = 1
    num_workers: int = 0
    train_split: float = 0.98
    add_bos: bool = False
    max_tokens: int = 0
    max_bytes: int = 0
    max_documents: int = 0
    text_field: str = "text"
    cache_dir: str = "data/cortex_cache"
    extensions: tuple[str, ...] = (".txt", ".md", ".jsonl")
    seed: int = 0

    def __post_init__(self) -> None:
        _tuple_fields(self, ("corpus", "extensions"))
        if isinstance(self.corpus, str):
            object.__setattr__(self, "corpus", (self.corpus,))
        if isinstance(self.extensions, str):
            object.__setattr__(self, "extensions", (self.extensions,))
        _require(self.tokenizer in {"hf", "byte"}, "data.tokenizer must be hf or byte")
        _require(self.block_size >= 2, "data.block_size must be >= 2")
        _require(self.batch_size >= 1, "data.batch_size must be >= 1")
        _require(self.eval_batch_size >= 1, "data.eval_batch_size must be >= 1")
        _require(self.num_workers >= 0, "data.num_workers must be >= 0")
        _require(
            0.0 < self.train_split < 1.0,
            "data.train_split must be in (0, 1)",
        )
        _require(self.max_tokens >= 0, "data.max_tokens must be >= 0")
        _require(self.max_bytes >= 0, "data.max_bytes must be >= 0")
        _require(self.max_documents >= 0, "data.max_documents must be >= 0")
        _require(bool(self.text_field), "data.text_field cannot be empty")
        _require(
            all(ext.startswith(".") for ext in self.extensions),
            "data.extensions entries must start with '.'",
        )
        _require(self.seed >= 0, "data.seed must be >= 0")


@dataclass(frozen=True)
class TrainConfig:
    output_dir: str = "runs/cortex"
    resume: str = ""
    max_steps: int = 20000
    lr: float = 2e-4
    brain_lr: float = 1e-3
    lora_lr: float = 2e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    grad_accum: int = 1
    amp: str = "bf16"
    eval_interval: int = 500
    eval_batches: int = 32
    save_interval: int = 500
    log_interval: int = 10
    sample_interval: int = 0
    sample_prompt: str = "中国的首都是"
    sample_tokens: int = 24
    sample_temperature: float = 0.8
    seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        _tuple_fields(self, ("betas",))
        _require(self.max_steps >= 1, "train.max_steps must be >= 1")
        _require(self.lr > 0, "train.lr must be > 0")
        _require(self.brain_lr > 0, "train.brain_lr must be > 0")
        _require(self.lora_lr > 0, "train.lora_lr must be > 0")
        _require(self.weight_decay >= 0, "train.weight_decay must be >= 0")
        _require(
            len(self.betas) == 2 and all(0.0 <= b < 1.0 for b in self.betas),
            "train.betas must contain two values in [0, 1)",
        )
        _require(self.warmup_steps >= 0, "train.warmup_steps must be >= 0")
        _require(
            0.0 <= self.min_lr_ratio <= 1.0,
            "train.min_lr_ratio must be in [0, 1]",
        )
        _require(self.grad_clip >= 0, "train.grad_clip must be >= 0")
        _require(self.grad_accum >= 1, "train.grad_accum must be >= 1")
        _require(self.amp in {"none", "bf16", "fp16"}, "train.amp must be none, bf16 or fp16")
        _require(self.eval_interval >= 1, "train.eval_interval must be >= 1")
        _require(self.eval_batches >= 1, "train.eval_batches must be >= 1")
        _require(self.save_interval >= 1, "train.save_interval must be >= 1")
        _require(self.log_interval >= 1, "train.log_interval must be >= 1")
        _require(self.sample_interval >= 0, "train.sample_interval must be >= 0")
        _require(self.sample_tokens >= 1, "train.sample_tokens must be >= 1")
        _require(
            0.0 <= self.sample_temperature,
            "train.sample_temperature must be >= 0",
        )
        _require(self.seed >= 0, "train.seed must be >= 0")
        _require(
            self.device in {"auto", "cpu", "cuda", "cuda:0"},
            "train.device must be auto, cpu, cuda or cuda:0",
        )


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "cortex"
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        _require(bool(self.name), "name cannot be empty")
        if not self.model.use_brain:
            trainable_backbone = (
                not self.backbone.freeze_decoder
                or not self.backbone.freeze_encoder
                or self.backbone.lora_rank > 0
            )
            _require(
                trainable_backbone,
                "use_brain=false leaves nothing trainable: unfreeze a backbone "
                "half or enable LoRA",
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        raw = dict(raw)
        unknown = sorted(
            set(raw) - {"name", "backbone", "brain", "model", "data", "train"}
        )
        if unknown:
            raise ConfigError(
                f"unknown ExperimentConfig field(s): {', '.join(unknown)}"
            )
        return cls(
            name=raw.get("name", "cortex"),
            backbone=BackboneConfig(
                **_check_unknown(BackboneConfig, raw.get("backbone", {}))
            ),
            brain=BrainConfig(
                **_check_unknown(BrainConfig, raw.get("brain", {}))
            ),
            model=ModelConfig(**_check_unknown(ModelConfig, raw.get("model", {}))),
            data=DataConfig(**_check_unknown(DataConfig, raw.get("data", {}))),
            train=TrainConfig(**_check_unknown(TrainConfig, raw.get("train", {}))),
        )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a JSON experiment config."""

    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ConfigError("the top-level config must be a JSON object")
    return ExperimentConfig.from_dict(raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _parse_override_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_overrides(
    raw: Mapping[str, Any],
    overrides: Iterable[str],
) -> dict[str, Any]:
    """Apply ``dotted.path=json_value`` overrides to a raw config dict.

    Values are parsed as JSON when possible, so ``train.amp=none`` stays a
    string because bare ``none`` is not valid JSON, while
    ``brain.ticks_per_token=4`` becomes an int and ``data.corpus=["a.txt"]``
    becomes a list.
    """

    result = json.loads(json.dumps(raw))
    for item in overrides:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ConfigError(f"override must look like key=value, got {item!r}")
        node: Any = result
        parts = key.split(".")
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"override path {key!r} does not exist")
            node = node[part]
        leaf = parts[-1]
        if not isinstance(node, dict) or leaf not in node:
            raise ConfigError(f"override path {key!r} does not exist")
        node[leaf] = _parse_override_value(value)
    return result
