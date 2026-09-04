from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .protocols import get_profile


@dataclass
class DataConfig:
    dataset: str = "fi2010"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    normalization: str = "decimal"
    validation_fraction: float = 0.20
    sequence_length: int = 100
    horizons: list[int] = field(default_factory=lambda: [10, 20, 50])


@dataclass
class ModelConfig:
    names: list[str] = field(
        default_factory=lambda: ["deeplob", "linear", "mlp", "cnn", "lstm", "mlplob"]
    )
    conv_channels: int = 32
    inception_channels: int = 64
    lstm_hidden: int = 64
    batch_norm: bool = False
    dropout: float = 0.20
    dropout_shared_time: bool = True
    dropout_at_inference: bool = True
    time_padding: str = "same"
    second_block_activation: str = "leaky_relu"
    output_activation: str = "logits"
    mlp_hidden: int = 128
    cnn_hidden: int = 64
    mlplob_hidden: int = 40
    mlplob_layers: int = 2


@dataclass
class TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 1e-4
    adam_eps: float = 1e-7
    max_epochs: int = 200
    patience: int = 20
    monitor: str = "val_loss"
    early_stopping: bool = False
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    deterministic: bool = True


@dataclass
class EvaluationConfig:
    mc_dropout_repeats: int = 5
    dropout_at_inference_override: bool | None = None
    checkpoint_source_run: str | None = None


@dataclass
class RunConfig:
    protocol: str = "author_tf1"
    run_name: str = "fi2010-author-tf1"
    output_dir: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _merge(obj: Any, patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if not hasattr(obj, key):
            raise ValueError(f"Unknown config field: {type(obj).__name__}.{key}")
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: str | Path) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    protocol = raw.get("protocol", "author_tf1")
    cfg = RunConfig(protocol=protocol)
    _merge(cfg, get_profile(protocol))
    _merge(cfg, raw)
    validate_config(cfg)
    return cfg


def validate_config(cfg: RunConfig) -> None:
    if cfg.data.dataset not in {"fi2010", "synthetic"}:
        raise ValueError("data.dataset must be 'fi2010' or 'synthetic'")
    if cfg.data.normalization not in {"decimal", "zscore"}:
        raise ValueError("data.normalization must be 'decimal' or 'zscore'")
    if not 0 < cfg.data.validation_fraction < 1:
        raise ValueError("validation_fraction must lie in (0, 1)")
    if cfg.data.sequence_length <= 0:
        raise ValueError("data.sequence_length must be positive")
    if not cfg.data.horizons:
        raise ValueError("data.horizons must not be empty")
    supported = {10, 20, 30, 50, 100}
    if not set(cfg.data.horizons).issubset(supported):
        raise ValueError(f"FI-2010 horizons must be a subset of {sorted(supported)}")
    if not cfg.models.names:
        raise ValueError("models.names must not be empty")
    if cfg.models.conv_channels <= 0 or cfg.models.inception_channels <= 0:
        raise ValueError("DeepLOB convolution channel counts must be positive")
    if cfg.models.lstm_hidden <= 0:
        raise ValueError("models.lstm_hidden must be positive")
    if not 0 <= cfg.models.dropout < 1:
        raise ValueError("models.dropout must lie in [0, 1)")
    if cfg.models.time_padding not in {"same", "valid"}:
        raise ValueError("models.time_padding must be same or valid")
    if cfg.models.second_block_activation not in {"leaky_relu", "tanh"}:
        raise ValueError("models.second_block_activation must be leaky_relu or tanh")
    if cfg.models.output_activation not in {"logits", "softmax"}:
        raise ValueError("models.output_activation must be logits or softmax")
    if cfg.training.batch_size <= 0:
        raise ValueError("training.batch_size must be positive")
    if cfg.training.learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive")
    if cfg.training.adam_eps <= 0:
        raise ValueError("training.adam_eps must be positive")
    if cfg.training.max_epochs <= 0:
        raise ValueError("training.max_epochs must be positive")
    if cfg.training.monitor not in {"val_loss", "val_accuracy"}:
        raise ValueError("training.monitor must be val_loss or val_accuracy")
    if cfg.training.patience < 1:
        raise ValueError("training.patience must be positive")
    if cfg.training.num_workers < 0:
        raise ValueError("training.num_workers must be non-negative")
    if cfg.evaluation.mc_dropout_repeats < 1:
        raise ValueError("evaluation.mc_dropout_repeats must be positive")
    known_models = {"deeplob", "linear", "mlp", "cnn", "lstm", "mlplob"}
    unknown = set(cfg.models.names) - known_models
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
