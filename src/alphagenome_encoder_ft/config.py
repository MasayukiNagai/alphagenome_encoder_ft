"""Normalized training configuration for encoder-only MPRA fine-tuning."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


def parse_hidden_sizes(value: int | str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, int):
        sizes = [value]
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("hidden_sizes must not be empty")
        sizes = [int(piece.strip()) for piece in stripped.split(",") if piece.strip()]
    else:
        sizes = [int(piece) for piece in value]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("hidden_sizes must contain positive integers")
    return sizes


def _ensure_mapping(value: Any, *, section: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected '{section}' to be a JSON object")
    return value


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class DataConfig:
    input_tsv: str | None = None
    sequence_length: int = 256
    construct_mode: str = "flanked"
    batch_size: int = 32
    reverse_complement: bool = False
    rc_prob: float = 0.5
    random_shift: bool = False
    shift_prob: float = 0.5
    max_shift: int = 15
    subset_frac: float = 1.0
    num_workers: int = 0
    pin_memory: bool = False
    left_adapter_seq: str | None = None
    right_adapter_seq: str | None = None
    promoter_seq: str | None = None
    barcode_seq: str | None = None

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("data.sequence_length must be > 0")
        if self.construct_mode not in {"core", "flanked", "full"}:
            raise ValueError("data.construct_mode must be one of core, flanked, full")
        if not 0 < self.subset_frac <= 1:
            raise ValueError("data.subset_frac must be in (0, 1]")
        if not 0 <= self.rc_prob <= 1:
            raise ValueError("data.rc_prob must be in [0, 1]")
        if not 0 <= self.shift_prob <= 1:
            raise ValueError("data.shift_prob must be in [0, 1]")
        if self.max_shift < 0:
            raise ValueError("data.max_shift must be >= 0")
        if self.batch_size <= 0:
            raise ValueError("data.batch_size must be > 0")
        if self.num_workers < 0:
            raise ValueError("data.num_workers must be >= 0")


@dataclass
class HeadConfig:
    pooling_type: str = "flatten"
    center_bp: int = 256
    hidden_sizes: list[int] = field(default_factory=lambda: [1024])
    dropout: float = 0.1
    activation: str = "relu"

    def __post_init__(self) -> None:
        self.hidden_sizes = parse_hidden_sizes(self.hidden_sizes)
        if self.pooling_type not in {"flatten", "center", "mean", "sum", "max"}:
            raise ValueError("head.pooling_type must be one of flatten, center, mean, sum, max")
        if self.center_bp <= 0:
            raise ValueError("head.center_bp must be > 0")
        if not 0 <= self.dropout < 1:
            raise ValueError("head.dropout must be in [0, 1)")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError("head.activation must be 'relu' or 'gelu'")


@dataclass
class OptimConfig:
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    lr_scheduler: str = "constant"
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    plateau_mode: str = "min"
    plateau_min_lr: float = 0.0
    gradient_accumulation_steps: int = 1
    gradient_clip: float | None = None

    def __post_init__(self) -> None:
        if self.optimizer not in {"adam", "adamw"}:
            raise ValueError("optim.optimizer must be 'adam' or 'adamw'")
        if self.learning_rate <= 0:
            raise ValueError("optim.learning_rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("optim.weight_decay must be >= 0")
        if self.lr_scheduler not in {"constant", "cosine", "plateau"}:
            raise ValueError("optim.lr_scheduler must be one of constant, cosine, plateau")
        if not 0 < self.plateau_factor < 1:
            raise ValueError("optim.plateau_factor must be in (0, 1)")
        if self.plateau_patience < 0:
            raise ValueError("optim.plateau_patience must be >= 0")
        if self.plateau_mode != "min":
            raise ValueError("optim.plateau_mode must be 'min'")
        if self.plateau_min_lr < 0:
            raise ValueError("optim.plateau_min_lr must be >= 0")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("optim.gradient_accumulation_steps must be > 0")
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("optim.gradient_clip must be > 0 when set")


@dataclass
class StageConfig:
    num_epochs: int = 10
    early_stopping_patience: int = 5
    val_evals_per_epoch: int = 1
    second_stage_lr: float | None = None
    second_stage_epochs: int = 10
    resume_from_stage2: bool = False

    def __post_init__(self) -> None:
        if self.num_epochs <= 0:
            raise ValueError("stage.num_epochs must be > 0")
        if self.early_stopping_patience < 0:
            raise ValueError("stage.early_stopping_patience must be >= 0")
        if self.val_evals_per_epoch <= 0:
            raise ValueError("stage.val_evals_per_epoch must be > 0")
        if self.second_stage_lr is not None and self.second_stage_lr <= 0:
            raise ValueError("stage.second_stage_lr must be > 0 when set")
        if self.second_stage_epochs <= 0:
            raise ValueError("stage.second_stage_epochs must be > 0")


@dataclass
class CheckpointConfig:
    pretrained_weights: str | None = None
    checkpoint_dir: str = "./checkpoints_mpra"
    save_mode: str = "minimal"

    def __post_init__(self) -> None:
        if self.save_mode not in {"minimal", "full", "head"}:
            raise ValueError("checkpoint.save_mode must be one of minimal, full, head")


@dataclass
class LoggingConfig:
    use_wandb: bool = False
    wandb_project: str = "alphagenome-mpra"
    wandb_name: str = "mpra-head-encoder"


@dataclass
class RuntimeConfig:
    device: str | None = None
    use_amp: bool = False
    seed: int = 42


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    stage: StageConfig = field(default_factory=StageConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        if not self.data.input_tsv:
            raise ValueError("data.input_tsv must be provided via config or CLI")
        if not self.checkpoint.pretrained_weights:
            raise ValueError("checkpoint.pretrained_weights must be provided via config or CLI")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def head_kwargs(self) -> dict[str, Any]:
        return {
            "pooling_type": self.head.pooling_type,
            "center_bp": self.head.center_bp,
            "hidden_sizes": list(self.head.hidden_sizes),
            "dropout": self.head.dropout,
            "activation": self.head.activation,
        }

    def construct_config(self) -> dict[str, Any]:
        return {
            "left_adapter": self.data.left_adapter_seq,
            "right_adapter": self.data.right_adapter_seq,
            "promoter_seq": self.data.promoter_seq,
            "barcode_seq": self.data.barcode_seq,
            "construct_mode": self.data.construct_mode,
            "sequence_length": self.data.sequence_length,
        }

    @classmethod
    def from_dict(cls, raw_config: Mapping[str, Any]) -> "TrainConfig":
        allowed_sections = {"data", "head", "optim", "stage", "checkpoint", "logging", "runtime"}
        unknown_sections = sorted(
            key for key in set(raw_config) - allowed_sections if not str(key).startswith("_")
        )
        if unknown_sections:
            raise ValueError(f"Unknown config sections: {', '.join(unknown_sections)}")

        return cls(
            data=DataConfig(**dict(_ensure_mapping(raw_config.get("data", {}), section="data"))),
            head=HeadConfig(**dict(_ensure_mapping(raw_config.get("head", {}), section="head"))),
            optim=OptimConfig(**dict(_ensure_mapping(raw_config.get("optim", {}), section="optim"))),
            stage=StageConfig(**dict(_ensure_mapping(raw_config.get("stage", {}), section="stage"))),
            checkpoint=CheckpointConfig(
                **dict(_ensure_mapping(raw_config.get("checkpoint", {}), section="checkpoint"))
            ),
            logging=LoggingConfig(**dict(_ensure_mapping(raw_config.get("logging", {}), section="logging"))),
            runtime=RuntimeConfig(**dict(_ensure_mapping(raw_config.get("runtime", {}), section="runtime"))),
        )


def load_train_config(path: str | Path | None) -> TrainConfig:
    if path is None:
        return TrainConfig()
    with open(path) as handle:
        raw_config = json.load(handle)
    return TrainConfig.from_dict(raw_config)


def merge_train_config(config: TrainConfig, overrides: Mapping[str, Any]) -> TrainConfig:
    merged = _deep_merge(config.to_dict(), overrides)
    return TrainConfig.from_dict(merged)
