"""Encoder-only AlphaGenome fine-tuning utilities."""

__all__ = [
    "DEFAULT_BARCODE_SEQ",
    "DEFAULT_FOLD_SPLITS",
    "DEFAULT_LEFT_ADAPTER_SEQ",
    "DEFAULT_PROMOTER_SEQ",
    "DEFAULT_RIGHT_ADAPTER_SEQ",
    "DataConfig",
    "HeadConfig",
    "OptimConfig",
    "StageConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "RuntimeConfig",
    "TrainConfig",
    "load_train_config",
    "merge_train_config",
    "parse_hidden_sizes",
    "LentiMPRADataset",
    "create_dataloader",
    "EncoderMPRAModel",
    "load_pretrained_model",
    "MPRAHead",
    "MPRAOracle",
    "load_oracle",
    "train_epoch",
    "evaluate",
    "run_training_stage",
    "run_two_stage_training",
    "save_checkpoint",
    "load_checkpoint",
    "set_encoder_trainable",
    "create_optimizer",
    "create_scheduler",
    "scheduler_stepper",
]


def __getattr__(name: str):
    if name in {
        "DEFAULT_BARCODE_SEQ",
        "DEFAULT_FOLD_SPLITS",
        "DEFAULT_LEFT_ADAPTER_SEQ",
        "DEFAULT_PROMOTER_SEQ",
        "DEFAULT_RIGHT_ADAPTER_SEQ",
        "LentiMPRADataset",
        "create_dataloader",
    }:
        from . import data

        return getattr(data, name)
    if name in {
        "DataConfig",
        "HeadConfig",
        "OptimConfig",
        "StageConfig",
        "CheckpointConfig",
        "LoggingConfig",
        "RuntimeConfig",
        "TrainConfig",
        "load_train_config",
        "merge_train_config",
        "parse_hidden_sizes",
    }:
        from . import config

        return getattr(config, name)
    if name in {"EncoderMPRAModel", "load_pretrained_model"}:
        from . import model

        return getattr(model, name)
    if name == "MPRAHead":
        from .heads import MPRAHead

        return MPRAHead
    if name in {"MPRAOracle", "load_oracle"}:
        from . import oracle

        return getattr(oracle, name)
    if name in {
        "train_epoch",
        "evaluate",
        "run_training_stage",
        "run_two_stage_training",
        "save_checkpoint",
        "load_checkpoint",
        "set_encoder_trainable",
        "create_optimizer",
        "create_scheduler",
        "scheduler_stepper",
    }:
        from . import train

        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
