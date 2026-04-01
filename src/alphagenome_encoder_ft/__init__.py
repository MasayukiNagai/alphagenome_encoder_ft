"""Encoder-only AlphaGenome fine-tuning utilities."""

__all__ = [
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
    "ConstructSpec",
    "LENTIMPRA_BARCODE",
    "LENTIMPRA_LEFT_ADAPTER",
    "LENTIMPRA_PROMOTER",
    "LENTIMPRA_RIGHT_ADAPTER",
    "EncoderMPRAModel",
    "MPRAHead",
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
        "LentiMPRADataset",
        "create_dataloader",
    }:
        from . import data

        return getattr(data, name)
    if name in {
        "ConstructSpec",
        "LENTIMPRA_BARCODE",
        "LENTIMPRA_LEFT_ADAPTER",
        "LENTIMPRA_PROMOTER",
        "LENTIMPRA_RIGHT_ADAPTER",
    }:
        from . import constructs

        return getattr(constructs, name)
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
    if name == "EncoderMPRAModel":
        from . import model

        return getattr(model, name)
    if name == "MPRAHead":
        from .heads import MPRAHead

        return MPRAHead
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
