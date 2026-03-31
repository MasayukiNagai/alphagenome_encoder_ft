from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from alphagenome_encoder_ft.config import TrainConfig
from alphagenome_encoder_ft.heads import MPRAHead
from alphagenome_encoder_ft.model import EncoderMPRAModel
from alphagenome_encoder_ft.train import evaluate, load_checkpoint, run_training_stage, run_two_stage_training


class DummyAlphaGenome(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1536),
        )

    def forward(self, sequences, organism_idx, encoder_only=False):
        del organism_idx
        if not encoder_only:
            raise ValueError("Dummy model only supports encoder_only=True")
        batch, length, channels = sequences.shape
        encoded = self.encoder(sequences.reshape(batch * length, channels)).reshape(batch, length, 1536)
        return {"encoder_output": encoded}


def _make_loader():
    torch.manual_seed(0)
    sequences = torch.randn(12, 2, 4)
    targets = sequences.sum(dim=(1, 2))
    return DataLoader(TensorDataset(sequences, targets), batch_size=4, shuffle=False)


def _make_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig.from_dict(
        {
            "data": {
                "input_tsv": "/tmp/mock.tsv",
                "sequence_length": 256,
                "batch_size": 4,
            },
            "head": {
                "pooling_type": "flatten",
                "hidden_sizes": [8],
                "center_bp": 256,
                "dropout": 0.1,
                "activation": "relu",
            },
            "optim": {
                "optimizer": "adam",
                "learning_rate": 1e-2,
                "weight_decay": 0.0,
                "lr_scheduler": "constant",
                "gradient_accumulation_steps": 1,
            },
            "stage": {
                "num_epochs": 2,
                "early_stopping_patience": 5,
                "val_eval_frequency": 1,
                "second_stage_lr": 1e-3,
                "second_stage_epochs": 1,
            },
            "checkpoint": {
                "pretrained_weights": "/tmp/weights.pt",
                "checkpoint_dir": str(tmp_path),
                "save_mode": "minimal",
            },
            "runtime": {
                "use_amp": False,
                "seed": 0,
            },
        }
    )


def _make_model() -> EncoderMPRAModel:
    model = EncoderMPRAModel(DummyAlphaGenome(), MPRAHead(pooling_type="flatten", hidden_sizes=8))
    model.initialize_head(sequence_length=2, device="cpu")
    return model


def test_run_training_stage_writes_minimal_checkpoint(tmp_path: Path):
    model = _make_model()
    loader = _make_loader()
    config = _make_config(tmp_path)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    result = run_training_stage(
        model,
        loader,
        optimizer=optimizer,
        config=config,
        device="cpu",
        num_epochs=2,
        stage="stage1",
        train_encoder=False,
        checkpoint_dir=tmp_path / "stage1",
    )

    assert (tmp_path / "stage1" / "best.pt").exists()
    assert result["best_checkpoint_path"] is not None


def test_two_stage_training_rejects_head_mode(tmp_path: Path):
    model = _make_model()
    loader = _make_loader()
    config = _make_config(tmp_path)
    config.checkpoint.save_mode = "head"
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    try:
        run_two_stage_training(
            model,
            loader,
            stage1_optimizer=optimizer,
            stage2_optimizer_factory=lambda model_obj: torch.optim.Adam(
                model_obj.trainable_parameters(include_encoder=True),
                lr=1e-3,
            ),
            config=config,
            device="cpu",
        )
    except ValueError as exc:
        assert "head save_mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for head save_mode")


def test_resume_from_stage2_loads_stage1_checkpoint(tmp_path: Path):
    model = _make_model()
    loader = _make_loader()
    config = _make_config(tmp_path)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    run_training_stage(
        model,
        loader,
        optimizer=optimizer,
        config=config,
        device="cpu",
        num_epochs=1,
        stage="stage1",
        train_encoder=False,
        checkpoint_dir=tmp_path / "stage1",
    )

    stage2_model = _make_model()
    stage2_config = _make_config(tmp_path)
    stage2_config.stage.resume_from_stage2 = True
    result = run_two_stage_training(
        stage2_model,
        loader,
        stage1_optimizer=torch.optim.Adam(stage2_model.head.parameters(), lr=1e-2),
        stage2_optimizer_factory=lambda model_obj: torch.optim.Adam(
            model_obj.trainable_parameters(include_encoder=True),
            lr=1e-3,
        ),
        config=stage2_config,
        device="cpu",
    )

    assert result["stage2"]["best_checkpoint_path"] is not None


def test_run_training_stage_respects_eval_frequency_and_emits_callbacks(tmp_path: Path):
    model = _make_model()
    train_loader = _make_loader()
    val_loader = _make_loader()
    config = _make_config(tmp_path)
    config.stage.num_epochs = 3
    config.stage.val_eval_frequency = 2
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)
    epoch_events = []

    result = run_training_stage(
        model,
        train_loader,
        optimizer=optimizer,
        config=config,
        device="cpu",
        num_epochs=3,
        stage="stage1",
        train_encoder=False,
        val_loader=val_loader,
        checkpoint_dir=tmp_path / "stage1",
        epoch_callback=epoch_events.append,
    )

    assert len(result["history"]["train_loss"]) == 3
    assert result["history"]["val_epoch"] == [2.0, 3.0]
    assert result["history"]["test_epoch"] == []
    assert [event["epoch"] for event in epoch_events] == [1.0, 2.0, 3.0]
    assert "val_loss" not in epoch_events[0]
    assert "test_loss" not in epoch_events[-1]
    assert epoch_events[1]["val_loss"] >= 0.0


def test_load_checkpoint_then_evaluate_best_checkpoint(tmp_path: Path):
    model = _make_model()
    train_loader = _make_loader()
    test_loader = _make_loader()
    config = _make_config(tmp_path)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    result = run_training_stage(
        model,
        train_loader,
        optimizer=optimizer,
        config=config,
        device="cpu",
        num_epochs=2,
        stage="stage1",
        train_encoder=False,
        checkpoint_dir=tmp_path / "stage1",
    )

    load_checkpoint(result["best_checkpoint_path"], model, map_location="cpu")
    metrics = evaluate(model, test_loader, device="cpu")

    assert metrics["loss"] >= 0.0
    assert "pearson" in metrics
