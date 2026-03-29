from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from alphagenome_encoder_ft.heads import MPRAHead
from alphagenome_encoder_ft.train import (
    load_checkpoint,
    run_training_stage,
    run_two_stage_training,
)


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


def test_run_training_stage_writes_minimal_checkpoint(tmp_path: Path):
    model = DummyAlphaGenome()
    head = MPRAHead(pooling_type="flatten", hidden_sizes=8)
    loader = _make_loader()
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)

    result = run_training_stage(
        model,
        head,
        loader,
        optimizer=optimizer,
        device="cpu",
        num_epochs=2,
        stage="stage1",
        encoder_trainable=False,
        checkpoint_dir=tmp_path / "stage1",
        save_mode="minimal",
        head_config={"pooling_type": "flatten", "hidden_sizes": 8, "center_bp": 256, "dropout": 0.1, "activation": "relu"},
        construct_config={"left_adapter": "A", "right_adapter": "C", "promoter_seq": "G", "barcode_seq": "T", "sequence_length": 256},
    )

    assert (tmp_path / "stage1" / "best.pt").exists()
    assert result["best_checkpoint_path"] is not None


def test_two_stage_training_rejects_head_mode(tmp_path: Path):
    model = DummyAlphaGenome()
    head = MPRAHead(pooling_type="flatten", hidden_sizes=8)
    loader = _make_loader()
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)

    try:
        run_two_stage_training(
            model,
            head,
            loader,
            stage1_optimizer=optimizer,
            stage2_optimizer_factory=lambda model_obj, head_obj: torch.optim.Adam(
                list(model_obj.encoder.parameters()) + list(head_obj.parameters()),
                lr=1e-3,
            ),
            device="cpu",
            stage1_num_epochs=1,
            stage2_num_epochs=1,
            checkpoint_dir=tmp_path,
            save_mode="head",
            head_config={"pooling_type": "flatten", "hidden_sizes": 8, "center_bp": 256, "dropout": 0.1, "activation": "relu"},
            construct_config={"left_adapter": "A", "right_adapter": "C", "promoter_seq": "G", "barcode_seq": "T", "sequence_length": 256},
        )
    except ValueError as exc:
        assert "head save_mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for head save_mode")


def test_resume_from_stage2_loads_stage1_checkpoint(tmp_path: Path):
    model = DummyAlphaGenome()
    head = MPRAHead(pooling_type="flatten", hidden_sizes=8)
    loader = _make_loader()
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)

    run_training_stage(
        model,
        head,
        loader,
        optimizer=optimizer,
        device="cpu",
        num_epochs=1,
        stage="stage1",
        encoder_trainable=False,
        checkpoint_dir=tmp_path / "stage1",
        save_mode="minimal",
        head_config={"pooling_type": "flatten", "hidden_sizes": 8, "center_bp": 256, "dropout": 0.1, "activation": "relu"},
        construct_config={"left_adapter": "A", "right_adapter": "C", "promoter_seq": "G", "barcode_seq": "T", "sequence_length": 256},
    )

    stage2_model = DummyAlphaGenome()
    stage2_head = MPRAHead(pooling_type="flatten", hidden_sizes=8)
    result = run_two_stage_training(
        stage2_model,
        stage2_head,
        loader,
        stage1_optimizer=torch.optim.Adam(stage2_head.parameters(), lr=1e-2),
        stage2_optimizer_factory=lambda model_obj, head_obj: torch.optim.Adam(
            list(model_obj.encoder.parameters()) + list(head_obj.parameters()),
            lr=1e-3,
        ),
        device="cpu",
        stage1_num_epochs=1,
        stage2_num_epochs=1,
        checkpoint_dir=tmp_path,
        save_mode="minimal",
        head_config={"pooling_type": "flatten", "hidden_sizes": 8, "center_bp": 256, "dropout": 0.1, "activation": "relu"},
        construct_config={"left_adapter": "A", "right_adapter": "C", "promoter_seq": "G", "barcode_seq": "T", "sequence_length": 256},
        resume_from_stage2=True,
    )

    assert result["stage2"]["best_checkpoint_path"] is not None
