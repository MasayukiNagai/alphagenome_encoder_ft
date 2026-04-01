from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from alphagenome_encoder_ft.config import TrainConfig
from alphagenome_encoder_ft.constructs import ConstructSpec
from alphagenome_encoder_ft.heads import MPRAHead
from alphagenome_encoder_ft.model import EncoderMPRAModel
from alphagenome_encoder_ft.train import save_checkpoint


class DummyAlphaGenome(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1536),
        )

    def forward(self, sequences, organism_idx, encoder_only=False):
        del organism_idx
        if not encoder_only:
            raise ValueError("Dummy model only supports encoder_only=True")
        batch, length, channels = sequences.shape
        encoded = self.encoder(sequences.reshape(batch * length, channels)).reshape(batch, length, 1536)
        return {"encoder_output": encoded}


def _make_config(tmp_path: Path, *, save_mode: str = "minimal") -> TrainConfig:
    return TrainConfig.from_dict(
        {
            "data": {
                "input_tsv": "/tmp/mock.tsv",
                "sequence_length": 2,
                "left_adapter_seq": "A",
                "right_adapter_seq": "C",
                "promoter_seq": "G",
                "barcode_seq": "T",
            },
            "head": {
                "pooling_type": "flatten",
                "hidden_sizes": [8],
                "center_bp": 256,
                "dropout": 0.1,
                "activation": "relu",
            },
            "checkpoint": {
                "pretrained_weights": "/tmp/weights.pt",
                "checkpoint_dir": str(tmp_path),
                "save_mode": save_mode,
            },
        }
    )


def test_construct_spec_assembles_sequences_for_all_modes():
    spec = ConstructSpec(left_adapter="A", right_adapter="C", promoter_seq="G", barcode_seq="T")

    assert spec.assemble_sequence("ac", mode="core") == "AACCGT"
    assert spec.assemble_sequence("ac", mode="flanked") == "ACGT"
    assert spec.assemble_sequence("ac", mode="full") == "AC"


def test_construct_spec_rejects_missing_required_parts():
    spec = ConstructSpec(left_adapter=None, right_adapter=None, promoter_seq=None, barcode_seq=None)

    with pytest.raises(ValueError, match="requires construct components"):
        spec.assemble_sequence("ac", mode="core")
    with pytest.raises(ValueError, match="requires construct components"):
        spec.assemble_sequence("ac", mode="flanked")
    assert spec.assemble_sequence("ac", mode="full") == "AC"


def test_construct_spec_assembles_onehot_for_rank_2_and_rank_3():
    spec = ConstructSpec(left_adapter="A", right_adapter="C", promoter_seq="G", barcode_seq="T")
    single = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    batch = torch.stack([single, single], dim=0)

    assembled_single = spec.assemble_onehot(single, mode="core")
    assembled_batch = spec.assemble_onehot(batch, mode="flanked")

    assert assembled_single.shape == (6, 4)
    assert assembled_batch.shape == (2, 4, 4)
    assert assembled_single.dtype == single.dtype
    assert assembled_batch.dtype == batch.dtype


def test_construct_spec_rejects_invalid_shapes_and_modes():
    spec = ConstructSpec()

    with pytest.raises(ValueError, match="Invalid mode"):
        spec.assemble_sequence("AC", mode="bad")
    with pytest.raises(ValueError, match="rank 2 or 3"):
        spec.assemble_onehot(torch.zeros(4))
    with pytest.raises(ValueError, match="Expected shape"):
        spec.assemble_onehot(torch.zeros(2, 5))


def test_construct_spec_rejects_missing_required_parts_for_onehot():
    spec = ConstructSpec(left_adapter="A", right_adapter="C", promoter_seq=None, barcode_seq="T")
    onehot = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="promoter_seq"):
        spec.assemble_onehot(onehot, mode="core")
    with pytest.raises(ValueError, match="promoter_seq"):
        spec.assemble_onehot(onehot, mode="flanked")


def test_from_checkpoint_roundtrip_minimal_without_pretrained_weights(tmp_path: Path):
    torch.manual_seed(0)
    construct_spec = ConstructSpec(left_adapter="A", right_adapter="C", promoter_seq="G", barcode_seq="T")
    model = EncoderMPRAModel(
        DummyAlphaGenome(),
        MPRAHead(pooling_type="flatten", hidden_sizes=8),
        construct_spec=construct_spec,
    )
    model.initialize_head(sequence_length=2, device="cpu")
    model.eval()

    insert = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]], dtype=torch.float32)
    construct = construct_spec.assemble_onehot(insert, mode="full")
    direct_preds = model(construct, torch.zeros(1, dtype=torch.long))

    checkpoint_path = save_checkpoint(
        tmp_path / "best.pt",
        model,
        config=_make_config(tmp_path, save_mode="minimal"),
        save_mode="minimal",
        stage="stage1",
        epoch=1,
    )

    restored = EncoderMPRAModel.from_checkpoint(
        checkpoint_path,
        device="cpu",
        backbone_factory=DummyAlphaGenome,
    )
    restored_preds = restored(construct, torch.zeros(1, dtype=torch.long))

    np.testing.assert_allclose(restored_preds.detach().numpy(), direct_preds.detach().numpy(), rtol=1e-5, atol=1e-5)
    assert restored.construct_spec == construct_spec


def test_from_checkpoint_roundtrip_full(tmp_path: Path):
    torch.manual_seed(0)
    construct_spec = ConstructSpec(left_adapter="A", right_adapter="C", promoter_seq="G", barcode_seq="T")
    model = EncoderMPRAModel(
        DummyAlphaGenome(),
        MPRAHead(pooling_type="flatten", hidden_sizes=8),
        construct_spec=construct_spec,
    )
    model.initialize_head(sequence_length=2, device="cpu")
    model.eval()

    checkpoint_path = save_checkpoint(
        tmp_path / "best_full.pt",
        model,
        config=_make_config(tmp_path, save_mode="full"),
        save_mode="full",
        stage="stage1",
        epoch=1,
    )

    restored = EncoderMPRAModel.from_checkpoint(
        checkpoint_path,
        device="cpu",
        backbone_factory=DummyAlphaGenome,
    )
    assert restored.construct_spec == construct_spec


def test_from_checkpoint_rejects_head_only_checkpoint(tmp_path: Path):
    model = EncoderMPRAModel(DummyAlphaGenome(), MPRAHead(pooling_type="flatten", hidden_sizes=8))
    model.initialize_head(sequence_length=2, device="cpu")

    checkpoint_path = save_checkpoint(
        tmp_path / "head_only.pt",
        model,
        config=_make_config(tmp_path, save_mode="head"),
        save_mode="head",
        stage="stage1",
        epoch=1,
    )

    with pytest.raises(ValueError, match="Head-only checkpoints"):
        EncoderMPRAModel.from_checkpoint(
            checkpoint_path,
            device="cpu",
            backbone_factory=DummyAlphaGenome,
        )
