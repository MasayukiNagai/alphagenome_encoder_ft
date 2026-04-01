from __future__ import annotations

import csv
from pathlib import Path

import torch

from alphagenome_pytorch.utils.sequence import onehot_to_sequence

from alphagenome_encoder_ft.constructs import ConstructSpec
from alphagenome_encoder_ft.data import LentiMPRADataset


def _write_dataset(path: Path) -> None:
    with open(path / "HepG2.tsv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seq", "rev", "fold", "mean_value"], delimiter="\t")
        writer.writeheader()
        writer.writerows(
            [
                {"seq": "AC", "rev": 0, "fold": 2, "mean_value": 1.0},
                {"seq": "GT", "rev": 0, "fold": 1, "mean_value": 2.0},
                {"seq": "AA", "rev": 1, "fold": 10, "mean_value": 3.0},
                {"seq": "CC", "rev": 0, "fold": 10, "mean_value": 4.0},
            ]
        )


def test_dataset_filters_split_and_rev(tmp_path: Path):
    _write_dataset(tmp_path)
    input_tsv = tmp_path / "HepG2.tsv"
    spec = ConstructSpec.lentimpra_default()

    train_ds = LentiMPRADataset(input_tsv, split="train", sequence_length=16, construct_spec=spec)
    val_ds = LentiMPRADataset(input_tsv, split="val", sequence_length=16, construct_spec=spec)
    test_ds = LentiMPRADataset(input_tsv, split="test", sequence_length=16, construct_spec=spec)

    assert len(train_ds) == 1
    assert len(val_ds) == 1
    assert len(test_ds) == 1


def test_dataset_returns_fixed_length_onehot(tmp_path: Path):
    _write_dataset(tmp_path)
    ds = LentiMPRADataset(
        tmp_path / "HepG2.tsv",
        split="train",
        sequence_length=20,
        construct_spec=ConstructSpec.lentimpra_default(),
    )
    seq, target = ds[0]

    assert isinstance(seq, torch.Tensor)
    assert seq.shape == (20, 4)
    assert target.shape == ()
    assert target.item() == 1.0


def test_dataset_leaves_sequence_unpadded_when_length_omitted(tmp_path: Path):
    _write_dataset(tmp_path)
    ds = LentiMPRADataset(
        tmp_path / "HepG2.tsv",
        split="train",
        construct_spec=ConstructSpec.lentimpra_default(),
    )

    seq, _ = ds[0]
    assert ds.sequence_length is None
    assert seq.shape[1] == 4


def test_dataset_supports_custom_folds(tmp_path: Path):
    _write_dataset(tmp_path)
    ds = LentiMPRADataset(
        tmp_path / "HepG2.tsv",
        split="train",
        train_folds=[10],
        sequence_length=20,
        construct_spec=ConstructSpec.lentimpra_default(),
    )

    assert len(ds) == 1
    _, target = ds[0]
    assert target.item() == 4.0


def test_dataset_optionally_includes_adapters(tmp_path: Path):
    _write_dataset(tmp_path)
    spec = ConstructSpec(left_adapter="AA", right_adapter="CC", promoter_seq="G", barcode_seq="T")
    ds = LentiMPRADataset(
        tmp_path / "HepG2.tsv",
        split="train",
        sequence_length=32,
        construct_spec=spec,
    )

    seq, _ = ds[0]
    assert torch.equal(seq[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(seq[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert onehot_to_sequence(seq[:7].numpy()) == spec.assemble_sequence("AC", mode="core")


def test_dataset_uses_construct_mode(tmp_path: Path):
    _write_dataset(tmp_path)
    spec = ConstructSpec(left_adapter="AA", right_adapter="CC", promoter_seq="G", barcode_seq="T")
    ds = LentiMPRADataset(
        tmp_path / "HepG2.tsv",
        split="train",
        sequence_length=32,
        construct_spec=spec,
        construct_mode="flanked",
    )

    seq, _ = ds[0]
    assert onehot_to_sequence(seq[:4].numpy()) == spec.assemble_sequence("AC", mode="flanked")


def test_dataset_requires_construct_spec(tmp_path: Path):
    _write_dataset(tmp_path)

    try:
        LentiMPRADataset(tmp_path / "HepG2.tsv", split="train", sequence_length=20)
    except ValueError as exc:
        assert "construct_spec must be provided" in str(exc)
    else:
        raise AssertionError("Expected ValueError when construct_spec is omitted")


def test_dataset_allows_variable_construct_lengths_when_length_omitted(tmp_path: Path):
    with open(tmp_path / "HepG2.tsv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seq", "rev", "fold", "mean_value"], delimiter="\t")
        writer.writeheader()
        writer.writerows(
            [
                {"seq": "A", "rev": 0, "fold": 2, "mean_value": 1.0},
                {"seq": "AC", "rev": 0, "fold": 2, "mean_value": 2.0},
            ]
        )

    ds = LentiMPRADataset(
        tmp_path / "HepG2.tsv",
        split="train",
        construct_spec=ConstructSpec(left_adapter=None, right_adapter=None, promoter_seq=None, barcode_seq=None),
        construct_mode="full",
    )

    seq0, _ = ds[0]
    seq1, _ = ds[1]
    assert seq0.shape == (1, 4)
    assert seq1.shape == (2, 4)
