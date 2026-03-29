from __future__ import annotations

import csv
from pathlib import Path

import torch

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

    train_ds = LentiMPRADataset(input_tsv, split="train", sequence_length=16)
    val_ds = LentiMPRADataset(input_tsv, split="val", sequence_length=16)
    test_ds = LentiMPRADataset(input_tsv, split="test", sequence_length=16)

    assert len(train_ds) == 1
    assert len(val_ds) == 1
    assert len(test_ds) == 1


def test_dataset_returns_fixed_length_onehot(tmp_path: Path):
    _write_dataset(tmp_path)
    ds = LentiMPRADataset(tmp_path / "HepG2.tsv", split="train", sequence_length=20)
    seq, target = ds[0]

    assert isinstance(seq, torch.Tensor)
    assert seq.shape == (20, 4)
    assert target.shape == ()
    assert target.item() == 1.0


def test_dataset_supports_custom_folds(tmp_path: Path):
    _write_dataset(tmp_path)
    ds = LentiMPRADataset(
        tmp_path / "HepG2.tsv",
        split="train",
        train_folds=[10],
        sequence_length=20,
    )

    assert len(ds) == 1
    _, target = ds[0]
    assert target.item() == 4.0


def test_dataset_optionally_includes_adapters(tmp_path: Path):
    _write_dataset(tmp_path)
    ds = LentiMPRADataset(
        tmp_path / "HepG2.tsv",
        split="train",
        sequence_length=32,
        promoter_seq="G",
        barcode_seq="T",
        left_adapter_seq="AA",
        right_adapter_seq="CC",
    )

    seq, _ = ds[0]
    assert torch.equal(seq[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(seq[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))
