from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from alphagenome_encoder_ft.heads import MPRAHead
from alphagenome_encoder_ft.oracle import MPRAOracle, load_oracle
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


class SumHead(torch.nn.Module):
    def forward(self, encoder_output):
        return encoder_output.sum(dim=(1, 2))


def test_oracle_modes_change_sequence_length():
    model = DummyAlphaGenome()
    head = SumHead()
    oracle = MPRAOracle(
        model,
        head,
        left_adapter="A",
        right_adapter="C",
        promoter_seq="G",
        barcode_seq="T",
        device="cpu",
    )
    payload = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    pred_full = oracle.forward(payload, mode="full")
    pred_flanked = oracle.forward(payload, mode="flanked")
    pred_core = oracle.forward(payload, mode="core")

    assert pred_full.shape == (1,)
    assert pred_flanked.shape == (1,)
    assert pred_core.shape == (1,)
    assert pred_core.item() != pred_flanked.item()
    assert pred_flanked.item() != pred_full.item()


def test_load_oracle_roundtrip_minimal_checkpoint(tmp_path: Path):
    torch.manual_seed(0)
    model = DummyAlphaGenome()
    head = MPRAHead(pooling_type="flatten", hidden_sizes=8)
    sample = torch.randn(1, 2, 4)
    model.eval()
    head.eval()
    direct_preds = head(model(sample, torch.zeros(1, dtype=torch.long), encoder_only=True)["encoder_output"])

    checkpoint_path = save_checkpoint(
        tmp_path / "best.pt",
        model,
        head,
        save_mode="minimal",
        stage="stage1",
        epoch=1,
        head_config={"pooling_type": "flatten", "hidden_sizes": 8, "center_bp": 256, "dropout": 0.1, "activation": "relu"},
        construct_config={"left_adapter": "A", "right_adapter": "C", "promoter_seq": "G", "barcode_seq": "T", "sequence_length": 256},
    )

    oracle = load_oracle(checkpoint_path, device="cpu", _model_factory=DummyAlphaGenome)
    oracle_preds = oracle.predict(sample.squeeze(0).numpy(), mode="full")

    np.testing.assert_allclose(oracle_preds, direct_preds.detach().numpy(), rtol=1e-5, atol=1e-5)


def test_oracle_forward_supports_autograd():
    model = DummyAlphaGenome()
    head = MPRAHead(pooling_type="flatten", hidden_sizes=8)
    oracle = MPRAOracle(model, head, device="cpu")
    onehot = torch.randn(1, 2, 4, requires_grad=True)
    preds = oracle.forward(onehot, mode="full")
    preds.sum().backward()
    assert onehot.grad is not None
