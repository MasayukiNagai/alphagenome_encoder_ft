from __future__ import annotations

import pytest
import torch

from alphagenome_encoder_ft.heads import MPRAHead


@pytest.mark.parametrize("pooling_type", ["flatten", "center", "mean", "sum", "max"])
def test_mpra_head_output_shape(pooling_type: str):
    head = MPRAHead(pooling_type=pooling_type, hidden_sizes=16, center_bp=256)
    encoder_output = torch.randn(4, 3, 1536)
    preds = head(encoder_output)
    assert preds.shape == (4,)


def test_mpra_head_lazy_init_for_flatten():
    head = MPRAHead(pooling_type="flatten", hidden_sizes=[32, 16])
    encoder_output = torch.randn(2, 5, 1536)
    preds = head(encoder_output)
    assert preds.shape == (2,)
