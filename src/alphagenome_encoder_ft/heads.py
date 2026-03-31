"""Utility MPRA head implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn

PoolingType = Literal["flatten", "center", "mean", "sum", "max"]
ENCODER_RESOLUTION_BP = 128
ENCODER_DIM = 1536


def _parse_hidden_sizes(hidden_sizes: int | Sequence[int]) -> list[int]:
    if isinstance(hidden_sizes, int):
        sizes = [hidden_sizes]
    else:
        sizes = list(hidden_sizes)
    if not sizes:
        raise ValueError("hidden_sizes must contain at least one layer")
    if any(size <= 0 for size in sizes):
        raise ValueError("hidden_sizes must be positive")
    return sizes


def _make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError("activation must be 'relu' or 'gelu'")


class MPRAHead(nn.Module):
    """Scalar regression head over encoder outputs."""

    def __init__(
        self,
        pooling_type: PoolingType = "flatten",
        center_bp: int = 256,
        hidden_sizes: int | Sequence[int] = 1024,
        dropout: float | None = 0.1,
        activation: Literal["relu", "gelu"] = "relu",
    ) -> None:
        super().__init__()
        if pooling_type not in {"flatten", "center", "mean", "sum", "max"}:
            raise ValueError(f"Unknown pooling_type: {pooling_type}")
        if center_bp <= 0:
            raise ValueError("center_bp must be > 0")
        if dropout is not None and not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.pooling_type = pooling_type
        self.center_bp = center_bp
        self.hidden_sizes = _parse_hidden_sizes(hidden_sizes)
        self.dropout = dropout
        self.activation = activation
        self.norm = nn.LayerNorm(ENCODER_DIM)

        layers: list[nn.Module] = []
        layers.append(nn.LazyLinear(self.hidden_sizes[0]))
        previous_size = self.hidden_sizes[0]
        for hidden_size in self.hidden_sizes[1:]:
            layers.append(_make_activation(self.activation))
            if self.dropout is not None:
                layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(previous_size, hidden_size))
            previous_size = hidden_size
        layers.append(_make_activation(self.activation))
        if self.dropout is not None:
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(previous_size, 1))
        self.mlp = nn.Sequential(*layers)

    def _pool(self, encoder_output: torch.Tensor) -> torch.Tensor:
        x = encoder_output
        if x.ndim == 3 and x.shape[-1] != ENCODER_DIM and x.shape[1] == ENCODER_DIM:
            x = x.transpose(1, 2)
        x = self.norm(x)
        if self.pooling_type == "flatten":
            return x.flatten(1)

        if x.ndim != 3:
            raise ValueError(f"Expected encoder output rank 3, got {x.ndim}")

        seq_len = x.shape[1]
        if self.pooling_type == "center":
            center_idx = seq_len // 2
            return x[:, center_idx, :]

        window_positions = max(1, self.center_bp // ENCODER_RESOLUTION_BP)
        window_positions = min(window_positions, seq_len)
        start = max((seq_len - window_positions) // 2, 0)
        center_window = x[:, start : start + window_positions, :]

        if self.pooling_type == "mean":
            return center_window.mean(dim=1)
        if self.pooling_type == "sum":
            return center_window.sum(dim=1)
        if self.pooling_type == "max":
            return center_window.max(dim=1).values
        raise RuntimeError(f"Unhandled pooling type: {self.pooling_type}")

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        pooled = self._pool(encoder_output)
        preds = self.mlp(pooled)
        return preds.squeeze(-1)
