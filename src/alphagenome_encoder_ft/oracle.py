"""Oracle API for MPRA inference."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor

from .data import (
    DEFAULT_BARCODE_SEQ,
    DEFAULT_LEFT_ADAPTER_SEQ,
    DEFAULT_PROMOTER_SEQ,
    DEFAULT_RIGHT_ADAPTER_SEQ,
)
from .heads import MPRAHead


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class MPRAOracle(nn.Module):
    """Differentiable MPRA oracle for encoder-only AlphaGenome checkpoints."""

    def __init__(
        self,
        model: nn.Module,
        head: nn.Module,
        *,
        left_adapter: str = DEFAULT_LEFT_ADAPTER_SEQ,
        right_adapter: str = DEFAULT_RIGHT_ADAPTER_SEQ,
        promoter_seq: str = DEFAULT_PROMOTER_SEQ,
        barcode_seq: str = DEFAULT_BARCODE_SEQ,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.head = head
        self.device = _resolve_device(device)
        self.left_adapter = left_adapter
        self.right_adapter = right_adapter
        self.promoter_seq = promoter_seq
        self.barcode_seq = barcode_seq

        self.register_buffer(
            "_left_adapter_onehot",
            sequence_to_onehot_tensor(left_adapter, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_right_adapter_onehot",
            sequence_to_onehot_tensor(right_adapter, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_promoter_onehot",
            sequence_to_onehot_tensor(promoter_seq, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_barcode_onehot",
            sequence_to_onehot_tensor(barcode_seq, dtype=torch.float32),
            persistent=False,
        )

        self.model.to(self.device)
        self.head.to(self.device)
        self.model.eval()
        self.head.eval()

    @staticmethod
    def _validate_mode(mode: str) -> str:
        normalized = mode.lower()
        if normalized not in {"core", "flanked", "full"}:
            raise ValueError(f"Invalid mode {mode!r}. Must be one of: core, flanked, full")
        return normalized

    @staticmethod
    def _normalize_onehot(onehot: torch.Tensor) -> torch.Tensor:
        if onehot.ndim == 2:
            if onehot.shape[-1] != 4:
                raise ValueError(f"Expected shape (L, 4), got {tuple(onehot.shape)}")
            return onehot.unsqueeze(0)
        if onehot.ndim == 3:
            if onehot.shape[-1] != 4:
                raise ValueError(f"Expected shape (B, L, 4), got {tuple(onehot.shape)}")
            return onehot
        raise ValueError(f"Expected rank 2 or 3 onehot input, got rank {onehot.ndim}")

    @staticmethod
    def _pad_batch(batch: list[torch.Tensor]) -> torch.Tensor:
        max_len = max(item.shape[0] for item in batch)
        padded: list[torch.Tensor] = []
        for item in batch:
            if item.shape[0] < max_len:
                pad = torch.zeros(max_len - item.shape[0], 4, dtype=item.dtype, device=item.device)
                item = torch.cat([item, pad], dim=0)
            padded.append(item)
        return torch.stack(padded, dim=0)

    def _assemble_construct(self, onehot: torch.Tensor, mode: str) -> torch.Tensor:
        mode = self._validate_mode(mode)
        batch = self._normalize_onehot(onehot).to(self.device, dtype=torch.float32)
        pieces: list[torch.Tensor] = []

        def expand_piece(piece: torch.Tensor) -> torch.Tensor:
            return piece.to(batch.device).unsqueeze(0).expand(batch.shape[0], -1, -1)

        if mode == "core":
            pieces.extend(
                [
                    expand_piece(self._left_adapter_onehot),
                    batch,
                    expand_piece(self._right_adapter_onehot),
                    expand_piece(self._promoter_onehot),
                    expand_piece(self._barcode_onehot),
                ]
            )
        elif mode == "flanked":
            pieces.extend(
                [
                    batch,
                    expand_piece(self._promoter_onehot),
                    expand_piece(self._barcode_onehot),
                ]
            )
        else:
            pieces.append(batch)

        return torch.cat(pieces, dim=1)

    @staticmethod
    def _normalize_sequences(sequences: str | Iterable[str]) -> list[str]:
        if isinstance(sequences, str):
            return [sequences.strip().upper()]
        normalized: list[str] = []
        for idx, seq in enumerate(sequences):
            if not isinstance(seq, str):
                raise TypeError(f"Sequence element {idx} must be str, got {type(seq).__name__}")
            normalized.append(seq.strip().upper())
        return normalized

    def _forward_batch(self, batch: torch.Tensor, mode: str) -> torch.Tensor:
        assembled = self._assemble_construct(batch, mode)
        organism_idx = torch.zeros(assembled.shape[0], dtype=torch.long, device=self.device)
        outputs = self.model(assembled, organism_idx, encoder_only=True)
        return self.head(outputs["encoder_output"])

    def forward(self, onehot: torch.Tensor, mode: str = "core") -> torch.Tensor:
        return self._forward_batch(onehot, mode)

    def forward_sequences(self, sequences: str | Iterable[str], mode: str = "core") -> torch.Tensor:
        encoded = [
            sequence_to_onehot_tensor(seq, dtype=torch.float32, device=self.device)
            for seq in self._normalize_sequences(sequences)
        ]
        batch = self._pad_batch(encoded)
        return self._forward_batch(batch, mode)

    def predict(self, onehot: torch.Tensor | np.ndarray, mode: str = "core", batch_size: int = 64) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        tensor = torch.as_tensor(onehot, dtype=torch.float32, device=self.device)
        batch = self._normalize_onehot(tensor)
        outputs: list[torch.Tensor] = []
        self.eval()
        with torch.no_grad():
            for start in range(0, batch.shape[0], batch_size):
                outputs.append(self._forward_batch(batch[start : start + batch_size], mode))
        return torch.cat(outputs, dim=0).detach().cpu().numpy()

    def predict_sequences(
        self,
        sequences: str | Iterable[str],
        mode: str = "core",
        batch_size: int = 64,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        seqs = self._normalize_sequences(sequences)
        outputs: list[torch.Tensor] = []
        self.eval()
        with torch.no_grad():
            for start in range(0, len(seqs), batch_size):
                encoded = [
                    sequence_to_onehot_tensor(seq, dtype=torch.float32, device=self.device)
                    for seq in seqs[start : start + batch_size]
                ]
                batch = self._pad_batch(encoded)
                outputs.append(self._forward_batch(batch, mode))
        return torch.cat(outputs, dim=0).detach().cpu().numpy()


def load_oracle(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
    _model_factory: Any = AlphaGenome,
) -> MPRAOracle:
    """Load an MPRA oracle from checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    save_mode = checkpoint.get("save_mode", "minimal")
    if save_mode == "head":
        raise ValueError("Head-only checkpoints cannot be loaded as an oracle")

    model = _model_factory()
    if save_mode == "minimal":
        encoder_state = checkpoint["encoder_state_dict"]
        model.encoder.load_state_dict(encoder_state)
    elif save_mode == "full":
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        raise ValueError(f"Unknown checkpoint save_mode: {save_mode}")

    head_config = checkpoint.get("head_config", {})
    head = MPRAHead(**head_config)
    head.load_state_dict(checkpoint["head_state_dict"])

    construct_config = checkpoint.get("construct_config", {})
    return MPRAOracle(
        model,
        head,
        left_adapter=construct_config.get("left_adapter", DEFAULT_LEFT_ADAPTER_SEQ),
        right_adapter=construct_config.get("right_adapter", DEFAULT_RIGHT_ADAPTER_SEQ),
        promoter_seq=construct_config.get("promoter_seq", DEFAULT_PROMOTER_SEQ),
        barcode_seq=construct_config.get("barcode_seq", DEFAULT_BARCODE_SEQ),
        device=device,
    )
