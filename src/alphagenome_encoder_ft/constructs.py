"""Construct assembly utilities for MPRA insert sequences."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch

from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor

LENTIMPRA_LEFT_ADAPTER = "AGGACCGGATCAACT"
LENTIMPRA_RIGHT_ADAPTER = "CATTGCGTGAACCGA"
LENTIMPRA_PROMOTER = "TCCATTATATACCCTCTAGTGTCGGTTCACGCAATG"
LENTIMPRA_BARCODE = "AGAGACTGAGGCCAC"


@dataclass(frozen=True)
class ConstructSpec:
    """Reusable construct assembly rules for MPRA insert sequences.

    Supported assembly modes (what gets added to the insert):
    - ``none``: insert only
    - ``adapters``: left adapter + insert + right adapter
    - ``promoter``: insert + promoter
    - ``promoter_barcode``: insert + promoter + barcode
    - ``all``: left adapter + insert + right adapter + promoter + barcode
    """

    left_adapter: str | None = LENTIMPRA_LEFT_ADAPTER
    right_adapter: str | None = LENTIMPRA_RIGHT_ADAPTER
    promoter_seq: str | None = LENTIMPRA_PROMOTER
    barcode_seq: str | None = LENTIMPRA_BARCODE

    _left_adapter_onehot: torch.Tensor | None = field(init=False, repr=False, compare=False, default=None)
    _right_adapter_onehot: torch.Tensor | None = field(init=False, repr=False, compare=False, default=None)
    _promoter_onehot: torch.Tensor | None = field(init=False, repr=False, compare=False, default=None)
    _barcode_onehot: torch.Tensor | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_left_adapter_onehot", self._encode_constant(self.left_adapter))
        object.__setattr__(self, "_right_adapter_onehot", self._encode_constant(self.right_adapter))
        object.__setattr__(self, "_promoter_onehot", self._encode_constant(self.promoter_seq))
        object.__setattr__(self, "_barcode_onehot", self._encode_constant(self.barcode_seq))

    @classmethod
    def lentimpra_default(cls) -> "ConstructSpec":
        return cls()

    @staticmethod
    def _encode_constant(sequence: str | None) -> torch.Tensor | None:
        if sequence is None:
            return None
        return sequence_to_onehot_tensor(sequence, dtype=torch.float32)

    # -------------------------
    # Mode handling
    # -------------------------

    @staticmethod
    def validate_mode(mode: str) -> str:
        normalized = mode.lower()
        valid = {"none", "adapters", "promoter", "promoter_barcode", "all"}
        if normalized not in valid:
            raise ValueError(
                "Invalid mode "
                f"{mode!r}. Must be one of: none, adapters, promoter, promoter_barcode, all"
            )
        return normalized

    def _validate_required_components(self, mode: str) -> None:
        missing: list[str] = []

        if mode in {"adapters", "all"}:
            if self.left_adapter is None:
                missing.append("left_adapter")
            if self.right_adapter is None:
                missing.append("right_adapter")

        if mode in {"promoter", "promoter_barcode", "all"}:
            if self.promoter_seq is None:
                missing.append("promoter_seq")
        if mode in {"promoter_barcode", "all"}:
            if self.barcode_seq is None:
                missing.append("barcode_seq")

        if missing:
            raise ValueError(
                f"Mode {mode!r} requires construct components: {', '.join(missing)}"
            )

    # -------------------------
    # String assembly
    # -------------------------

    @staticmethod
    def _normalize_insert_sequence(insert_seq: str) -> str:
        return insert_seq.strip().upper()

    def assemble_sequence(self, insert_seq: str, mode: str = "all") -> str:
        normalized_mode = self.validate_mode(mode)
        self._validate_required_components(normalized_mode)

        insert = self._normalize_insert_sequence(insert_seq)

        parts: list[str] = []

        if normalized_mode in {"adapters", "all"}:
            parts.append(self.left_adapter)

        parts.append(insert)

        if normalized_mode in {"adapters", "all"}:
            parts.append(self.right_adapter)

        if normalized_mode in {"promoter", "promoter_barcode", "all"}:
            parts.append(self.promoter_seq)
        if normalized_mode in {"promoter_barcode", "all"}:
            parts.append(self.barcode_seq)

        return "".join(parts)

    def assemble_sequences(self, insert_seqs: Iterable[str], mode: str = "all") -> list[str]:
        normalized_mode = self.validate_mode(mode)
        self._validate_required_components(normalized_mode)
        return [self.assemble_sequence(seq, normalized_mode) for seq in insert_seqs]

    # -------------------------
    # One-hot utilities
    # -------------------------

    @staticmethod
    def _normalize_onehot(onehot: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if onehot.ndim == 2:
            if onehot.shape[-1] != 4:
                raise ValueError(f"Expected shape (L, 4), got {tuple(onehot.shape)}")
            return onehot.unsqueeze(0), True
        if onehot.ndim == 3:
            if onehot.shape[-1] != 4:
                raise ValueError(f"Expected shape (B, L, 4), got {tuple(onehot.shape)}")
            return onehot, False
        raise ValueError(f"Expected rank 2 or 3 input, got rank {onehot.ndim}")

    @staticmethod
    def _expand_piece(
        piece: torch.Tensor | None,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if piece is None:
            return None
        return piece.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    # -------------------------
    # One-hot assembly
    # -------------------------

    def assemble_onehot(self, onehot: torch.Tensor, mode: str = "all") -> torch.Tensor:
        normalized_mode = self.validate_mode(mode)
        self._validate_required_components(normalized_mode)

        batch, squeeze = self._normalize_onehot(onehot)
        batch_size = batch.shape[0]

        kwargs = {
            "batch_size": batch_size,
            "dtype": batch.dtype,
            "device": batch.device,
        }

        pieces: list[torch.Tensor] = []

        if normalized_mode in {"adapters", "all"}:
            pieces.append(self._expand_piece(self._left_adapter_onehot, **kwargs))

        pieces.append(batch)

        if normalized_mode in {"adapters", "all"}:
            pieces.append(self._expand_piece(self._right_adapter_onehot, **kwargs))

        if normalized_mode in {"promoter", "promoter_barcode", "all"}:
            pieces.append(self._expand_piece(self._promoter_onehot, **kwargs))
        if normalized_mode in {"promoter_barcode", "all"}:
            pieces.append(self._expand_piece(self._barcode_onehot, **kwargs))

        assembled = torch.cat(pieces, dim=1)
        return assembled.squeeze(0) if squeeze else assembled
