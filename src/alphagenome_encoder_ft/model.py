"""Wrapped AlphaGenome encoder + MPRA head model."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.extensions.finetuning.transfer import load_trunk, remove_all_heads
from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor

from .config import HeadConfig, build_head
from .constructs import ConstructSpec
from .heads import MPRAHead


class AlphaGenomeEncoderModel(nn.Module):
    """Thin wrapper around an AlphaGenome backbone and an MPRA regression head."""

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        *,
        construct_spec: ConstructSpec | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.construct_spec = construct_spec

    @property
    def encoder(self) -> nn.Module:
        if not hasattr(self.backbone, "encoder"):
            raise AttributeError("Backbone does not expose an 'encoder' module")
        return self.backbone.encoder

    def encode(
        self,
        sequences: torch.Tensor,
        organism_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if organism_idx is None:
            organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=sequences.device)
        outputs = self.backbone(sequences, organism_idx, encoder_only=True)
        return outputs["encoder_output"]

    def predict_from_encoder(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return self.head(encoder_output)

    def forward(
        self,
        sequences: torch.Tensor,
        organism_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.predict_from_encoder(self.encode(sequences, organism_idx))

    def predict_sequences(
        self,
        sequences: Sequence[str],
        *,
        construct_mode: str | None = None,
        organism_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_sequences = list(sequences)
        if not batch_sequences:
            raise ValueError("predict_sequences requires at least one sequence")

        if construct_mode is not None:
            if self.construct_spec is None:
                raise ValueError("construct_mode requires model.construct_spec to be set")
            batch_sequences = self.construct_spec.assemble_sequences(batch_sequences, mode=construct_mode)
        else:
            batch_sequences = [seq.strip().upper() for seq in batch_sequences]

        lengths = {len(seq) for seq in batch_sequences}
        if len(lengths) != 1:
            raise ValueError("All sequences must have the same length")

        device = next(self.parameters()).device
        onehot = torch.stack(
            [sequence_to_onehot_tensor(seq, dtype=torch.float32, device=device) for seq in batch_sequences],
            dim=0,
        )

        with torch.no_grad():
            return self(onehot, organism_idx)

    def initialize_head(self, sequence_length: int, device: torch.device | str) -> None:
        with torch.no_grad():
            device = torch.device(device)
            dummy_sequence = torch.zeros(1, sequence_length, 4, device=device)
            dummy_organism_idx = torch.zeros(1, dtype=torch.long, device=device)
            encoder_output = self.encode(dummy_sequence, dummy_organism_idx)
            _ = self.predict_from_encoder(encoder_output)

    def set_encoder_trainable(self, trainable: bool) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = trainable

    def trainable_parameters(self, include_encoder: bool) -> list[nn.Parameter]:
        params = list(self.head.parameters())
        if include_encoder:
            params = list(self.encoder.parameters()) + params
        deduped: list[nn.Parameter] = []
        seen: set[int] = set()
        for param in params:
            if param.requires_grad and id(param) not in seen:
                deduped.append(param)
                seen.add(id(param))
        return deduped

    @staticmethod
    def _resolve_device(device: torch.device | str | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_weights: str | Path,
        head_config: HeadConfig,
        *,
        device: torch.device | str | None = None,
        construct_spec: ConstructSpec | None = None,
        backbone_factory=AlphaGenome,
        head_type: str | None = None,
    ) -> "AlphaGenomeEncoderModel":
        device = cls._resolve_device(device)
        backbone = backbone_factory()
        backbone = load_trunk(backbone, pretrained_weights, exclude_heads=True)
        backbone = remove_all_heads(backbone)
        resolved_head_type = head_type or getattr(head_config, "head_type", "mpra")
        head = build_head(resolved_head_type, head_config.__dict__)
        model = cls(backbone, head, construct_spec=construct_spec)
        model.set_encoder_trainable(False)
        model.to(device)
        return model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str | None = None,
        backbone_factory=AlphaGenome,
    ) -> "AlphaGenomeEncoderModel":
        device = cls._resolve_device(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        save_mode = checkpoint.get("save_mode", "minimal")
        if save_mode == "head":
            raise ValueError("Head-only checkpoints cannot be loaded standalone")

        head_config_dict = dict(checkpoint.get("head_config", {}))
        # backward compat: historical ckpts omit head_type; default to "mpra".
        head_type = checkpoint.get("head_type", head_config_dict.get("head_type", "mpra"))
        construct_config = checkpoint.get("construct_config", {})
        construct_spec = ConstructSpec(
            left_adapter=construct_config.get("left_adapter"),
            right_adapter=construct_config.get("right_adapter"),
            promoter_seq=construct_config.get("promoter_seq"),
            barcode_seq=construct_config.get("barcode_seq"),
        )

        model = cls(
            backbone_factory(),
            build_head(head_type, head_config_dict),
            construct_spec=construct_spec,
        )
        model.to(device)
        sequence_length = construct_config.get("sequence_length")
        if sequence_length is None:
            config = checkpoint.get("config", {})
            if isinstance(config, dict):
                sequence_length = config.get("data", {}).get("sequence_length")
        if sequence_length is None:
            raise ValueError("Checkpoint is missing construct sequence_length needed to initialize the head")
        model.initialize_head(int(sequence_length), device)

        if save_mode == "minimal":
            model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        elif save_mode == "full":
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            raise ValueError(f"Unknown checkpoint save_mode: {save_mode}")

        model.head.load_state_dict(checkpoint["head_state_dict"])
        model.set_encoder_trainable(False)
        model.to(device)
        model.eval()
        return model


# Backward compatibility alias.
EncoderMPRAModel = AlphaGenomeEncoderModel
