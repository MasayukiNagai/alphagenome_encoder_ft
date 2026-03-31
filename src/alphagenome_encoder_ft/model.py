"""Wrapped AlphaGenome encoder + MPRA head model."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.extensions.finetuning.transfer import load_trunk, remove_all_heads

from .config import HeadConfig, TrainConfig
from .heads import MPRAHead


class EncoderMPRAModel(nn.Module):
    """Thin wrapper around an AlphaGenome backbone and an MPRA regression head."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

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

    @classmethod
    def from_pretrained(
        cls,
        pretrained_weights: str | Path,
        head_config: HeadConfig,
        *,
        device: torch.device | str,
        backbone_factory=AlphaGenome,
    ) -> "EncoderMPRAModel":
        backbone = backbone_factory()
        backbone = load_trunk(backbone, pretrained_weights, exclude_heads=True)
        backbone = remove_all_heads(backbone)
        model = cls(backbone, MPRAHead(**head_config.__dict__))
        model.set_encoder_trainable(False)
        model.to(device)
        return model


def load_pretrained_model(
    config: TrainConfig,
    *,
    device: torch.device | str,
    backbone_factory=AlphaGenome,
) -> EncoderMPRAModel:
    model = EncoderMPRAModel.from_pretrained(
        config.checkpoint.pretrained_weights,
        config.head,
        device=device,
        backbone_factory=backbone_factory,
    )
    model.initialize_head(config.data.sequence_length, device)
    model.eval()
    return model
