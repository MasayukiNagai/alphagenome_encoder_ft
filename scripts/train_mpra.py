#!/usr/bin/env python
"""Train an encoder-only MPRA model on lentiMPRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.extensions.finetuning.transfer import load_trunk, remove_all_heads

from alphagenome_encoder_ft.data import (
    DEFAULT_BARCODE_SEQ,
    DEFAULT_LEFT_ADAPTER_SEQ,
    DEFAULT_PROMOTER_SEQ,
    DEFAULT_RIGHT_ADAPTER_SEQ,
    LentiMPRADataset,
    create_dataloader,
)
from alphagenome_encoder_ft.heads import MPRAHead
from alphagenome_encoder_ft.train import (
    run_training_stage,
    run_two_stage_training,
    set_encoder_trainable,
)


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path) as handle:
        return json.load(handle)


def _apply_config_defaults(parser: argparse.ArgumentParser, config: dict[str, Any]) -> None:
    if not config:
        return
    parser.set_defaults(**config)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train encoder-only AlphaGenome MPRA model")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--input_tsv", type=str, required=True)
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sequence_length", type=int, default=256)
    parser.add_argument("--reverse_complement", action="store_true")
    parser.add_argument("--rc_prob", type=float, default=0.5)
    parser.add_argument("--random_shift", action="store_true")
    parser.add_argument("--shift_prob", type=float, default=0.5)
    parser.add_argument("--max_shift", type=int, default=15)
    parser.add_argument("--subset_frac", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--pooling_type", type=str, default="flatten", choices=["flatten", "center", "mean", "sum", "max"])
    parser.add_argument("--center_bp", type=int, default=256)
    parser.add_argument("--hidden_sizes", type=str, default="1024")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"])
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_clip", type=float, default=None)
    parser.add_argument("--lr_scheduler", type=str, default="constant", choices=["constant", "cosine", "plateau"])
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--val_eval_frequency", type=int, default=1)
    parser.add_argument("--test_eval_frequency", type=int, default=1)
    parser.add_argument("--second_stage_lr", type=float, default=None)
    parser.add_argument("--second_stage_epochs", type=int, default=10)
    parser.add_argument("--resume_from_stage2", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_mpra")
    parser.add_argument("--save_mode", type=str, default="minimal", choices=["minimal", "full", "head"])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="alphagenome-mpra")
    parser.add_argument("--wandb_name", type=str, default="mpra-head-encoder")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _parse_hidden_sizes(text: str) -> int | list[int]:
    text = text.strip()
    if "," not in text:
        return int(text)
    return [int(piece.strip()) for piece in text.split(",") if piece.strip()]


def _make_optimizer(
    name: str,
    params,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if name == "adam":
        return Adam(params, lr=learning_rate, weight_decay=weight_decay)
    return AdamW(params, lr=learning_rate, weight_decay=weight_decay)


def _make_scheduler(name: str, optimizer: torch.optim.Optimizer, total_epochs: int):
    if name == "constant":
        return None
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=2)
    raise ValueError(f"Unknown lr_scheduler: {name}")


def _scheduler_stepper(name: str):
    if name == "plateau":
        return lambda scheduler, metrics: scheduler.step(metrics["loss"]) if scheduler is not None else None
    return lambda scheduler, metrics: scheduler.step() if scheduler is not None else None


def main() -> dict[str, Any]:
    parser = build_arg_parser()
    initial_args, _ = parser.parse_known_args()
    _apply_config_defaults(parser, load_config(initial_args.config))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = Path(args.checkpoint_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    hidden_sizes = _parse_hidden_sizes(args.hidden_sizes)
    head_config = {
        "pooling_type": args.pooling_type,
        "center_bp": args.center_bp,
        "hidden_sizes": hidden_sizes,
        "dropout": args.dropout,
        "activation": args.activation,
    }
    construct_config = {
        "left_adapter": DEFAULT_LEFT_ADAPTER_SEQ,
        "right_adapter": DEFAULT_RIGHT_ADAPTER_SEQ,
        "promoter_seq": DEFAULT_PROMOTER_SEQ,
        "barcode_seq": DEFAULT_BARCODE_SEQ,
        "sequence_length": args.sequence_length,
    }
    training_config = vars(args).copy()
    training_config["hidden_sizes"] = hidden_sizes

    model = AlphaGenome()
    model = load_trunk(model, args.pretrained_weights, exclude_heads=True)
    model = remove_all_heads(model)
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)

    head = MPRAHead(**head_config).to(device)

    train_dataset = LentiMPRADataset(
        args.input_tsv,
        split="train",
        sequence_length=args.sequence_length,
        reverse_complement=args.reverse_complement,
        rc_prob=args.rc_prob,
        random_shift=args.random_shift,
        shift_prob=args.shift_prob,
        max_shift=args.max_shift,
        subset_frac=args.subset_frac,
        seed=args.seed,
    )
    val_dataset = LentiMPRADataset(
        args.input_tsv,
        split="val",
        sequence_length=args.sequence_length,
        subset_frac=args.subset_frac,
        seed=args.seed,
    )
    test_dataset = LentiMPRADataset(
        args.input_tsv,
        split="test",
        sequence_length=args.sequence_length,
        subset_frac=args.subset_frac,
        seed=args.seed,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    stage1_optimizer = _make_optimizer(
        args.optimizer,
        head.parameters(),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    stage1_scheduler = _make_scheduler(args.lr_scheduler, stage1_optimizer, args.num_epochs)
    stage1_scheduler_step = _scheduler_stepper(args.lr_scheduler)

    if args.use_wandb:
        try:
            import wandb

            wandb.init(project=args.wandb_project, name=args.wandb_name, config=training_config)
        except ImportError:
            print("wandb is not installed; continuing without wandb")
            args.use_wandb = False

    if args.second_stage_lr is not None:
        def stage2_optimizer_factory(model_obj, head_obj):
            params = [param for param in model_obj.parameters() if param.requires_grad] + list(head_obj.parameters())
            deduped = []
            seen = set()
            for param in params:
                if id(param) not in seen:
                    deduped.append(param)
                    seen.add(id(param))
            return _make_optimizer(
                args.optimizer,
                deduped,
                learning_rate=args.second_stage_lr,
                weight_decay=args.weight_decay,
            )

        def stage2_scheduler_factory(optimizer):
            return _make_scheduler(args.lr_scheduler, optimizer, args.second_stage_epochs)

        results = run_two_stage_training(
            model,
            head,
            train_loader,
            stage1_optimizer=stage1_optimizer,
            stage2_optimizer_factory=stage2_optimizer_factory,
            device=device,
            stage1_num_epochs=args.num_epochs,
            stage2_num_epochs=args.second_stage_epochs,
            val_loader=val_loader,
            test_loader=test_loader,
            stage1_scheduler=stage1_scheduler,
            stage2_scheduler_factory=stage2_scheduler_factory,
            stage1_scheduler_step=stage1_scheduler_step,
            stage2_scheduler_step=_scheduler_stepper(args.lr_scheduler),
            checkpoint_dir=run_dir,
            save_mode=args.save_mode,
            head_config=head_config,
            construct_config=construct_config,
            training_config=training_config,
            early_stopping_patience=args.early_stopping_patience,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_amp=args.use_amp,
            val_eval_frequency=args.val_eval_frequency,
            test_eval_frequency=args.test_eval_frequency,
            grad_clip=args.gradient_clip,
            resume_from_stage2=args.resume_from_stage2,
        )
    else:
        if args.save_mode == "head":
            set_encoder_trainable(model, False)
        results = run_training_stage(
            model,
            head,
            train_loader,
            optimizer=stage1_optimizer,
            device=device,
            num_epochs=args.num_epochs,
            stage="stage1",
            encoder_trainable=False,
            val_loader=val_loader,
            test_loader=test_loader,
            scheduler=stage1_scheduler,
            scheduler_step=stage1_scheduler_step,
            checkpoint_dir=run_dir / "stage1",
            save_mode=args.save_mode,
            head_config=head_config,
            construct_config=construct_config,
            training_config=training_config,
            early_stopping_patience=args.early_stopping_patience,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_amp=args.use_amp,
            val_eval_frequency=args.val_eval_frequency,
            test_eval_frequency=args.test_eval_frequency,
            grad_clip=args.gradient_clip,
        )

    with open(run_dir / "config.json", "w") as handle:
        json.dump(training_config, handle, indent=2)
    with open(run_dir / "history.json", "w") as handle:
        json.dump(results["history"], handle, indent=2)

    if args.use_wandb:
        import wandb

        history = results["history"]
        if history["train_loss"]:
            wandb.log(
                {
                    "train_loss": history["train_loss"][-1],
                    "train_pearson": history["train_pearson"][-1],
                }
            )
        wandb.finish()

    return results


if __name__ == "__main__":
    main()
