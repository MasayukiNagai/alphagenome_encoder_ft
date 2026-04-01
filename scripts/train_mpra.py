#!/usr/bin/env python
"""Train an encoder-only MPRA model on lentiMPRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from alphagenome_encoder_ft import (
    ConstructSpec,
    EncoderMPRAModel,
    LentiMPRADataset,
    TrainConfig,
    create_dataloader,
    create_optimizer,
    create_scheduler,
    evaluate,
    load_checkpoint,
    load_train_config,
    merge_train_config,
    parse_hidden_sizes,
    run_training_stage,
    run_two_stage_training,
    scheduler_stepper,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train encoder-only AlphaGenome MPRA model")
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--input_tsv", type=str, default=None)
    parser.add_argument("--pretrained_weights", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--save_mode", type=str, default=None, choices=["minimal", "full", "head"])

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--sequence_length", type=int, default=None)
    parser.add_argument("--construct_mode", type=str, default=None, choices=["core", "flanked", "full"])
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_shift", type=int, default=None)
    parser.add_argument("--subset_frac", type=float, default=None)
    parser.add_argument("--rc_prob", type=float, default=None)
    parser.add_argument("--shift_prob", type=float, default=None)
    parser.add_argument("--reverse_complement", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--random_shift", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--pooling_type", type=str, default=None, choices=["flatten", "center", "mean", "sum", "max"])
    parser.add_argument("--center_bp", type=int, default=None)
    parser.add_argument("--hidden_sizes", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--activation", type=str, default=None, choices=["relu", "gelu"])

    parser.add_argument("--optimizer", type=str, default=None, choices=["adam", "adamw"])
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--lr_scheduler", type=str, default=None, choices=["constant", "cosine", "plateau"])
    parser.add_argument("--plateau_factor", type=float, default=None)
    parser.add_argument("--plateau_patience", type=int, default=None)
    parser.add_argument("--plateau_mode", type=str, default=None, choices=["min"])
    parser.add_argument("--plateau_min_lr", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--gradient_clip", type=float, default=None)

    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--val_evals_per_epoch", type=int, default=None)
    parser.add_argument("--second_stage_lr", type=float, default=None)
    parser.add_argument("--second_stage_epochs", type=int, default=None)
    parser.add_argument("--resume_from_stage2", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show_progress", action=argparse.BooleanOptionalAction, default=False)
    return parser


def _resolve_construct_defaults(config: TrainConfig) -> None:
    default_spec = ConstructSpec.lentimpra_default()
    if config.data.promoter_seq is None:
        config.data.promoter_seq = default_spec.promoter_seq
    if config.data.barcode_seq is None:
        config.data.barcode_seq = default_spec.barcode_seq

def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "data": {},
        "head": {},
        "optim": {},
        "stage": {},
        "checkpoint": {},
        "logging": {},
        "runtime": {},
    }

    data_pairs = {
        "input_tsv": args.input_tsv,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "construct_mode": args.construct_mode,
        "reverse_complement": args.reverse_complement,
        "rc_prob": args.rc_prob,
        "random_shift": args.random_shift,
        "shift_prob": args.shift_prob,
        "max_shift": args.max_shift,
        "subset_frac": args.subset_frac,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    head_pairs = {
        "pooling_type": args.pooling_type,
        "center_bp": args.center_bp,
        "hidden_sizes": parse_hidden_sizes(args.hidden_sizes) if args.hidden_sizes is not None else None,
        "dropout": args.dropout,
        "activation": args.activation,
    }
    optim_pairs = {
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lr_scheduler": args.lr_scheduler,
        "plateau_factor": args.plateau_factor,
        "plateau_patience": args.plateau_patience,
        "plateau_mode": args.plateau_mode,
        "plateau_min_lr": args.plateau_min_lr,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clip": args.gradient_clip,
    }
    stage_pairs = {
        "num_epochs": args.num_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "val_evals_per_epoch": args.val_evals_per_epoch,
        "second_stage_lr": args.second_stage_lr,
        "second_stage_epochs": args.second_stage_epochs,
        "resume_from_stage2": args.resume_from_stage2,
    }
    checkpoint_pairs = {
        "pretrained_weights": args.pretrained_weights,
        "checkpoint_dir": args.checkpoint_dir,
        "save_mode": args.save_mode,
    }
    logging_pairs = {
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "wandb_name": args.wandb_name,
    }
    runtime_pairs = {
        "device": args.device,
        "use_amp": args.use_amp,
        "seed": args.seed,
    }

    for section_name, values in (
        ("data", data_pairs),
        ("head", head_pairs),
        ("optim", optim_pairs),
        ("stage", stage_pairs),
        ("checkpoint", checkpoint_pairs),
        ("logging", logging_pairs),
        ("runtime", runtime_pairs),
    ):
        overrides[section_name] = {key: value for key, value in values.items() if value is not None}
    return overrides


def _make_dataset(config: TrainConfig, split: str) -> LentiMPRADataset:
    use_augment = split == "train"
    construct_spec = ConstructSpec(
        left_adapter=config.data.left_adapter_seq,
        right_adapter=config.data.right_adapter_seq,
        promoter_seq=config.data.promoter_seq,
        barcode_seq=config.data.barcode_seq,
    )
    return LentiMPRADataset(
        config.data.input_tsv,
        split=split,
        sequence_length=config.data.sequence_length,
        construct_spec=construct_spec,
        construct_mode=config.data.construct_mode,
        reverse_complement=config.data.reverse_complement if use_augment else False,
        rc_prob=config.data.rc_prob,
        random_shift=config.data.random_shift if use_augment else False,
        shift_prob=config.data.shift_prob,
        max_shift=config.data.max_shift,
        subset_frac=config.data.subset_frac,
        seed=config.runtime.seed,
    )


def main() -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        config = merge_train_config(load_train_config(args.config), _build_overrides(args))
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))

    _resolve_construct_defaults(config)

    torch.manual_seed(config.runtime.seed)
    device = torch.device(config.runtime.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}", flush=True)
    run_dir = Path(config.checkpoint.checkpoint_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as handle:
        json.dump(config.to_dict(), handle, indent=2)
    print(f"Saved config to {run_dir / 'config.json'}")

    print(f"Loading pretrained weights from {config.checkpoint.pretrained_weights}...")
    construct_spec = ConstructSpec(
        left_adapter=config.data.left_adapter_seq,
        right_adapter=config.data.right_adapter_seq,
        promoter_seq=config.data.promoter_seq,
        barcode_seq=config.data.barcode_seq,
    )
    model = EncoderMPRAModel.from_pretrained(
        config.checkpoint.pretrained_weights,
        config.head,
        device=device,
        construct_spec=construct_spec,
    )
    model.initialize_head(config.data.sequence_length, device)
    model.eval()

    n_trainable = sum(p.numel() for p in model.head.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print("EncoderMPRAModel created.")
    print(f"  Trainable (head)   : {n_trainable:,}")
    print(f"  Frozen (backbone)  : {n_total - n_trainable:,}")
    print(f"  Total parameters   : {n_total:,}")
    print(f"  Trainable fraction : {100 * n_trainable / n_total:.4f}%")
    print()
    print("Head architecture:")
    print(model.head)

    train_dataset = _make_dataset(config, "train")
    val_dataset = _make_dataset(config, "val")
    test_dataset = _make_dataset(config, "test")

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    print(f"Datasets loaded from {config.data.input_tsv}")
    print(f"  Train batches : {len(train_loader):,}")
    print(f"  Val batches   : {len(val_loader):,}")
    print(f"  Test batches  : {len(test_loader):,}")

    stage1_optimizer = create_optimizer(config.optim, model.trainable_parameters(include_encoder=False))
    stage1_scheduler = create_scheduler(config.optim, stage1_optimizer, config.stage.num_epochs)
    stage1_scheduler_step = scheduler_stepper(config.optim.lr_scheduler)

    wandb_epoch_logger = None
    if config.logging.use_wandb:
        try:
            import wandb

            wandb.init(
                project=config.logging.wandb_project,
                name=config.logging.wandb_name,
                config=config.to_dict(),
            )

            def wandb_epoch_logger(metrics: dict[str, Any]) -> None:
                stage = str(metrics["stage"])
                epoch = float(metrics["epoch"])
                payload = {"epoch": epoch}
                for key, value in metrics.items():
                    if key in {"stage", "epoch"}:
                        continue
                    payload[f"{stage}/{key}"] = value
                wandb.log(payload)
        except ImportError:
            print("wandb is not installed; continuing without wandb")
            config.logging.use_wandb = False

    if config.stage.second_stage_lr is not None:
        def stage2_optimizer_factory(model_obj):
            return create_optimizer(
                config.optim,
                model_obj.trainable_parameters(include_encoder=True),
                learning_rate=config.stage.second_stage_lr,
            )

        def stage2_scheduler_factory(optimizer):
            return create_scheduler(config.optim, optimizer, config.stage.second_stage_epochs)

        results = run_two_stage_training(
            model,
            train_loader,
            stage1_optimizer=stage1_optimizer,
            stage2_optimizer_factory=stage2_optimizer_factory,
            config=config,
            device=device,
            val_loader=val_loader,
            stage1_scheduler=stage1_scheduler,
            stage2_scheduler_factory=stage2_scheduler_factory,
            stage1_scheduler_step=stage1_scheduler_step,
            stage2_scheduler_step=scheduler_stepper(config.optim.lr_scheduler),
            epoch_callback=wandb_epoch_logger,
            show_progress=args.show_progress,
        )
    else:
        results = run_training_stage(
            model,
            train_loader,
            optimizer=stage1_optimizer,
            config=config,
            device=device,
            num_epochs=config.stage.num_epochs,
            stage="stage1",
            train_encoder=False,
            val_loader=val_loader,
            scheduler=stage1_scheduler,
            scheduler_step=stage1_scheduler_step,
            checkpoint_dir=run_dir / "stage1",
            epoch_callback=wandb_epoch_logger,
            show_progress=args.show_progress,
        )

    test_targets: list[tuple[str, dict[str, Any]]] = [("stage1", results)]
    if "stage2" in results:
        test_targets = [("stage1", results["stage1"]), ("stage2", results["stage2"])]

    final_stage = test_targets[-1][0]
    final_metrics: dict[str, float] | None = None
    final_epoch = 0.0
    for stage_name, stage_result in test_targets:
        checkpoint_path = stage_result.get("best_checkpoint_path")
        test_epoch = float(stage_result.get("best_epoch", 0))
        load_checkpoint(checkpoint_path, model, map_location=device)
        test_metrics = evaluate(model, test_loader, device, use_amp=config.runtime.use_amp)
        results[f"{stage_name}_test_metrics"] = test_metrics
        print(
            f"[{stage_name}] final test | epoch {test_epoch:g} | "
            f"test_loss={test_metrics['loss']:.4f} | "
            f"test_pearson={test_metrics.get('pearson', float('nan')):.4f}"
        )
        if wandb_epoch_logger is not None:
            wandb_epoch_logger(
                {
                    "stage": stage_name,
                    "epoch": test_epoch,
                    "test_loss": test_metrics["loss"],
                    "test_pearson": test_metrics.get("pearson", float("nan")),
                    "event": "final_test",
                }
            )
        if stage_name == final_stage:
            final_metrics = test_metrics
            final_epoch = test_epoch

    assert final_metrics is not None
    results["test_metrics"] = final_metrics
    results["history"]["test_loss"].append(final_metrics["loss"])
    results["history"]["test_pearson"].append(final_metrics.get("pearson", float("nan")))
    results["history"]["test_epoch"].append(final_epoch)

    with open(run_dir / "history.json", "w") as handle:
        json.dump(results["history"], handle, indent=2)

    if config.logging.use_wandb:
        import wandb

        wandb.finish()

    return results


if __name__ == "__main__":
    main()
