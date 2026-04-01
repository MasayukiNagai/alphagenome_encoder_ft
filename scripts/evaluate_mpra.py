#!/usr/bin/env python
"""Evaluate an encoder-only MPRA checkpoint on the test split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alphagenome_encoder_ft import (
    ConstructSpec,
    EncoderMPRAModel,
    LentiMPRADataset,
    TrainConfig,
    create_dataloader,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an encoder-only AlphaGenome MPRA checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--input_tsv", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=None)
    return parser


def _resolve_construct_defaults(config: TrainConfig) -> None:
    default_spec = ConstructSpec.lentimpra_default()
    if config.data.promoter_seq is None:
        config.data.promoter_seq = default_spec.promoter_seq
    if config.data.barcode_seq is None:
        config.data.barcode_seq = default_spec.barcode_seq


def _load_config_from_checkpoint(checkpoint_path: Path) -> tuple[TrainConfig, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_config = checkpoint.get("config")
    if raw_config is None:
        raise ValueError(f"Checkpoint does not contain a serialized config: {checkpoint_path}")
    config = TrainConfig.from_dict(raw_config)
    _resolve_construct_defaults(config)
    return config, checkpoint


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)

    start = 0
    while start < sorted_values.shape[0]:
        end = start + 1
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def compute_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size < 2:
        return float("nan")

    true_centered = y_true - y_true.mean()
    pred_centered = y_pred - y_pred.mean()
    denom = np.linalg.norm(true_centered) * np.linalg.norm(pred_centered)
    if denom == 0.0:
        return float("nan")
    return float(np.dot(true_centered, pred_centered) / denom)


def compute_spearmanr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size < 2:
        return float("nan")
    return compute_pearsonr(_average_ranks(y_true), _average_ranks(y_pred))


@torch.no_grad()
def collect_predictions(
    model,
    data_loader,
    *,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    for sequences, targets in data_loader:
        sequences = sequences.to(device)
        targets = targets.to(device).float()
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = model(sequences, organism_idx)
        else:
            preds = model(sequences, organism_idx)

        all_targets.append(targets.detach().cpu().numpy().reshape(-1))
        all_predictions.append(preds.detach().float().cpu().numpy().reshape(-1))

    if not all_targets:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    return (
        np.concatenate(all_targets, axis=0).astype(np.float32, copy=False),
        np.concatenate(all_predictions, axis=0).astype(np.float32, copy=False),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_pred - y_true
    mse = float(np.mean(np.square(residual))) if y_true.size else float("nan")
    mae = float(np.mean(np.abs(residual))) if y_true.size else float("nan")
    return {
        "n_samples": int(y_true.size),
        "mse": mse,
        "rmse": float(math.sqrt(mse)) if not math.isnan(mse) else float("nan"),
        "mae": mae,
        "pearsonr": compute_pearsonr(y_true, y_pred),
        "spearmanr": compute_spearmanr(y_true, y_pred),
    }


def save_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "y", "y_pred"])
        for idx, (target, pred) in enumerate(zip(y_true.tolist(), y_pred.tolist(), strict=True)):
            writer.writerow([idx, target, pred])


def save_scatter_plot(path: Path, y_true: np.ndarray, y_pred: np.ndarray, metrics: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=10, alpha=0.6, edgecolors="none")

    finite_values = np.concatenate([y_true, y_pred])
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size:
        lower = float(finite_values.min())
        upper = float(finite_values.max())
        ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.0, color="black")

    ax.set_xlabel("y")
    ax.set_ylabel("y_pred")
    ax.set_title("Test Set: y vs y_pred")
    ax.text(
        0.03,
        0.97,
        "\n".join(
            [
                f"n = {metrics['n_samples']}",
                f"Pearson r = {metrics['pearsonr']:.4f}",
                f"Spearman rho = {metrics['spearmanr']:.4f}",
                f"RMSE = {metrics['rmse']:.4f}",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not checkpoint_path.exists():
        parser.error(f"Checkpoint not found: {checkpoint_path}")

    config, checkpoint = _load_config_from_checkpoint(checkpoint_path)

    if args.input_tsv is not None:
        config.data.input_tsv = args.input_tsv
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.pin_memory is not None:
        config.data.pin_memory = args.pin_memory
    if args.use_amp is not None:
        config.runtime.use_amp = args.use_amp
    if args.device is not None:
        config.runtime.device = args.device

    if not config.data.input_tsv:
        parser.error("data.input_tsv must be present in the checkpoint config or provided via --input_tsv")

    device = torch.device(config.runtime.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    default_output_dir = checkpoint_path.parent / f"{checkpoint_path.stem}_test_eval"
    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else default_output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = EncoderMPRAModel.from_checkpoint(checkpoint_path, device=device)
    construct_spec = model.construct_spec or ConstructSpec(
        left_adapter=config.data.left_adapter_seq,
        right_adapter=config.data.right_adapter_seq,
        promoter_seq=config.data.promoter_seq,
        barcode_seq=config.data.barcode_seq,
    )

    test_dataset = LentiMPRADataset(
        config.data.input_tsv,
        split="test",
        sequence_length=config.data.sequence_length,
        construct_spec=construct_spec,
        construct_mode=config.data.construct_mode,
        reverse_complement=False,
        random_shift=False,
        subset_frac=1.0,
        seed=config.runtime.seed,
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    y_true, y_pred = collect_predictions(
        model,
        test_loader,
        device=device,
        use_amp=config.runtime.use_amp,
    )
    metrics = compute_metrics(y_true, y_pred)
    metrics.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "output_dir": str(output_dir),
            "save_mode": checkpoint.get("save_mode"),
            "checkpoint_stage": checkpoint.get("stage"),
            "checkpoint_epoch": checkpoint.get("epoch"),
        }
    )

    predictions_path = output_dir / "test_predictions.csv"
    metrics_path = output_dir / "test_metrics.json"
    plot_path = output_dir / "y_vs_y_pred.png"

    save_predictions(predictions_path, y_true, y_pred)
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)
    save_scatter_plot(plot_path, y_true, y_pred, metrics)

    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved plot to {plot_path}")
    return metrics


if __name__ == "__main__":
    main()
