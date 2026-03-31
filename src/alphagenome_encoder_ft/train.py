"""Reusable encoder-only training primitives."""

from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from .config import OptimConfig, TrainConfig
from .model import EncoderMPRAModel


def _autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _default_loss_fn(preds: Tensor, targets: Tensor) -> Tensor:
    return F.mse_loss(preds.float(), targets.float())


def _pearson_r(preds: Tensor, targets: Tensor, eps: float = 1e-8) -> Tensor:
    preds = preds.float()
    targets = targets.float()
    if preds.numel() < 2:
        return torch.tensor(float("nan"), device=preds.device)
    preds_centered = preds - preds.mean()
    targets_centered = targets - targets.mean()
    denom = preds_centered.pow(2).sum().sqrt() * targets_centered.pow(2).sum().sqrt()
    return (preds_centered * targets_centered).sum() / (denom + eps)


def _compute_metrics(
    preds: Tensor,
    targets: Tensor,
    metric_fns: dict[str, Callable[[Tensor, Tensor], Tensor | float]] | None,
) -> dict[str, float]:
    functions = metric_fns or {"pearson": _pearson_r}
    metrics: dict[str, float] = {}
    for name, fn in functions.items():
        value = fn(preds, targets)
        if isinstance(value, Tensor):
            value = value.detach().float().cpu().item()
        metrics[name] = float(value)
    return metrics


def _gather_predictions(preds: list[Tensor], targets: list[Tensor]) -> tuple[Tensor, Tensor]:
    pred_tensor = torch.cat(preds, dim=0) if preds else torch.empty(0)
    target_tensor = torch.cat(targets, dim=0) if targets else torch.empty(0)
    return pred_tensor, target_tensor


def set_encoder_trainable(model: EncoderMPRAModel, trainable: bool) -> None:
    model.set_encoder_trainable(trainable)


def create_optimizer(
    optim_config: OptimConfig,
    params,
    *,
    learning_rate: float | None = None,
) -> torch.optim.Optimizer:
    lr = optim_config.learning_rate if learning_rate is None else learning_rate
    if optim_config.optimizer == "adam":
        return Adam(params, lr=lr, weight_decay=optim_config.weight_decay)
    return AdamW(params, lr=lr, weight_decay=optim_config.weight_decay)


def create_scheduler(
    lr_scheduler: str,
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
):
    if lr_scheduler == "constant":
        return None
    if lr_scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))
    if lr_scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=2)
    raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}")


def scheduler_stepper(name: str):
    if name == "plateau":
        return lambda scheduler, metrics: scheduler.step(metrics["loss"]) if scheduler is not None else None
    return lambda scheduler, metrics: scheduler.step() if scheduler is not None else None


def _num_batches(data_loader) -> int | None:
    try:
        return len(data_loader)
    except TypeError:
        return None


def train_epoch(
    model: EncoderMPRAModel,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    metric_fns: dict[str, Callable[[Tensor, Tensor], Tensor | float]] | None = None,
    gradient_accumulation_steps: int = 1,
    use_amp: bool = True,
    train_encoder: bool = False,
    grad_clip: float | None = None,
    show_progress: bool = False,
) -> dict[str, float]:
    """Train for one epoch."""

    device = torch.device(device)
    loss_fn = loss_fn or _default_loss_fn
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0")

    if train_encoder:
        model.train()
    else:
        model.eval()
        model.head.train()

    total_loss = 0.0
    total_samples = 0
    all_preds: list[Tensor] = []
    all_targets: list[Tensor] = []

    optimizer.zero_grad(set_to_none=True)

    num_batches = _num_batches(train_loader)
    batch_iterator = train_loader
    if tqdm is not None and show_progress:
        batch_iterator = tqdm(
            train_loader,
            total=num_batches,
            desc="train",
            leave=False,
        )

    for batch_idx, (sequences, targets) in enumerate(batch_iterator, start=1):
        sequences = sequences.to(device)
        targets = targets.to(device).float()
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)
        autocast_ctx = _autocast_context(device, use_amp)

        if train_encoder:
            with autocast_ctx:
                preds = model(sequences, organism_idx)
                loss = loss_fn(preds, targets)
        else:
            with torch.no_grad():
                with autocast_ctx:
                    encoder_output = model.encode(sequences, organism_idx)
            with autocast_ctx:
                preds = model.predict_from_encoder(encoder_output)
                loss = loss_fn(preds, targets)

        (loss / gradient_accumulation_steps).backward()

        if batch_idx % gradient_accumulation_steps == 0 or batch_idx == len(train_loader):
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), grad_clip)
                if train_encoder:
                    torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = targets.shape[0]
        total_samples += batch_size
        total_loss += loss.detach().float().cpu().item() * batch_size
        all_preds.append(preds.detach().float().cpu())
        all_targets.append(targets.detach().float().cpu())

        if tqdm is not None and show_progress:
            batch_iterator.set_postfix(loss=total_loss / max(1, total_samples))

    preds_cat, targets_cat = _gather_predictions(all_preds, all_targets)
    metrics = _compute_metrics(preds_cat, targets_cat, metric_fns)
    metrics["loss"] = total_loss / max(1, total_samples)
    return metrics


@torch.no_grad()
def evaluate(
    model: EncoderMPRAModel,
    data_loader,
    device: torch.device | str,
    *,
    loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    metric_fns: dict[str, Callable[[Tensor, Tensor], Tensor | float]] | None = None,
    use_amp: bool = True,
) -> dict[str, float]:
    """Evaluate on a data loader."""

    device = torch.device(device)
    loss_fn = loss_fn or _default_loss_fn
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_preds: list[Tensor] = []
    all_targets: list[Tensor] = []

    for sequences, targets in data_loader:
        sequences = sequences.to(device)
        targets = targets.to(device).float()
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)
        with _autocast_context(device, use_amp):
            preds = model(sequences, organism_idx)
            loss = loss_fn(preds, targets)

        batch_size = targets.shape[0]
        total_samples += batch_size
        total_loss += loss.detach().float().cpu().item() * batch_size
        all_preds.append(preds.detach().float().cpu())
        all_targets.append(targets.detach().float().cpu())

    preds_cat, targets_cat = _gather_predictions(all_preds, all_targets)
    metrics = _compute_metrics(preds_cat, targets_cat, metric_fns)
    metrics["loss"] = total_loss / max(1, total_samples)
    return metrics


def save_checkpoint(
    path: str | Path,
    model: EncoderMPRAModel,
    *,
    config: TrainConfig,
    save_mode: str,
    stage: str,
    epoch: int,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Save a checkpoint following the repo checkpoint contract."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "save_mode": save_mode,
        "stage": stage,
        "epoch": epoch,
        "config": config.to_dict(),
        "head_state_dict": model.head.state_dict(),
        "head_config": config.head_kwargs(),
        "construct_config": config.construct_config(),
        "metrics": metrics or {},
    }

    if save_mode == "minimal":
        payload["encoder_state_dict"] = model.encoder.state_dict()
    elif save_mode == "full":
        payload["model_state_dict"] = model.state_dict()
    elif save_mode != "head":
        raise ValueError(f"Unknown save_mode: {save_mode}")

    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    model: EncoderMPRAModel,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into ``model``."""

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    save_mode = checkpoint["save_mode"]

    if save_mode == "minimal":
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    elif save_mode == "full":
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif save_mode != "head":
        raise ValueError(f"Unknown save_mode: {save_mode}")

    model.head.load_state_dict(checkpoint["head_state_dict"])
    return checkpoint


def _default_scheduler_step(scheduler, metrics: dict[str, float]) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metrics["loss"])
    else:
        scheduler.step()


def _history_template() -> dict[str, list[float]]:
    return {
        "train_loss": [],
        "train_pearson": [],
        "train_epoch": [],
        "val_loss": [],
        "val_pearson": [],
        "val_epoch": [],
        "test_loss": [],
        "test_pearson": [],
        "test_epoch": [],
    }


def _append_stage_history(history: dict[str, list[float]], stage_history: dict[str, list[float]]) -> None:
    for key, values in stage_history.items():
        history.setdefault(key, []).extend(values)


def run_training_stage(
    model: EncoderMPRAModel,
    train_loader,
    *,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    device: torch.device | str,
    num_epochs: int,
    stage: str,
    train_encoder: bool,
    val_loader=None,
    scheduler=None,
    scheduler_step: Callable[[Any, dict[str, float]], None] | None = None,
    loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    metric_fns: dict[str, Callable[[Tensor, Tensor], Tensor | float]] | None = None,
    checkpoint_dir: str | Path | None = None,
    start_epoch: int = 0,
    epoch_callback: Callable[[dict[str, Any]], None] | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Run a single training stage."""

    device = torch.device(device)
    scheduler_step = scheduler_step or _default_scheduler_step
    stage_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if stage_dir is not None:
        stage_dir.mkdir(parents=True, exist_ok=True)

    history = _history_template()
    best_monitor = math.inf
    best_epoch = start_epoch
    best_checkpoint_path: Path | None = None
    evals_without_improvement = 0

    for epoch_idx in range(num_epochs):
        epoch_number = start_epoch + epoch_idx + 1
        is_last_epoch = epoch_idx == num_epochs - 1
        should_run_val = val_loader is not None and (
            config.stage.val_eval_frequency <= 1
            or epoch_number % config.stage.val_eval_frequency == 0
            or is_last_epoch
        )
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_fn=loss_fn,
            metric_fns=metric_fns,
            gradient_accumulation_steps=config.optim.gradient_accumulation_steps,
            use_amp=config.runtime.use_amp,
            train_encoder=train_encoder,
            grad_clip=config.optim.gradient_clip,
            show_progress=show_progress,
        )
        history["train_loss"].append(train_metrics["loss"])
        history["train_pearson"].append(train_metrics.get("pearson", float("nan")))
        history["train_epoch"].append(float(epoch_number))

        current_monitor = train_metrics["loss"]
        latest_eval_metrics = {"loss": current_monitor}
        val_metrics = None
        should_update_monitor = val_loader is None

        if should_run_val:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                loss_fn=loss_fn,
                metric_fns=metric_fns,
                use_amp=config.runtime.use_amp,
            )
            history["val_loss"].append(val_metrics["loss"])
            history["val_pearson"].append(val_metrics.get("pearson", float("nan")))
            history["val_epoch"].append(float(epoch_number))
            current_monitor = val_metrics["loss"]
            latest_eval_metrics = val_metrics
            should_update_monitor = True

        scheduler_step(scheduler, latest_eval_metrics)

        if should_update_monitor:
            if current_monitor < best_monitor:
                best_monitor = current_monitor
                best_epoch = epoch_number
                evals_without_improvement = 0
                if stage_dir is not None:
                    best_checkpoint_path = save_checkpoint(
                        stage_dir / "best.pt",
                        model,
                        config=config,
                        save_mode=config.checkpoint.save_mode,
                        stage=stage,
                        epoch=epoch_number,
                        metrics=latest_eval_metrics,
                    )
            else:
                evals_without_improvement += 1

        metrics_parts = [
            f"[{stage}] epoch {epoch_number}",
            f"train_loss={train_metrics['loss']:.4f}",
            f"train_pearson={train_metrics.get('pearson', float('nan')):.4f}",
        ]
        if val_metrics is not None:
            metrics_parts.append(f"val_loss={val_metrics['loss']:.4f}")
            metrics_parts.append(f"val_pearson={val_metrics.get('pearson', float('nan')):.4f}")
        print(" | ".join(metrics_parts))

        if epoch_callback is not None:
            payload: dict[str, Any] = {
                "stage": stage,
                "epoch": float(epoch_number),
                "train_loss": train_metrics["loss"],
                "train_pearson": train_metrics.get("pearson", float("nan")),
            }
            if val_metrics is not None:
                payload["val_loss"] = val_metrics["loss"]
                payload["val_pearson"] = val_metrics.get("pearson", float("nan"))
            epoch_callback(payload)

        if should_run_val and evals_without_improvement >= config.stage.early_stopping_patience:
            break

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_monitor": best_monitor,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
    }


def run_two_stage_training(
    model: EncoderMPRAModel,
    train_loader,
    *,
    stage1_optimizer: torch.optim.Optimizer,
    stage2_optimizer_factory: Callable[[EncoderMPRAModel], torch.optim.Optimizer] | None,
    config: TrainConfig,
    device: torch.device | str,
    val_loader=None,
    stage1_scheduler=None,
    stage2_scheduler_factory: Callable[[torch.optim.Optimizer], Any] | None = None,
    stage1_scheduler_step: Callable[[Any, dict[str, float]], None] | None = None,
    stage2_scheduler_step: Callable[[Any, dict[str, float]], None] | None = None,
    loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    metric_fns: dict[str, Callable[[Tensor, Tensor], Tensor | float]] | None = None,
    epoch_callback: Callable[[dict[str, Any]], None] | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Run stage 1 head-only training followed by stage 2 encoder+head training."""

    if config.stage.second_stage_lr is None:
        raise ValueError("stage.second_stage_lr must be set for two-stage training")
    if stage2_optimizer_factory is None:
        raise ValueError("stage2_optimizer_factory is required for two-stage training")
    if config.checkpoint.save_mode == "head":
        raise ValueError("head save_mode is not allowed for two-stage training")

    checkpoint_dir = Path(config.checkpoint.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    combined_history = _history_template()

    stage1_dir = checkpoint_dir / "stage1"
    stage2_dir = checkpoint_dir / "stage2"
    stage1_result: dict[str, Any]

    if not config.stage.resume_from_stage2:
        model.set_encoder_trainable(False)
        stage1_result = run_training_stage(
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
            loss_fn=loss_fn,
            metric_fns=metric_fns,
            checkpoint_dir=stage1_dir,
            epoch_callback=epoch_callback,
            show_progress=show_progress,
        )
        _append_stage_history(combined_history, stage1_result["history"])
    else:
        stage1_checkpoint = stage1_dir / "best.pt"
        if not stage1_checkpoint.exists():
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_checkpoint}")
        load_checkpoint(stage1_checkpoint, model)
        stage1_result = {
            "history": _history_template(),
            "best_epoch": 0,
            "best_monitor": math.inf,
            "best_checkpoint_path": str(stage1_checkpoint),
        }

    best_stage1_path = stage1_result["best_checkpoint_path"] or str(stage1_dir / "best.pt")
    load_checkpoint(best_stage1_path, model)

    model.set_encoder_trainable(True)
    stage2_optimizer = stage2_optimizer_factory(model)
    stage2_scheduler = stage2_scheduler_factory(stage2_optimizer) if stage2_scheduler_factory is not None else None
    stage2_result = run_training_stage(
        model,
        train_loader,
        optimizer=stage2_optimizer,
        config=config,
        device=device,
        num_epochs=config.stage.second_stage_epochs,
        stage="stage2",
        train_encoder=True,
        val_loader=val_loader,
        scheduler=stage2_scheduler,
        scheduler_step=stage2_scheduler_step,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        checkpoint_dir=stage2_dir,
        start_epoch=stage1_result["best_epoch"],
        epoch_callback=epoch_callback,
        show_progress=show_progress,
    )
    _append_stage_history(combined_history, stage2_result["history"])

    return {
        "history": combined_history,
        "stage1": stage1_result,
        "stage2": stage2_result,
    }
