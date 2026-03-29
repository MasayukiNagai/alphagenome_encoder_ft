"""Reusable encoder-only training primitives."""

from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor


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


def set_encoder_trainable(model: torch.nn.Module, trainable: bool) -> None:
    """Toggle trainability of ``model.encoder`` parameters."""

    if not hasattr(model, "encoder"):
        raise AttributeError("Model does not have an 'encoder' attribute")
    for param in model.encoder.parameters():
        param.requires_grad = trainable


def train_epoch(
    model: torch.nn.Module,
    head: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    metric_fns: dict[str, Callable[[Tensor, Tensor], Tensor | float]] | None = None,
    gradient_accumulation_steps: int = 1,
    use_amp: bool = True,
    encoder_trainable: bool = False,
    grad_clip: float | None = None,
) -> dict[str, float]:
    """Train for one epoch."""

    device = torch.device(device)
    loss_fn = loss_fn or _default_loss_fn
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0")

    if encoder_trainable:
        model.train()
    else:
        model.eval()
    head.train()

    total_loss = 0.0
    total_samples = 0
    all_preds: list[Tensor] = []
    all_targets: list[Tensor] = []

    optimizer.zero_grad(set_to_none=True)
    autocast_ctx = _autocast_context(device, use_amp)

    for batch_idx, (sequences, targets) in enumerate(train_loader, start=1):
        sequences = sequences.to(device)
        targets = targets.to(device).float()
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

        model_context = nullcontext() if encoder_trainable else torch.no_grad()
        with model_context:
            with autocast_ctx:
                encoder_output = model(sequences, organism_idx, encoder_only=True)["encoder_output"]

        with autocast_ctx:
            preds = head(encoder_output)
            loss = loss_fn(preds, targets)

        (loss / gradient_accumulation_steps).backward()

        if batch_idx % gradient_accumulation_steps == 0 or batch_idx == len(train_loader):
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
                if encoder_trainable and hasattr(model, "encoder"):
                    torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = targets.shape[0]
        total_samples += batch_size
        total_loss += loss.detach().float().cpu().item() * batch_size
        all_preds.append(preds.detach().float().cpu())
        all_targets.append(targets.detach().float().cpu())

    preds_cat, targets_cat = _gather_predictions(all_preds, all_targets)
    metrics = _compute_metrics(preds_cat, targets_cat, metric_fns)
    metrics["loss"] = total_loss / max(1, total_samples)
    return metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    head: torch.nn.Module,
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
    head.eval()

    total_loss = 0.0
    total_samples = 0
    all_preds: list[Tensor] = []
    all_targets: list[Tensor] = []

    autocast_ctx = _autocast_context(device, use_amp)
    for sequences, targets in data_loader:
        sequences = sequences.to(device)
        targets = targets.to(device).float()
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

        with autocast_ctx:
            encoder_output = model(sequences, organism_idx, encoder_only=True)["encoder_output"]
            preds = head(encoder_output)
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
    model: torch.nn.Module,
    head: torch.nn.Module,
    *,
    save_mode: str,
    stage: str,
    epoch: int,
    head_config: dict[str, Any],
    construct_config: dict[str, Any],
    training_config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Save a checkpoint following the repo checkpoint contract."""

    save_mode = save_mode.lower()
    if save_mode not in {"minimal", "full", "head"}:
        raise ValueError(f"Unknown save_mode: {save_mode}")

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "save_mode": save_mode,
        "stage": stage,
        "epoch": epoch,
        "head_state_dict": head.state_dict(),
        "head_config": head_config,
        "construct_config": construct_config,
        "training_config": training_config or {},
        "metrics": metrics or {},
    }

    if save_mode == "minimal":
        payload["encoder_state_dict"] = model.encoder.state_dict()
    elif save_mode == "full":
        payload["model_state_dict"] = model.state_dict()

    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    head: torch.nn.Module | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into ``model`` and optionally ``head``."""

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    save_mode = checkpoint["save_mode"]

    if save_mode == "minimal":
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    elif save_mode == "full":
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif save_mode != "head":
        raise ValueError(f"Unknown save_mode: {save_mode}")

    if head is not None:
        head.load_state_dict(checkpoint["head_state_dict"])
    return checkpoint


def _default_scheduler_step(scheduler, metrics: dict[str, float]) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metrics["loss"])
    else:
        scheduler.step()


def _stage_eval_points(num_batches: int, frequency: int) -> list[int]:
    if frequency <= 1:
        return [num_batches]
    interval = max(1, num_batches // frequency)
    points = sorted({min(i * interval, num_batches) for i in range(1, frequency + 1)})
    if points[-1] != num_batches:
        points.append(num_batches)
    return points


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
    model: torch.nn.Module,
    head: torch.nn.Module,
    train_loader,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    num_epochs: int,
    stage: str,
    encoder_trainable: bool,
    val_loader=None,
    test_loader=None,
    scheduler=None,
    scheduler_step: Callable[[Any, dict[str, float]], None] | None = None,
    loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    metric_fns: dict[str, Callable[[Tensor, Tensor], Tensor | float]] | None = None,
    checkpoint_dir: str | Path | None = None,
    save_mode: str = "minimal",
    head_config: dict[str, Any] | None = None,
    construct_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    early_stopping_patience: int = 5,
    gradient_accumulation_steps: int = 1,
    use_amp: bool = True,
    val_eval_frequency: int = 1,
    test_eval_frequency: int = 1,
    start_epoch: int = 0,
    grad_clip: float | None = None,
) -> dict[str, Any]:
    """Run a single training stage."""

    device = torch.device(device)
    head_config = head_config or {}
    construct_config = construct_config or {}
    training_config = training_config or {}
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
        train_metrics = train_epoch(
            model,
            head,
            train_loader,
            optimizer,
            device,
            loss_fn=loss_fn,
            metric_fns=metric_fns,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_amp=use_amp,
            encoder_trainable=encoder_trainable,
            grad_clip=grad_clip,
        )
        history["train_loss"].append(train_metrics["loss"])
        history["train_pearson"].append(train_metrics.get("pearson", float("nan")))
        history["train_epoch"].append(float(epoch_number))

        current_monitor = train_metrics["loss"]
        latest_eval_metrics = {"loss": current_monitor}

        if val_loader is not None:
            val_metrics = evaluate(
                model,
                head,
                val_loader,
                device,
                loss_fn=loss_fn,
                metric_fns=metric_fns,
                use_amp=use_amp,
            )
            history["val_loss"].append(val_metrics["loss"])
            history["val_pearson"].append(val_metrics.get("pearson", float("nan")))
            history["val_epoch"].append(float(epoch_number))
            current_monitor = val_metrics["loss"]
            latest_eval_metrics = val_metrics

        if test_loader is not None:
            test_metrics = evaluate(
                model,
                head,
                test_loader,
                device,
                loss_fn=loss_fn,
                metric_fns=metric_fns,
                use_amp=use_amp,
            )
            history["test_loss"].append(test_metrics["loss"])
            history["test_pearson"].append(test_metrics.get("pearson", float("nan")))
            history["test_epoch"].append(float(epoch_number))

        scheduler_step(scheduler, latest_eval_metrics)

        if current_monitor < best_monitor:
            best_monitor = current_monitor
            best_epoch = epoch_number
            evals_without_improvement = 0
            if stage_dir is not None:
                best_checkpoint_path = save_checkpoint(
                    stage_dir / "best.pt",
                    model,
                    head,
                    save_mode=save_mode,
                    stage=stage,
                    epoch=epoch_number,
                    head_config=head_config,
                    construct_config=construct_config,
                    training_config=training_config,
                    metrics=latest_eval_metrics,
                )
        else:
            evals_without_improvement += 1

        if val_loader is not None and evals_without_improvement >= early_stopping_patience:
            break

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_monitor": best_monitor,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
    }


def run_two_stage_training(
    model: torch.nn.Module,
    head: torch.nn.Module,
    train_loader,
    *,
    stage1_optimizer: torch.optim.Optimizer,
    stage2_optimizer_factory: Callable[[torch.nn.Module, torch.nn.Module], torch.optim.Optimizer] | None,
    device: torch.device | str,
    stage1_num_epochs: int,
    stage2_num_epochs: int,
    val_loader=None,
    test_loader=None,
    stage1_scheduler=None,
    stage2_scheduler_factory: Callable[[torch.optim.Optimizer], Any] | None = None,
    stage1_scheduler_step: Callable[[Any, dict[str, float]], None] | None = None,
    stage2_scheduler_step: Callable[[Any, dict[str, float]], None] | None = None,
    loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    metric_fns: dict[str, Callable[[Tensor, Tensor], Tensor | float]] | None = None,
    checkpoint_dir: str | Path | None = None,
    save_mode: str = "minimal",
    head_config: dict[str, Any] | None = None,
    construct_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    early_stopping_patience: int = 5,
    gradient_accumulation_steps: int = 1,
    use_amp: bool = True,
    val_eval_frequency: int = 1,
    test_eval_frequency: int = 1,
    grad_clip: float | None = None,
    resume_from_stage2: bool = False,
) -> dict[str, Any]:
    """Run stage 1 head-only training followed by stage 2 encoder+head training."""

    if stage2_num_epochs <= 0:
        raise ValueError("stage2_num_epochs must be > 0")
    if stage2_optimizer_factory is None:
        raise ValueError("stage2_optimizer_factory is required for two-stage training")
    if save_mode == "head":
        raise ValueError("head save_mode is not allowed for two-stage training")
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required for two-stage training")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    combined_history = _history_template()

    stage1_dir = checkpoint_dir / "stage1"
    stage2_dir = checkpoint_dir / "stage2"
    stage1_result: dict[str, Any]

    if not resume_from_stage2:
        set_encoder_trainable(model, False)
        stage1_result = run_training_stage(
            model,
            head,
            train_loader,
            optimizer=stage1_optimizer,
            device=device,
            num_epochs=stage1_num_epochs,
            stage="stage1",
            encoder_trainable=False,
            val_loader=val_loader,
            test_loader=test_loader,
            scheduler=stage1_scheduler,
            scheduler_step=stage1_scheduler_step,
            loss_fn=loss_fn,
            metric_fns=metric_fns,
            checkpoint_dir=stage1_dir,
            save_mode=save_mode,
            head_config=head_config,
            construct_config=construct_config,
            training_config=training_config,
            early_stopping_patience=early_stopping_patience,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_amp=use_amp,
            val_eval_frequency=val_eval_frequency,
            test_eval_frequency=test_eval_frequency,
            grad_clip=grad_clip,
        )
        _append_stage_history(combined_history, stage1_result["history"])
    else:
        stage1_checkpoint = stage1_dir / "best.pt"
        if not stage1_checkpoint.exists():
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_checkpoint}")
        load_checkpoint(stage1_checkpoint, model, head)
        stage1_result = {
            "history": _history_template(),
            "best_epoch": 0,
            "best_monitor": math.inf,
            "best_checkpoint_path": str(stage1_checkpoint),
        }

    best_stage1_path = stage1_result["best_checkpoint_path"]
    if best_stage1_path is None:
        best_stage1_path = str(stage1_dir / "best.pt")
    load_checkpoint(best_stage1_path, model, head)

    set_encoder_trainable(model, True)
    stage2_optimizer = stage2_optimizer_factory(model, head)
    stage2_scheduler = stage2_scheduler_factory(stage2_optimizer) if stage2_scheduler_factory is not None else None
    stage2_result = run_training_stage(
        model,
        head,
        train_loader,
        optimizer=stage2_optimizer,
        device=device,
        num_epochs=stage2_num_epochs,
        stage="stage2",
        encoder_trainable=True,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=stage2_scheduler,
        scheduler_step=stage2_scheduler_step,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        checkpoint_dir=stage2_dir,
        save_mode=save_mode,
        head_config=head_config,
        construct_config=construct_config,
        training_config=training_config,
        early_stopping_patience=early_stopping_patience,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_amp=use_amp,
        val_eval_frequency=val_eval_frequency,
        test_eval_frequency=test_eval_frequency,
        start_epoch=stage1_result["best_epoch"],
        grad_clip=grad_clip,
    )
    _append_stage_history(combined_history, stage2_result["history"])

    return {
        "history": combined_history,
        "stage1": stage1_result,
        "stage2": stage2_result,
    }
