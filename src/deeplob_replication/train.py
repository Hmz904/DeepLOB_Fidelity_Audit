from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import classification_metrics, expected_calibration_error
from .utils import atomic_torch_save


@dataclass
class TrainResult:
    best_epoch: int
    best_val_loss: float
    best_val_accuracy: float
    epochs_ran: int
    seconds: float


def _validation_stats(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += float(criterion(logits, y))
            correct += int((logits.argmax(dim=1) == y).sum())
            n += len(y)
    if n == 0:
        raise ValueError("validation loader is empty")
    return total_loss / n, correct / n


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    adam_eps: float,
    max_epochs: int,
    patience: int,
    checkpoint: str | Path,
    monitor: str = "val_loss",
    early_stopping: bool = True,
) -> TrainResult:
    if monitor not in {"val_loss", "val_accuracy"}:
        raise ValueError("monitor must be val_loss or val_accuracy")
    checkpoint = Path(checkpoint)
    partial = checkpoint.with_suffix(checkpoint.suffix + ".partial")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=adam_eps)
    criterion = nn.CrossEntropyLoss()
    best_value = float("inf") if monitor == "val_loss" else -float("inf")
    best_epoch = -1
    best_val_loss = float("inf")
    best_val_accuracy = -float("inf")
    stale = 0
    t0 = time.perf_counter()

    for epoch in range(max_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        val_loss, val_accuracy = _validation_stats(model, val_loader, device)
        value = val_loss if monitor == "val_loss" else val_accuracy
        improved = value < best_value - 1e-8 if monitor == "val_loss" else value > best_value + 1e-8
        if improved:
            best_value = value
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            stale = 0
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "monitor": monitor,
                    "monitor_value": value,
                },
                partial,
            )
        else:
            stale += 1

        if early_stopping and stale >= patience:
            break

    if best_epoch < 0 or not partial.exists():
        raise RuntimeError("training completed without producing a checkpoint")
    seconds = time.perf_counter() - t0
    epochs_ran = epoch + 1
    state = torch.load(partial, map_location="cpu", weights_only=True)
    state["completed"] = True
    state["epochs_ran"] = epochs_ran
    state["train_seconds"] = seconds
    atomic_torch_save(state, checkpoint)
    partial.unlink()
    return TrainResult(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_accuracy=best_val_accuracy,
        epochs_ran=epochs_ran,
        seconds=seconds,
    )


def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs, preds, targets = [], [], []
    with torch.inference_mode():
        for x, y in loader:
            scores = model(x.to(device))
            if getattr(model, "output_activation", "logits") == "softmax":
                p = scores.cpu().numpy()
            else:
                p = torch.softmax(scores, dim=1).cpu().numpy()
            probs.append(p)
            preds.append(p.argmax(axis=1))
            targets.append(y.numpy())
    if not targets:
        raise ValueError("prediction loader is empty")
    return np.concatenate(targets), np.concatenate(preds), np.concatenate(probs)


def evaluate_predictions(
    y: np.ndarray, pred: np.ndarray, probs: np.ndarray
) -> dict[str, float | int]:
    out = classification_metrics(y, pred)
    out["ece"] = expected_calibration_error(probs, y)
    return out
