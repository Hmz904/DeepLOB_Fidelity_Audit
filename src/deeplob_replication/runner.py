from __future__ import annotations

import ast
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import RunConfig
from .data import (
    CLASS_NAMES,
    LOBWindowDataset,
    load_panel,
    processed_panel_paths,
    split_train_validation,
)
from .dataset_report import horizon_class_rows, test_window_targets
from .metrics import confusion
from .models import (
    MLPLOB,
    CNNBaseline,
    DeepLOB,
    LinearBaseline,
    LSTMBaseline,
    MLPBaseline,
)
from .reference import PAPER_SETUP2
from .train import evaluate_predictions, predict, train_model
from .utils import (
    derived_seed,
    environment_manifest,
    file_sha256,
    git_commit,
    resolve_device,
    set_seed,
    stable_hash,
)


def _model(cfg: RunConfig, name: str) -> torch.nn.Module:
    if name == "deeplob":
        return DeepLOB(
            conv_channels=cfg.models.conv_channels,
            inception_channels=cfg.models.inception_channels,
            lstm_hidden=cfg.models.lstm_hidden,
            batch_norm=cfg.models.batch_norm,
            dropout=cfg.models.dropout,
            dropout_shared_time=cfg.models.dropout_shared_time,
            dropout_at_inference=cfg.models.dropout_at_inference,
            time_padding=cfg.models.time_padding,
            second_block_activation=cfg.models.second_block_activation,
            output_activation=cfg.models.output_activation,
        )
    if name == "linear":
        return LinearBaseline(cfg.data.sequence_length)
    if name == "mlp":
        return MLPBaseline(cfg.data.sequence_length, cfg.models.mlp_hidden)
    if name == "cnn":
        return CNNBaseline(cfg.models.cnn_hidden)
    if name == "lstm":
        return LSTMBaseline(cfg.models.lstm_hidden)
    if name == "mlplob":
        return MLPLOB(
            cfg.data.sequence_length,
            cfg.models.mlplob_hidden,
            cfg.models.mlplob_layers,
        )
    raise ValueError(name)


def _architecture_spec(model: torch.nn.Module) -> dict:
    return {
        "repr": repr(model),
        "output_activation": getattr(model, "output_activation", None),
        "state_shapes": {
            name: list(tensor.shape) for name, tensor in model.state_dict().items()
        },
    }


def _implementation_digest(model_name: str) -> str:
    """Hash training-critical source semantics to invalidate stale checkpoints safely.

    We intentionally over-invalidate: changing orchestration, protocol resolution, data
    slicing, training, utilities, or any model implementation should never silently reuse
    weights fitted under older code. Evaluation/report-only modules are excluded.
    """
    del model_name  # retained in the signature for backward compatibility/tests
    root = Path(__file__).parent
    files = [
        root / "config.py",
        root / "data.py",
        root / "protocols.py",
        root / "runner.py",
        root / "train.py",
        root / "utils.py",
        *sorted((root / "models").glob("*.py")),
    ]
    normalized = {}
    for path in files:
        tree = ast.parse(path.read_text())
        normalized[str(path.relative_to(root))] = ast.dump(tree, include_attributes=False)
    return stable_hash(normalized)


def _fingerprint(
    cfg: RunConfig,
    model_name: str,
    horizon: int,
    train_data_hash: str | None = None,
) -> str:
    """Hash only choices that can alter fitted weights/predictions before evaluation overrides."""
    model = _model(cfg, model_name)
    payload = {
        "protocol": cfg.protocol,
        "dataset": cfg.data.dataset,
        "normalization": cfg.data.normalization,
        "validation_fraction": cfg.data.validation_fraction,
        "sequence_length": cfg.data.sequence_length,
        "horizon": horizon,
        "architecture": _architecture_spec(model),
        "implementation": _implementation_digest(model_name),
        "train_data_hash": train_data_hash,
        "training": {
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "adam_eps": cfg.training.adam_eps,
            "max_epochs": cfg.training.max_epochs,
            "patience": cfg.training.patience,
            "monitor": cfg.training.monitor,
            "early_stopping": cfg.training.early_stopping,
            "seed": cfg.training.seed,
            "deterministic": cfg.training.deterministic,
            "fit_seed": derived_seed(
                cfg.training.seed, cfg.protocol, model_name, horizon, "fit"
            ),
            "loader": {"shuffle": True, "drop_last": False},
        },
    }
    return stable_hash(payload)


def _loader(
    dataset: LOBWindowDataset,
    cfg: RunConfig,
    *,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(cfg.training.seed if seed is None else seed)
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=cfg.training.num_workers,
        generator=generator,
    )


def _require_panel(path: Path, config_path_hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Processed panel not found: {path}. {config_path_hint}")


def _effective_inference_dropout(cfg: RunConfig) -> bool:
    override = cfg.evaluation.dropout_at_inference_override
    return cfg.models.dropout_at_inference if override is None else override


def _apply_evaluation_overrides(model: torch.nn.Module, cfg: RunConfig) -> None:
    if not isinstance(model, DeepLOB):
        return
    override = cfg.evaluation.dropout_at_inference_override
    if override is not None:
        model.set_inference_dropout(override)


def _dropout_mc_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: RunConfig,
    model_name: str,
    horizon: int,
    primary_metrics: dict[str, float | int],
) -> dict[str, float | int | str]:
    active = _effective_inference_dropout(cfg)
    if (
        model_name != "deeplob"
        or cfg.models.dropout <= 0
        or not active
        or cfg.evaluation.mc_dropout_repeats <= 1
    ):
        return {"mc_dropout_repeats": 1}

    f1s = [float(primary_metrics["weighted_f1"])]
    accuracies = [float(primary_metrics["accuracy"])]
    for repeat in range(1, cfg.evaluation.mc_dropout_repeats):
        seed = derived_seed(
            cfg.training.seed, cfg.protocol, model_name, horizon, "eval_mc", repeat
        )
        set_seed(seed, cfg.training.deterministic)
        y, pred, probs = predict(model, loader, device)
        metrics = evaluate_predictions(y, pred, probs)
        f1s.append(float(metrics["weighted_f1"]))
        accuracies.append(float(metrics["accuracy"]))
    return {
        "mc_dropout_repeats": len(f1s),
        "mc_weighted_f1_mean": float(np.mean(f1s)),
        "mc_weighted_f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "mc_weighted_f1_min": float(np.min(f1s)),
        "mc_weighted_f1_max": float(np.max(f1s)),
        "mc_weighted_f1_values_json": json.dumps(f1s),
        "mc_accuracy_mean": float(np.mean(accuracies)),
        "mc_accuracy_std": (
            float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0
        ),
        "mc_accuracy_min": float(np.min(accuracies)),
        "mc_accuracy_max": float(np.max(accuracies)),
        "mc_accuracy_values_json": json.dumps(accuracies),
    }


def run(cfg: RunConfig) -> pd.DataFrame:
    device = resolve_device(cfg.training.device)
    train_path, test_path = processed_panel_paths(
        cfg.data.processed_dir, cfg.data.dataset, cfg.data.normalization
    )
    hint = (
        "Run `deeplob-rep prepare --normalization ...` first."
        if cfg.data.dataset == "fi2010"
        else "Run `deeplob-rep synthetic` first."
    )
    _require_panel(train_path, hint)
    _require_panel(test_path, hint)

    train_data_hash = file_sha256(train_path)
    test_data_hash = file_sha256(test_path)
    train_panel = load_panel(train_path)
    test_panel = load_panel(test_path)
    (train_start, train_end), (val_start, val_end) = split_train_validation(
        train_panel, cfg.data.validation_fraction
    )
    out_dir = Path(cfg.output_dir) / cfg.run_name
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "confusion").mkdir(parents=True, exist_ok=True)

    manifest = {
        "protocol": cfg.protocol,
        "dataset": cfg.data.dataset,
        "device": str(device),
        "base_seed": cfg.training.seed,
        "git_commit": git_commit(),
        "resolved_config": asdict(cfg),
        "data_hashes": {"train": train_data_hash, "test": test_data_hash},
        "environment": environment_manifest(),
        "class_mapping": {str(i): name for i, name in enumerate(CLASS_NAMES)},
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    rows: list[dict] = []
    distribution_rows: list[dict] = []
    baseline_rows: list[dict] = []

    for horizon in cfg.data.horizons:
        train_ds = LOBWindowDataset(
            train_panel, horizon, cfg.data.sequence_length, train_start, train_end
        )
        val_ds = LOBWindowDataset(
            train_panel, horizon, cfg.data.sequence_length, val_start, val_end
        )
        test_ds = LOBWindowDataset(test_panel, horizon, cfg.data.sequence_length)
        val_loader = _loader(val_ds, cfg, shuffle=False)
        test_loader = _loader(test_ds, cfg, shuffle=False)

        y_test = test_window_targets(test_ds, horizon)
        horizon_distribution, horizon_baseline = horizon_class_rows(
            y_test, horizon, cfg.data.dataset
        )
        distribution_rows.extend(horizon_distribution)
        baseline_rows.append(horizon_baseline)
        # Persist cheap diagnostics before model fitting so partial runs remain inspectable.
        pd.DataFrame(distribution_rows).to_csv(out_dir / "class_distribution.csv", index=False)
        pd.DataFrame(baseline_rows).to_csv(out_dir / "baseline_metrics.csv", index=False)

        for model_name in cfg.models.names:
            fit_seed = derived_seed(
                cfg.training.seed, cfg.protocol, model_name, horizon, "fit"
            )
            eval_seed = derived_seed(
                cfg.training.seed, cfg.protocol, model_name, horizon, "eval"
            )
            fp = _fingerprint(cfg, model_name, horizon, train_data_hash)
            checkpoint_run = cfg.evaluation.checkpoint_source_run or cfg.run_name
            checkpoint_dir = Path(cfg.output_dir) / checkpoint_run / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt = checkpoint_dir / f"{model_name}_k{horizon}_{fp}.pt"

            set_seed(fit_seed, cfg.training.deterministic)
            model = _model(cfg, model_name)
            if not ckpt.exists():
                if cfg.evaluation.checkpoint_source_run is not None:
                    raise FileNotFoundError(
                        f"Evaluation-only checkpoint not found: {ckpt}. "
                        f"Run the source config for {checkpoint_run!r} first."
                    )
                train_loader = _loader(train_ds, cfg, shuffle=True, seed=fit_seed)
                tr = train_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    cfg.training.learning_rate,
                    cfg.training.adam_eps,
                    cfg.training.max_epochs,
                    cfg.training.patience,
                    ckpt,
                    monitor=cfg.training.monitor,
                    early_stopping=cfg.training.early_stopping,
                )
            else:
                tr = None

            state = torch.load(ckpt, map_location=device, weights_only=True)
            if not state.get("completed", False):
                raise RuntimeError(f"Refusing incomplete checkpoint: {ckpt}")
            model.load_state_dict(state["model_state"])
            model.to(device)
            _apply_evaluation_overrides(model, cfg)
            set_seed(eval_seed, cfg.training.deterministic)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_start = time.perf_counter()
            y, pred, probs = predict(model, test_loader, device)
            if not np.array_equal(y, y_test):
                raise RuntimeError("Prediction targets are misaligned with the test-window targets")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - inference_start
            metrics = evaluate_predictions(y, pred, probs)
            mc_metrics = _dropout_mc_metrics(
                model, test_loader, device, cfg, model_name, horizon, metrics
            )
            cm = pd.DataFrame(confusion(y, pred), index=CLASS_NAMES, columns=CLASS_NAMES)
            cm.index.name = "true"
            cm.columns.name = "predicted"
            cm.to_csv(out_dir / "confusion" / f"{model_name}_k{horizon}.csv")

            ref = (
                PAPER_SETUP2.get(horizon)
                if cfg.data.dataset == "fi2010" and model_name == "deeplob"
                else None
            )
            row = {
                "dataset": cfg.data.dataset,
                "protocol": cfg.protocol,
                "normalization": cfg.data.normalization,
                "model": model_name,
                "horizon": horizon,
                "fingerprint": fp,
                "n_parameters": sum(p.numel() for p in model.parameters()),
                **metrics,
                **mc_metrics,
                "paper_f1": None if ref is None else ref["f1"],
                "weighted_f1_gap_vs_paper": (
                    None if ref is None else metrics["weighted_f1"] - ref["f1"]
                ),
                "best_epoch": int(state["epoch"]),
                "best_val_loss": float(state["val_loss"]),
                "best_val_accuracy": float(state["val_accuracy"]),
                "fit_seed": fit_seed,
                "eval_seed": eval_seed,
                "evaluation_dropout_at_inference": (
                    _effective_inference_dropout(cfg) if model_name == "deeplob" else None
                ),
                "checkpoint_source_run": checkpoint_run,
                "train_seconds": float(
                    state.get("train_seconds", tr.seconds if tr else float("nan"))
                ),
                "inference_seconds": inference_seconds,
                "inference_ms_per_sample": 1000.0 * inference_seconds / len(y),
                "events_per_second": len(y) / inference_seconds,
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)

    return pd.DataFrame(rows)
