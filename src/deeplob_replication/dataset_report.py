"""Training-free class-balance and majority-baseline diagnostics.

These are properties of the benchmark itself, not of any fitted model, so they can be
computed from the processed panels alone. ``runner.run`` emits the same rows for the
horizons it trains on; ``deeplob-rep dataset-report`` emits them for any horizon set
without fitting anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    CLASS_NAMES,
    HORIZONS,
    LOBWindowDataset,
    load_panel,
    processed_panel_paths,
)
from .metrics import classification_metrics


def test_window_targets(dataset: LOBWindowDataset, horizon: int) -> np.ndarray:
    """Labels of every test window, derived independently of any model prediction."""
    horizon_idx = HORIZONS.index(horizon)
    return np.asarray(
        dataset.labels[dataset.first_target : dataset.end_event, horizon_idx],
        dtype=np.int64,
    )


def horizon_class_rows(
    y_test: np.ndarray, horizon: int, dataset_name: str
) -> tuple[list[dict], dict]:
    """Return (class-distribution rows, majority-baseline row) for one horizon."""
    counts = np.bincount(y_test, minlength=3)
    total = int(counts.sum())
    majority_class = int(counts.argmax())
    distribution = [
        {
            "dataset": dataset_name,
            "horizon": horizon,
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "n": int(count),
            "fraction": float(count / total),
            "majority_class": CLASS_NAMES[majority_class],
            "majority_accuracy": float(counts[majority_class] / total),
        }
        for class_id, count in enumerate(counts)
    ]
    baseline = {
        "dataset": dataset_name,
        "model": "majority",
        "horizon": horizon,
        **classification_metrics(y_test, np.full_like(y_test, majority_class)),
    }
    return distribution, baseline


def build_dataset_report(
    processed_dir: str | Path,
    dataset: str = "fi2010",
    normalization: str = "decimal",
    horizons: tuple[int, ...] = (10, 20, 50),
    sequence_length: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute test-set class balance and majority baselines without training."""
    _, test_path = processed_panel_paths(processed_dir, dataset, normalization)
    if not Path(test_path).exists():
        raise FileNotFoundError(
            f"Processed test panel not found: {test_path}. "
            "Run `deeplob-rep prepare` for FI-2010 or `deeplob-rep synthetic` first."
        )
    panel = load_panel(test_path)
    distribution_rows: list[dict] = []
    baseline_rows: list[dict] = []
    for horizon in horizons:
        windows = LOBWindowDataset(panel, horizon, sequence_length)
        y_test = test_window_targets(windows, horizon)
        distribution, baseline = horizon_class_rows(y_test, horizon, dataset)
        distribution_rows.extend(distribution)
        baseline_rows.append(baseline)
    return pd.DataFrame(distribution_rows), pd.DataFrame(baseline_rows)


def write_dataset_report(
    out_dir: str | Path,
    processed_dir: str | Path,
    dataset: str = "fi2010",
    normalization: str = "decimal",
    horizons: tuple[int, ...] = (10, 20, 50),
    sequence_length: int = 100,
) -> tuple[Path, Path]:
    distribution, baseline = build_dataset_report(
        processed_dir, dataset, normalization, horizons, sequence_length
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    distribution_path = out_dir / f"{dataset}_{normalization}_class_distribution.csv"
    baseline_path = out_dir / f"{dataset}_{normalization}_majority_baseline.csv"
    distribution.to_csv(distribution_path, index=False)
    baseline.to_csv(baseline_path, index=False)
    return distribution_path, baseline_path
