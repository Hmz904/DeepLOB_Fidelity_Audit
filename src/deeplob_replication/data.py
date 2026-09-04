from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

HORIZONS = (10, 20, 30, 50, 100)
CLASS_NAMES = ("up", "stationary", "down")
DECIMAL_FILES = {
    "train": "Train_Dst_NoAuction_DecPre_CF_7.txt",
    "test7": "Test_Dst_NoAuction_DecPre_CF_7.txt",
    "test8": "Test_Dst_NoAuction_DecPre_CF_8.txt",
    "test9": "Test_Dst_NoAuction_DecPre_CF_9.txt",
}
ZSCORE_FILES = {
    "train": "Train_Dst_NoAuction_ZScore_CF_7.txt",
    "test7": "Test_Dst_NoAuction_ZScore_CF_7.txt",
    "test8": "Test_Dst_NoAuction_ZScore_CF_8.txt",
    "test9": "Test_Dst_NoAuction_ZScore_CF_9.txt",
}


@dataclass(frozen=True)
class EventPanel:
    features: np.ndarray  # [events, 40]
    labels: np.ndarray  # [events, 5], encoded 0/1/2

    def __post_init__(self) -> None:
        if self.features.ndim != 2 or self.features.shape[1] != 40:
            raise ValueError(f"Expected [N,40] features, got {self.features.shape}")
        if self.labels.ndim != 2 or self.labels.shape[1] != 5:
            raise ValueError(f"Expected [N,5] labels, got {self.labels.shape}")
        if len(self.features) != len(self.labels):
            raise ValueError("features and labels must have identical event counts")
        values = np.unique(self.labels)
        if not set(values.tolist()).issubset({0, 1, 2}):
            raise ValueError(f"Labels must map to 0/1/2, got {values.tolist()}")


class LOBWindowDataset(Dataset):
    """Lazy FI-2010 windows confined to one raw event segment.

    ``start_event`` and ``end_event`` delimit the raw segment before windowing, matching the
    author notebooks, which split train/validation first and construct windows second.
    """

    def __init__(
        self,
        panel: EventPanel,
        horizon: int,
        sequence_length: int = 100,
        start_event: int = 0,
        end_event: int | None = None,
    ) -> None:
        if horizon not in HORIZONS:
            raise ValueError(f"Unknown horizon {horizon}")
        self.features = panel.features
        self.labels = panel.labels
        self.horizon_idx = HORIZONS.index(horizon)
        self.sequence_length = sequence_length
        self.segment_start = max(0, start_event)
        self.first_target = self.segment_start + sequence_length - 1
        self.end_event = (
            len(panel.features)
            if end_event is None
            else min(end_event, len(panel.features))
        )
        if self.end_event <= self.first_target:
            raise ValueError("Empty window dataset")

    def __len__(self) -> int:
        return self.end_event - self.first_target

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        event_idx = self.first_target + idx
        lo = event_idx - self.sequence_length + 1
        x = np.asarray(self.features[lo : event_idx + 1], dtype=np.float32)
        y = int(self.labels[event_idx, self.horizon_idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def _orient(raw: np.ndarray) -> np.ndarray:
    if raw.ndim != 2:
        raise ValueError("FI-2010 text data must be a 2D matrix")
    if raw.shape[0] == 149:
        return raw.T
    if raw.shape[1] == 149:
        return raw
    raise ValueError(f"Expected one FI-2010 dimension to equal 149, got {raw.shape}")


def parse_fi2010_matrix(raw: np.ndarray) -> EventPanel:
    matrix = _orient(raw)
    features = np.asarray(matrix[:, :40], dtype=np.float32)
    labels = np.asarray(matrix[:, -5:], dtype=np.int64) - 1
    if not np.isfinite(features).all():
        raise ValueError("LOB features contain NaN or inf")
    return EventPanel(features, labels)


def load_text(path: str | Path) -> EventPanel:
    return parse_fi2010_matrix(np.loadtxt(path, dtype=np.float32))


def processed_panel_paths(
    processed_dir: str | Path, dataset: str, normalization: str
) -> tuple[Path, Path]:
    if dataset not in {"fi2010", "synthetic"}:
        raise ValueError(f"Unknown dataset {dataset!r}")
    stem = f"{dataset}_{normalization}"
    root = Path(processed_dir)
    return root / f"{stem}_train7.npz", root / f"{stem}_test3.npz"


def save_panel(panel: EventPanel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=panel.features, labels=panel.labels)


def load_panel(path: str | Path) -> EventPanel:
    with np.load(path) as z:
        return EventPanel(z["features"].astype(np.float32), z["labels"].astype(np.int64))


def concatenate(panels: Sequence[EventPanel]) -> EventPanel:
    return EventPanel(
        np.concatenate([p.features for p in panels], axis=0),
        np.concatenate([p.labels for p in panels], axis=0),
    )


def profile_filenames(normalization: str) -> dict[str, str]:
    if normalization == "decimal":
        return DECIMAL_FILES.copy()
    if normalization == "zscore":
        return ZSCORE_FILES.copy()
    raise ValueError(normalization)


def convert_setup2(raw_dir: str | Path, processed_dir: str | Path, normalization: str) -> None:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    names = profile_filenames(normalization)
    missing = [name for name in names.values() if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing FI-2010 files: {missing}")
    train = load_text(raw_dir / names["train"])
    test = concatenate([load_text(raw_dir / names[f"test{i}"]) for i in (7, 8, 9)])
    train_path, test_path = processed_panel_paths(processed_dir, "fi2010", normalization)
    save_panel(train, train_path)
    save_panel(test, test_path)


def split_train_validation(
    panel: EventPanel, validation_fraction: float
) -> tuple[tuple[int, int], tuple[int, int]]:
    cut = int(np.floor(len(panel.features) * (1.0 - validation_fraction)))
    return (0, cut), (cut, len(panel.features))


def make_synthetic_panel(n_events: int, seed: int = 0) -> EventPanel:
    """Generate a small learnable FI-2010-shaped panel for CI/integration tests.

    This is not a market simulator and must never be reported as an empirical result.
    Labels are deterministic noisy functions of contemporaneous synthetic features.
    """
    if n_events < 120:
        raise ValueError("synthetic panel needs at least 120 events")
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_events, 40)).astype(np.float32)
    labels = np.empty((n_events, len(HORIZONS)), dtype=np.int64)
    for j, horizon in enumerate(HORIZONS):
        signal = (
            features[:, 0]
            + 0.6 * features[:, 1]
            - 0.3 * features[:, 2]
            + rng.normal(scale=0.35 + 0.002 * horizon, size=n_events)
        )
        lo, hi = np.quantile(signal, [1 / 3, 2 / 3])
        labels[:, j] = np.where(signal > hi, 0, np.where(signal < lo, 2, 1))
    return EventPanel(features, labels)


def create_synthetic_setup2(
    processed_dir: str | Path,
    *,
    normalization: str = "decimal",
    train_events: int = 600,
    test_events: int = 300,
    seed: int = 7,
) -> tuple[Path, Path]:
    """Write compact synthetic train/test panels compatible with ``run``."""
    processed_dir = Path(processed_dir)
    train_path, test_path = processed_panel_paths(processed_dir, "synthetic", normalization)
    save_panel(make_synthetic_panel(train_events, seed), train_path)
    save_panel(make_synthetic_panel(test_events, seed + 1), test_path)
    return train_path, test_path
