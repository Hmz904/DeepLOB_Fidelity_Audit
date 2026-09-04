"""Training-free benchmark diagnostics and release-metadata consistency."""

import re
from pathlib import Path

import numpy as np
import pytest
import tomllib

import deeplob_replication
from deeplob_replication.data import create_synthetic_setup2
from deeplob_replication.dataset_report import (
    build_dataset_report,
    horizon_class_rows,
    write_dataset_report,
)

ROOT = Path(__file__).resolve().parents[1]


def test_majority_baseline_accuracy_equals_majority_class_fraction():
    y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 2, 2])
    distribution, baseline = horizon_class_rows(y, 10, "synthetic")
    stationary = next(r for r in distribution if r["class_name"] == "stationary")
    assert stationary["fraction"] == pytest.approx(0.6)
    assert baseline["accuracy"] == pytest.approx(0.6)
    # Support-weighted recall is accuracy by identity; this is the paper-comparison metric.
    assert baseline["weighted_recall"] == pytest.approx(baseline["accuracy"])
    assert baseline["model"] == "majority"


def test_class_fractions_sum_to_one_per_horizon():
    y = np.array([0, 1, 2, 1, 1, 0])
    distribution, _ = horizon_class_rows(y, 20, "synthetic")
    assert sum(r["fraction"] for r in distribution) == pytest.approx(1.0)
    assert sum(r["n"] for r in distribution) == len(y)


def test_report_runs_without_training_and_writes_both_files(tmp_path):
    processed = tmp_path / "processed"
    create_synthetic_setup2(processed, train_events=300, test_events=260, seed=5)
    distribution, baseline = build_dataset_report(
        processed, dataset="synthetic", horizons=(10, 20), sequence_length=100
    )
    assert set(distribution["horizon"]) == {10, 20}
    assert len(baseline) == 2
    assert (baseline["accuracy"] <= 1).all()

    distribution_path, baseline_path = write_dataset_report(
        tmp_path / "results",
        processed,
        dataset="synthetic",
        horizons=(10, 20),
        sequence_length=100,
    )
    assert distribution_path.exists() and baseline_path.exists()


def test_missing_panel_reports_the_command_that_creates_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="deeplob-rep prepare"):
        build_dataset_report(tmp_path, dataset="fi2010")


def test_package_version_matches_pyproject_and_citation():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    declared = pyproject["project"]["version"]
    assert deeplob_replication.__version__ == declared
    citation = (ROOT / "CITATION.cff").read_text()
    assert re.search(rf"^version: {re.escape(declared)}$", citation, re.MULTILINE)
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert re.search(rf"^## {re.escape(declared)}", changelog, re.MULTILINE)
