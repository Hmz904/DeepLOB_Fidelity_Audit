import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from deeplob_replication.config import RunConfig
from deeplob_replication.data import create_synthetic_setup2
from deeplob_replication.runner import run


def test_runner_end_to_end_on_synthetic_panel(tmp_path: Path):
    processed = tmp_path / "processed"
    create_synthetic_setup2(
        processed,
        normalization="decimal",
        train_events=260,
        test_events=180,
        seed=11,
    )

    cfg = RunConfig()
    cfg.run_name = "integration"
    cfg.output_dir = str(tmp_path / "outputs")
    cfg.data.dataset = "synthetic"
    cfg.data.processed_dir = str(processed)
    cfg.data.sequence_length = 20
    cfg.data.horizons = [10]
    cfg.models.names = ["linear"]
    cfg.training.batch_size = 32
    cfg.training.max_epochs = 1
    cfg.training.patience = 1
    cfg.training.learning_rate = 1e-2
    cfg.training.device = "cpu"
    cfg.evaluation.mc_dropout_repeats = 1

    result = run(cfg)
    out = Path(cfg.output_dir) / cfg.run_name

    assert len(result) == 1
    assert result.iloc[0]["model"] == "linear"
    assert result.iloc[0]["dataset"] == "synthetic"
    assert pd.isna(result.iloc[0]["paper_f1"])
    assert pd.isna(result.iloc[0]["weighted_f1_gap_vs_paper"])
    assert result.iloc[0]["n_obs"] > 0
    assert (out / "metrics.csv").exists()
    assert (out / "baseline_metrics.csv").exists()
    assert (out / "class_distribution.csv").exists()
    assert (out / "confusion" / "linear_k10.csv").exists()
    assert len(list((out / "checkpoints").glob("linear_k10_*.pt"))) == 1

    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["dataset"] == "synthetic"
    assert manifest["data_hashes"]["train"]
    assert manifest["data_hashes"]["test"]
    assert manifest["resolved_config"]["data"]["sequence_length"] == 20
    assert manifest["environment"]["torch"]

    distribution = pd.read_csv(out / "class_distribution.csv")
    assert distribution["n"].sum() == result.iloc[0]["n_obs"]
    majority = pd.read_csv(out / "baseline_metrics.csv")
    assert majority.iloc[0]["model"] == "majority"
    assert 0 <= majority.iloc[0]["accuracy"] <= 1


def test_evaluation_run_can_reuse_checkpoint_from_a_different_run_name(tmp_path: Path):
    processed = tmp_path / "processed"
    create_synthetic_setup2(processed, train_events=220, test_events=150, seed=21)

    base = RunConfig()
    base.run_name = "base"
    base.output_dir = str(tmp_path / "outputs")
    base.data.dataset = "synthetic"
    base.data.processed_dir = str(processed)
    base.data.sequence_length = 20
    base.data.horizons = [10]
    base.models.names = ["linear"]
    base.training.batch_size = 32
    base.training.max_epochs = 1
    base.training.patience = 1
    base.training.device = "cpu"
    base.evaluation.mc_dropout_repeats = 1
    run(base)

    source_ckpt = next((Path(base.output_dir) / "base" / "checkpoints").glob("linear_k10_*.pt"))
    before = source_ckpt.stat().st_mtime_ns

    evaluation = deepcopy(base)
    evaluation.run_name = "evaluation-only"
    evaluation.evaluation.checkpoint_source_run = "base"
    result = run(evaluation)

    assert len(result) == 1
    assert source_ckpt.stat().st_mtime_ns == before
    assert not list(
        (Path(base.output_dir) / "evaluation-only" / "checkpoints").glob("linear_k10_*.pt")
    )
