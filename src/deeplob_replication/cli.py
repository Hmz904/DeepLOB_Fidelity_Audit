from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import convert_setup2, create_synthetic_setup2
from .dataset_report import build_dataset_report, write_dataset_report
from .runner import run


def _run_multiseed(config_path: str, seeds: list[int]) -> pd.DataFrame:
    base = load_config(config_path)
    frames = []
    for seed in seeds:
        cfg = deepcopy(base)
        cfg.training.seed = seed
        cfg.run_name = f"{base.run_name}-seed{seed}"
        if base.evaluation.checkpoint_source_run is not None:
            cfg.evaluation.checkpoint_source_run = (
                f"{base.evaluation.checkpoint_source_run}-seed{seed}"
            )
        result = run(cfg).copy()
        result["base_seed"] = seed
        frames.append(result)
    all_rows = pd.concat(frames, ignore_index=True)
    out_dir = Path(base.output_dir) / f"{base.run_name}-multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(out_dir / "metrics_by_seed.csv", index=False)
    summary = all_rows.groupby(
        ["dataset", "protocol", "normalization", "model", "horizon"], as_index=False
    ).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std"),
        ece_mean=("ece", "mean"),
        ece_std=("ece", "std"),
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(prog="deeplob-rep")
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("prepare", help="Convert FI-2010 text files into compact NPZ panels")
    convert.add_argument("--raw-dir", default="data/raw")
    convert.add_argument("--processed-dir", default="data/processed")
    convert.add_argument("--normalization", choices=["decimal", "zscore"], default="decimal")

    synthetic = sub.add_parser(
        "synthetic", help="Generate a small learnable FI-2010-shaped smoke-test dataset"
    )
    synthetic.add_argument("--processed-dir", default="data/processed")
    synthetic.add_argument("--normalization", choices=["decimal", "zscore"], default="decimal")
    synthetic.add_argument("--train-events", type=int, default=600)
    synthetic.add_argument("--test-events", type=int, default=300)
    synthetic.add_argument("--seed", type=int, default=7)

    train = sub.add_parser("run", help="Train/evaluate configured models")
    train.add_argument("config")

    report = sub.add_parser(
        "dataset-report",
        help="Write test-set class balance and majority baselines without training",
    )
    report.add_argument("--processed-dir", default="data/processed")
    report.add_argument("--out-dir", default="results")
    report.add_argument("--dataset", choices=["fi2010", "synthetic"], default="fi2010")
    report.add_argument("--normalization", choices=["decimal", "zscore"], default="decimal")
    report.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 50])
    report.add_argument("--sequence-length", type=int, default=100)

    multiseed = sub.add_parser("multiseed", help="Run a config over multiple independent seeds")
    multiseed.add_argument("config")
    multiseed.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])

    args = parser.parse_args()
    if args.command == "prepare":
        convert_setup2(args.raw_dir, args.processed_dir, args.normalization)
    elif args.command == "synthetic":
        train_path, test_path = create_synthetic_setup2(
            args.processed_dir,
            normalization=args.normalization,
            train_events=args.train_events,
            test_events=args.test_events,
            seed=args.seed,
        )
        print(f"Wrote SYNTHETIC smoke data: {train_path} and {test_path}")
    elif args.command == "dataset-report":
        distribution_path, baseline_path = write_dataset_report(
            args.out_dir,
            args.processed_dir,
            dataset=args.dataset,
            normalization=args.normalization,
            horizons=tuple(args.horizons),
            sequence_length=args.sequence_length,
        )
        distribution, baseline = build_dataset_report(
            args.processed_dir,
            dataset=args.dataset,
            normalization=args.normalization,
            horizons=tuple(args.horizons),
            sequence_length=args.sequence_length,
        )
        print(distribution.to_string(index=False))
        print()
        print(baseline.to_string(index=False))
        print(f"\nWrote {distribution_path} and {baseline_path}")
    elif args.command == "multiseed":
        print(_run_multiseed(args.config, args.seeds).to_string(index=False))
    else:
        print(run(load_config(args.config)).to_string(index=False))


if __name__ == "__main__":
    main()
