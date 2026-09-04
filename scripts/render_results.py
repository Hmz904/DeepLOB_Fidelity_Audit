"""Render paper-vs-replication weighted-F1 bars with optional multi-seed error bars."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if "dataset" in df.columns:
        df = df[df["dataset"] == "fi2010"]
    z = df[(df["model"] == "deeplob") & df["paper_f1"].notna()].copy()
    if z.empty:
        raise ValueError("input contains no FI-2010 DeepLOB rows with paper reference values")
    grouped = (
        z.groupby("horizon", as_index=False)
        .agg(
            paper_f1=("paper_f1", "first"),
            ours_mean=("weighted_f1", "mean"),
            ours_std=("weighted_f1", "std"),
            n_runs=("weighted_f1", "size"),
        )
        .sort_values("horizon")
    )
    grouped["ours_std"] = grouped["ours_std"].fillna(0.0)
    return grouped


def _independent_label(z: pd.DataFrame) -> str:
    """State how many runs the mean/sd summarizes so a bar is never ambiguous."""
    runs = sorted(set(int(n) for n in z["n_runs"]))
    if runs == [1]:
        return "Independent run (single run)"
    if len(runs) == 1:
        return f"Independent run (mean ± sd, n={runs[0]})"
    return f"Independent run (mean ± sd, n={min(runs)}-{max(runs)})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", help="metrics.csv or metrics_by_seed.csv")
    parser.add_argument("--out", default="docs/replication_gap.png")
    args = parser.parse_args()

    z = _aggregate(pd.read_csv(args.metrics))
    x = np.arange(len(z))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - width / 2, z["paper_f1"], width, label="Paper")
    ax.bar(
        x + width / 2,
        z["ours_mean"],
        width,
        yerr=z["ours_std"],
        capsize=4,
        label=_independent_label(z),
    )
    for i, row in z.reset_index(drop=True).iterrows():
        gap = row["ours_mean"] - row["paper_f1"]
        top = row["ours_mean"] + row["ours_std"]
        ax.text(i + width / 2, top + 0.01, f"{gap:+.3f}", ha="center")
    ax.set_xticks(x, [str(int(h)) for h in z["horizon"]])
    ax.set_xlabel("Prediction horizon k (events)")
    ax.set_ylabel("Weighted F1")
    upper = max(z["paper_f1"].max(), (z["ours_mean"] + z["ours_std"]).max()) + 0.12
    ax.set_ylim(0, min(1.0, upper))
    ax.legend()
    ax.set_title("DeepLOB FI-2010: published vs independent replication")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)


if __name__ == "__main__":
    main()
