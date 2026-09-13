"""Rebuild outputs/metrics_combined.csv from the per-run metric files.

`render_results.py` aggregates whatever it is handed, so the combined file decides what the
published figure shows. It had been assembled by hand, which meant the figure could silently go
stale whenever a new seed landed -- and did: after the k=20 three-seed run finished, the file still
carried the single seed-42 row and the figure redrew unchanged.

Sources are given as `horizon:path` so the composition is visible in the command that produced it.
A multiseed directory's `metrics_by_seed.csv` contributes one row per seed; a single-run
directory's `metrics.csv` contributes one row.

Usage
  python scripts/combine_metrics.py \
      --source 10:outputs/fi2010-author-tf1-k10-multiseed/metrics_by_seed.csv \
      --source 20:outputs/fi2010-author-tf1-k20-multiseed/metrics_by_seed.csv \
      --source 50:outputs/fi2010-author-tf1-k50/metrics.csv \
      --out outputs/metrics_combined.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--source", action="append", required=True, help="'horizon:path', repeatable")
parser.add_argument("--out", default="outputs/metrics_combined.csv")
args = parser.parse_args()

frames = []
for spec in args.source:
    horizon, _, raw = spec.partition(":")
    if not raw:
        raise ValueError(f"source spec {spec!r} must be 'horizon:path'")
    horizon, path = int(horizon), Path(raw.strip())
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)

    # A stale path is the failure this script exists to prevent, so refuse a file whose horizon
    # does not match what the caller said it is rather than writing it under the wrong label.
    found = sorted(frame["horizon"].unique())
    if found != [horizon]:
        raise ValueError(f"{path} holds horizons {found}, not [{horizon}]")

    frames.append(frame)
    seeds = frame["base_seed"].tolist() if "base_seed" in frame.columns else ["(unlabelled)"]
    print(f"k={horizon:<3} {len(frame)} row(s) from {path}   seeds: {seeds}")

combined = pd.concat(frames, ignore_index=True, sort=False).sort_values(
    ["horizon", "base_seed"] if any("base_seed" in f.columns for f in frames) else ["horizon"]
)

dup = combined.duplicated(["horizon", "fit_seed"]).sum()
if dup:
    raise ValueError(f"{dup} rows share a (horizon, fit_seed); a source is listed twice")

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(out, index=False)

print()
print(combined.groupby("horizon").agg(
    runs=("weighted_f1", "size"),
    wf1_mean=("weighted_f1", "mean"),
    wf1_sd=("weighted_f1", "std"),
    paper=("paper_f1", "first"),
).assign(gap=lambda d: d["wf1_mean"] - d["paper"]).round(6).to_string())
print()
print(out)
