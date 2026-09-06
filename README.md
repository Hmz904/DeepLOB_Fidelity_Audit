# DeepLOB Fidelity Audit

[![CI](https://github.com/Hmz904/DeepLOB_Fidelity_Audit/actions/workflows/ci.yml/badge.svg)](https://github.com/Hmz904/DeepLOB_Fidelity_Audit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A fidelity audit of DeepLOB: the 2019 paper, the authors' TensorFlow-1 notebook, and the later PyTorch notebook are three different executable specifications.**

This repository asks a narrower and more useful question than “can I implement DeepLOB?”:
**how much do those specification differences change the FI-2010 result, and which choices are actually material?**

The implementation is built to make expensive experiments inspectable: named protocols, data/code fingerprints, atomic checkpoints, strict split tests, complete run manifests, multi-seed aggregation, MC-dropout diagnostics, and CI.

## First verified result

`author_tf1`, horizon k = 10, single seed, full 200-epoch run on real FI-2010 Setup-2 data
(run fingerprint `809e019c02e94906`):

| | weighted F1 | accuracy |
|---|---:|---:|
| Paper (Setup 2, k=10) | 0.8340 | 0.8447 |
| **This repository** | **0.8077** | **0.8197** |
| Majority-class floor | 0.5855 | 0.7069 |
| **Gap to paper** | **−0.0263** | −0.0250 |
| **Lift over floor** | **+0.2222** | +0.1128 |

The floor row is not decoration. Predicting *stationary* for every window already scores
0.7069 accuracy on this benchmark, so an accuracy near 0.82 is a lift of 0.11 over a constant
predictor, not 82% of the problem solved. Every table in [`RESULTS.md`](RESULTS.md) reports
the floor alongside the model.

A gap below the published value is **not** treated as failure. The output of this project is
the attribution of that gap — protocol drift, seed variance, inference-time dropout, padding,
initialization — through the Ablation Ledger. With one seed and one horizon, none of those
has been isolated yet, and everything not yet run is marked pending rather than estimated.

> **Result status.** Verified and committed: the data pipeline reproduces the authors'
> exact Setup-2 window counts (203,701 / 50,851 / 139,488) and the 142,691 parameter count;
> the test-set class balance and majority baselines are measured; `author_tf1` k=10 is
> trained and reported above. Still pending: k=20 and k=50, the five-seed table, and the
> Ablation Ledger. Paper targets elsewhere in this README are reference values, not results.
> [`RESULTS.md`](RESULTS.md) keeps the two apart.

**The baseline every DeepLOB number must be read against** (FI-2010 Setup-2 test set,
139,488 windows, no training required — `deeplob-rep dataset-report`):

| k | majority class share | majority accuracy | majority weighted F1 | paper accuracy | lift |
|---:|---:|---:|---:|---:|---:|
| 10 | 70.7% stationary | 0.7069 | 0.5855 | 0.8447 | +0.1378 |
| 20 | 62.1% stationary | 0.6208 | 0.4755 | 0.7485 | +0.1277 |
| 50 | 47.3% stationary | 0.4731 | 0.3039 | 0.8051 | +0.3320 |

## 30-second quickstart

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
pip install -e ".[dev]"
python scripts/download_author_setup2.py
deeplob-rep prepare --normalization decimal
deeplob-rep run configs/author_tf1.yaml
```

On a Chinese/Japanese Windows install also set `PYTHONUTF8=1`; several text reads currently
resolve to the system locale and fail under GBK. See *Known defects* below.

The primary quickstart trains only DeepLOB. Baselines are deliberately separated into `configs/baselines.yaml` so a first reproduction run does not silently expand into 18 model/horizon fits.

Training-free benchmark diagnostics (seconds, CPU):

```bash
deeplob-rep dataset-report --horizons 10 20 50 --out-dir results
```

For a no-data integration smoke test:

```bash
deeplob-rep synthetic
deeplob-rep run configs/synthetic.yaml
```

Synthetic panels are written as `synthetic_<normalization>_*.npz`, never to the FI-2010 filenames. `data.dataset` is persisted in the fingerprint, manifest and metrics; paper-reference columns are forcibly blank for non-FI-2010 runs.

For the conventional five-seed experiment:

```bash
deeplob-rep multiseed configs/author_tf1.yaml --seeds 1 2 3 4 5
```

## What is being audited

DeepLOB consumes the 100 most recent LOB states, each with 40 features (10 price levels × ask price/volume + bid price/volume), and predicts one of three future mid-price movement classes.

`LOB window -> convolution blocks -> Inception -> LSTM -> 3-class logits`

FI-2010 supplies five horizons (`10, 20, 30, 50, 100` events). The headline Setup-2 experiment develops on the first seven days and tests on the final three.

### Three DeepLOB specifications

| Choice | Paper | Author TF1 notebook | Author PyTorch notebook |
|---|---:|---:|---:|
| FI-2010 normalization | z-score | decimal precision | decimal precision |
| first conv channels | 16 (Figure 3) | 32 | 32 |
| Inception branch channels | 32 | 64 | 64 |
| BatchNorm | not specified | no | yes |
| post-Inception dropout | not specified | 0.2 | none |
| dropout active at inference | not specified | **yes** (`training=True`) | no |
| time convolution padding | SAME | SAME | VALID |
| Adam LR | 0.01 | 1e-4 | 1e-4 |
| Adam epsilon | 1 | framework default | framework default |
| batch size | 32 | 128 | 64 |
| stopping/checkpoint | val accuracy patience 20 | 200 epochs, best val loss | best val loss |
| output/loss semantics | softmax + categorical CE | softmax + categorical CE equivalent | **softmax -> `CrossEntropyLoss`** in literal notebook |

Profiles:

- `configs/author_tf1.yaml` — primary author-code replication target; runs **DeepLOB only** by default.
- `configs/author_tf1_k10.yaml` — the single-horizon config that produced the verified k=10 result above.
- `configs/baselines.yaml` — optional linear/MLP/CNN/LSTM/MLPLOB comparisons under the TF1 data/training protocol.
- `configs/paper.yaml` — literal paper choices where they are specified; max epoch is 200 and the stated patience rule decides when training stops.
- `configs/author_pytorch_literal.yaml` — literal later official PyTorch notebook, including softmax-before-cross-entropy.
- `configs/author_pytorch_corrected.yaml` — same PyTorch architecture/protocol with the loss input corrected to logits.
- `configs/ablations/author_tf1_dropout_off.yaml` — **evaluation-only** dropout-off counterfactual; reads the fitted checkpoint from the base run. That checkpoint now exists, so this ablation runs in seconds without retraining.
- `configs/ablations/author_tf1_validation_dropout_off.yaml` — validation/checkpoint-selection dropout-off protocol; retrains by design.
- `configs/ablations/author_tf1_zscore.yaml`, `author_tf1_valid_padding.yaml`, and `author_tf1_paper_channels.yaml` — executable one-factor fidelity ablations for the remaining headline ledger rows.

See [`FIDELITY.md`](FIDELITY.md) for the exact ledger of reproduced and intentionally unreproduced behavior.

## Published Setup-2 targets — **not our results**

| k | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 10 | 84.47 | 84.00 | 84.47 | **83.40** |
| 20 | 74.85 | 74.06 | 74.85 | **72.82** |
| 50 | 80.51 | 80.38 | 80.51 | **80.35** |

The direct comparison column is `weighted_f1_gap_vs_paper`. Macro F1 is also retained because it is more revealing under class imbalance.

## Outputs that make a run auditable

Each run writes:

- `metrics.csv` — dataset identity, model metrics, paper gap (FI-2010 only), parameter count, training/inference time;
- `class_distribution.csv` — class counts/fractions and majority-class accuracy by horizon;
- `baseline_metrics.csv` — majority-class baseline;
- `confusion/*.csv` — fixed 3×3 confusion matrices;
- `run_manifest.json` — **resolved config, git SHA, train/test SHA256, Python/NumPy/pandas/sklearn/Torch versions, CUDA/cuDNN/GPU metadata, seed and class mapping**;
- atomic completed checkpoints per `(model, horizon, prediction fingerprint)`.

For `author_tf1`, forced inference-time dropout makes a test score a Monte-Carlo draw. The primary eval seed remains fixed, and the runner additionally reports mean/std/min/max **and the raw per-draw values** for weighted F1 and accuracy over `evaluation.mc_dropout_repeats` (default 5). Set `evaluation.dropout_at_inference_override: false` to evaluate the **same checkpoint** with inference dropout disabled; this override intentionally does not enter the training fingerprint.

Measured at k=10, that MC spread is a weighted-F1 standard deviation of **0.000213** across five draws — two orders of magnitude below the 2.6-point gap to the paper. Reporting a single draw is therefore not materially misleading here, but that statement is licensed by the measurement, not assumed in advance.

## Models

- `deeplob` — CNN + Inception + LSTM replication target.
- `linear` — flattened multinomial linear classifier.
- `mlp` — simple flattened MLP.
- `cnn` — temporal CNN without Inception/LSTM.
- `lstm` — raw-LOB LSTM.
- `mlplob` — compact MLPLOB-style feature/time mixer; a modern comparison baseline, **not** a claimed bit-for-bit reproduction of Berti & Kasneci (2025).

## Data and memory

The canonical FI-2010 dataset is public through Fairdata/ETSIN. The download script needs roughly 1 GB of free disk: the pinned 56 MB archive expands to about 940 MB of text, which `prepare` then compresses to under 10 MB of NPZ panels. The archive is deleted after extraction. The DeepLOB authors also mirror a 53.7 MB archive containing the four decimal-precision Setup-2 files used by their notebooks.

Raw files are converted once to `[N,40]` events and `[N,5]` labels. Windows are sliced lazily in `Dataset.__getitem__`; the repository deliberately avoids materializing all `[N,100,40]` windows in memory.

The author notebook concatenates the three test-day matrices **before** windowing. This creates 99 windows across each of the two day boundaries (198 cross-day windows total for a 100-event lookback). The author profiles preserve that behavior for fidelity; [`FIDELITY.md`](FIDELITY.md) records it explicitly.

## Reproducibility semantics

`training.deterministic: true` is a strict request, not a marketing label:

- Python/NumPy/Torch/CUDA seeds are fixed;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set;
- cuDNN benchmarking is disabled;
- `torch.use_deterministic_algorithms(True)` is enabled.

On hardware/operations for which PyTorch cannot supply a deterministic algorithm, strict mode may raise instead of silently falling back. Exact floating-point identity across different GPU architectures, CUDA/cuDNN versions, and historical TensorFlow kernels is **not** promised; the manifest exists so those differences are visible.

This machinery got an unplanned end-to-end test. A first attempt at the k=10 run was killed at
75 minutes by a forced operating-system reboot, leaving a partial checkpoint at **epoch 69,
validation loss 0.6693622545938956**. The completed rerun on the restarted machine selected
**epoch 69, validation loss 0.669362** — the same epoch and the same loss to every printed
digit, across two separate process lifetimes and a full system restart.

## Known defects

Logged in full in [`RESULTS.md`](RESULTS.md); summarized here because two of them affect anyone
who clones this repository:

- **`patience` is a dead parameter.** `author_tf1` carries `patience: 20` alongside
  `early_stopping: False`, so every run executes all 200 epochs. At k=10 the best validation
  loss occurred at epoch 69 and the remaining 131 epochs consumed 2.4 hours without changing
  the selected checkpoint.
- **Text I/O without an explicit encoding.** Six `read_text()`/`write_text()` call sites
  resolve to the system locale and raise `UnicodeDecodeError` under GBK. One of them is in
  the code-fingerprint path, which would abort a finished training run as it writes its
  manifest. UTF-8 CI never reproduces this.
- **No per-epoch progress output and no resume.** A 3.5-hour run prints nothing between
  start and finish, and the partial checkpoint stores weights only — no optimizer or RNG
  state.

## Why there is no FI-2010 PnL

This is intentional. FI-2010 is already filtered, transformed and labelled; it is not a reconstructible execution tape. A defensible PnL requires executable bid/ask states, a causal normalization policy, fill/latency assumptions, fees, spread/slippage and inventory constraints. Reporting mid-price PnL without those ingredients would make the repository look more quantitative while making the result less defensible.

The planned raw-LOB extension is specified in [`ROADMAP.md`](ROADMAP.md).

## Benchmark caveat: the labels are part of the model specification

Recent work such as **TLOB (Berti & Kasneci, 2025)** argues that the conventional LOB trend label has a horizon bias and proposes an alternative labelling method. That matters here: this repository first reproduces the historical FI-2010 benchmark as published, then treats label construction as an explicit future ablation rather than pretending the benchmark target is uniquely correct.

## Compute

Measured, not estimated. One full `author_tf1` run at a single horizon:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop, 6 GB, 105 W |
| Software | PyTorch 2.5.1+cu121, CUDA 12.1 |
| Training | 200 epochs, 12,730 s (3 h 32 m), **63.7 s/epoch** |
| Peak GPU memory | about 1.8 GB of 6 GB |
| Inference | 9.79 s for 139,488 windows, 0.070 ms/window |

GPU utilisation sits near 50%: the model is small at 142,691 parameters but the 100-step LSTM
is sequential, so the run is latency-bound on recurrence rather than throughput-bound on the
convolutions. Remaining work (k=20/50, five seeds, three retraining ablations) is roughly
32 GPU-hours at these settings, or about 11 with early stopping enabled — see the `patience`
defect above. CPU is suitable for CI and smoke tests, not the headline experiment.

## Repository credibility checks

```bash
ruff check .
pytest -q
python -m build
```

CI pins Ruff to the exact version declared in `pyproject.toml`; changes in lint-tool sorting rules cannot silently turn a previously green repository red. Core runtime dependencies are also bounded below the next major release to reduce accidental environment drift.

The normal CI suite is network-free. A separate monthly/manual **Real FI-2010 data audit** downloads the author archive, prepares it, and verifies the exact author-notebook train/validation/test window counts without training a model. Archive integrity rests on pinning the URL to an immutable commit SHA; the byte-size check is a secondary guard against truncated downloads.

## Results and ablations

See [`RESULTS.md`](RESULTS.md). The intended headline artifacts are:

1. paper vs independent weighted-F1 gap by horizon (k=10 done, k=20/50 pending);
2. five-seed mean ± standard deviation (pending);
3. class distribution + majority baseline (done);
4. an **Ablation Ledger** measuring the F1 cost of normalization, padding and channel-count ambiguities, while splitting inference dropout into (a) same-checkpoint test-time dropout noise and (b) validation/checkpoint-selection drift. (a) is measured at k=10; (b) is pending.

```bash
pip install -e ".[report]"
python scripts/render_results.py outputs/<run>/metrics.csv --out docs/replication_gap.png
# or, for error bars across seeds:
python scripts/render_results.py outputs/<run>-multiseed/metrics_by_seed.csv --out docs/replication_gap.png
```

## References

- Zhang, Z., Zohren, S., & Roberts, S. (2019). *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books*. IEEE Transactions on Signal Processing, 67(11), 3001–3012.
- Ntakaris, A. et al. (2018). *Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods*. Journal of Forecasting, 37(8), 852–866.
- Berti, L. & Kasneci, G. (2025). *TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data*.

## License

MIT. FI-2010 and third-party reference code retain their own terms; no raw FI-2010 data or third-party weights are redistributed.
