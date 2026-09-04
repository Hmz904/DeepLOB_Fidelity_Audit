# Fidelity ledger

DeepLOB has an unusual reproducibility problem: the 2019 paper, the authors' original TensorFlow-1 notebook, and the later official PyTorch notebook are not the same executable specification. This repository preserves those differences as named protocols instead of collapsing them into one vaguely “faithful” implementation.

## Primary sources

- Paper: https://arxiv.org/abs/1808.03668
- Authors' repository: https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
- TensorFlow-1 notebook: `jupyter_tensorflow/run_train_tensorflow-version1.ipynb`
- PyTorch notebook: `jupyter_pytorch/run_train_pytorch.ipynb`
- FI-2010 canonical dataset: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649

## Paper profile

The paper explicitly states:

- z-score-normalized FI-2010;
- 100 most recent observations × 40 raw LOB features;
- Figure-3 convolution width 16 and Inception width 32 per branch;
- time convolutions are padded to preserve the temporal dimension;
- LeakyReLU slope 0.01;
- 64-unit LSTM and roughly 60k parameters;
- categorical cross-entropy + Adam, LR 0.01, epsilon 1;
- batch size 32;
- stop after validation accuracy fails to improve for 20 epochs; ~100 epochs is reported as the observed FI-2010 behavior, **not the stopping rule**;
- Setup 2 develops on the first seven days and tests on the final three.

`paper` therefore uses `max_epochs=200` and lets the stated patience rule decide the actual stopping epoch. The paper does not fully specify the validation carve-out, so the chronological final-20% split used by both released notebooks is inherited and explicitly recorded.

## TensorFlow-1 author profile

The original released TF1 notebook instead uses:

- decimal-precision FI-2010 files;
- chronological 80/20 split inside `Train_..._CF_7` **before** constructing windows;
- 32 convolution channels and 64 Inception channels;
- TensorFlow/Keras SAME padding for the 4×1 time convolutions;
- dropout 0.2 after Inception with a mask shared across time;
- `training=True` hard-coded on Dropout, so dropout remains active during validation and prediction;
- 64-unit CuDNNLSTM;
- Adam LR `1e-4`;
- 200 epochs, batch size 128;
- best validation-loss checkpoint, no early stopping.

The notebook reports 142,691 trainable parameters; a unit test locks that count. It reports train/validation/test window counts of 203,701 / 50,851 / 139,488 for the convenience decimal Setup-2 data.

### Inference-time dropout is a stochastic test protocol

Keeping `training=True` means a test score is a Monte-Carlo dropout draw. A fixed eval seed makes one draw reproducible, but does not make it representative. The runner therefore reports the primary fixed-seed metric **and** raw MC draws plus mean/std/min/max. The audit deliberately separates two counterfactuals:

1. `configs/ablations/author_tf1_dropout_off.yaml` changes **evaluation only**. It reuses the same fitted weights and disables dropout only at test time, isolating the pure test-time stochasticity/noise effect.
2. `configs/ablations/author_tf1_validation_dropout_off.yaml` disables inference-style dropout during validation as part of the model protocol. This can select a different best epoch, therefore it changes the fingerprint and retrains.

These two effects must not be collapsed into one “dropout-off” number.

### Random streams and initialization remain numerical fidelity limits

The released TF1 notebook explicitly seeds NumPy with 1 and TensorFlow with 2. A modern PyTorch port cannot share those historical framework random streams exactly, so this repository uses deterministic derived fit/evaluation seeds and reports multi-seed results rather than claiming seed identity.

The current compatibility implementation also uses modern PyTorch module defaults rather than reconstructing historical Keras/CuDNN initialization exactly. Keras Conv/Dense layers traditionally default to Glorot-style initialization, whereas PyTorch Conv/Linear defaults differ; historical CuDNNLSTM input/recurrent initializers also differ from `torch.nn.LSTM` defaults. This is a real source of seed-level and possibly headline-result variation and belongs in the Ablation Ledger if it proves material.

Finally, the TF1 notebook calls `predict(testX_CNN)` without specifying a prediction batch size. Because its Dropout layer is forced active at inference, changing batch partitioning can change the exact random-number consumption for a single Monte-Carlo draw even when the dropout distribution is unchanged. The audit therefore treats single-draw equality as a numerical detail, not a fidelity target.

The repository therefore claims **architectural/training-protocol fidelity**, not bit-identical TensorFlow-1 numerical reproduction.

## Later author PyTorch profiles

The later official PyTorch notebook differs again:

- decimal-precision data and the same chronological 80/20 development split;
- batch 64, Adam LR `1e-4`;
- 32 convolution channels, 64 Inception channels, BatchNorm after activations;
- 4×1 time convolutions use VALID padding, reducing 100 events to 82;
- second convolution block uses Tanh while the first/third use LeakyReLU;
- best validation-loss checkpoint rather than the paper's validation-accuracy early stop.

The notebook architecture has 143,907 parameters, locked by a test here.

The released notebook also applies `softmax` inside `forward()` and then passes those probabilities to `CrossEntropyLoss`, which mathematically expects logits. The audit now preserves both interpretations instead of silently fixing one:

- `author_pytorch_literal` reproduces the released notebook behavior, including softmax-before-`CrossEntropyLoss`;
- `author_pytorch_corrected` keeps the same architecture/training protocol but returns logits, isolating the effect of that objective bug.

The shipped configs use explicit `author_pytorch_literal` and `author_pytorch_corrected` names. The unsuffixed protocol name remains only as a legacy programmatic alias.

## Author convenience archive pinning

`scripts/download_author_setup2.py` no longer follows the mutable `master` branch. It pins the authors' 53.7 MB convenience archive to commit `d45844b022209bd9d7985de97076f2e80c5144dc`, the final commit in the file history that changed `data/data.zip`, and verifies the 56,278,154-byte size recorded by the released notebook before extraction.

The repository does **not** claim a SHA256 for that third-party archive yet: the checksum has not been independently obtained in this release. The immutable commit URL plus exact-size check prevents silent branch drift, while processed train/test SHA256 values are still recorded in every run manifest. A future release may add an independently verified archive SHA256 without changing the data protocol.

A separate opt-in workflow, `.github/workflows/real-data-audit.yml`, downloads and prepares the pinned archive and checks the released notebook's 203,701 / 50,851 / 139,488 train/validation/test window counts. Ordinary CI remains network-free.

## Window-boundary behavior

### Train/validation boundary

The author code splits the seven-day raw event matrix first and then creates windows separately. Validation windows therefore do **not** reach backward into the training segment. This repository matches that behavior and has a regression test for it.

### Test-day boundaries

The author code does something different for test data: it horizontally concatenates Test CF7, CF8 and CF9 and **then** creates 100-event windows. With a 100-event lookback, that means 99 windows cross the CF7→CF8 boundary and 99 cross CF8→CF9: **198 cross-day windows**.

The author profiles preserve this arguably artificial behavior for benchmark fidelity. A future robustness run should reset the window at each trading-day boundary and report the delta rather than silently changing the historical benchmark.

## Metrics

The paper's Setup-2 DeepLOB rows have recall exactly equal to accuracy at k=10, 20 and 50. In single-label multiclass classification, support-weighted recall equals accuracy by identity. This is evidence that the headline table used support-weighted aggregation, although the paper does not explicitly name the averaging rule.

The repository reports both macro and support-weighted metrics and labels the direct paper comparison `weighted_f1_gap_vs_paper`.

## Memory behavior

Released notebooks materialize every `[100,40]` input window. The TF1 notebook reports a training tensor `(203701,100,40,1)`. This repository stores only `[N,40]` events plus labels and slices windows lazily. This changes memory behavior, not sample definition.

## Determinism

Strict deterministic mode fixes Python/NumPy/Torch/CUDA seeds, configures `CUBLAS_WORKSPACE_CONFIG` before CUDA RNG/work is initialized, disables cuDNN benchmarking, and calls `torch.use_deterministic_algorithms(True)`. The quickstart also exports the variable before Python starts. If a requested CUDA operation cannot be executed deterministically, the preferred behavior is to fail rather than silently promise determinism.

This does **not** imply bit identity across GPU architectures, CUDA/cuDNN releases, or TensorFlow vs PyTorch kernels. `run_manifest.json` records the relevant environment so those comparisons remain auditable.

## Benchmark target itself is contestable

The audit reproduces the historical FI-2010 labels first. More recent work (e.g. Berti & Kasneci, 2025, TLOB) argues that the conventional trend-label construction creates horizon bias and proposes an alternative. That is treated as a future target-definition ablation, not retroactively substituted into the historical reproduction.

## MLPLOB-style baseline

`mlplob` is a compact feature/time-mixing baseline inspired by the public TLOB/MLPLOB work. It is not a reproduction claim for that separate paper. It exists to test whether the specialized CNN-Inception-LSTM stack remains useful against a simpler modern mixer under the same data protocol.
