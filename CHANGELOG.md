# Changelog

## 0.1.4 — first committed results

- record the **first independently verified results**: FI-2010 Setup-2 window counts
  (203,701 / 50,851 / 139,488) reproduced from the real decimal files, and the test-set
  class balance plus majority-class baselines for k = 10/20/50;
- add `deeplob-rep dataset-report`, a training-free command that regenerates those
  class-balance numbers, so `RESULTS.md` is populated from a command rather than by hand;
- share the class-balance/majority-baseline computation between the runner and the new
  command instead of duplicating it;
- fix the remaining Ruff import-order findings so `ruff check .` is clean, and pin the
  linter with `required-version` so a drifting tool cannot silently change the verdict;
- align `pyproject.toml`, `__init__.__version__`, `CITATION.cff` and `CHANGELOG.md`, and
  add a regression test that fails if they ever disagree again;
- cover `scripts/render_results.py` with tests (aggregation, seed dispersion, filtering,
  error path and end-to-end figure write) and move `matplotlib` into the dev extra so CI
  exercises the figure path that synthetic data cannot reach;
- print the number of runs behind the mean ± sd in the figure legend;
- delete the downloaded archive after extraction, and state that integrity comes from the
  immutable commit pin rather than from the byte-size check.

## 0.1.3 — public-repo hardening

- make `author_tf1.yaml` run only DeepLOB and move optional comparison models to `configs/baselines.yaml`;
- pin the authors' convenience archive to immutable commit `d45844b...`, validate the notebook-recorded byte size before extraction, and avoid claiming an unverified archive SHA256;
- expand checkpoint code fingerprints across training-critical orchestration/config/data/model code so runner changes cannot silently reuse stale weights;
- split later-author PyTorch behavior into literal softmax-before-cross-entropy and corrected-logits protocols/configs;
- add opt-in monthly/manual real FI-2010 integrity CI checking the released 203,701 / 50,851 / 139,488 window counts;
- strengthen numeric/config validation and add regression tests for the new protocol and fingerprint semantics;
- document TF1 random-stream, initialization, and inference-batch numerical fidelity limits;
- bound runtime dependencies below their next major versions.

## 0.1.2 — dataset identity and evaluation-fidelity hardening

- pin development linting to `ruff==0.16.5` and normalize imports for that rule set;
- isolate synthetic files as `synthetic_<normalization>_train7/test3.npz`, so smoke data can never overwrite FI-2010 processed panels;
- add explicit `data.dataset: fi2010 | synthetic`, include it in checkpoints/manifests/metrics, and suppress paper reference/gap columns outside FI-2010;
- split dropout ablation into an evaluation-only override that reuses the same checkpoint and a validation/checkpoint-selection ablation that retrains;
- persist raw MC-dropout draws plus mean/std/min/max and skip MC repeats when configured dropout is zero;
- assert prediction targets equal independently derived test-window targets;
- persist class-distribution and majority-baseline files before expensive model fitting for each horizon;
- add `docs/.gitkeep` so the documented figure destination exists in Git;
- make the result renderer aggregate repeated seeds and draw standard-deviation error bars;
- expand integration/regression coverage for dataset isolation, fingerprint semantics, evaluation overrides and MC-dropout diagnostics.

## 0.1.1 — fidelity-audit framing

- add `RESULTS.md`, `ROADMAP.md`, `CITATION.cff` and CI badge;
- pin Ruff in dev dependencies;
- add runnable synthetic smoke-data generation and runner/CLI integration tests;
- enrich manifests with resolved config, data hashes, environment and git metadata;
- enable strict deterministic-algorithm mode;
- document initialization differences and cross-day test windows;
- set paper-profile max epochs to 200 and retain the paper's patience rule;
- add MC-dropout diagnostics, majority baseline and class-distribution outputs;
- add the initial paper-vs-replication result renderer.

## 0.1.0

Initial DeepLOB replication framework with paper, author-TF1 and author-PyTorch profiles.
