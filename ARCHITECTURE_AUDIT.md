# Static architecture audit

Findings obtained **without training a single step**. Every number below is reproduced by
instantiating the model and inspecting shapes, parameter counts, and perturbation-based
receptive fields. Reproduce with `pytest tests/test_architecture_audit.py`.

Sources compared:

- Paper: https://arxiv.org/abs/1808.03668
- Authors' TF1 notebook: `jupyter_tensorflow/run_train_tensorflow-version1.ipynb`
  (from https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books)

---

## 1. The published results cannot come from the architecture the paper describes

| specification | conv / inception channels | trainable parameters |
|---|---|---|
| Paper, Figure 3 | 16 / 32 | **60,947** |
| Authors' TF1 notebook | 32 / 64 | **142,691** |

The paper states roughly 60k parameters, and 16/32 channels yield 60,947 — so the paper is
**internally consistent**: its parameter count matches its own figure. The released notebook
prints 142,691, which this implementation reproduces to the digit at 32/64 channels.

The released model therefore has **2.34x the parameters of the architecture described in the
paper**. This is not an implementation detail; the two are different models, and the reported
FI-2010 numbers were produced by the larger one.

Consequence for this repository: `paper` and `author_tf1` are separate protocols by necessity,
and any comparison of their scores conflates a channel-width change with everything else.

## 2. The convolution stack is not causal

`SamePadConv2d` pads a 4x1 kernel with **1 step before and 2 steps after**, matching Keras
`padding='same'` for even kernels. Each 4x1 layer therefore reads `[t-1, t+2]`.

Perturbing the input at a single step `t=50` changes conv-stack activations across
**`t=38` to `t=56`** — the representation at `t=44` depends on input from `t=50`, six steps
ahead.

This is **not look-ahead bias**: the whole 100-step window precedes the prediction point, so
looking forward inside the window never crosses the label boundary. But it does mean DeepLOB's
convolution stack is not a causal filter and must not be described as a step-by-step online
predictor. `configs/ablations/author_tf1_valid_padding.yaml` switches to VALID padding, which
*is* causal — so that ablation changes causality and sequence length at once (see §3).

## 3. Zero-padding contamination is concentrated exactly where the classifier reads

The full temporal receptive field after conv + inception is 23 steps. Counting how many of
those 23 positions carry real data rather than SAME zero-padding:

| output step | real inputs / 23 | padding share |
|---|---|---|
| t = 10 … 90 | 23 / 23 | 0% |
| t = 95 | 13 / 23 | 43% |
| t = 97 | 11 / 23 | 52% |
| t = 98 | 10 / 23 | 57% |
| **t = 99** | **9 / 23** | **61%** |

22 of the 100 timesteps fed to the LSTM touch padding. The middle of the sequence is clean;
degradation is confined to the ends and is **asymmetric** — the head sees only 35% padding at
`t=0` because SAME pads 1 step before but 2 steps after.

`DeepLOB.forward` classifies from `x[:, -1]`, the LSTM output at the final step. The last input
to the LSTM is thus the single most padding-contaminated element of the sequence. The LSTM
still integrates all 100 steps, so the final hidden state is not purely padding-driven — but
the terminal features are the most degraded ones in the window.

Testable consequence: under VALID padding the sequence is causal and free of padding, but 18%
shorter (100 -> 82 steps). Any difference between the two protocols mixes these two effects,
and attributing it to "padding" alone is unsound.

## 4. BatchNorm exists in neither published specification

Searching the authors' TF1 notebook for `BatchNormalization` returns **nothing**. The
convolution block is `Conv2D -> LeakyReLU` throughout, with no normalization anywhere.

`ConvActNorm` in this repository accepts `batch_norm=True`, placing normalization *after* the
activation. There is no counterpart in the authors' code to compare that ordering against.

**`batch_norm` is an extension of this repository, not a reconstruction of anything published.**
`author_tf1` and `paper` must keep `batch_norm: false`. Any ablation that enables it is
measuring a model neither the authors nor the paper ever ran, and its result cannot be
attributed to a choice made by the authors.

## 5. Dropout matches the authors' code exactly

The authors' line is:

```python
keras.layers.Dropout(0.2, noise_shape=(None, 1, C))(conv_reshape, training=True)
```

`noise_shape=(None, 1, C)` makes the mask independent across batch and channels but
**broadcast across time** — one channel mask reused for the whole sequence. This is
dimension-for-dimension what `TemporalSharedDropout` builds with
`torch.empty((x.shape[0], 1, x.shape[2]))`.

`training=True` is hard-coded, so dropout stays active during validation and prediction. A
test score is one Monte-Carlo draw. Splitting this into evaluation-only and
validation-inclusive ablations is correct: the latter can select a different best epoch.

## 6. Also confirmed against the authors' notebook

Inception branches at 64 channels (`1x1 -> 3x1`, `1x1 -> 5x1`, `MaxPool(3,1) -> 1x1`);
`CuDNNLSTM(64)`; `Dense(3, activation='softmax')`; `Adam(lr=1e-4)`; 200 epochs; batch size 128;
`save_best_only` on `val_loss` with **no early stopping**; chronological 80/20 split of
`Train_..._CF_7` taken **before** windowing.

Layer shapes verified: LOB width collapses 40 -> 20 (price/volume pairing) -> 10 (level
pairing) -> 1 (aggregation across levels), and the temporal dimension is preserved at 100
under SAME padding, contracting to 82 under VALID.

### Softmax vs logits is mathematically equivalent and numerically is not

The authors emit softmax and use `categorical_crossentropy`; `torch.nn.CrossEntropyLoss`
expects logits and fuses log-softmax for stability. `output_activation` already exposes this,
and `author_pytorch_literal` should keep `softmax`. Worth a line in the ablation ledger as a
clean example of an equivalence that does not survive contact with floating point.

---

## Scope

These findings concern architecture only. They say nothing about achievable accuracy, and none
of them is a bug in the authors' code — items 1 through 3 are properties of the published
design that this repository is the first place, as far as the author of this audit is aware, to
quantify. Item 4 is a property of *this* repository that must not be mistaken for a finding
about the authors.

Not yet checked: whether the paper's prose mentions normalization that the code omits; Keras
vs PyTorch initialization differences (flagged in `FIDELITY.md` as an open numerical-fidelity
limit).
