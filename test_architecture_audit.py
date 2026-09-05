"""Static architecture audit: properties provable without training a single step.

Each test locks one finding documented in ARCHITECTURE_AUDIT.md. They are cheap -- no
optimisation, no data -- so they can run in CI on every commit and will fail loudly if the
model is edited in a way that silently changes its published-specification fidelity.
"""

from __future__ import annotations

import pytest
import torch

from deeplob_replication.models.deeplob import ConvActNorm, DeepLOB, SamePadConv2d


def _n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------------------
# 1. The paper and the released code are different models.
# --------------------------------------------------------------------------------------


def test_author_tf1_parameter_count_matches_the_released_notebook():
    """The TF1 notebook prints 142,691 trainable parameters. Reproduce it exactly."""
    model = DeepLOB(conv_channels=32, inception_channels=64, lstm_hidden=64)
    assert _n_params(model) == 142_691


def test_paper_figure3_parameter_count_matches_the_papers_own_claim():
    """Figure 3 specifies 16 conv / 32 inception channels and 'roughly 60k' parameters.

    The paper is internally consistent; it is the released code that departs from it.
    """
    model = DeepLOB(conv_channels=16, inception_channels=32, lstm_hidden=64)
    assert _n_params(model) == 60_947


def test_released_model_is_materially_larger_than_the_documented_one():
    big = _n_params(DeepLOB(conv_channels=32, inception_channels=64))
    small = _n_params(DeepLOB(conv_channels=16, inception_channels=32))
    assert big / small == pytest.approx(2.34, abs=0.01)


# --------------------------------------------------------------------------------------
# 2 & 3. SAME padding is non-causal, and the contamination lands on the classified step.
# --------------------------------------------------------------------------------------


def test_same_padding_matches_keras_asymmetry_for_even_kernels():
    """Keras SAME pads an even kernel 1 step before and 2 after. That asymmetry is why
    the forward-looking span is twice the backward span."""
    conv = SamePadConv2d(1, 1, (4, 1))
    left, right, top, bottom = conv.pad
    assert (top, bottom) == (1, 2)
    assert (left, right) == (0, 0)


def test_convolution_stack_is_not_causal():
    """Perturbing one input step must move outputs at EARLIER steps too: the stack looks
    forward inside the window. This is not look-ahead bias -- the whole window precedes
    the label -- but it means DeepLOB is not a step-by-step online predictor."""
    model = DeepLOB(dropout=0.0).eval()
    x = torch.randn(1, 100, 40)
    with torch.no_grad():
        base = model.conv(x.unsqueeze(1))
        bumped = x.clone()
        bumped[0, 50, :] += 10.0
        moved = (model.conv(bumped.unsqueeze(1)) - base).abs().amax(dim=(0, 1, 3)) > 1e-6
    idx = torch.nonzero(moved).flatten()
    assert idx.min().item() == 38
    assert idx.max().item() == 56  # six steps ahead of the perturbed input


def test_final_timestep_receptive_field_is_mostly_zero_padding():
    """The classifier reads x[:, -1]. Its conv-stack receptive field is 19 steps wide but
    only 7 carry real data; the other 12 are SAME zero-padding past the window end."""
    model = DeepLOB(dropout=0.0).eval()
    x = torch.randn(1, 100, 40)
    with torch.no_grad():
        base = model.conv(x.unsqueeze(1))[0, :, -1, 0]
        supporting = []
        for t in range(100):
            bumped = x.clone()
            bumped[0, t, :] += 10.0
            out = model.conv(bumped.unsqueeze(1))[0, :, -1, 0]
            if (out - base).abs().max() > 1e-6:
                supporting.append(t)
    assert supporting == list(range(93, 100))
    assert len(supporting) == 7  # of a 19-wide field: 63% padding


def test_head_is_contaminated_but_less_than_the_tail():
    """Padding is asymmetric: 1 step before, 2 after, so the sequence head is cleaner than
    the tail. Locking this prevents a 'symmetric padding' refactor slipping through."""
    model = DeepLOB(dropout=0.0).eval()
    x = torch.randn(1, 100, 40)

    def support(step):
        with torch.no_grad():
            base = model.conv(x.unsqueeze(1))[0, :, step, 0]
            n = 0
            for t in range(100):
                bumped = x.clone()
                bumped[0, t, :] += 10.0
                out = model.conv(bumped.unsqueeze(1))[0, :, step, 0]
                if (out - base).abs().max() > 1e-6:
                    n += 1
        return n

    assert support(0) > support(99)


# --------------------------------------------------------------------------------------
# Shape contract.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("padding,expected_steps", [("same", 100), ("valid", 82)])
def test_temporal_length_into_the_lstm(padding, expected_steps):
    """Six 4x1 convolutions consume 3 steps each under VALID: 100 -> 82. So the VALID
    ablation changes causality AND sequence length; the two cannot be separated."""
    model = DeepLOB(time_padding=padding, dropout=0.0).eval()
    with torch.no_grad():
        h = model.conv(torch.zeros(2, 100, 40).unsqueeze(1))
        merged = torch.cat(
            [model.inception1(h), model.inception2(h), model.inception3(h)], dim=1
        )
    assert merged.shape[2] == expected_steps
    assert merged.shape[3] == 1  # LOB width must collapse: 40 -> 20 -> 10 -> 1


def test_lob_width_collapses_in_the_documented_stages():
    model = DeepLOB(dropout=0.0).eval()
    widths = []
    h = torch.zeros(1, 1, 100, 40)
    with torch.no_grad():
        for block in model.conv:
            h = block(h)
            widths.append(h.shape[3])
    # price/volume pairing, then level pairing, then aggregation across the 10 levels
    assert widths[0] == 20
    assert widths[3] == 10
    assert widths[6] == 1


# --------------------------------------------------------------------------------------
# 4. BatchNorm belongs to this repository, not to either published specification.
# --------------------------------------------------------------------------------------


def test_batch_norm_is_off_by_default():
    """Neither the paper nor the TF1 notebook contains any normalisation layer; searching
    the released notebook for BatchNormalization returns nothing. Any config enabling it
    is measuring a model the authors never ran."""
    model = DeepLOB()
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())


def test_batch_norm_when_enabled_follows_the_activation():
    """Documenting the ordering this repository chose, since there is no author code to
    compare it against."""
    block = ConvActNorm(1, 4, (4, 1), batch_norm=True)
    assert isinstance(block[1], torch.nn.LeakyReLU)
    assert isinstance(block[2], torch.nn.BatchNorm2d)


# --------------------------------------------------------------------------------------
# 5. Dropout reproduces the authors' noise_shape and hard-coded training=True.
# --------------------------------------------------------------------------------------


def test_dropout_mask_is_shared_across_time():
    """The authors use noise_shape=(None, 1, C): independent per batch and channel,
    broadcast across time. A channel dropped at one step is dropped at every step."""
    model = DeepLOB(dropout=0.5, dropout_shared_time=True).train()
    x = torch.ones(8, 100, 192)
    out = model.dropout(x)
    for sample in out:
        # every channel column is constant down the time axis
        assert torch.all(sample.std(dim=0) < 1e-6)


def test_inference_dropout_is_opt_in_and_toggleable():
    """training=True is hard-coded in the notebook, so a test score is one Monte-Carlo
    draw. The override must change evaluation only."""
    model = DeepLOB(dropout=0.5, dropout_shared_time=True).eval()
    x = torch.ones(4, 100, 192)
    assert torch.allclose(model.dropout(x), x)
    model.set_inference_dropout(True)
    assert not torch.allclose(model.dropout(x), x)


def test_output_activation_contract():
    """The authors emit softmax with categorical_crossentropy; torch CrossEntropyLoss
    wants logits. Mathematically equivalent, numerically not."""
    logits = DeepLOB(output_activation="logits", dropout=0.0).eval()
    probs = DeepLOB(output_activation="softmax", dropout=0.0).eval()
    x = torch.randn(3, 100, 40)
    with torch.no_grad():
        assert not torch.allclose(logits(x).sum(dim=1), torch.ones(3))
        assert torch.allclose(probs(x).sum(dim=1), torch.ones(3), atol=1e-5)
