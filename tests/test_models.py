import torch

from deeplob_replication.models import (
    MLPLOB,
    CNNBaseline,
    DeepLOB,
    LinearBaseline,
    LSTMBaseline,
    MLPBaseline,
)
from deeplob_replication.models.deeplob import TemporalSharedDropout


def test_all_models_emit_three_logits():
    x = torch.randn(2, 100, 40)
    models = [
        DeepLOB(conv_channels=8, inception_channels=8, lstm_hidden=8),
        LinearBaseline(100),
        MLPBaseline(100, 16),
        CNNBaseline(16),
        LSTMBaseline(8),
        MLPLOB(100, 16, 1),
    ]
    for model in models:
        y = model(x)
        assert y.shape == (2, 3)
        assert torch.isfinite(y).all()


def test_author_tf1_architecture_matches_reference_parameter_count():
    # The authors' TensorFlow-1 notebook prints exactly 142,691 trainable parameters.
    n = sum(p.numel() for p in DeepLOB().parameters())
    assert n == 142_691


def test_author_pytorch_batchnorm_profile_matches_reference_parameter_count():
    model = DeepLOB(batch_norm=True, dropout=0.0, dropout_shared_time=False)
    n = sum(p.numel() for p in model.parameters())
    assert n == 143_907


def test_literal_paper_figure_profile_matches_reported_rough_parameter_count():
    # Table III reports DeepLOB as approximately 60k parameters.
    model = DeepLOB(
        conv_channels=16,
        inception_channels=32,
        lstm_hidden=64,
        dropout=0.0,
        dropout_shared_time=False,
    )
    n = sum(p.numel() for p in model.parameters())
    assert 59_000 < n < 62_000


def test_temporal_shared_dropout_uses_one_mask_per_channel_across_time():
    torch.manual_seed(0)
    layer = TemporalSharedDropout(0.5)
    layer.train()
    x = torch.ones(8, 12, 16)
    y = layer(x)
    assert torch.equal(y, y[:, :1, :].expand_as(y))
    assert (y == 0).any()
    assert (y != 0).any()


def test_author_tf1_dropout_can_remain_active_at_inference():
    torch.manual_seed(0)
    layer = TemporalSharedDropout(0.5, active_at_inference=True)
    layer.eval()
    y = layer(torch.ones(8, 4, 16))
    assert (y == 0).any()


def test_author_pytorch_profile_reduces_time_axis_to_82_and_uses_tanh_block():
    model = DeepLOB(
        batch_norm=True,
        dropout=0.0,
        dropout_shared_time=False,
        time_padding="valid",
        second_block_activation="tanh",
    )
    z = model.conv(torch.randn(2, 1, 100, 40))
    assert z.shape == (2, 32, 82, 1)
    assert sum(isinstance(layer, torch.nn.Tanh) for layer in model.modules()) == 3


def test_tensorflow_profile_preserves_time_axis_with_same_padding():
    model = DeepLOB(dropout=0.0)
    z = model.conv(torch.randn(2, 1, 100, 40))
    assert z.shape == (2, 32, 100, 1)


def test_author_pytorch_literal_can_emit_probabilities():
    model = DeepLOB(
        conv_channels=8,
        inception_channels=8,
        lstm_hidden=8,
        batch_norm=True,
        dropout=0.0,
        time_padding="valid",
        second_block_activation="tanh",
        output_activation="softmax",
    )
    model.eval()
    out = model(torch.randn(3, 100, 40))
    assert torch.allclose(out.sum(dim=1), torch.ones(3), atol=1e-6)
    assert (out >= 0).all()
