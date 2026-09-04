from deeplob_replication.config import load_config
from deeplob_replication.protocols import get_profile


def test_profile_values_are_explicit():
    paper = get_profile("paper")
    tf1 = get_profile("author_tf1")
    pytorch_literal = get_profile("author_pytorch_literal")
    pytorch_corrected = get_profile("author_pytorch_corrected")

    assert paper["training"]["batch_size"] == 32
    assert paper["training"]["monitor"] == "val_accuracy"
    assert paper["training"]["early_stopping"] is True
    assert tf1["models"]["conv_channels"] == 32
    assert tf1["models"]["dropout_shared_time"] is True
    assert tf1["models"]["dropout_at_inference"] is True
    assert tf1["training"]["max_epochs"] == 200
    assert tf1["training"]["early_stopping"] is False
    assert pytorch_literal["models"]["batch_norm"] is True
    assert pytorch_literal["models"]["time_padding"] == "valid"
    assert pytorch_literal["models"]["second_block_activation"] == "tanh"
    assert pytorch_literal["models"]["output_activation"] == "softmax"
    assert pytorch_corrected["models"]["output_activation"] == "logits"
    assert pytorch_literal["training"]["max_epochs"] == 50
    assert pytorch_literal["training"]["early_stopping"] is False


def test_config_override_preserves_profile_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("protocol: author_tf1\ntraining:\n  seed: 99\n")
    cfg = load_config(p)
    assert cfg.training.seed == 99
    assert cfg.training.batch_size == 128
    assert cfg.models.inception_channels == 64


def test_dataset_identity_is_explicit(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("protocol: author_tf1\ndata:\n  dataset: synthetic\n")
    cfg = load_config(p)
    assert cfg.data.dataset == "synthetic"


def test_all_shipped_ablation_configs_parse():
    from pathlib import Path

    for path in sorted(Path("configs/ablations").glob("*.yaml")):
        cfg = load_config(path)
        assert cfg.models.names == ["deeplob"]
        assert cfg.data.dataset == "fi2010"


def test_primary_author_tf1_config_runs_only_replication_target():
    cfg = load_config("configs/author_tf1.yaml")
    assert cfg.models.names == ["deeplob"]


def test_pytorch_literal_and_corrected_configs_are_unambiguous():
    literal = load_config("configs/author_pytorch_literal.yaml")
    corrected = load_config("configs/author_pytorch_corrected.yaml")
    assert literal.protocol == "author_pytorch_literal"
    assert literal.models.output_activation == "softmax"
    assert corrected.protocol == "author_pytorch_corrected"
    assert corrected.models.output_activation == "logits"


def test_legacy_author_pytorch_profile_name_maps_to_literal_behavior():
    legacy = get_profile("author_pytorch")
    assert legacy["models"]["output_activation"] == "softmax"
