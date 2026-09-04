from copy import deepcopy

from deeplob_replication.config import RunConfig
from deeplob_replication.runner import _fingerprint, _implementation_digest


def test_model_list_and_execution_settings_do_not_invalidate_existing_model():
    cfg = RunConfig()
    a = _fingerprint(cfg, "deeplob", 10)
    cfg2 = deepcopy(cfg)
    cfg2.models.names.append("mlp")
    cfg2.training.num_workers = 8
    cfg2.training.device = "cpu"
    b = _fingerprint(cfg2, "deeplob", 10)
    assert a == b


def test_prediction_parameter_change_invalidates_checkpoint():
    cfg = RunConfig()
    a = _fingerprint(cfg, "deeplob", 10)
    cfg.models.conv_channels = 16
    b = _fingerprint(cfg, "deeplob", 10)
    assert a != b


def test_unrelated_baseline_parameter_does_not_invalidate_deeplob():
    cfg = RunConfig()
    a = _fingerprint(cfg, "deeplob", 10)
    cfg.models.mlp_hidden = 999
    assert _fingerprint(cfg, "deeplob", 10) == a


def test_training_data_change_invalidates_checkpoint():
    cfg = RunConfig()
    a = _fingerprint(cfg, "deeplob", 10, train_data_hash="aaa")
    b = _fingerprint(cfg, "deeplob", 10, train_data_hash="bbb")
    assert a != b


def test_evaluation_mc_repeats_do_not_invalidate_checkpoint():
    cfg = RunConfig()
    a = _fingerprint(cfg, "deeplob", 10)
    cfg.evaluation.mc_dropout_repeats = 11
    assert _fingerprint(cfg, "deeplob", 10) == a


def test_evaluation_dropout_override_does_not_invalidate_checkpoint():
    cfg = RunConfig()
    a = _fingerprint(cfg, "deeplob", 10)
    cfg.evaluation.dropout_at_inference_override = False
    assert _fingerprint(cfg, "deeplob", 10) == a


def test_dataset_identity_invalidates_checkpoint():
    cfg = RunConfig()
    a = _fingerprint(cfg, "deeplob", 10)
    cfg.data.dataset = "synthetic"
    assert _fingerprint(cfg, "deeplob", 10) != a


def test_checkpoint_source_run_is_evaluation_only():
    cfg = RunConfig()
    a = _fingerprint(cfg, "deeplob", 10)
    cfg.evaluation.checkpoint_source_run = "some-base-run"
    assert _fingerprint(cfg, "deeplob", 10) == a


def test_implementation_digest_includes_runner_orchestration(monkeypatch):
    from pathlib import Path

    baseline = _implementation_digest("deeplob")
    original = Path.read_text

    def changed_runner(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.name == "runner.py":
            return text + "\n_CHECKPOINT_DIGEST_TEST_SENTINEL = 1\n"
        return text

    monkeypatch.setattr(Path, "read_text", changed_runner)
    assert _implementation_digest("deeplob") != baseline


def test_output_activation_change_invalidates_checkpoint():
    cfg = RunConfig()
    a = _fingerprint(cfg, "deeplob", 10)
    cfg.models.output_activation = "softmax"
    assert _fingerprint(cfg, "deeplob", 10) != a
