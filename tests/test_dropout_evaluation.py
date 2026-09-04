from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from deeplob_replication.config import RunConfig
from deeplob_replication.models import DeepLOB
from deeplob_replication.runner import (
    _apply_evaluation_overrides,
    _dropout_mc_metrics,
    _effective_inference_dropout,
    _fingerprint,
)
from deeplob_replication.utils import set_seed


def test_dropout_off_evaluation_reuses_same_weights_and_changes_only_eval_behavior():
    cfg = RunConfig()
    cfg.models.names = ["deeplob"]
    baseline_fp = _fingerprint(cfg, "deeplob", 10)

    counterfactual = deepcopy(cfg)
    counterfactual.evaluation.dropout_at_inference_override = False
    assert _fingerprint(counterfactual, "deeplob", 10) == baseline_fp
    assert _effective_inference_dropout(counterfactual) is False

    set_seed(1)
    model = DeepLOB(dropout=0.2, dropout_at_inference=True)
    x = torch.randn(2, 100, 40)
    model.eval()
    set_seed(10)
    a = model(x)
    set_seed(11)
    b = model(x)
    assert not torch.equal(a, b)

    _apply_evaluation_overrides(model, counterfactual)
    set_seed(10)
    c = model(x)
    set_seed(11)
    d = model(x)
    torch.testing.assert_close(c, d)


def test_mc_dropout_records_raw_values_and_skips_when_dropout_is_zero(monkeypatch):
    cfg = RunConfig()
    cfg.models.dropout = 0.2
    cfg.models.dropout_at_inference = True
    cfg.evaluation.mc_dropout_repeats = 3
    loader = DataLoader(
        TensorDataset(torch.zeros(2, 100, 40), torch.tensor([0, 1])), batch_size=2
    )
    model = DeepLOB(conv_channels=2, inception_channels=2, lstm_hidden=2, dropout=0.2)
    calls = {"n": 0}

    def fake_predict(*args, **kwargs):
        calls["n"] += 1
        y = np.array([0, 1])
        pred = np.array([0, 1]) if calls["n"] % 2 else np.array([1, 1])
        probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
        return y, pred, probs

    monkeypatch.setattr("deeplob_replication.runner.predict", fake_predict)
    out = _dropout_mc_metrics(
        model,
        loader,
        torch.device("cpu"),
        cfg,
        "deeplob",
        10,
        {"weighted_f1": 1.0, "accuracy": 1.0},
    )
    assert out["mc_dropout_repeats"] == 3
    assert calls["n"] == 2
    assert len(__import__("json").loads(out["mc_weighted_f1_values_json"])) == 3
    assert out["mc_weighted_f1_min"] <= out["mc_weighted_f1_max"]

    cfg.models.dropout = 0.0
    calls["n"] = 0
    out = _dropout_mc_metrics(
        model,
        loader,
        torch.device("cpu"),
        cfg,
        "deeplob",
        10,
        {"weighted_f1": 1.0, "accuracy": 1.0},
    )
    assert out == {"mc_dropout_repeats": 1}
    assert calls["n"] == 0
