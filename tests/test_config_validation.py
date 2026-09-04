from copy import deepcopy

import pytest

from deeplob_replication.config import RunConfig, validate_config


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("learning_rate", 0.0),
        ("adam_eps", 0.0),
        ("max_epochs", 0),
        ("num_workers", -1),
    ],
)
def test_training_numeric_validation(field, value):
    cfg = RunConfig()
    setattr(cfg.training, field, value)
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_model_and_data_validation_rejects_empty_or_invalid_values():
    cases = []
    cfg = RunConfig()
    cfg.data.sequence_length = 0
    cases.append(cfg)
    cfg = RunConfig()
    cfg.data.horizons = []
    cases.append(cfg)
    cfg = RunConfig()
    cfg.models.names = []
    cases.append(cfg)
    cfg = RunConfig()
    cfg.models.dropout = 1.0
    cases.append(cfg)
    cfg = RunConfig()
    cfg.models.output_activation = "mystery"
    cases.append(cfg)
    for bad in cases:
        with pytest.raises(ValueError):
            validate_config(deepcopy(bad))
