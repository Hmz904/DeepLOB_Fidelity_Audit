from __future__ import annotations

from copy import deepcopy

PROFILES: dict[str, dict] = {
    "author_tf1": {
        "data": {"normalization": "decimal", "validation_fraction": 0.20},
        "models": {
            "conv_channels": 32,
            "inception_channels": 64,
            "lstm_hidden": 64,
            "batch_norm": False,
            "dropout": 0.20,
            "dropout_shared_time": True,
            "dropout_at_inference": True,
            "time_padding": "same",
            "second_block_activation": "leaky_relu",
        },
        "training": {
            "batch_size": 128,
            "learning_rate": 1e-4,
            "adam_eps": 1e-7,
            "max_epochs": 200,
            "patience": 20,
            "monitor": "val_loss",
            "early_stopping": False,
        },
    },
    "author_pytorch_literal": {
        "data": {"normalization": "decimal", "validation_fraction": 0.20},
        "models": {
            "conv_channels": 32,
            "inception_channels": 64,
            "lstm_hidden": 64,
            "batch_norm": True,
            "dropout": 0.0,
            "dropout_shared_time": False,
            "dropout_at_inference": False,
            "time_padding": "valid",
            "second_block_activation": "tanh",
            "output_activation": "softmax",
        },
        "training": {
            "batch_size": 64,
            "learning_rate": 1e-4,
            "adam_eps": 1e-8,
            "max_epochs": 50,
            "patience": 20,
            "monitor": "val_loss",
            "early_stopping": False,
        },
    },
    "author_pytorch_corrected": {
        "data": {"normalization": "decimal", "validation_fraction": 0.20},
        "models": {
            "conv_channels": 32,
            "inception_channels": 64,
            "lstm_hidden": 64,
            "batch_norm": True,
            "dropout": 0.0,
            "dropout_shared_time": False,
            "dropout_at_inference": False,
            "time_padding": "valid",
            "second_block_activation": "tanh",
            "output_activation": "logits",
        },
        "training": {
            "batch_size": 64,
            "learning_rate": 1e-4,
            "adam_eps": 1e-8,
            "max_epochs": 50,
            "patience": 20,
            "monitor": "val_loss",
            "early_stopping": False,
        },
    },
    "paper": {
        "data": {"normalization": "zscore", "validation_fraction": 0.20},
        "models": {
            "conv_channels": 16,
            "inception_channels": 32,
            "lstm_hidden": 64,
            "batch_norm": False,
            "dropout": 0.0,
            "dropout_shared_time": False,
            "dropout_at_inference": False,
            "time_padding": "same",
            "second_block_activation": "leaky_relu",
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-2,
            "adam_eps": 1.0,
            "max_epochs": 200,
            "patience": 20,
            "monitor": "val_accuracy",
            "early_stopping": True,
        },
    },
}

# Legacy unsuffixed protocol alias for older callers. Shipped configs use the
# explicit ``author_pytorch_literal`` or ``author_pytorch_corrected`` names.
PROFILES["author_pytorch"] = deepcopy(PROFILES["author_pytorch_literal"])


def get_profile(name: str) -> dict:
    if name not in PROFILES:
        raise ValueError(f"Unknown protocol profile {name!r}; choose from {sorted(PROFILES)}")
    return deepcopy(PROFILES[name])
