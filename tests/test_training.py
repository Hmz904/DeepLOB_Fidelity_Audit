from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deeplob_replication.train import train_model


def test_successful_training_promotes_partial_checkpoint(tmp_path: Path):
    x = torch.randn(24, 4)
    y = (x[:, 0] > 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    model = nn.Linear(4, 2)
    checkpoint = tmp_path / "model.pt"

    result = train_model(
        model,
        loader,
        loader,
        torch.device("cpu"),
        learning_rate=1e-2,
        adam_eps=1e-8,
        max_epochs=2,
        patience=1,
        checkpoint=checkpoint,
        monitor="val_loss",
        early_stopping=False,
    )

    assert result.epochs_ran == 2
    assert checkpoint.exists()
    assert not checkpoint.with_suffix(".pt.partial").exists()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["monitor"] == "val_loss"
    assert state["completed"] is True
    assert state["epochs_ran"] == 2
    assert state["train_seconds"] >= 0
    assert "val_accuracy" in state
