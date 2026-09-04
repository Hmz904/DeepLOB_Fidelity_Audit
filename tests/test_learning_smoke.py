import numpy as np
import torch
from torch.utils.data import DataLoader

from deeplob_replication.data import EventPanel, LOBWindowDataset
from deeplob_replication.metrics import classification_metrics
from deeplob_replication.models import DeepLOB
from deeplob_replication.utils import set_seed


def test_small_deeplob_learns_strong_synthetic_lob_signal():
    set_seed(0)
    rng = np.random.default_rng(0)
    n, sequence_length = 1200, 20
    classes = (np.arange(n) // 40) % 3
    labels = np.tile(classes[:, None], (1, 5)).astype(np.int64)
    features = rng.normal(0, 0.05, (n, 40)).astype(np.float32)
    features += (classes[:, None] - 1).astype(np.float32) * 2.0
    panel = EventPanel(features, labels)
    train = LOBWindowDataset(panel, 10, sequence_length, 0, 800)
    test = LOBWindowDataset(panel, 10, sequence_length, 900, 1200)
    model = DeepLOB(conv_channels=4, inception_channels=4, lstm_hidden=8, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(0)
    train_loader = DataLoader(train, batch_size=32, shuffle=True, generator=generator)
    for _ in range(5):
        model.train()
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for x, y in DataLoader(test, batch_size=64):
            y_true.extend(y.tolist())
            y_pred.extend(model(x).argmax(dim=1).tolist())
    metrics = classification_metrics(np.asarray(y_true), np.asarray(y_pred))
    assert metrics["macro_f1"] > 0.90
