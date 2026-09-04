from __future__ import annotations

import torch
from torch import nn


class LinearBaseline(nn.Module):
    """Multinomial logistic-regression baseline on the flattened LOB window."""

    def __init__(self, sequence_length: int = 100) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(sequence_length * 40, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPBaseline(nn.Module):
    def __init__(self, sequence_length: int = 100, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sequence_length * 40, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNBaseline(nn.Module):
    """Small temporal CNN baseline with no Inception or recurrent layer."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(40, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x.transpose(1, 2)).squeeze(-1)
        return self.head(z)


class LSTMBaseline(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(40, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        return self.head(x[:, -1])
