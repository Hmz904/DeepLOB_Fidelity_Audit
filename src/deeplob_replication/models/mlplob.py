from __future__ import annotations

import torch
from torch import nn


class BiN(nn.Module):
    """Stable bilinear normalization adapted for [batch, features, time]."""

    def __init__(self, features: int, time: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.feature_scale = nn.Parameter(torch.ones(features, 1))
        self.feature_bias = nn.Parameter(torch.zeros(features, 1))
        self.time_scale = nn.Parameter(torch.ones(time, 1))
        self.time_bias = nn.Parameter(torch.zeros(time, 1))
        self.mix_logits = nn.Parameter(torch.zeros(2))

    def extra_repr(self) -> str:
        return f"eps={self.eps}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = (x - x.mean(dim=2, keepdim=True)) / x.std(
            dim=2, keepdim=True, unbiased=False
        ).clamp_min(self.eps)
        temporal = temporal * self.feature_scale + self.feature_bias
        feature = (x - x.mean(dim=1, keepdim=True)) / x.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(self.eps)
        feature = feature * self.time_scale.T + self.time_bias.T
        w = torch.softmax(self.mix_logits, dim=0)
        return w[0] * temporal + w[1] * feature


class MixingMLP(nn.Module):
    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(x + self.fc2(self.act(self.fc1(x)))))


class MLPLOB(nn.Module):
    """Compact MLPLOB-style feature/time mixer for a modern comparison baseline."""

    def __init__(self, sequence_length: int = 100, hidden: int = 40, layers: int = 2) -> None:
        super().__init__()
        self.bin = BiN(40, sequence_length)
        self.feature_proj = nn.Linear(40, hidden)
        self.feature_mix = nn.ModuleList([MixingMLP(hidden) for _ in range(layers)])
        self.time_mix = nn.ModuleList([MixingMLP(sequence_length) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sequence_length * hidden, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bin(x.transpose(1, 2)).transpose(1, 2)
        x = self.feature_proj(x)
        for fm, tm in zip(self.feature_mix, self.time_mix, strict=True):
            x = fm(x)
            x = tm(x.transpose(1, 2)).transpose(1, 2)
        return self.head(x)
