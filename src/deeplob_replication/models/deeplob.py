from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SamePadConv2d(nn.Module):
    """TensorFlow-style SAME padding for stride-1 convolutions, including even kernels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
    ) -> None:
        super().__init__()
        kh, kw = kernel_size
        total_h, total_w = kh - 1, kw - 1
        self.pad = (
            total_w // 2,
            total_w - total_w // 2,
            total_h // 2,
            total_h - total_h // 2,
        )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, self.pad))


class ConvActNorm(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] | str = (0, 0),
        batch_norm: bool = False,
        activation: str = "leaky_relu",
    ) -> None:
        if padding == "same":
            if stride != (1, 1):
                raise ValueError("SamePadConv2d currently supports stride=(1,1) only")
            conv: nn.Module = SamePadConv2d(in_channels, out_channels, kernel_size)
        else:
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
            )
        if activation == "leaky_relu":
            act: nn.Module = nn.LeakyReLU(0.01)
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation {activation!r}")
        layers: list[nn.Module] = [conv, act]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        super().__init__(*layers)


class TemporalSharedDropout(nn.Module):
    """Drop whole channels with one mask shared across the time dimension."""

    def __init__(self, p: float, active_at_inference: bool = False) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0,1)")
        self.p = p
        self.active_at_inference = active_at_inference

    def extra_repr(self) -> str:
        return f"p={self.p}, active_at_inference={self.active_at_inference}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active = self.training or self.active_at_inference
        if self.p == 0.0 or not active:
            return x
        keep = 1.0 - self.p
        mask = torch.empty((x.shape[0], 1, x.shape[2]), device=x.device, dtype=x.dtype)
        mask.bernoulli_(keep).div_(keep)
        return x * mask


class DeepLOB(nn.Module):
    """CNN-Inception-LSTM DeepLOB family with explicit protocol knobs."""

    def __init__(
        self,
        conv_channels: int = 32,
        inception_channels: int = 64,
        lstm_hidden: int = 64,
        batch_norm: bool = False,
        dropout: float = 0.2,
        dropout_shared_time: bool = True,
        dropout_at_inference: bool = False,
        time_padding: str = "same",
        second_block_activation: str = "leaky_relu",
        output_activation: str = "logits",
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        if time_padding not in {"same", "valid"}:
            raise ValueError("time_padding must be 'same' or 'valid'")
        if output_activation not in {"logits", "softmax"}:
            raise ValueError("output_activation must be 'logits' or 'softmax'")
        self.output_activation = output_activation
        c = conv_channels
        temporal_padding: tuple[int, int] | str = (
            "same" if time_padding == "same" else (0, 0)
        )
        self.conv = nn.Sequential(
            ConvActNorm(1, c, (1, 2), (1, 2), batch_norm=batch_norm),
            ConvActNorm(c, c, (4, 1), padding=temporal_padding, batch_norm=batch_norm),
            ConvActNorm(c, c, (4, 1), padding=temporal_padding, batch_norm=batch_norm),
            ConvActNorm(
                c,
                c,
                (1, 2),
                (1, 2),
                batch_norm=batch_norm,
                activation=second_block_activation,
            ),
            ConvActNorm(
                c,
                c,
                (4, 1),
                padding=temporal_padding,
                batch_norm=batch_norm,
                activation=second_block_activation,
            ),
            ConvActNorm(
                c,
                c,
                (4, 1),
                padding=temporal_padding,
                batch_norm=batch_norm,
                activation=second_block_activation,
            ),
            ConvActNorm(c, c, (1, 10), batch_norm=batch_norm),
            ConvActNorm(c, c, (4, 1), padding=temporal_padding, batch_norm=batch_norm),
            ConvActNorm(c, c, (4, 1), padding=temporal_padding, batch_norm=batch_norm),
        )
        ic = inception_channels
        self.inception1 = nn.Sequential(
            ConvActNorm(c, ic, (1, 1), padding="same", batch_norm=batch_norm),
            ConvActNorm(ic, ic, (3, 1), padding="same", batch_norm=batch_norm),
        )
        self.inception2 = nn.Sequential(
            ConvActNorm(c, ic, (1, 1), padding="same", batch_norm=batch_norm),
            ConvActNorm(ic, ic, (5, 1), padding="same", batch_norm=batch_norm),
        )
        self.inception3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            ConvActNorm(c, ic, (1, 1), padding="same", batch_norm=batch_norm),
        )
        if dropout <= 0:
            self.dropout: nn.Module = nn.Identity()
        elif dropout_shared_time:
            self.dropout = TemporalSharedDropout(dropout, dropout_at_inference)
        else:
            self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(3 * ic, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, num_classes)

    def set_inference_dropout(self, active: bool) -> None:
        """Override only test-time dropout; fitted weights/checkpoint selection are unchanged."""
        if isinstance(self.dropout, TemporalSharedDropout):
            self.dropout.active_at_inference = active
        elif active and isinstance(self.dropout, nn.Dropout) and self.dropout.p > 0:
            raise ValueError(
                "Inference-dropout override is only supported for TemporalSharedDropout"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 40:
            raise ValueError(f"DeepLOB expects [batch,time,40], got {tuple(x.shape)}")
        x = self.conv(x.unsqueeze(1))
        x = torch.cat(
            [self.inception1(x), self.inception2(x), self.inception3(x)], dim=1
        )
        if x.shape[-1] != 1:
            raise RuntimeError(f"LOB-width should collapse to 1, got {tuple(x.shape)}")
        x = x.squeeze(-1).transpose(1, 2)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        scores = self.head(x[:, -1])
        if self.output_activation == "softmax":
            return torch.softmax(scores, dim=1)
        return scores
