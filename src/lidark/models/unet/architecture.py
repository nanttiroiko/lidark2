"""U-Net architecture, ported with topology parity from the original script.

Same depth (4 encoders, bottleneck, 4 decoders), same
``Conv2d(use_bias=False) -> BatchNorm -> Activation`` blocks, same
``Add``-skip decoders, same base feature count (16). Only the output head is
task-dependent: 1 channel when ``num_classes == 1`` (original behavior),
``(num_classes + 1)`` channels otherwise.

Torch conventions differ from the Keras original in two documented ways:
inputs are channels-first ``(B, F, H, W)``, and the head emits **logits** (the
loss applies sigmoid/softmax numerically stably; inference applies them
explicitly). Like the original, the window size must be a multiple of 16
(the Add skips require the four pool/upscale levels to meet shape-exactly).

This module is imported only from inside the U-Net backend's lazy
``build``/``load`` methods, so listing backends never imports torch.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class ConvBnAct(nn.Module):
    """Conv2d (no bias; BN learns it) -> BatchNorm -> optional activation."""

    def __init__(
        self,
        in_features: int,
        features: int,
        kernel: int = 1,
        stride: int = 1,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_features, features, kernel, stride=stride, padding=kernel // 2, bias=False
        )
        self.batchnorm = nn.BatchNorm2d(features)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SimpleConvBlock(nn.Module):
    """Two 3x3 Conv-BN-Act passes (the original ``simple_conv_block``)."""

    def __init__(self, in_features: int, features: int, activation: nn.Module | None) -> None:
        super().__init__()
        self.c1 = ConvBnAct(in_features, features, 3, 1, activation)
        self.c2 = ConvBnAct(features, features, 3, 1, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.c2(self.c1(x)))


class PoolEncoderBlock(nn.Module):
    """Encoder block using max pooling (the original ``pool_encoder_block``)."""

    def __init__(self, in_features: int, features: int, activation: nn.Module | None) -> None:
        super().__init__()
        self.block = SimpleConvBlock(in_features, features, activation)
        self.down_pool = nn.MaxPool2d((2, 2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.block(x)
        return skip, self.down_pool(skip)


class DecoderBlockAddSkip(nn.Module):
    """Upscale convolution + Add skip + optional dropout + conv block."""

    def __init__(
        self,
        in_features: int,
        features: int,
        activation: nn.Module | None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # stride 2 with padding/output_padding 1 doubles H, W - the torch
        # equivalent of Keras' ``Conv2DTranspose(strides=2, padding="same")``.
        self.conv_up = nn.ConvTranspose2d(
            in_features, features, (3, 3), stride=2, padding=1, output_padding=1
        )
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else None
        self.block = SimpleConvBlock(features, features, activation)

    def forward(self, x: torch.Tensor, skip_features: torch.Tensor) -> torch.Tensor:
        x = self.conv_up(x)
        x = x + skip_features
        if self.drop is not None:
            x = self.drop(x)
        return cast(torch.Tensor, self.block(x))


class UNet(nn.Module):
    """The original U-Net: 4 encoders, bottleneck, 4 Add-skip decoders."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "relu",
        dropout: float = 0.0,
        base_features: int = 16,
    ) -> None:
        super().__init__()
        act: nn.Module | None = _activation(activation)
        features = base_features
        self.encoder1 = PoolEncoderBlock(in_channels, features, act)
        self.encoder2 = PoolEncoderBlock(features, features * 2, act)
        self.encoder3 = PoolEncoderBlock(features * 2, features * 4, act)
        self.encoder4 = PoolEncoderBlock(features * 4, features * 8, act)
        self.b1 = SimpleConvBlock(features * 8, features * 16, act)
        self.decoder1 = DecoderBlockAddSkip(features * 16, features * 8, act, dropout)
        self.decoder2 = DecoderBlockAddSkip(features * 8, features * 4, act, dropout)
        self.decoder3 = DecoderBlockAddSkip(features * 4, features * 2, act, dropout)
        self.decoder4 = DecoderBlockAddSkip(features * 2, features, act, dropout)
        self.head = nn.Conv2d(features, out_channels, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, p1 = self.encoder1(x)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)
        b1 = self.b1(p4)
        d1 = self.decoder1(b1, s4)
        d2 = self.decoder2(d1, s3)
        d3 = self.decoder3(d2, s2)
        d4 = self.decoder4(d3, s1)
        return cast(torch.Tensor, self.head(d4))


def _activation(name: str) -> nn.Module | None:
    """Named hidden activation module (original: relu)."""
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"unsupported activation {name!r}")


def get_unet_model(
    img_size: tuple[int, int, int],
    num_classes: int,
    activation_function: str = "relu",
    dropout: float = 0.0,
    base_features: int = 16,
) -> UNet:
    """Build the original U-Net.

    Args:
        img_size: (H, W, F) input shape; H, W must be multiples of 16.
        num_classes: Number of **output channels**: 1 for the original
            binary sigmoid head, ``C + 1`` for a multi-class softmax head.
        activation_function: Hidden activation (original: relu).
        dropout: Dropout in the decoder blocks (original: 0).
        base_features: Filters in the first encoder (original: 16).

    Returns:
        A ``UNet`` emitting ``(B, num_classes, H, W)`` logits.
    """
    if len(img_size) != 3:
        raise ValueError(f"img_size must be (H, W, F), got {img_size!r}")
    return UNet(
        in_channels=img_size[2],
        out_channels=num_classes,
        activation=activation_function,
        dropout=dropout,
        base_features=base_features,
    )


__all__ = [
    "ConvBnAct",
    "DecoderBlockAddSkip",
    "PoolEncoderBlock",
    "SimpleConvBlock",
    "UNet",
    "get_unet_model",
]
