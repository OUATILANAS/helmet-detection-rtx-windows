"""Small depthwise-separable CNN shared by training and inference."""

from __future__ import annotations


def build_tiny_helmet_net():
    """Build a compact two-class CNN without a torchvision dependency."""
    import torch.nn as nn

    def block(in_channels: int, out_channels: int, stride: int = 1):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, 3, stride=stride, padding=1,
                groups=in_channels, bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    return nn.Sequential(
        nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        block(16, 24, 2),
        block(24, 32, 2),
        block(32, 48, 2),
        block(48, 64, 2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 2),
    )
