"""1D ResNet for single-lead ECG classification.

Architecture adapted from:
    Hannun AY, Rajpurkar P, et al. Cardiologist-level arrhythmia detection and
    classification in ambulatory electrocardiograms using a deep neural network.
    Nat Med 25, 65-69 (2019). https://doi.org/10.1038/s41591-018-0268-3

Notes:
    - Uses 1-D convolutions over time; depth ~18 layers.
    - GroupNorm rather than BatchNorm: works better with the small per-batch
      effective sample size of long 1-D signals on a single GPU.
    - The final ``feature_map`` attribute exposes the last conv activations
      so that GradCAM-1D can hook into them without modifying the forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv1d_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 16,
    stride: int = 1,
    groups_norm: int = 8,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        ),
        nn.GroupNorm(num_groups=groups_norm, num_channels=out_channels),
        nn.ReLU(inplace=True),
    )


class ResidualBlock1D(nn.Module):
    """Pre-activation 1-D residual block with optional downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 16,
        stride: int = 1,
        dropout: float = 0.2,
        groups_norm: int = 8,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(groups_norm, out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.gn2 = nn.GroupNorm(groups_norm, out_channels)
        self.dropout = nn.Dropout(dropout)

        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(groups_norm, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.gn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        out = self.gn2(self.conv2(out))
        out = out + residual
        return F.relu(out, inplace=True)


class ResNet1D(nn.Module):
    """1-D ResNet for ECG classification.

    Args:
        num_classes: number of output classes (4 for PhysioNet 2017).
        base_channels: number of channels in the first stem; doubled at each stage.
        block_counts: number of residual blocks per stage.
        kernel_size: temporal kernel size (16 ≈ 50 ms @ 300 Hz, ~ one QRS width).
        dropout: dropout probability inside residual blocks.
    """

    def __init__(
        self,
        num_classes: int = 4,
        base_channels: int = 64,
        block_counts: tuple[int, ...] = (2, 2, 2, 2),
        kernel_size: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.stem = _conv1d_block(1, base_channels, kernel_size=kernel_size, stride=1)

        stages: list[nn.Module] = []
        in_c = base_channels
        for stage_idx, count in enumerate(block_counts):
            out_c = base_channels * (2**stage_idx)
            for block_idx in range(count):
                stride = 2 if (block_idx == 0 and stage_idx > 0) else 2 if block_idx == 0 else 1
                stages.append(
                    ResidualBlock1D(
                        in_c,
                        out_c,
                        kernel_size=kernel_size,
                        stride=stride,
                        dropout=dropout,
                    )
                )
                in_c = out_c
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_c, num_classes)

        self._feature_map: torch.Tensor | None = None

    @property
    def feature_map(self) -> torch.Tensor | None:
        """Last conv activations from the most recent forward pass (for GradCAM)."""
        return self._feature_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        self._feature_map = x  # cache for explainability
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def resnet18_1d(num_classes: int = 4, **kwargs: object) -> ResNet1D:
    """Standard 18-layer 1-D ResNet for ECG."""
    return ResNet1D(num_classes=num_classes, block_counts=(2, 2, 2, 2), **kwargs)
