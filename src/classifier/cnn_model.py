"""CNN-based accent classifier operating on mel-spectrograms.

A lightweight Conv2d architecture suitable for training on limited data.
Uses AdaptiveAvgPool2d to handle variable-length inputs gracefully.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AccentCNN(nn.Module):
    """3-block CNN for accent classification from mel-spectrograms.

    Architecture:
        3x [Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d(2,2)]
        -> AdaptiveAvgPool2d(1,1) -> Flatten -> Linear(n_classes)

    Input shape: (batch, 1, n_mels, n_frames)
    Output shape: (batch, n_classes) — raw logits, no softmax.

    Args:
        n_classes: Number of accent classes (e.g. 5 for IBGE macro-regions).
        n_mels: Number of mel filterbank channels (must match dataset).
        conv_channels: Number of output channels per conv block.
    """

    def __init__(
        self,
        n_classes: int,
        n_mels: int = 80,
        conv_channels: list[int] | None = None,
    ) -> None:
        super().__init__()

        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")

        if conv_channels is None:
            conv_channels = [32, 64, 128]

        if len(conv_channels) != 3:
            raise ValueError(
                f"conv_channels must have exactly 3 elements, got {len(conv_channels)}"
            )

        self.n_classes = n_classes
        self.n_mels = n_mels

        # Block 1: (1, conv_channels[0])
        self.block1 = nn.Sequential(
            nn.Conv2d(1, conv_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Block 2: (conv_channels[0], conv_channels[1])
        self.block2 = nn.Sequential(
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Block 3: (conv_channels[1], conv_channels[2])
        self.block3 = nn.Sequential(
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Adaptive pooling handles variable spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Linear(conv_channels[-1], n_classes)

        logger.info(
            "AccentCNN: n_classes=%d, n_mels=%d, channels=%s, params=%d",
            n_classes,
            n_mels,
            conv_channels,
            sum(p.numel() for p in self.parameters()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Mel-spectrogram tensor of shape (batch, 1, n_mels, n_frames).

        Returns:
            Logits of shape (batch, n_classes). No softmax applied.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)           # (batch, conv_channels[-1], 1, 1)
        x = x.flatten(start_dim=1)  # (batch, conv_channels[-1])
        x = self.classifier(x)     # (batch, n_classes)
        return x
