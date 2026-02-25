"""Tests for accent classifier model architectures.

Only tests AccentCNN. AccentWav2Vec2 is excluded because it requires
downloading a pre-trained model from HuggingFace at instantiation time.
"""

import pytest
import torch

from src.classifier.cnn_model import AccentCNN


class TestAccentCNN:
    def test_forward_shape_default(self):
        """Default params: input (2,1,80,300) -> output (2,5) for 5 classes."""
        # Arrange
        model = AccentCNN(n_classes=5)
        x = torch.randn(2, 1, 80, 300)

        # Act
        out = model(x)

        # Assert
        assert out.shape == (2, 5)

    def test_forward_shape_different_n_classes(self):
        """Output second dimension matches n_classes=3."""
        # Arrange
        model = AccentCNN(n_classes=3)
        x = torch.randn(2, 1, 80, 300)

        # Act
        out = model(x)

        # Assert
        assert out.shape == (2, 3)

    def test_forward_variable_length(self):
        """AdaptiveAvgPool handles shorter time dimension without error."""
        # Arrange
        model = AccentCNN(n_classes=5)
        x = torch.randn(2, 1, 80, 150)

        # Act
        out = model(x)

        # Assert
        assert out.shape == (2, 5)

    def test_gradient_flow(self):
        """All trainable parameters receive gradients after backward pass."""
        # Arrange
        model = AccentCNN(n_classes=5)
        x = torch.randn(2, 1, 80, 300)

        # Act
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Assert
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"Parameter '{name}' has requires_grad=True but grad is None"
                )

    def test_n_classes_too_small_raises(self):
        """n_classes=1 raises ValueError (need at least 2 for classification)."""
        with pytest.raises(ValueError, match="n_classes must be >= 2"):
            AccentCNN(n_classes=1)

    def test_conv_channels_wrong_length_raises(self):
        """conv_channels with != 3 elements raises ValueError."""
        with pytest.raises(ValueError, match="conv_channels must have exactly 3"):
            AccentCNN(n_classes=5, conv_channels=[32, 64])

    def test_custom_conv_channels(self):
        """Custom conv_channels=[16,32,64] produces correct output shape."""
        # Arrange
        model = AccentCNN(n_classes=5, conv_channels=[16, 32, 64])
        x = torch.randn(4, 1, 80, 300)

        # Act
        out = model(x)

        # Assert
        assert out.shape == (4, 5)
        # Verify final linear layer matches last conv channel
        assert model.classifier.in_features == 64
