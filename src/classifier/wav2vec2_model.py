"""Wav2Vec2-based accent classifier operating on raw waveforms.

Uses a pre-trained Wav2Vec2 model as feature extractor with a linear
classification head. The CNN feature encoder can be frozen to reduce
trainable parameters.
"""

import logging

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

logger = logging.getLogger(__name__)


class AccentWav2Vec2(nn.Module):
    """Wav2Vec2 backbone + linear head for accent classification.

    The pre-trained Wav2Vec2 model extracts contextualized representations
    from raw audio. Mean pooling over the time dimension produces a
    fixed-size vector, which is projected to class logits.

    Input: raw waveform tensor of shape (batch, n_samples)
    Output: logits of shape (batch, n_classes) — no softmax.

    Args:
        n_classes: Number of accent classes.
        model_name: HuggingFace model identifier for Wav2Vec2.
        freeze_feature_extractor: If True, freeze the CNN feature encoder
            (first layers) to save memory and training time.
    """

    def __init__(
        self,
        n_classes: int,
        model_name: str = "facebook/wav2vec2-base",
        freeze_feature_extractor: bool = True,
    ) -> None:
        super().__init__()

        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")

        self.n_classes = n_classes
        self.model_name = model_name

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_feature_extractor:
            self._freeze_feature_extractor()

        hidden_size = self.get_hidden_size()
        self.classifier = nn.Linear(hidden_size, n_classes)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "AccentWav2Vec2: model=%s, n_classes=%d, hidden=%d, "
            "trainable=%d/%d params, freeze_feat=%s",
            model_name,
            n_classes,
            hidden_size,
            trainable,
            total,
            freeze_feature_extractor,
        )

    def _freeze_feature_extractor(self) -> None:
        """Freeze the CNN feature encoder parameters."""
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False

    def get_hidden_size(self) -> int:
        """Return the hidden size of the Wav2Vec2 model.

        Returns:
            Integer dimension of the last hidden state.
        """
        return self.wav2vec2.config.hidden_size

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_values: Raw waveform tensor of shape (batch, n_samples).
            attention_mask: Optional mask of shape (batch, n_samples).
                1 for real samples, 0 for padding.

        Returns:
            Logits of shape (batch, n_classes). No softmax applied.
        """
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        # last_hidden_state: (batch, seq_len, hidden_size)
        hidden_states = outputs.last_hidden_state

        # Mean pooling over time dimension, respecting attention mask
        if attention_mask is not None:
            # Wav2Vec2 downsamples time, so we need the output lengths
            # Use the hidden_states length directly
            mask = attention_mask[:, :hidden_states.shape[1]].unsqueeze(-1).float()
            sum_hidden = (hidden_states * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1.0)
            pooled = sum_hidden / count
        else:
            pooled = hidden_states.mean(dim=1)

        logits = self.classifier(pooled)
        return logits
