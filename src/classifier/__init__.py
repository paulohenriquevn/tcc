"""Accent classifier for external evaluation of generated audio (Stages 2-3)."""

from src.classifier.cnn_model import AccentCNN
from src.classifier.wav2vec2_model import AccentWav2Vec2
from src.classifier.trainer import train_classifier, evaluate_classifier, TrainingConfig, TrainingResult
from src.classifier.inference import load_classifier, classify_audio, classify_batch
