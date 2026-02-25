"""Pipeline cache for Google Drive persistence across Colab sessions.

Stores manifest and extracted features under a directory keyed by the
SHA-256 hash of the dataset filter config — any config change produces
a new cache directory, preventing stale data.
"""

import hashlib
import logging
from pathlib import Path

import numpy as np
import yaml

from src.data.manifest import ManifestEntry, compute_file_hash, read_manifest

logger = logging.getLogger(__name__)


def compute_filter_hash(dataset_config: dict) -> str:
    """Compute a 12-char hex hash of the dataset filters section.

    Uses canonical YAML dump (sorted keys) so key ordering doesn't
    affect the hash.

    Args:
        dataset_config: The 'dataset' section of the config dict.

    Returns:
        12-character hex string (first 12 chars of SHA-256).
    """
    filters = dataset_config.get("filters", {})
    canonical = yaml.dump(filters, default_flow_style=False, sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:12]


class PipelineCache:
    """Cache layer for manifest and features on Google Drive.

    Directory layout under drive_base:
        <filter_hash>/
            manifest.jsonl
            manifest.jsonl.sha256
            audio/              (WAV files)
            features/
                acoustic.npz
                ecapa.npz
                wavlm_layer_0.npz
                ...
                backbone_layer_0.npz
                ...

    Args:
        config: Full experiment config dict.
        drive_base: Base directory for cache (e.g. /content/drive/MyDrive/tcc-cache).
    """

    def __init__(self, config: dict, drive_base: str) -> None:
        dataset_config = config.get("dataset", {})
        self._filter_hash = compute_filter_hash(dataset_config)
        self._base = Path(drive_base) / self._filter_hash
        self._manifest_path = self._base / "manifest.jsonl"
        self._features_dir = self._base / "features"
        self._audio_dir = self._base / "audio"

    @property
    def filter_hash(self) -> str:
        return self._filter_hash

    # ── Manifest ──

    def has_manifest(self) -> bool:
        """Check if cached manifest exists and passes SHA-256 validation."""
        if not self._manifest_path.exists():
            return False

        sha_path = self._manifest_path.with_suffix(
            self._manifest_path.suffix + ".sha256"
        )
        if not sha_path.exists():
            logger.warning("Manifest exists but SHA-256 sidecar missing — rebuilding")
            return False

        # Parse expected hash from sidecar (format: "<hash>  <filename>\n")
        expected_hash = sha_path.read_text().strip().split()[0]
        actual_hash = compute_file_hash(self._manifest_path)

        if actual_hash != expected_hash:
            logger.warning(
                f"Manifest SHA-256 mismatch (expected {expected_hash[:12]}..., "
                f"got {actual_hash[:12]}...) — rebuilding"
            )
            return False

        return True

    def load_manifest(self) -> list[ManifestEntry]:
        """Load manifest from cache.

        Raises:
            FileNotFoundError: If manifest doesn't exist.
            ValueError: If manifest is corrupt.
        """
        return read_manifest(self._manifest_path)

    def get_manifest_path(self) -> Path:
        """Return path where manifest should be written."""
        self._base.mkdir(parents=True, exist_ok=True)
        return self._manifest_path

    # ── Audio ──

    def get_audio_dir(self) -> Path:
        """Return audio directory, creating if needed."""
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        return self._audio_dir

    # ── Features ──

    def has_features(self, feature_type: str) -> bool:
        """Check if cached features exist for the given type.

        Args:
            feature_type: e.g. "acoustic", "ecapa", "wavlm_layer_0",
                          "backbone_layer_12".
        """
        npz_path = self._features_dir / f"{feature_type}.npz"
        return npz_path.exists()

    def load_features(self, feature_type: str) -> dict[str, np.ndarray]:
        """Load features from cached NPZ.

        Args:
            feature_type: Feature type identifier.

        Returns:
            Dict mapping utt_id (or "{layer}_{utt_id}") to numpy arrays.

        Raises:
            FileNotFoundError: If NPZ doesn't exist.
        """
        npz_path = self._features_dir / f"{feature_type}.npz"
        data = np.load(str(npz_path))
        return dict(data)

    def save_features(
        self, feature_type: str, features: dict[str, np.ndarray]
    ) -> None:
        """Save features to NPZ (compressed).

        Args:
            feature_type: Feature type identifier.
            features: Dict mapping keys to numpy arrays.
        """
        self._features_dir.mkdir(parents=True, exist_ok=True)
        npz_path = self._features_dir / f"{feature_type}.npz"
        np.savez_compressed(str(npz_path), **features)
        logger.info(
            f"Cached {len(features)} vectors to {npz_path} "
            f"({npz_path.stat().st_size / 1e6:.1f} MB)"
        )

    # ── Status ──

    def report(self) -> str:
        """Return a human-readable cache status summary."""
        lines = [
            f"PipelineCache (filter_hash={self._filter_hash})",
            f"  Base: {self._base}",
            f"  Manifest: {'CACHED' if self.has_manifest() else 'MISSING'}",
        ]

        if self._features_dir.exists():
            npz_files = sorted(self._features_dir.glob("*.npz"))
            if npz_files:
                lines.append(f"  Features cached ({len(npz_files)}):")
                for npz in npz_files:
                    size_mb = npz.stat().st_size / 1e6
                    lines.append(f"    {npz.stem}: {size_mb:.1f} MB")
            else:
                lines.append("  Features: NONE")
        else:
            lines.append("  Features: NONE")

        if self._audio_dir.exists():
            wav_count = len(list(self._audio_dir.glob("*.wav")))
            lines.append(f"  Audio: {wav_count} WAV files")
        else:
            lines.append("  Audio: NONE")

        return "\n".join(lines)
