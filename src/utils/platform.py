"""Platform detection for multi-environment notebook support.

Detects whether the notebook is running on Google Colab, Lightning.ai,
Paperspace Gradient, or a local machine, and provides appropriate paths
and setup functions.

Usage:
    from src.utils.platform import detect_platform, PlatformConfig

    platform = detect_platform()
    print(platform.name)        # "colab", "lightning", "paperspace", or "local"
    print(platform.cache_base)  # Platform-appropriate cache directory
"""

import os
import sys
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PlatformConfig:
    """Platform-specific configuration for notebook execution."""

    name: str                # "colab", "lightning", "paperspace", or "local"
    repo_dir: Path           # Root of the cloned repository
    cache_base: Path         # Persistent cache directory for manifests/features
    has_gpu: bool            # Whether a GPU is available
    needs_clone: bool        # Whether the repo needs to be git-cloned
    needs_drive_mount: bool  # Whether Google Drive needs mounting


def detect_platform(
    cache_override: str | None = None,
) -> PlatformConfig:
    """Detect the current execution platform and return config.

    Detection order:
    1. Lightning.ai — /teamspace/studios/this_studio exists
    2. Paperspace Gradient — PAPERSPACE env var is set
    3. Google Colab — google.colab module is importable
    4. Local — fallback

    Args:
        cache_override: Optional path to override the default cache location.

    Returns:
        PlatformConfig with platform-appropriate paths.
    """
    import torch

    has_gpu = torch.cuda.is_available()

    # Check Lightning.ai first
    lightning_studio = Path("/teamspace/studios/this_studio")
    if lightning_studio.exists():
        repo_dir = lightning_studio / "TCC"
        cache_base = Path(cache_override) if cache_override else (lightning_studio / "cache")
        config = PlatformConfig(
            name="lightning",
            repo_dir=repo_dir,
            cache_base=cache_base,
            has_gpu=has_gpu,
            needs_clone=not (repo_dir / ".git").exists(),
            needs_drive_mount=False,
        )
        logger.info("Platform: Lightning.ai (studio=%s)", lightning_studio)
        return config

    # Check Paperspace Gradient
    # Gradient sets PAPERSPACE env var; /storage/ is persistent across instances
    if os.environ.get("PAPERSPACE"):
        repo_dir = Path("/notebooks/TCC")
        storage_dir = Path("/storage")
        cache_base = Path(cache_override) if cache_override else (storage_dir / "tcc-cache")
        config = PlatformConfig(
            name="paperspace",
            repo_dir=repo_dir,
            cache_base=cache_base,
            has_gpu=has_gpu,
            needs_clone=not (repo_dir / ".git").exists(),
            needs_drive_mount=False,
        )
        logger.info("Platform: Paperspace Gradient (gpu=%s)", has_gpu)
        return config

    # Check Google Colab
    try:
        import google.colab  # noqa: F401
        repo_dir = Path("/content/TCC")
        cache_base = Path(cache_override) if cache_override else Path("/content/drive/MyDrive/tcc-cache")
        config = PlatformConfig(
            name="colab",
            repo_dir=repo_dir,
            cache_base=cache_base,
            has_gpu=has_gpu,
            needs_clone=not (repo_dir / ".git").exists(),
            needs_drive_mount=True,
        )
        logger.info("Platform: Google Colab")
        return config
    except ImportError:
        pass

    # Fallback: local machine
    # Try to find repo root from current working directory
    cwd = Path.cwd()
    if (cwd / ".git").exists() and (cwd / "src").exists():
        repo_dir = cwd
    elif (cwd.parent / ".git").exists() and (cwd.parent / "src").exists():
        repo_dir = cwd.parent
    else:
        repo_dir = cwd

    cache_base = Path(cache_override) if cache_override else (repo_dir / "cache")
    config = PlatformConfig(
        name="local",
        repo_dir=repo_dir,
        cache_base=cache_base,
        has_gpu=has_gpu,
        needs_clone=False,
        needs_drive_mount=False,
    )
    logger.info("Platform: local (%s)", repo_dir)
    return config


def setup_environment(platform: PlatformConfig) -> None:
    """Perform platform-specific environment setup.

    Handles: repo cloning, Drive mounting, sys.path, working directory.

    Args:
        platform: PlatformConfig from detect_platform().
    """
    # Clone repo if needed
    if platform.needs_clone:
        import subprocess
        logger.info("Cloning repository to %s", platform.repo_dir)
        if platform.repo_dir.exists():
            import shutil
            shutil.rmtree(platform.repo_dir)
        subprocess.run(
            ["git", "clone", "https://github.com/paulohenriquevn/tcc.git",
             str(platform.repo_dir)],
            check=True,
        )

    # Mount Google Drive (Colab only)
    if platform.needs_drive_mount:
        from google.colab import drive
        drive.mount("/content/drive")

    # Set working directory
    os.chdir(platform.repo_dir)

    # Ensure repo on sys.path
    repo_str = str(platform.repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    # Create cache directory
    platform.cache_base.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Environment ready: platform=%s, repo=%s, cache=%s, gpu=%s",
        platform.name, platform.repo_dir, platform.cache_base, platform.has_gpu,
    )
