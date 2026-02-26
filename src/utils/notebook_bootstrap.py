"""Notebook bootstrap — stdlib-only, no third-party imports.

This module bootstraps the execution environment for Jupyter notebooks.
It handles: platform detection, repo cloning, dependency installation,
and NumPy ABI compatibility checks.

IMPORTANT: This module must NOT import any third-party packages because
it runs BEFORE pip install. Only Python stdlib is allowed.

Usage (first cell of every notebook):
    from src.utils.notebook_bootstrap import bootstrap
    bootstrap()
"""

import os
import subprocess
import sys
from pathlib import Path


def _detect_platform_raw() -> tuple[str, str]:
    """Detect platform using only stdlib (no torch, no google.colab import).

    Returns:
        Tuple of (platform_name, repo_dir).
    """
    # Lightning.ai
    lightning_studio = "/teamspace/studios/this_studio"
    if os.path.exists(lightning_studio):
        return "lightning", os.path.join(lightning_studio, "TCC")

    # Paperspace Gradient
    if os.environ.get("PAPERSPACE"):
        return "paperspace", "/notebooks/TCC"

    # Google Colab — check sys.modules (avoids importing google.colab)
    if "google.colab" in sys.modules or os.path.exists("/content"):
        return "colab", "/content/TCC"

    # Local
    return "local", os.getcwd()


def _clone_if_needed(repo_dir: str) -> None:
    """Clone the TCC repo if not already present."""
    if not os.path.exists(os.path.join(repo_dir, ".git")):
        subprocess.run(["rm", "-rf", repo_dir], check=False)
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/paulohenriquevn/tcc.git",
                repo_dir,
            ],
            check=True,
        )


def _install_deps(repo_dir: str) -> None:
    """Install requirements.txt quietly."""
    req_path = os.path.join(repo_dir, "requirements.txt")
    if os.path.exists(req_path):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_path, "-q"],
            check=True,
        )


def _check_numpy_abi() -> None:
    """Check for NumPy ABI mismatch (common on Colab).

    Colab pre-loads numpy 2.x in memory, but requirements.txt may pin
    an older version. After pip downgrades, stale C-extensions cause
    binary incompatibility. Fix: restart runtime ONCE.
    """
    installed_np = subprocess.check_output(
        [sys.executable, "-c", "import numpy; print(numpy.__version__)"],
        text=True,
    ).strip()

    try:
        import numpy as _np

        loaded_np = _np.__version__
    except Exception:
        loaded_np = None

    if loaded_np != installed_np:
        print(f"\nNumPy ABI mismatch: loaded={loaded_np}, installed={installed_np}")
        print(
            "Restarting runtime... After restart, re-run this cell (no second restart)."
        )
        os.kill(os.getpid(), 9)


def bootstrap() -> str:
    """Bootstrap notebook environment. Returns platform name.

    Steps:
    1. Detect platform (Lightning.ai, Paperspace, Colab, local)
    2. Clone repo if needed
    3. Set working directory and sys.path
    4. Install pip dependencies
    5. Check NumPy ABI compatibility

    Returns:
        Platform name string ("lightning", "paperspace", "colab", "local").
    """
    platform_name, repo_dir = _detect_platform_raw()

    _clone_if_needed(repo_dir)

    os.chdir(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    _install_deps(repo_dir)
    _check_numpy_abi()

    print(f"\nPlatform: {platform_name}")
    print(f"Repo: {repo_dir}")
    print("Environment OK")

    return platform_name
