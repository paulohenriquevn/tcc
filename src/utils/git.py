"""Git utilities for experiment provenance tracking.

Provides a safe way to get the current commit hash for logging
in experiment reports, without cluttering notebooks with subprocess calls.
"""

import subprocess


def get_commit_hash() -> str:
    """Get the current git HEAD commit hash.

    Returns:
        Full SHA-1 hex string, or 'unknown' if not in a git repo.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip()
    except Exception:
        return "unknown"
