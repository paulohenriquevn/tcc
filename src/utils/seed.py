"""Deterministic seed setup for reproducible experiments.

Sets seeds for: random, numpy, torch (CPU + CUDA).
Configures cuDNN for deterministic behavior.

Usage:
    from src.utils.seed import set_global_seed, seed_worker

    set_global_seed(42)
    loader = DataLoader(..., worker_init_fn=seed_worker, generator=generator)
"""

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> torch.Generator:
    """Set deterministic seeds for all RNGs.

    Args:
        seed: Integer seed value.

    Returns:
        torch.Generator configured with the seed (for DataLoader).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator


def seed_worker(worker_id: int) -> None:
    """Seed function for DataLoader workers.

    Ensures each worker gets a deterministic but unique seed
    derived from the global state.

    Args:
        worker_id: Worker index assigned by DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
