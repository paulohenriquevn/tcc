"""Tests for seed utility — determinism verification."""

import random

import numpy as np
import pytest
import torch

from src.utils.seed import set_global_seed


class TestSetGlobalSeed:
    def test_random_module_deterministic(self):
        set_global_seed(42)
        a = [random.random() for _ in range(10)]

        set_global_seed(42)
        b = [random.random() for _ in range(10)]

        assert a == b

    def test_numpy_deterministic(self):
        set_global_seed(42)
        a = np.random.rand(10).tolist()

        set_global_seed(42)
        b = np.random.rand(10).tolist()

        assert a == b

    def test_torch_deterministic(self):
        set_global_seed(42)
        a = torch.rand(10).tolist()

        set_global_seed(42)
        b = torch.rand(10).tolist()

        assert a == b

    def test_returns_generator(self):
        gen = set_global_seed(42)
        assert isinstance(gen, torch.Generator)

    def test_different_seeds_give_different_results(self):
        set_global_seed(42)
        a = random.random()

        set_global_seed(1337)
        b = random.random()

        assert a != b
