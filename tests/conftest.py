# tests/conftest.py
"""Minimal shared fixtures for test suite."""

import numpy as np
import pytest


@pytest.fixture
def seed():
    """Fixed seed for reproducibility."""
    return 42


@pytest.fixture
def rng(seed):
    """Numpy random generator with fixed seed."""
    return np.random.default_rng(seed)
