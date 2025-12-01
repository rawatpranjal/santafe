# tests/behavior/conftest.py
"""Shared fixtures for behavioral tests."""

import pytest

from traders.legacy.bgan import BGAN
from traders.legacy.ledyard import Ledyard
from traders.legacy.ringuette import Ringuette
from traders.legacy.skeleton import Skeleton
from traders.legacy.staecker import Staecker
from traders.legacy.zic import ZIC


@pytest.fixture
def ringuette_buyer():
    """Standard Ringuette buyer for testing."""
    return Ringuette(
        player_id=1,
        is_buyer=True,
        num_tokens=4,
        valuations=[100, 90, 80, 70],
        price_min=0,
        price_max=200,
        num_times=100,
        seed=42,
    )


@pytest.fixture
def ringuette_seller():
    """Standard Ringuette seller for testing."""
    return Ringuette(
        player_id=1,
        is_buyer=False,
        num_tokens=4,
        valuations=[30, 40, 50, 60],
        price_min=0,
        price_max=200,
        num_times=100,
        seed=42,
    )


@pytest.fixture
def zic_buyer():
    """ZIC buyer opponent."""
    return ZIC(
        player_id=2,
        is_buyer=True,
        num_tokens=4,
        valuations=[95, 85, 75, 65],
        price_min=0,
        price_max=200,
        seed=43,
    )


@pytest.fixture
def zic_seller():
    """ZIC seller opponent."""
    return ZIC(
        player_id=3,
        is_buyer=False,
        num_tokens=4,
        valuations=[35, 45, 55, 65],
        price_min=0,
        price_max=200,
        seed=44,
    )


@pytest.fixture
def skeleton_buyer():
    """Skeleton buyer opponent."""
    return Skeleton(
        player_id=4,
        is_buyer=True,
        num_tokens=4,
        valuations=[92, 82, 72, 62],
        price_min=0,
        price_max=200,
        num_times=100,
        seed=45,
    )


@pytest.fixture
def skeleton_seller():
    """Skeleton seller opponent."""
    return Skeleton(
        player_id=5,
        is_buyer=False,
        num_tokens=4,
        valuations=[38, 48, 58, 68],
        price_min=0,
        price_max=200,
        num_times=100,
        seed=46,
    )


@pytest.fixture
def bgan_buyer():
    """BGAN buyer for testing."""
    return BGAN(
        player_id=6,
        is_buyer=True,
        num_tokens=4,
        valuations=[100, 90, 80, 70],
        price_min=0,
        price_max=200,
        num_times=100,
        seed=47,
    )


@pytest.fixture
def bgan_seller():
    """BGAN seller for testing."""
    return BGAN(
        player_id=7,
        is_buyer=False,
        num_tokens=4,
        valuations=[30, 40, 50, 60],
        price_min=0,
        price_max=200,
        num_times=100,
        seed=48,
    )


@pytest.fixture
def staecker_buyer():
    """Staecker buyer for testing."""
    return Staecker(
        player_id=8,
        is_buyer=True,
        num_tokens=4,
        valuations=[100, 90, 80, 70],
        price_min=0,
        price_max=200,
        num_times=100,
        seed=49,
    )


@pytest.fixture
def staecker_seller():
    """Staecker seller for testing."""
    return Staecker(
        player_id=9,
        is_buyer=False,
        num_tokens=4,
        valuations=[30, 40, 50, 60],
        price_min=0,
        price_max=200,
        num_times=100,
        seed=50,
    )


@pytest.fixture
def ledyard_buyer():
    """Ledyard buyer for testing."""
    return Ledyard(
        player_id=10,
        is_buyer=True,
        num_tokens=4,
        valuations=[100, 90, 80, 70],
        price_min=0,
        price_max=200,
        num_times=100,
        num_buyers=4,
        num_sellers=4,
        seed=51,
    )


@pytest.fixture
def ledyard_seller():
    """Ledyard seller for testing."""
    return Ledyard(
        player_id=11,
        is_buyer=False,
        num_tokens=4,
        valuations=[30, 40, 50, 60],
        price_min=0,
        price_max=200,
        num_times=100,
        num_buyers=4,
        num_sellers=4,
        seed=52,
    )
