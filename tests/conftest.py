# tests/conftest.py
import pytest
import sys
import os
import random
import numpy as np

# Add src_code directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src_code'))

from traders.zip import ZIPBuyer, ZIPSeller
from traders.zic import ZICBuyer, ZICSeller
from traders.base import BaseTrader
from auction import Auction


@pytest.fixture(scope="function")
def reset_random():
    """Reset random seeds for reproducible tests."""
    random.seed(42)
    np.random.seed(42)
    yield
    # Reset after test
    random.seed()
    np.random.seed()


@pytest.fixture
def sample_market_params():
    """Standard market parameters for testing."""
    return {
        'min_price': 1,
        'max_price': 1000,
        'num_tokens': 4,
        'num_periods': 3,
        'num_steps': 25,
        'gametype': 6453
    }


@pytest.fixture
def zip_buyer(sample_market_params, reset_random):
    """Create a ZIP buyer for testing."""
    buyer = ZIPBuyer(
        name="TestZIPBuyer",
        is_buyer=True,
        private_values=[100, 90, 80, 70],
        zip_beta=0.3,
        zip_gamma=0.05,
        zip_buyer_margin_low=-0.2,
        zip_buyer_margin_high=-0.1
    )
    buyer.update_game_params(sample_market_params)
    buyer.tokens_left = buyer.max_tokens  # Ensure tokens_left is set properly
    return buyer


@pytest.fixture
def zip_seller(sample_market_params, reset_random):
    """Create a ZIP seller for testing."""
    seller = ZIPSeller(
        name="TestZIPSeller", 
        is_buyer=False,
        private_values=[50, 60, 70, 80],
        zip_beta=0.3,
        zip_gamma=0.05,
        zip_seller_margin_low=0.1,
        zip_seller_margin_high=0.2
    )
    seller.update_game_params(sample_market_params)
    return seller


@pytest.fixture
def zic_buyer(sample_market_params):
    """Create a ZIC buyer for testing."""
    buyer = ZICBuyer(
        name="TestZICBuyer",
        is_buyer=True, 
        private_values=[100, 90, 80, 70]
    )
    buyer.update_game_params(sample_market_params)
    return buyer


@pytest.fixture
def zic_seller(sample_market_params):
    """Create a ZIC seller for testing.""" 
    seller = ZICSeller(
        name="TestZICSeller",
        is_buyer=False,
        private_values=[50, 60, 70, 80]
    )
    seller.update_game_params(sample_market_params)
    return seller


@pytest.fixture
def simple_auction_config():
    """Simple auction configuration for integration tests."""
    return {
        "experiment_name": "test_auction",
        "num_rounds": 1,
        "num_periods": 1, 
        "num_steps": 5,
        "num_buyers": 1,
        "num_sellers": 1,
        "num_tokens": 1,
        "min_price": 1,
        "max_price": 200,
        "gametype": 0,
        "buyers": [{"type": "zic"}],
        "sellers": [{"type": "zic"}],
        "rng_seed_auction": 42,
        "rng_seed_values": 123,
    }


@pytest.fixture
def mock_market_history():
    """Mock market history for trader testing."""
    return {
        'last_trade_info_for_period': None,
        'lasttime': -1,
        'all_bids_this_step': [],
        'all_asks_this_step': []
    }


@pytest.fixture
def mock_step_outcome():
    """Mock step outcome for observe_reward testing."""
    return {
        'last_trade_info': {
            'price': 75,
            'type': 'buy_accepts_ask',
            'buyer': None,
            'seller': None,
            'step': 1
        }
    }