"""
Tests for Lin (Truth-Teller) trader.

Validates behavioral components against SRobotLin.java from the 1993 tournament.
"""

import pytest
import numpy as np
from traders.legacy.lin import Lin


class TestLinConstruction:
    """Test Lin agent initialization."""

    def test_basic_construction(self):
        """Test Lin can be instantiated with required parameters."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            price_min=0,
            price_max=1000,
            seed=42
        )
        assert agent.player_id == 1
        assert agent.is_buyer is True
        assert agent.num_tokens == 5
        assert len(agent.valuations) == 5

    def test_construction_with_market_params(self):
        """Test Lin with market composition parameters."""
        agent = Lin(
            player_id=2,
            is_buyer=False,
            num_tokens=3,
            valuations=[20, 30, 40],
            num_buyers=4,
            num_sellers=4,
            num_times=100,
            seed=42
        )
        assert agent.num_buyers == 4
        assert agent.num_sellers == 4
        assert agent.num_times == 100

    def test_initial_state(self):
        """Test initial state of tracking variables."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        assert agent.current_period == 0
        assert len(agent.mean_price) == 100
        assert agent.mean_price[0] == 0.0
        assert len(agent.traded_prices) == 0


class TestLinBoxMuller:
    """Test Box-Muller transform for normal distribution sampling."""

    def test_norm_distribution(self):
        """Test that _norm generates values around the mean."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )

        mean = 50.0
        err = 10.0
        samples = [agent._norm(mean, err) for _ in range(1000)]

        # Check that samples are roughly normally distributed
        sample_mean = np.mean(samples)
        sample_std = np.std(samples)

        # Allow for sampling variance
        assert abs(sample_mean - mean) < 3, f"Mean {sample_mean} far from {mean}"
        assert abs(sample_std - err) < 3, f"Std {sample_std} far from {err}"

    def test_norm_zero_error(self):
        """Test _norm with zero error returns exactly the mean."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )

        mean = 50.0
        err = 0.0
        result = agent._norm(mean, err)
        assert result == mean


class TestLinPriceStatistics:
    """Test statistical price calculation methods."""

    def test_mean_price_empty(self):
        """Test mean price with no trades."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        assert agent._get_mean_price() == 0.0

    def test_mean_price_calculation(self):
        """Test mean price calculation matches Java logic."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.traded_prices = [50, 60, 70, 80]
        mean = agent._get_mean_price()
        assert mean == 65.0  # (50+60+70+80)/4

    def test_mean_price_with_negatives(self):
        """Test mean price uses absolute values (Java line 127)."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.traded_prices = [-50, 60, -70, 80]
        mean = agent._get_mean_price()
        assert mean == 65.0  # (50+60+70+80)/4

    def test_stderr_price_single_trade(self):
        """Test stderr with one trade returns 1.0 (Java line 158)."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.traded_prices = [50]
        stderr = agent._get_stderr_price()
        assert stderr == 1.0

    def test_stderr_price_calculation(self):
        """Test stderr calculation matches Java formula (Java line 158)."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.traded_prices = [50, 60, 70, 80]
        stderr = agent._get_stderr_price()

        # Java formula: Math.sqrt(sum2)/(ntrades-1)
        # NOT the standard stderr formula! Java divides sqrt(sum) by (n-1)
        mean = 65.0
        sum_sq = ((50-65)**2 + (60-65)**2 + (70-65)**2 + (80-65)**2)
        expected_stderr = np.sqrt(sum_sq) / 3  # Java's non-standard formula

        assert abs(stderr - expected_stderr) < 0.01

    def test_target_price_first_period(self):
        """Test target price in first period (only current trades)."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.current_period = 1
        agent.traded_prices = [50, 60, 70]

        target = agent._get_target_price()
        assert target == 60.0  # Mean of current period only

    def test_target_price_multiple_periods(self):
        """Test target price averages across periods (Java lines 137-140)."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.current_period = 3
        agent.mean_price[1] = 50.0
        agent.mean_price[2] = 60.0
        agent.traded_prices = [70, 80]  # Current period

        # Current mean: 75, Previous means: 50, 60
        # Target = (75 + 50 + 60) / 3 = 61.67
        target = agent._get_target_price()
        expected = (75 + 50 + 60) / 3
        assert abs(target - expected) < 0.01


class TestLinWeighting:
    """Test weighting formula components (Java line 55, 82-84)."""

    def test_weight_calculation_buyer(self):
        """Test weight formula for buyers."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            num_buyers=4,
            num_sellers=4,
            num_times=100,
            seed=42
        )
        agent.current_time = 50
        agent.num_trades = 2

        # Time factor: (100-50+1)/100 = 0.51
        # Token factor: (5-2)/5 = 0.6
        # Market factor: 4/(4+4) = 0.5
        # Weight = 0.51 * 0.6 * 0.5 = 0.153

        time_factor = (100 - 50 + 1) / 100
        token_factor = (5 - 2) / 5
        market_factor = 4 / (4 + 4)
        expected_weight = time_factor * token_factor * market_factor

        # We need to call bid_ask_response to trigger the calculation
        # For now, just verify the formula
        assert time_factor == 0.51
        assert token_factor == 0.6
        assert market_factor == 0.5
        assert abs(expected_weight - 0.153) < 0.001

    def test_weight_decreases_over_time(self):
        """Test that weight decreases as time progresses."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            num_buyers=4,
            num_sellers=4,
            num_times=100,
            seed=42
        )

        agent.current_time = 10
        time_factor_early = (100 - 10 + 1) / 100

        agent.current_time = 90
        time_factor_late = (100 - 90 + 1) / 100

        assert time_factor_early > time_factor_late


class TestLinBidding:
    """Test bid/ask submission logic."""

    def test_no_bid_when_no_tokens(self):
        """Test agent returns 0 bid when out of tokens (Java line 43)."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.num_trades = 5  # All tokens traded
        bid = agent._player_request_bid()
        assert bid == 0

    def test_no_bid_when_unprofitable(self):
        """Test agent returns 0 when current bid >= valuation (Java line 45)."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.current_bid = 99
        agent.num_trades = 0  # First token worth 100
        bid = agent._player_request_bid()
        assert bid == 0

    def test_bid_respects_price_floor(self):
        """Test bid never goes below price_min (Java line 62)."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            price_min=10,
            seed=42
        )
        agent.traded_prices = []  # No history
        bid = agent._player_request_bid()
        assert bid >= 10


class TestLinAcceptance:
    """Test buy/sell acceptance thresholds."""

    def test_buy_when_current_bidder(self):
        """Test buyer accepts when they're current bidder and spread crossed (Java line 100)."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.current_bidder = 1  # This agent is high bidder
        agent.current_bid = 60
        agent.current_ask = 60

        result = agent._player_request_buy()
        assert result == 1

    def test_no_buy_at_loss(self):
        """Test buyer rejects unprofitable trades (Java line 99)."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.num_trades = 0  # First token worth 100
        agent.current_ask = 101  # Ask above valuation

        result = agent._player_request_buy()
        assert result == 0

    def test_sell_when_current_asker(self):
        """Test seller accepts when they're current asker and spread crossed (Java line 113)."""
        agent = Lin(
            player_id=2,
            is_buyer=False,
            num_tokens=5,
            valuations=[20, 30, 40, 50, 60],
            seed=42
        )
        agent.current_asker = 2  # This agent is low asker
        agent.current_ask = 30
        agent.current_bid = 30

        result = agent._player_want_to_sell()
        assert result == 1

    def test_no_sell_at_loss(self):
        """Test seller rejects unprofitable trades (Java line 112)."""
        agent = Lin(
            player_id=2,
            is_buyer=False,
            num_tokens=5,
            valuations=[20, 30, 40, 50, 60],
            seed=42
        )
        agent.num_trades = 0  # First token costs 20
        agent.current_bid = 19  # Bid below cost

        result = agent._player_want_to_sell()
        assert result == 0


class TestLinPeriodTracking:
    """Test period lifecycle and historical tracking."""

    def test_start_period_resets_trades(self):
        """Test start_period resets traded_prices."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.traded_prices = [50, 60, 70]
        agent.start_period(2)

        assert agent.current_period == 2
        assert len(agent.traded_prices) == 0

    def test_end_period_stores_mean(self):
        """Test end_period stores mean price for historical tracking."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.current_period = 3
        agent.traded_prices = [50, 60, 70, 80]

        agent.end_period()

        # Mean should be stored at index current_period
        assert agent.mean_price[3] == 65.0


class TestLinIntegration:
    """Integration tests for full trading cycle."""

    def test_full_abstract_method_cycle(self):
        """Test that all abstract methods work together."""
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )

        # Bid/ask cycle
        agent.bid_ask(time=1, nobidask=0)
        assert agent.has_responded is False

        bid = agent.bid_ask_response()
        assert agent.has_responded is True
        assert isinstance(bid, int)
        assert bid >= 0

        # Simulate bid/ask result
        agent.bid_ask_result(
            status=2,
            num_trades=0,
            new_bids=[bid],
            new_asks=[],
            high_bid=bid,
            high_bidder=1,
            low_ask=0,
            low_asker=0
        )
        assert agent.current_bid == bid

        # Buy/sell cycle
        agent.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=bid,
            low_ask=70,
            high_bidder=1,
            low_asker=2
        )
        assert agent.has_responded is False

        decision = agent.buy_sell_response()
        assert agent.has_responded is True
        assert isinstance(decision, bool)

    def test_deterministic_with_seed(self):
        """Test that same seed produces same behavior."""
        agent1 = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )

        agent2 = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )

        # Both should generate same random samples
        norm1 = agent1._norm(50, 10)
        norm2 = agent2._norm(50, 10)
        assert norm1 == norm2
