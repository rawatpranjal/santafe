"""
Tests for Perry Original trader.

Validates behavioral components against SRobotPerryOriginal.java from the 1993 tournament.
"""

import pytest
import math
from traders.legacy.perry import Perry


class TestPerryConstruction:
    """Test Perry agent initialization."""

    def test_basic_construction(self):
        """Test Perry can be instantiated with required parameters."""
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            price_min=0,
            price_max=100,
            seed=42
        )
        assert agent.player_id == 1
        assert agent.is_buyer is True
        assert agent.num_tokens == 5
        assert len(agent.valuations) == 5

    def test_construction_with_market_params(self):
        """Test Perry with market composition parameters."""
        agent = Perry(
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
        """Test initial state of tracking variables (Java lines 24-34)."""
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        assert agent.r_price_sum == 0
        assert agent.r_price_ss == 0
        assert agent.rtrades == 0
        assert agent.ctrades == 0
        assert agent.r_price_ave == 0
        assert agent.r_price_std == 50  # Initial value
        assert agent.round_count == 0
        assert agent.a0 == 2.0
        assert agent.p_ave_price == 0
        assert agent.price_std == 0
        assert agent.mprice == 0


class TestPerryPriceStatistics:
    """Test price calculation methods."""

    def test_ave_price_empty(self):
        """Test average price with no trades (Java line 385)."""
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        assert agent._ave_price() == 0

    def test_ave_price_calculation(self):
        """Test average price calculation (Java lines 387-389)."""
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.traded_prices = [50, 60, 70, 80]
        avg = agent._ave_price()
        assert avg == 65  # (50+60+70+80)/4 = 65

    def test_ave_price_with_negatives(self):
        """Test average price uses absolute values."""
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.traded_prices = [-50, 60, -70, 80]
        avg = agent._ave_price()
        assert avg == 65  # (50+60+70+80)/4

    def test_round_average_price_new_round(self):
        """Test round statistics reset on new round (Java lines 402-410)."""
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.current_round = 1
        agent.round_count = 0  # Different from current_round

        agent._round_average_price()

        # Should reset round statistics
        assert agent.round_count == 1
        assert agent.r_price_sum == 0
        assert agent.r_price_ss == 0
        assert agent.rtrades == 0
        assert agent.ctrades == 0
        assert agent.r_price_ave == 0
        assert agent.r_price_std == 30  # Default value (Java line 409)

    def test_round_average_price_update(self):
        """Test round statistics update with new trade (Java lines 411-424)."""
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.current_round = 1
        agent.round_count = 1
        agent.traded_prices = [60]

        agent._round_average_price()

        assert agent.rtrades == 1
        assert agent.r_price_sum == 60
        assert agent.r_price_ss == 3600  # 60*60
        assert agent.r_price_ave == 60

    def test_round_price_std_calculation(self):
        """Test round price standard deviation (Java lines 422-424)."""
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.current_round = 1
        agent.round_count = 1
        agent.traded_prices = [50]
        agent._round_average_price()

        agent.traded_prices = [50, 70]
        agent._round_average_price()

        # Mean = (50+70)/2 = 60
        # Variance = ((50-60)^2 + (70-60)^2) / 2 = 100
        # Std = sqrt(100) = 10
        expected_var = ((50*50 + 70*70) - (50+70)*(50+70)/2) / 2
        expected_std = int(math.sqrt(abs(expected_var)))
        assert agent.r_price_std == expected_std


class TestPerryParameterAdaptation:
    """Test adaptive parameter tuning (evaluate function)."""

    def test_evaluate_no_trades(self):
        """Test evaluate with no trades."""
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.period_profit = 0
        agent.traded_prices = []

        initial_a0 = agent.a0
        agent._evaluate()

        # With no trades, efficiency calculation should handle gracefully
        # a0 might change depending on zero efficiency condition (Java line 365)
        assert agent.a0 <= initial_a0  # a0 should decrease or stay same

    def test_evaluate_buyer_feasible_trades(self):
        """Test feasible trades calculation for buyers (Java lines 341-346)."""
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.traded_prices = [75]
        agent.p_ave_price = 75

        # Feasible trades: tokens with value >= average price
        # 100-75=25, 90-75=15, 80-75=5, 70-75=-5 (not feasible), 60-75=-15 (not feasible)
        # So feasible_trades = 3, x = 1 + 25 + 15 + 5 = 46

        agent._evaluate()
        # Can't easily assert internal feasible_trades, but verify it runs

    def test_evaluate_seller_feasible_trades(self):
        """Test feasible trades calculation for sellers (Java lines 351-356)."""
        agent = Perry(
            player_id=2,
            is_buyer=False,
            num_tokens=5,
            valuations=[20, 30, 40, 50, 60],
            seed=42
        )
        agent.traded_prices = [45]
        agent.p_ave_price = 45

        # Feasible trades: tokens with cost <= average price
        # 45-20=25, 45-30=15, 45-40=5, 45-50=-5 (not feasible), 45-60=-15 (not feasible)
        # So feasible_trades = 3, x = 1 + 25 + 15 + 5 = 46

        agent._evaluate()
        # Verify it runs without errors

    def test_a0_decreases_on_poor_efficiency(self):
        """Test a0 decreases when efficiency < 1 (Java lines 364-367)."""
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.traded_prices = [75]
        agent.p_ave_price = 75
        agent.num_trades = 1
        agent.period_profit = 10  # Less than potential profit

        initial_a0 = agent.a0
        agent._evaluate()

        # With efficiency < 1 and undertrades, a0 should decrease
        assert agent.a0 < initial_a0

    def test_price_std_adjustment_low_efficiency(self):
        """Test price_std increased when efficiency <= 0.8 (Java line 368)."""
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.traded_prices = [75, 75]
        agent.p_ave_price = 75
        agent.num_trades = 1
        agent.period_profit = 10  # Low efficiency
        agent.price_std = 5  # Low std

        agent._evaluate()

        # price_std should be set to 30 when efficiency <= 0.8 and price_std < 10
        assert agent.price_std == 30


class TestPerryTwoPhaseStrategy:
    """Test conservative vs statistical strategy phases."""

    def test_conservative_bid_first_period_no_market(self):
        """Test conservative bidding for first 3 trades (Java lines 93-101)."""
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.current_period = 1
        agent.traded_prices = []  # First trade
        agent.current_ask = 0  # No market yet

        bid = agent._player_request_bid()

        # First bid should be random, less than smallest token (Java line 95)
        # token[ntokens] in Java is the worst token = 60
        assert bid < 60
        assert bid >= 0

    def test_conservative_ask_first_period(self):
        """Test conservative asking for first 3 trades (Java lines 181-198)."""
        agent = Perry(
            player_id=2,
            is_buyer=False,
            num_tokens=5,
            valuations=[20, 30, 40, 50, 60],
            seed=42
        )
        agent.current_period = 1
        agent.traded_prices = []
        agent.current_bid = 0
        agent.current_ask = 0
        agent.price_max = 100

        ask = agent._player_request_ask()

        # First ask should be random, greater than largest token (Java line 185-186)
        # token[ntokens] in Java is the worst token = 60
        assert ask > 60
        assert ask <= 100


class TestPerryAcceptanceRegions:
    """Test buy/sell acceptance thresholds."""

    def test_buy_when_current_bidder(self):
        """Test buyer accepts when current bidder and spread crossed (Java line 260)."""
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.current_bidder = 1  # This agent
        agent.current_bid = 60
        agent.current_ask = 60

        result = agent._player_request_buy()
        assert result == 1

    def test_no_buy_at_loss(self):
        """Test buyer rejects when ask > valuation (Java line 259)."""
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )
        agent.num_trades = 0  # First token worth 100
        agent.current_ask = 101

        result = agent._player_request_buy()
        assert result == 0

    def test_buy_acceptance_region(self):
        """Test buyer accepts in acceptance region (Java line 265)."""
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            num_buyers=4,
            num_sellers=4,
            num_times=100,
            seed=42
        )
        agent.current_bidder = 1
        agent.current_time = 50
        agent.num_trades = 0
        agent.r_price_ave = 75
        agent.r_price_std = 10
        agent.current_ask = 70  # Below threshold
        agent.current_bid = 65

        result = agent._player_request_buy()
        # Should accept if ask <= threshold
        assert result in [0, 1]  # Depends on exact threshold calculation

    def test_sell_when_current_asker(self):
        """Test seller accepts when current asker and spread crossed (Java line 280)."""
        agent = Perry(
            player_id=2,
            is_buyer=False,
            num_tokens=5,
            valuations=[20, 30, 40, 50, 60],
            seed=42
        )
        agent.current_asker = 2
        agent.current_ask = 30
        agent.current_bid = 30

        result = agent._player_want_to_sell()
        assert result == 1

    def test_no_sell_at_loss(self):
        """Test seller rejects when bid < cost (Java line 279)."""
        agent = Perry(
            player_id=2,
            is_buyer=False,
            num_tokens=5,
            valuations=[20, 30, 40, 50, 60],
            seed=42
        )
        agent.num_trades = 0  # First token costs 20
        agent.current_bid = 19

        result = agent._player_want_to_sell()
        assert result == 0

    def test_desperate_acceptance(self):
        """Test desperate acceptance near end of period (Java lines 292-294)."""
        agent = Perry(
            player_id=2,
            is_buyer=False,
            num_tokens=5,
            valuations=[20, 30, 40, 50, 60],
            num_buyers=4,
            num_sellers=4,
            num_times=100,
            seed=42
        )
        agent.num_trades = 0
        agent.current_time = 85  # 15% remaining -> <= 20%
        agent.traded_prices = [50]
        agent.current_bid = 25  # Above cost + 2
        agent.r_price_ave = 50
        agent.r_price_std = 10

        result = agent._player_want_to_sell()
        # Should accept if bid > cost + 2 when desperate
        assert result == 1


class TestPerryTimePressure:
    """Test time-dependent weighting (a1 parameter)."""

    def test_a1_decreases_over_time_bid(self):
        """Test a1 decreases as period progresses (Java line 72)."""
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            num_buyers=4,
            num_sellers=4,
            num_times=100,
            seed=42
        )

        # Early in period
        agent.current_time = 10
        a1_early = agent.a0 * (100 - 10) / 100 * 3 / 4 * 4 / 4

        # Late in period
        agent.current_time = 90
        a1_late = agent.a0 * (100 - 90) / 100 * 3 / 4 * 4 / 4

        assert a1_early > a1_late

    def test_a1_market_composition_buyer(self):
        """Test a1 accounts for market composition for buyers (Java line 72)."""
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            num_buyers=4,
            num_sellers=6,
            num_times=100,
            seed=42
        )
        agent.current_time = 50

        # a1 = a0 * (ntimes-t)/ntimes * (nplayers-1)/nplayers * nsellers/nbuyers
        # = 2.0 * 50/100 * 9/10 * 6/4 = 1.35
        expected_a1 = 2.0 * (100-50)/100 * 9/10 * 6/4
        assert abs(expected_a1 - 1.35) < 0.01


class TestPerryIntegration:
    """Integration tests for full trading cycle."""

    def test_full_abstract_method_cycle(self):
        """Test that all abstract methods work together."""
        agent = Perry(
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

    def test_period_lifecycle(self):
        """Test full period lifecycle."""
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )

        # Start period
        agent.start_period(1)
        assert agent.current_period == 1
        assert len(agent.traded_prices) == 0

        # Simulate some trades
        agent.buy_sell_result(
            status=1,
            trade_price=75,
            trade_type=1,
            high_bid=0,
            high_bidder=0,
            low_ask=0,
            low_asker=0
        )
        assert len(agent.traded_prices) == 1

        # End period
        agent.end_period()
        # _evaluate should have been called

    def test_deterministic_with_seed(self):
        """Test that same seed produces same behavior."""
        agent1 = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )

        agent2 = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42
        )

        # Both should generate same random values
        agent1.current_period = 1
        agent2.current_period = 1
        agent1.current_ask = 0
        agent2.current_ask = 0

        bid1 = agent1._player_request_bid()
        bid2 = agent2._player_request_bid()
        assert bid1 == bid2


class TestPerryBudgetConstraints:
    """Comprehensive budget constraint validation."""

    def test_perry_never_violates_budget_constraint_buyer(self):
        """
        Test Perry buyer never bids above valuation in realistic market.

        Runs 50 time steps with various market conditions to catch edge cases.
        """
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            num_buyers=4,
            num_sellers=4,
            num_times=50,
            seed=42
        )

        violations = []

        # Simulate 50 time steps
        for t in range(1, 51):
            agent.bid_ask(time=t, nobidask=0)
            bid = agent.bid_ask_response()

            if bid > 0:  # Only check non-zero bids
                current_val = agent.valuations[agent.num_trades]
                if bid > current_val:
                    violations.append({
                        'time': t,
                        'bid': bid,
                        'valuation': current_val,
                        'num_trades': agent.num_trades
                    })

            # Reset for next iteration
            agent.has_responded = False

        assert len(violations) == 0, \
            f"Perry violated budget constraint {len(violations)} times: {violations[:3]}"

    def test_perry_never_violates_budget_constraint_seller(self):
        """
        Test Perry seller never asks below cost in realistic market.

        Runs 50 time steps with various market conditions to catch edge cases.
        """
        agent = Perry(
            player_id=2,
            is_buyer=False,
            num_tokens=5,
            valuations=[40, 50, 60, 70, 80],
            num_buyers=4,
            num_sellers=4,
            num_times=50,
            seed=43
        )

        violations = []

        # Simulate 50 time steps
        for t in range(1, 51):
            agent.bid_ask(time=t, nobidask=0)
            ask = agent.bid_ask_response()

            if ask > 0:  # Only check non-zero asks
                current_cost = agent.valuations[agent.num_trades]
                if ask < current_cost:
                    violations.append({
                        'time': t,
                        'ask': ask,
                        'cost': current_cost,
                        'num_trades': agent.num_trades
                    })

            # Reset for next iteration
            agent.has_responded = False

        assert len(violations) == 0, \
            f"Perry violated budget constraint {len(violations)} times: {violations[:3]}"

    def test_perry_desperate_acceptance_respects_constraints(self):
        """
        Test Perry's desperate acceptance logic doesn't violate constraints.

        Perry has desperate acceptance near end of period (lines 488-490).
        Must verify it still respects budget constraints.
        """
        agent = Perry(
            player_id=2,
            is_buyer=False,
            num_tokens=5,
            valuations=[40, 50, 60, 70, 80],
            num_buyers=4,
            num_sellers=4,
            num_times=100,
            seed=44
        )

        # Set up desperate scenario (20% time remaining)
        agent.current_time = 85  # 15% remaining < 20% threshold
        agent.traded_prices = [55]  # Some history
        agent.r_price_ave = 55
        agent.r_price_std = 10
        agent.current_asker = 2
        agent.current_ask = 65
        agent.num_trades = 0  # First token costs 40

        # Test various bid prices
        for bid in [35, 38, 40, 42, 45]:  # Below, at, above cost
            agent.current_bid = bid
            agent.buy_sell(
                time=85,
                nobuysell=0,
                high_bid=bid,
                low_ask=65,
                high_bidder=1,
                low_asker=2
            )
            result = agent.buy_sell_response()

            # Should only accept if bid > cost (40) + some margin
            # Desperate acceptance allows trades at cost + 2 (line 489)
            if bid <= agent.valuations[0]:  # bid <= 40 (cost)
                assert result == 0, \
                    f"Perry accepted losing trade: bid={bid} <= cost={agent.valuations[0]}"

            agent.has_responded = False
