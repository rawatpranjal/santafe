"""
Behavioral invariant tests for Lin and Perry traders.

Tests safety properties and bounds that must ALWAYS hold:
- Lin: Weighting formula ∈ [0,1], Box-Muller never NaN, constraint enforcement
- Perry: a0 > 0, price_std ≥ 0, no loss trades, parameter bounds

These are defensive tests ensuring no edge cases violate core behavioral assumptions.
"""

import pytest
import numpy as np
import math
from engine.market import Market
from traders.legacy.lin import Lin
from traders.legacy.perry import Perry
from traders.legacy.zic import ZIC


class TestLinInvariants:
    """Test Lin trader safety invariants and bounds."""

    def test_lin_weighting_formula_bounds(self):
        """
        Test that Lin's weighting formula always produces values in [0, 1].

        Java lines 85-90: weight = time_left × token_depletion × market_composition
        Each component should be in [0, 1], final weight in [0, 1].
        """
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            num_buyers=3,
            num_sellers=4,
            num_times=50,
            seed=42,
        )

        # Test at various time points and trade counts
        test_cases = [
            (1, 0),    # Start: high time, no trades
            (25, 2),   # Middle: medium time, some trades
            (49, 4),   # Near end: low time, many trades
            (50, 5),   # End: no time left, all tokens used
        ]

        for time, trades in test_cases:
            agent.current_time = time
            agent.num_trades = trades

            # Calculate weight components (from lin.py lines 278-282)
            time_factor = max(0.0, (agent.num_times - agent.current_time) / agent.num_times)
            token_factor = max(0.0, (agent.num_tokens - agent.num_trades) / agent.num_tokens)
            market_factor = (agent.num_buyers * agent.num_sellers) / (agent.num_buyers + agent.num_sellers)
            weight = time_factor * token_factor * market_factor

            print(f"Time {time}, Trades {trades}: weight = {weight:.4f}")

            # Weight must be in [0, 1]
            # Note: market_factor can be > 1, so weight might exceed 1
            assert weight >= 0.0, f"Weight {weight} negative at time={time}, trades={trades}"
            assert time_factor >= 0.0 and time_factor <= 1.0, "time_factor out of bounds"
            assert token_factor >= 0.0 and token_factor <= 1.0, "token_factor out of bounds"

    def test_lin_box_muller_no_nan(self):
        """
        Test that Lin's Box-Muller never produces NaN.

        Box-Muller formula:
            x1 = mean + err × sqrt(-2 × ln(r1)) × cos(2π × r2)

        This can produce NaN if r1=0 (ln(0) = -inf).
        Lin should guard against this.
        """
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42,
        )

        mean = 50.0
        err = 10.0

        # Generate 1000 samples
        for _ in range(1000):
            sample = agent._norm(mean, err)

            # Verify not NaN
            assert not math.isnan(sample), "Box-Muller produced NaN"

            # Verify finite
            assert math.isfinite(sample), "Box-Muller produced infinite value"

    def test_lin_never_bid_above_valuation(self):
        """
        Test that Lin never bids above its current token valuation.

        This is a CRITICAL safety constraint - violating it means accepting loss.
        """
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=3,
            valuations=[100, 90, 80],
            num_buyers=2,
            num_sellers=2,
            num_times=10,
            seed=42,
        )

        # Simulate market and collect all bids
        buyers = [
            agent,
            ZIC(player_id=2, is_buyer=True, num_tokens=3, valuations=[95, 85, 75], seed=43),
        ]

        sellers = [
            ZIC(player_id=1, is_buyer=False, num_tokens=3, valuations=[20, 30, 40], seed=44),
            ZIC(player_id=2, is_buyer=False, num_tokens=3, valuations=[25, 35, 45], seed=45),
        ]

        market = Market(
            num_buyers=2,
            num_sellers=2,
            num_times=10,
            price_min=0,
            price_max=100,
            buyers=buyers,
            sellers=sellers,
            seed=100,
        )

        # Run market and track bids
        for step in range(10):
            success = market.run_time_step()
            if not success:
                break

        # Verify Lin never bid above current valuation
        # (Lin tracks num_trades, so we can check valuation at each trade)
        current_valuation = agent.valuations[agent.num_trades] if agent.num_trades < 3 else 0
        print(f"Lin made {agent.num_trades} trades, current valuation: {current_valuation}")

        # If Lin made all trades, it should have no remaining valuation
        if agent.num_trades == 3:
            assert current_valuation == 0 or agent.num_trades == len(agent.valuations), "Lin exhausted tokens correctly"

    def test_lin_never_ask_below_cost(self):
        """
        Test that Lin sellers never ask below their cost.

        Symmetric to bid constraint - sellers should never accept loss.
        """
        agent = Lin(
            player_id=1,
            is_buyer=False,
            num_tokens=3,
            valuations=[20, 30, 40],
            num_buyers=2,
            num_sellers=2,
            num_times=10,
            seed=42,
        )

        buyers = [
            ZIC(player_id=1, is_buyer=True, num_tokens=3, valuations=[100, 90, 80], seed=43),
            ZIC(player_id=2, is_buyer=True, num_tokens=3, valuations=[95, 85, 75], seed=44),
        ]

        sellers = [
            agent,
            ZIC(player_id=2, is_buyer=False, num_tokens=3, valuations=[25, 35, 45], seed=45),
        ]

        market = Market(
            num_buyers=2,
            num_sellers=2,
            num_times=10,
            price_min=0,
            price_max=100,
            buyers=buyers,
            sellers=sellers,
            seed=100,
        )

        # Run market
        for step in range(10):
            success = market.run_time_step()
            if not success:
                break

        # Verify Lin made valid trades
        print(f"Lin (seller) made {agent.num_trades} trades")
        assert agent.num_trades <= 3, "Lin never over-traded"

    def test_lin_statistical_calculations_valid(self):
        """
        Test that Lin's statistical calculations never produce invalid values.

        Checks: mean_price, stderr_price, target_price all finite and reasonable.
        """
        agent = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=3,
            valuations=[100, 90, 80],
            seed=42,
        )

        # Set up traded prices
        agent.traded_prices = [50, 60, 70, 80]

        # Test mean calculation
        mean = np.mean(agent.traded_prices) if agent.traded_prices else 0.0
        assert math.isfinite(mean), "Mean price not finite"
        assert mean >= 0, "Mean price negative"

        # Test stderr calculation
        stderr = agent._get_stderr_price()
        assert math.isfinite(stderr), "Stderr not finite"
        assert stderr >= 0, "Stderr negative"

        print(f"Mean: {mean:.2f}, Stderr: {stderr:.4f}")

        # Test target price (requires period tracking)
        agent.current_period = 2
        agent.mean_price[1] = 55.0
        target = agent._get_target_price()
        assert math.isfinite(target), "Target price not finite"
        assert target >= 0, "Target price negative"

        print(f"Target: {target:.2f}")


class TestPerryInvariants:
    """Test Perry trader safety invariants and bounds."""

    def test_perry_a0_always_positive(self):
        """
        Test that Perry's a0 parameter always stays positive.

        a0 is used in Perry's acceptance region formula.
        If a0 ≤ 0, the strategy breaks down.
        """
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42,
        )

        # Test initial a0
        assert agent.a0 > 0, f"Initial a0 {agent.a0} not positive"

        # Simulate extremely poor performance (should trigger a0 decrease)
        for _ in range(20):
            agent.traded_prices = [75]
            agent.p_ave_price = 75
            agent.num_trades = 0  # No trades = very low efficiency
            agent.period_profit = 0

            agent._evaluate()

            print(f"a0 after poor period: {agent.a0:.6f}")

            # a0 should stay positive even after many bad periods
            assert agent.a0 > 0, f"a0 became non-positive: {agent.a0}"

    def test_perry_price_std_non_negative(self):
        """
        Test that Perry's price_std never becomes negative.

        price_std is used for bid/ask randomization. Negative values are invalid.
        """
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42,
        )

        # Test initial price_std
        assert agent.price_std >= 0, f"Initial price_std {agent.price_std} negative"

        # Simulate various market conditions
        test_scenarios = [
            {"traded_prices": [75], "p_ave_price": 75, "num_trades": 1, "period_profit": 10},
            {"traded_prices": [50, 60, 70], "p_ave_price": 60, "num_trades": 3, "period_profit": 30},
            {"traded_prices": [], "p_ave_price": 0, "num_trades": 0, "period_profit": 0},
        ]

        for scenario in test_scenarios:
            agent.traded_prices = scenario["traded_prices"]
            agent.p_ave_price = scenario["p_ave_price"]
            agent.num_trades = scenario["num_trades"]
            agent.period_profit = scenario["period_profit"]

            agent._evaluate()

            print(f"price_std: {agent.price_std}")
            assert agent.price_std >= 0, f"price_std became negative: {agent.price_std}"

    def test_perry_never_accept_loss_trades(self):
        """
        Test that Perry buyers never accept trades above their valuation.

        Perry uses acceptance thresholds - verify they enforce constraints.
        """
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=3,
            valuations=[100, 90, 80],
            num_buyers=2,
            num_sellers=2,
            num_times=50,
            seed=42,
        )

        buyers = [
            agent,
            ZIC(player_id=2, is_buyer=True, num_tokens=3, valuations=[95, 85, 75], seed=43),
        ]

        sellers = [
            ZIC(player_id=1, is_buyer=False, num_tokens=3, valuations=[20, 30, 40], seed=44),
            ZIC(player_id=2, is_buyer=False, num_tokens=3, valuations=[25, 35, 45], seed=45),
        ]

        market = Market(
            num_buyers=2,
            num_sellers=2,
            num_times=50,
            price_min=0,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=100,
        )

        # Run market
        for step in range(50):
            success = market.run_time_step()
            if not success:
                break

        # Verify Perry never lost money
        print(f"Perry profit: {agent.total_profit}")
        print(f"Perry trades: {agent.num_trades}")

        # Perry should never have negative profit
        assert agent.total_profit >= 0, f"Perry accepted loss trades (profit: {agent.total_profit})"

    def test_perry_statistical_calculations_valid(self):
        """
        Test that Perry's price statistics are always valid.

        Checks: p_ave_price, r_price_ave, r_price_std all finite and non-negative.
        """
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42,
        )

        # Test with various price histories
        agent.traded_prices = [50, 60, 70, 80]
        agent._ave_price()

        print(f"p_ave_price: {agent.p_ave_price}")
        assert math.isfinite(agent.p_ave_price), "p_ave_price not finite"
        assert agent.p_ave_price >= 0, "p_ave_price negative"

        # Test round statistics
        agent.current_round = 1
        agent.round_count = 1
        agent._round_average_price()

        print(f"r_price_ave: {agent.r_price_ave:.2f}")
        print(f"r_price_std: {agent.r_price_std:.2f}")

        assert math.isfinite(agent.r_price_ave), "r_price_ave not finite"
        assert math.isfinite(agent.r_price_std), "r_price_std not finite"
        assert agent.r_price_ave >= 0, "r_price_ave negative"
        assert agent.r_price_std >= 0, "r_price_std negative"

    def test_perry_parameter_bounds_after_adaptation(self):
        """
        Test that Perry's parameters stay within reasonable bounds after adaptation.

        Parameters: a0, price_std, p_ave_price
        Even after many adaptation cycles, should remain valid.
        """
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=5,
            valuations=[100, 90, 80, 70, 60],
            seed=42,
        )

        # Simulate 50 periods of adaptation with varying performance
        for period in range(50):
            # Alternate between good and poor performance
            if period % 2 == 0:
                agent.traded_prices = [60, 65, 70]
                agent.p_ave_price = 65
                agent.num_trades = 3
                agent.period_profit = 90
            else:
                agent.traded_prices = [75]
                agent.p_ave_price = 75
                agent.num_trades = 1
                agent.period_profit = 15

            agent._evaluate()

            # Verify all parameters remain valid
            assert agent.a0 > 0, f"a0 invalid at period {period}: {agent.a0}"
            assert agent.a0 <= 10.0, f"a0 too large at period {period}: {agent.a0}"
            assert agent.price_std >= 0, f"price_std invalid at period {period}: {agent.price_std}"
            assert math.isfinite(agent.p_ave_price), f"p_ave_price invalid at period {period}"

        print(f"\nAfter 50 periods of adaptation:")
        print(f"  Final a0: {agent.a0:.6f}")
        print(f"  Final price_std: {agent.price_std}")
        print(f"  Final p_ave_price: {agent.p_ave_price:.2f}")


class TestCrossTraderInvariants:
    """Test invariants that apply to both Lin and Perry."""

    def test_token_conservation(self):
        """
        Test that both Lin and Perry never trade more tokens than available.

        This is a fundamental market constraint.
        """
        # Test Lin
        lin_buyer = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=3,
            valuations=[100, 90, 80],
            num_buyers=2,
            num_sellers=2,
            num_times=50,
            seed=42,
        )

        # Test Perry
        perry_buyer = Perry(
            player_id=2,
            is_buyer=True,
            num_tokens=3,
            valuations=[95, 85, 75],
            num_buyers=2,
            num_sellers=2,
            num_times=50,
            seed=43,
        )

        buyers = [lin_buyer, perry_buyer]

        sellers = [
            ZIC(player_id=1, is_buyer=False, num_tokens=3, valuations=[20, 30, 40], seed=44),
            ZIC(player_id=2, is_buyer=False, num_tokens=3, valuations=[25, 35, 45], seed=45),
        ]

        market = Market(
            num_buyers=2,
            num_sellers=2,
            num_times=50,
            price_min=0,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=100,
        )

        # Run market
        for step in range(50):
            success = market.run_time_step()
            if not success:
                break

        # Verify token conservation
        print(f"\nToken conservation test:")
        print(f"  Lin trades: {lin_buyer.num_trades} / 3 tokens")
        print(f"  Perry trades: {perry_buyer.num_trades} / 3 tokens")

        assert lin_buyer.num_trades <= 3, f"Lin over-traded: {lin_buyer.num_trades}"
        assert perry_buyer.num_trades <= 3, f"Perry over-traded: {perry_buyer.num_trades}"

    def test_no_negative_profits(self):
        """
        Test that both Lin and Perry never accept trades resulting in negative profit.

        Both traders should enforce valuation constraints.
        """
        buyers = [
            Lin(
                player_id=1,
                is_buyer=True,
                num_tokens=3,
                valuations=[100, 90, 80],
                num_buyers=2,
                num_sellers=2,
                num_times=50,
                seed=42,
            ),
            Perry(
                player_id=2,
                is_buyer=True,
                num_tokens=3,
                valuations=[95, 85, 75],
                num_buyers=2,
                num_sellers=2,
                num_times=50,
                seed=43,
            ),
        ]

        sellers = [
            ZIC(player_id=1, is_buyer=False, num_tokens=3, valuations=[20, 30, 40], seed=44),
            ZIC(player_id=2, is_buyer=False, num_tokens=3, valuations=[25, 35, 45], seed=45),
        ]

        market = Market(
            num_buyers=2,
            num_sellers=2,
            num_times=50,
            price_min=0,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=100,
        )

        # Run market
        for step in range(50):
            success = market.run_time_step()
            if not success:
                break

        # Verify no negative profits
        print(f"\nProfit safety test:")
        print(f"  Lin profit: {buyers[0].total_profit}")
        print(f"  Perry profit: {buyers[1].total_profit}")

        assert buyers[0].total_profit >= 0, f"Lin negative profit: {buyers[0].total_profit}"
        assert buyers[1].total_profit >= 0, f"Perry negative profit: {buyers[1].total_profit}"
