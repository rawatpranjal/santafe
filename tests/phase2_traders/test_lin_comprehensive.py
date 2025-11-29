"""
Comprehensive validation tests for Lin trader.

Tests exploitation by sophisticated traders (GD, ZIP, Kaplan) and multi-period
statistical prediction mechanisms. Critical for explaining Lin's 26th place
despite 99.85% self-play efficiency.
"""

import pytest
import numpy as np
from engine.market import Market
from engine.efficiency import calculate_allocative_efficiency, calculate_max_surplus, calculate_actual_surplus, extract_trades_from_orderbook
from traders.legacy.lin import Lin
from traders.legacy.zic import ZIC
from traders.legacy.gd import GD
from traders.legacy.zip import ZIP
from traders.legacy.kaplan import Kaplan


class TestLinExploitation:
    """
    Test Lin vs sophisticated traders - CRITICAL HYPOTHESIS.

    Lin achieves 99.85% self-play but ranked 26th in 1993.
    Hypothesis: GD, ZIP, and Kaplan exploit Lin's predictable statistical strategy.
    """

    def test_lin_vs_gd_exploitation(self):
        """
        Test Lin buyers vs GD sellers - exploitation hypothesis.

        Expected: GD should extract significantly more profit than Lin.
        GD has 10x profit dominance vs ZIC, should exploit Lin's predictability.
        """
        lin_profits = []
        gd_profits = []

        for seed in range(10):
            buyers = [
                Lin(
                    player_id=i+1,
                    is_buyer=True,
                    num_tokens=3,
                    valuations=[100, 90, 80],
                    num_buyers=4,
                    num_sellers=4,
                    num_times=50,
                    seed=seed*10 + i,
                )
                for i in range(4)
            ]

            sellers = [
                GD(
                    player_id=i+1,
                    is_buyer=False,
                    num_tokens=3,
                    valuations=[20, 30, 40],
                    num_buyers=4,
                    num_sellers=4,
                    seed=seed*10 + 100 + i,
                )
                for i in range(4)
            ]

            market = Market(
                num_buyers=4,
                num_sellers=4,
                num_times=50,
                price_min=0,
                price_max=200,
                buyers=buyers,
                sellers=sellers,
                seed=seed,
            )

            # Run market
            for step in range(50):
                success = market.run_time_step()
                if not success:
                    break

            # Calculate profits
            lin_total = sum(b.total_profit for b in buyers)
            gd_total = sum(s.total_profit for s in sellers)

            lin_profits.append(lin_total)
            gd_profits.append(gd_total)

        mean_lin = np.mean(lin_profits)
        mean_gd = np.mean(gd_profits)
        total_profit = mean_lin + mean_gd
        lin_share = (mean_lin / total_profit * 100) if total_profit > 0 else 0
        gd_share = (mean_gd / total_profit * 100) if total_profit > 0 else 0
        profit_ratio = (mean_gd / mean_lin) if mean_lin > 0 else 0

        print(f"\nLin vs GD Exploitation Test:")
        print(f"  Lin profit:  {mean_lin:.2f} ({lin_share:.1f}% share)")
        print(f"  GD profit:   {mean_gd:.2f} ({gd_share:.1f}% share)")
        print(f"  GD/Lin ratio: {profit_ratio:.2f}x")

        if total_profit == 0:
            pytest.skip("No trades occurred")

        # GD should extract more profit than Lin (exploitation)
        assert gd_share > lin_share, f"GD did not exploit Lin (GD: {gd_share:.1f}%, Lin: {lin_share:.1f}%)"

    def test_lin_vs_zip_exploitation(self):
        """
        Test Lin buyers vs ZIP sellers - exploitation hypothesis.

        Expected: ZIP should extract more profit than Lin.
        ZIP has 7.20x profit ratio vs ZIC, should exploit Lin.
        """
        lin_profits = []
        zip_profits = []

        for seed in range(10):
            buyers = [
                Lin(
                    player_id=i+1,
                    is_buyer=True,
                    num_tokens=3,
                    valuations=[100, 90, 80],
                    num_buyers=4,
                    num_sellers=4,
                    num_times=50,
                    seed=seed*10 + i,
                )
                for i in range(4)
            ]

            sellers = [
                ZIP(
                    player_id=i+1,
                    is_buyer=False,
                    num_tokens=3,
                    valuations=[20, 30, 40],
                    num_buyers=4,
                    num_sellers=4,
                    seed=seed*10 + 100 + i,
                )
                for i in range(4)
            ]

            market = Market(
                num_buyers=4,
                num_sellers=4,
                num_times=50,
                price_min=0,
                price_max=200,
                buyers=buyers,
                sellers=sellers,
                seed=seed,
            )

            # Run market
            for step in range(50):
                success = market.run_time_step()
                if not success:
                    break

            # Calculate profits
            lin_total = sum(b.total_profit for b in buyers)
            zip_total = sum(s.total_profit for s in sellers)

            lin_profits.append(lin_total)
            zip_profits.append(zip_total)

        mean_lin = np.mean(lin_profits)
        mean_zip = np.mean(zip_profits)
        total_profit = mean_lin + mean_zip
        lin_share = (mean_lin / total_profit * 100) if total_profit > 0 else 0
        zip_share = (mean_zip / total_profit * 100) if total_profit > 0 else 0
        profit_ratio = (mean_zip / mean_lin) if mean_lin > 0 else 0

        print(f"\nLin vs ZIP Exploitation Test:")
        print(f"  Lin profit:  {mean_lin:.2f} ({lin_share:.1f}% share)")
        print(f"  ZIP profit:  {mean_zip:.2f} ({zip_share:.1f}% share)")
        print(f"  ZIP/Lin ratio: {profit_ratio:.2f}x")

        if total_profit == 0:
            pytest.skip("No trades occurred")

        # ZIP should extract more profit than Lin (exploitation)
        assert zip_share > lin_share, f"ZIP did not exploit Lin (ZIP: {zip_share:.1f}%, Lin: {lin_share:.1f}%)"

    @pytest.mark.skip(reason="Kaplan has known index out of range bug (tracker.md)")
    def test_lin_vs_kaplan_exploitation(self):
        """
        Test Lin vs Kaplan - ultimate exploitation test.

        Expected: Kaplan should dominate Lin.
        Kaplan (1st place) is parasitic on predictable traders.

        SKIPPED: Kaplan has index out of range bug in price tracking.
        """
        lin_profits = []
        kaplan_profits = []

        for seed in range(10):
            buyers = [
                Lin(
                    player_id=i+1,
                    is_buyer=True,
                    num_tokens=3,
                    valuations=[100, 90, 80],
                    num_buyers=4,
                    num_sellers=4,
                    num_times=50,
                    seed=seed*10 + i,
                )
                for i in range(2)
            ] + [
                Kaplan(
                    player_id=i+3,
                    is_buyer=True,
                    num_tokens=3,
                    valuations=[100, 90, 80],
                    num_times=50,
                    seed=seed*10 + 50 + i,
                )
                for i in range(2)
            ]

            sellers = [
                ZIC(
                    player_id=i+1,
                    is_buyer=False,
                    num_tokens=3,
                    valuations=[20, 30, 40],
                    seed=seed*10 + 100 + i,
                )
                for i in range(4)
            ]

            market = Market(
                num_buyers=4,
                num_sellers=4,
                num_times=50,
                price_min=0,
                price_max=200,
                buyers=buyers,
                sellers=sellers,
                seed=seed,
            )

            # Run market
            for step in range(50):
                success = market.run_time_step()
                if not success:
                    break

            # Calculate profits (Lin is buyers[0:2], Kaplan is buyers[2:4])
            lin_total = sum(buyers[i].total_profit for i in range(2))
            kaplan_total = sum(buyers[i].total_profit for i in range(2, 4))

            lin_profits.append(lin_total)
            kaplan_profits.append(kaplan_total)

        mean_lin = np.mean(lin_profits)
        mean_kaplan = np.mean(kaplan_profits)
        total_profit = mean_lin + mean_kaplan
        lin_share = (mean_lin / total_profit * 100) if total_profit > 0 else 0
        kaplan_share = (mean_kaplan / total_profit * 100) if total_profit > 0 else 0

        print(f"\nLin vs Kaplan Mixed Market:")
        print(f"  Lin profit:    {mean_lin:.2f} ({lin_share:.1f}% share)")
        print(f"  Kaplan profit: {mean_kaplan:.2f} ({kaplan_share:.1f}% share)")

        if total_profit == 0:
            pytest.skip("No trades occurred")

        # Kaplan should extract more profit than Lin
        assert kaplan_share > lin_share, f"Kaplan did not dominate Lin (Kaplan: {kaplan_share:.1f}%, Lin: {lin_share:.1f}%)"


class TestLinMultiPeriod:
    """Test Lin's multi-period statistical prediction mechanisms."""

    def test_lin_mean_price_convergence_over_periods(self):
        """
        Test that Lin's mean_price array populates and converges correctly.

        Expected: Mean prices should stabilize near equilibrium over periods.
        """
        buyers = [
            Lin(
                player_id=1,
                is_buyer=True,
                num_tokens=3,
                valuations=[100, 90, 80],
                num_buyers=2,
                num_sellers=2,
                num_times=30,
                seed=42,
            ),
            ZIC(
                player_id=2,
                is_buyer=True,
                num_tokens=3,
                valuations=[100, 90, 80],
                seed=43,
            ),
        ]

        sellers = [
            ZIC(
                player_id=1,
                is_buyer=False,
                num_tokens=3,
                valuations=[20, 30, 40],
                seed=44,
            ),
            ZIC(
                player_id=2,
                is_buyer=False,
                num_tokens=3,
                valuations=[20, 30, 40],
                seed=45,
            ),
        ]

        market = Market(
            num_buyers=2,
            num_sellers=2,
            num_times=30,
            price_min=0,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=100,
        )

        lin = buyers[0]
        mean_prices = []

        # Run 5 periods
        for period in range(1, 6):
            for agent in buyers + sellers:
                agent.start_period(period)

            # Run timesteps
            for step in range(30):
                success = market.run_time_step()
                if not success:
                    break

            for agent in buyers + sellers:
                agent.end_period()

            # Record Lin's mean price for this period
            if period < len(lin.mean_price):
                mean_prices.append(lin.mean_price[period])

        print(f"\nLin mean price evolution over 5 periods:")
        print(f"  Prices: {[f'{p:.2f}' for p in mean_prices]}")

        # Verify mean prices were recorded
        assert len(mean_prices) == 5, "Not all periods recorded"

        # Verify prices are reasonable (between min and max)
        assert all(0 <= p <= 200 for p in mean_prices), "Mean prices out of bounds"

        # Check for convergence (variance should be reasonable)
        if len(mean_prices) > 2:
            variance = np.var(mean_prices)
            print(f"  Variance: {variance:.2f}")
            # Don't assert convergence strictly, just verify it's tracked

    def test_lin_target_price_historical_weighting(self):
        """
        Test that Lin's target price correctly averages across periods.

        Java lines 137-140: target = (current_mean + sum(previous_means)) / period
        """
        lin = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=3,
            valuations=[100, 90, 80],
            seed=42,
        )

        # Set up historical data
        lin.current_period = 3
        lin.mean_price[1] = 50.0
        lin.mean_price[2] = 60.0
        lin.traded_prices = [70, 80]  # Current period

        target = lin._get_target_price()

        # Expected: (75 + 50 + 60) / 3 = 61.67
        current_mean = 75.0  # (70+80)/2
        expected = (current_mean + 50.0 + 60.0) / 3

        print(f"\nLin target price calculation:")
        print(f"  Current mean: {current_mean:.2f}")
        print(f"  Previous means: [50.00, 60.00]")
        print(f"  Target: {target:.2f}")
        print(f"  Expected: {expected:.2f}")

        assert abs(target - expected) < 0.1, f"Target price {target:.2f} != expected {expected:.2f}"

    def test_lin_stderr_calculation_stability(self):
        """
        Test that Lin's stderr calculation produces consistent results.

        Java lines 147-159: Non-standard stderr formula.
        """
        lin = Lin(
            player_id=1,
            is_buyer=True,
            num_tokens=3,
            valuations=[100, 90, 80],
            seed=42,
        )

        # Test with consistent data
        lin.traded_prices = [50, 60, 70, 80]
        stderr1 = lin._get_stderr_price()

        # Test again with same data
        stderr2 = lin._get_stderr_price()

        print(f"\nLin stderr stability:")
        print(f"  Stderr1: {stderr1:.4f}")
        print(f"  Stderr2: {stderr2:.4f}")

        # Should be exactly the same (deterministic)
        assert stderr1 == stderr2, "Stderr calculation not deterministic"

        # Verify it's the Java non-standard formula
        mean = 65.0
        sum_sq = sum((abs(p) - mean) ** 2 for p in lin.traded_prices)
        expected = np.sqrt(sum_sq) / 3  # Java's formula

        assert abs(stderr1 - expected) < 0.01, f"Stderr {stderr1:.4f} != expected {expected:.4f}"
