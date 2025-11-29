"""
Edge case tests for Lin and Perry traders.

Tests degenerate and extreme market conditions:
- Zero surplus markets (no gains from trade)
- Single token per agent
- Extreme valuations (very high/very low)
- Degenerate markets (1v1, no trading opportunities)

These tests ensure traders don't crash in pathological conditions.
"""

import pytest
import numpy as np
from engine.market import Market
from engine.efficiency import calculate_allocative_efficiency, calculate_max_surplus, calculate_actual_surplus, extract_trades_from_orderbook
from traders.legacy.lin import Lin
from traders.legacy.perry import Perry
from traders.legacy.zic import ZIC


class TestZeroSurplusMarkets:
    """Test Lin and Perry in markets with zero gains from trade."""

    def test_lin_zero_surplus_market(self):
        """
        Test Lin in zero surplus market (all buyers value < all sellers cost).

        Valuations: Buyers [30, 20, 10], Sellers [40, 50, 60]
        No profitable trades exist - Lin should not crash.
        """
        buyers = [
            Lin(
                player_id=i+1,
                is_buyer=True,
                num_tokens=3,
                valuations=[30, 20, 10],
                num_buyers=2,
                num_sellers=2,
                num_times=20,
                seed=42 + i,
            )
            for i in range(2)
        ]

        sellers = [
            ZIC(
                player_id=i+1,
                is_buyer=False,
                num_tokens=3,
                valuations=[40, 50, 60],
                seed=100 + i,
            )
            for i in range(2)
        ]

        market = Market(
            num_buyers=2,
            num_sellers=2,
            num_times=20,
            price_min=0,
            price_max=100,
            buyers=buyers,
            sellers=sellers,
            seed=200,
        )

        # Run market
        for step in range(20):
            success = market.run_time_step()
            if not success:
                break

        # Count trades
        num_trades = sum(1 for t in range(1, 21) if market.orderbook.trade_price[t] > 0)

        print(f"\nLin Zero Surplus Market:")
        print(f"  Trades: {num_trades}")
        print(f"  Lin[0] profit: {buyers[0].total_profit}")
        print(f"  Lin[1] profit: {buyers[1].total_profit}")

        # No trades should occur (or if they do, they're irrational)
        # Lin should not crash
        assert buyers[0].total_profit >= 0, "Lin accepted loss trade"
        assert buyers[1].total_profit >= 0, "Lin accepted loss trade"

    def test_perry_zero_surplus_market(self):
        """
        Test Perry in zero surplus market.

        Perry's conservative strategy should prevent any trades.
        """
        buyers = [
            Perry(
                player_id=i+1,
                is_buyer=True,
                num_tokens=3,
                valuations=[30, 20, 10],
                num_buyers=2,
                num_sellers=2,
                num_times=20,
                seed=42 + i,
            )
            for i in range(2)
        ]

        sellers = [
            ZIC(
                player_id=i+1,
                is_buyer=False,
                num_tokens=3,
                valuations=[40, 50, 60],
                seed=100 + i,
            )
            for i in range(2)
        ]

        market = Market(
            num_buyers=2,
            num_sellers=2,
            num_times=20,
            price_min=0,
            price_max=100,
            buyers=buyers,
            sellers=sellers,
            seed=200,
        )

        # Run market
        for step in range(20):
            success = market.run_time_step()
            if not success:
                break

        # Count trades
        num_trades = sum(1 for t in range(1, 21) if market.orderbook.trade_price[t] > 0)

        print(f"\nPerry Zero Surplus Market:")
        print(f"  Trades: {num_trades}")
        print(f"  Perry[0] profit: {buyers[0].total_profit}")
        print(f"  Perry[1] profit: {buyers[1].total_profit}")

        # Perry should not accept loss trades
        assert buyers[0].total_profit >= 0, "Perry accepted loss trade"
        assert buyers[1].total_profit >= 0, "Perry accepted loss trade"


class TestSingleTokenMarkets:
    """Test Lin and Perry with single token per agent."""

    def test_lin_single_token_market(self):
        """
        Test Lin with only 1 token per agent.

        Tests token depletion formula when num_tokens = 1.
        """
        buyers = [
            Lin(
                player_id=i+1,
                is_buyer=True,
                num_tokens=1,
                valuations=[100],
                num_buyers=3,
                num_sellers=3,
                num_times=20,
                seed=42 + i,
            )
            for i in range(3)
        ]

        sellers = [
            ZIC(
                player_id=i+1,
                is_buyer=False,
                num_tokens=1,
                valuations=[20],
                seed=100 + i,
            )
            for i in range(3)
        ]

        market = Market(
            num_buyers=3,
            num_sellers=3,
            num_times=20,
            price_min=0,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=200,
        )

        # Run market
        for step in range(20):
            success = market.run_time_step()
            if not success:
                break

        # Verify Lin handled single token correctly
        total_lin_trades = sum(b.num_trades for b in buyers)

        print(f"\nLin Single Token Market:")
        print(f"  Total Lin trades: {total_lin_trades} / 3 possible")

        # Each Lin should trade at most 1 token
        for i, buyer in enumerate(buyers):
            assert buyer.num_trades <= 1, f"Lin[{i}] over-traded: {buyer.num_trades}"

    def test_perry_single_token_market(self):
        """
        Test Perry with only 1 token per agent.

        Perry's two-phase strategy requires 3 trades to reach Phase 2.
        With only 1 token, Perry stays in conservative Phase 1.
        """
        buyers = [
            Perry(
                player_id=i+1,
                is_buyer=True,
                num_tokens=1,
                valuations=[100],
                num_buyers=3,
                num_sellers=3,
                num_times=20,
                seed=42 + i,
            )
            for i in range(3)
        ]

        sellers = [
            ZIC(
                player_id=i+1,
                is_buyer=False,
                num_tokens=1,
                valuations=[20],
                seed=100 + i,
            )
            for i in range(3)
        ]

        market = Market(
            num_buyers=3,
            num_sellers=3,
            num_times=20,
            price_min=0,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=200,
        )

        # Run market
        for step in range(20):
            success = market.run_time_step()
            if not success:
                break

        # Verify Perry handled single token
        total_perry_trades = sum(b.num_trades for b in buyers)

        print(f"\nPerry Single Token Market:")
        print(f"  Total Perry trades: {total_perry_trades} / 3 possible")

        # Each Perry should trade at most 1 token
        for i, buyer in enumerate(buyers):
            assert buyer.num_trades <= 1, f"Perry[{i}] over-traded: {buyer.num_trades}"


class TestExtremeValuations:
    """Test Lin and Perry with extreme valuation ranges."""

    def test_lin_very_high_valuations(self):
        """
        Test Lin with very high valuations.

        Valuations: Buyers [10000, 9000, 8000], Sellers [100, 200, 300]
        Tests numerical stability with large numbers.
        """
        buyers = [
            Lin(
                player_id=i+1,
                is_buyer=True,
                num_tokens=3,
                valuations=[10000, 9000, 8000],
                num_buyers=2,
                num_sellers=2,
                num_times=20,
                seed=42 + i,
            )
            for i in range(2)
        ]

        sellers = [
            ZIC(
                player_id=i+1,
                is_buyer=False,
                num_tokens=3,
                valuations=[100, 200, 300],
                seed=100 + i,
            )
            for i in range(2)
        ]

        market = Market(
            num_buyers=2,
            num_sellers=2,
            num_times=20,
            price_min=0,
            price_max=20000,
            buyers=buyers,
            sellers=sellers,
            seed=200,
        )

        # Run market
        for step in range(20):
            success = market.run_time_step()
            if not success:
                break

        # Verify Lin handled large values
        print(f"\nLin Very High Valuations:")
        print(f"  Lin[0] trades: {buyers[0].num_trades}")
        print(f"  Lin[0] profit: {buyers[0].total_profit}")

        # Lin should not crash with large numbers
        assert buyers[0].num_trades <= 3, "Lin over-traded"
        assert buyers[0].total_profit >= 0, "Lin negative profit"

    def test_perry_very_low_valuations(self):
        """
        Test Perry with very low valuations.

        Valuations: Buyers [10, 9, 8], Sellers [1, 2, 3]
        Tests numerical stability with small numbers.
        """
        buyers = [
            Perry(
                player_id=i+1,
                is_buyer=True,
                num_tokens=3,
                valuations=[10, 9, 8],
                num_buyers=2,
                num_sellers=2,
                num_times=20,
                seed=42 + i,
            )
            for i in range(2)
        ]

        sellers = [
            ZIC(
                player_id=i+1,
                is_buyer=False,
                num_tokens=3,
                valuations=[1, 2, 3],
                seed=100 + i,
            )
            for i in range(2)
        ]

        market = Market(
            num_buyers=2,
            num_sellers=2,
            num_times=20,
            price_min=0,
            price_max=20,
            buyers=buyers,
            sellers=sellers,
            seed=200,
        )

        # Run market
        for step in range(20):
            success = market.run_time_step()
            if not success:
                break

        # Verify Perry handled small values
        print(f"\nPerry Very Low Valuations:")
        print(f"  Perry[0] trades: {buyers[0].num_trades}")
        print(f"  Perry[0] profit: {buyers[0].total_profit}")

        # Perry should not crash with small numbers
        assert buyers[0].num_trades <= 3, "Perry over-traded"
        assert buyers[0].total_profit >= 0, "Perry negative profit"


class TestDegenerateMarkets:
    """Test Lin and Perry in degenerate market structures."""

    def test_lin_1v1_market(self):
        """
        Test Lin in 1 buyer vs 1 seller market.

        Tests market_factor calculation when num_buyers = num_sellers = 1.
        """
        buyers = [
            Lin(
                player_id=1,
                is_buyer=True,
                num_tokens=3,
                valuations=[100, 90, 80],
                num_buyers=1,
                num_sellers=1,
                num_times=20,
                seed=42,
            ),
        ]

        sellers = [
            ZIC(
                player_id=1,
                is_buyer=False,
                num_tokens=3,
                valuations=[20, 30, 40],
                seed=43,
            ),
        ]

        market = Market(
            num_buyers=1,
            num_sellers=1,
            num_times=20,
            price_min=0,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=200,
        )

        # Run market
        for step in range(20):
            success = market.run_time_step()
            if not success:
                break

        # Verify Lin survived 1v1 market
        print(f"\nLin 1v1 Market:")
        print(f"  Lin trades: {buyers[0].num_trades}")
        print(f"  Lin profit: {buyers[0].total_profit}")

        # Lin should not crash
        assert buyers[0].total_profit >= 0, "Lin negative profit"

    def test_perry_1v1_market(self):
        """
        Test Perry in 1 buyer vs 1 seller market.

        Perry should handle minimal market structure.
        """
        buyers = [
            Perry(
                player_id=1,
                is_buyer=True,
                num_tokens=3,
                valuations=[100, 90, 80],
                num_buyers=1,
                num_sellers=1,
                num_times=20,
                seed=42,
            ),
        ]

        sellers = [
            ZIC(
                player_id=1,
                is_buyer=False,
                num_tokens=3,
                valuations=[20, 30, 40],
                seed=43,
            ),
        ]

        market = Market(
            num_buyers=1,
            num_sellers=1,
            num_times=20,
            price_min=0,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=200,
        )

        # Run market
        for step in range(20):
            success = market.run_time_step()
            if not success:
                break

        # Verify Perry survived 1v1 market
        print(f"\nPerry 1v1 Market:")
        print(f"  Perry trades: {buyers[0].num_trades}")
        print(f"  Perry profit: {buyers[0].total_profit}")

        # Perry should not crash
        assert buyers[0].total_profit >= 0, "Perry negative profit"

    def test_lin_no_trading_opportunities(self):
        """
        Test Lin when buyer valuations exactly equal seller costs.

        Valuations: Buyers [50, 40, 30], Sellers [50, 40, 30]
        Zero surplus but trades theoretically possible at equilibrium.
        """
        buyers = [
            Lin(
                player_id=1,
                is_buyer=True,
                num_tokens=3,
                valuations=[50, 40, 30],
                num_buyers=1,
                num_sellers=1,
                num_times=20,
                seed=42,
            ),
        ]

        sellers = [
            ZIC(
                player_id=1,
                is_buyer=False,
                num_tokens=3,
                valuations=[50, 40, 30],
                seed=43,
            ),
        ]

        market = Market(
            num_buyers=1,
            num_sellers=1,
            num_times=20,
            price_min=0,
            price_max=100,
            buyers=buyers,
            sellers=sellers,
            seed=200,
        )

        # Run market
        for step in range(20):
            success = market.run_time_step()
            if not success:
                break

        # Count trades
        num_trades = sum(1 for t in range(1, 21) if market.orderbook.trade_price[t] > 0)

        print(f"\nLin No Trading Opportunities:")
        print(f"  Trades: {num_trades}")
        print(f"  Lin profit: {buyers[0].total_profit}")

        # Lin should not crash (profit might be 0)
        assert buyers[0].total_profit >= 0, "Lin negative profit"

    def test_perry_empty_traded_prices(self):
        """
        Test Perry when no trades occur (empty traded_prices array).

        Perry's statistical calculations should handle empty arrays gracefully.
        """
        agent = Perry(
            player_id=1,
            is_buyer=True,
            num_tokens=3,
            valuations=[100, 90, 80],
            seed=42,
        )

        # Simulate period with no trades
        agent.traded_prices = []
        agent.p_ave_price = 0
        agent.num_trades = 0
        agent.period_profit = 0

        # Evaluate should not crash
        try:
            agent._evaluate()
            print("\nPerry Empty Traded Prices: No crash ✅")
        except Exception as e:
            pytest.fail(f"Perry crashed with empty traded_prices: {e}")

        # Parameters should remain valid
        assert agent.a0 > 0, "a0 invalid after empty period"
        assert agent.price_std >= 0, "price_std invalid after empty period"
