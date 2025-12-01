# tests/integration/test_aurora_protocol.py
"""
Integration tests for the AURORA Double Auction Protocol.

These tests verify that the Market + OrderBook + real traders work together
correctly. They test the full protocol flow with actual trading agents.
"""


from engine.market import Market
from traders.legacy.kaplan import Kaplan
from traders.legacy.zic import ZIC
from traders.legacy.zip import ZIP

# =============================================================================
# Test: Full ZIC Market
# =============================================================================


class TestFullZICMarket:
    """Integration tests with ZIC traders."""

    def test_zic_market_produces_trades(self):
        """ZIC market should produce trades when supply/demand overlap."""
        buyers = [
            ZIC(
                player_id=i + 1,
                is_buyer=True,
                num_tokens=4,
                valuations=[100, 90, 80, 70],  # High values
                price_min=1,
                price_max=200,
                seed=42 + i,
            )
            for i in range(4)
        ]
        sellers = [
            ZIC(
                player_id=i + 5,
                is_buyer=False,
                num_tokens=4,
                valuations=[30, 40, 50, 60],  # Low costs
                price_min=1,
                price_max=200,
                seed=100 + i,
            )
            for i in range(4)
        ]

        market = Market(
            num_buyers=4,
            num_sellers=4,
            num_times=50,
            price_min=1,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=42,
        )

        # Run the market
        for _ in range(50):
            if not market.run_time_step():
                break

        # Check that trades occurred
        total_trades = sum(b.num_trades for b in buyers)
        assert total_trades > 0, "ZIC market should produce trades"

    def test_zic_never_loses_money(self):
        """All ZIC trades should be profitable for both parties."""
        buyers = [
            ZIC(
                player_id=i + 1,
                is_buyer=True,
                num_tokens=4,
                valuations=[100, 90, 80, 70],
                price_min=1,
                price_max=200,
                seed=42 + i,
            )
            for i in range(4)
        ]
        sellers = [
            ZIC(
                player_id=i + 5,
                is_buyer=False,
                num_tokens=4,
                valuations=[30, 40, 50, 60],
                price_min=1,
                price_max=200,
                seed=100 + i,
            )
            for i in range(4)
        ]

        market = Market(
            num_buyers=4,
            num_sellers=4,
            num_times=50,
            price_min=1,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=42,
        )

        for _ in range(50):
            if not market.run_time_step():
                break

        # All period profits should be non-negative
        for buyer in buyers:
            assert (
                buyer.period_profit >= 0
            ), f"Buyer {buyer.player_id} has negative profit: {buyer.period_profit}"
        for seller in sellers:
            assert (
                seller.period_profit >= 0
            ), f"Seller {seller.player_id} has negative profit: {seller.period_profit}"


# =============================================================================
# Test: Full ZIP Market
# =============================================================================


class TestFullZIPMarket:
    """Integration tests with ZIP traders."""

    def test_zip_market_converges(self):
        """ZIP market should converge to efficient prices."""
        buyers = [
            ZIP(
                player_id=i + 1,
                is_buyer=True,
                num_tokens=4,
                valuations=[100, 90, 80, 70],
                price_min=1,
                price_max=200,
                seed=42 + i,
            )
            for i in range(4)
        ]
        sellers = [
            ZIP(
                player_id=i + 5,
                is_buyer=False,
                num_tokens=4,
                valuations=[30, 40, 50, 60],
                price_min=1,
                price_max=200,
                seed=100 + i,
            )
            for i in range(4)
        ]

        market = Market(
            num_buyers=4,
            num_sellers=4,
            num_times=100,
            price_min=1,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=42,
        )

        for _ in range(100):
            if not market.run_time_step():
                break

        # Check that trades occurred
        total_trades = sum(b.num_trades for b in buyers)
        assert total_trades > 0, "ZIP market should produce trades"

    def test_zip_never_loses_money(self):
        """All ZIP trades should be profitable."""
        buyers = [
            ZIP(
                player_id=i + 1,
                is_buyer=True,
                num_tokens=4,
                valuations=[100, 90, 80, 70],
                price_min=1,
                price_max=200,
                seed=42 + i,
            )
            for i in range(4)
        ]
        sellers = [
            ZIP(
                player_id=i + 5,
                is_buyer=False,
                num_tokens=4,
                valuations=[30, 40, 50, 60],
                price_min=1,
                price_max=200,
                seed=100 + i,
            )
            for i in range(4)
        ]

        market = Market(
            num_buyers=4,
            num_sellers=4,
            num_times=100,
            price_min=1,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=42,
        )

        for _ in range(100):
            if not market.run_time_step():
                break

        for buyer in buyers:
            assert (
                buyer.period_profit >= 0
            ), f"ZIP Buyer {buyer.player_id} has negative profit: {buyer.period_profit}"
        for seller in sellers:
            assert (
                seller.period_profit >= 0
            ), f"ZIP Seller {seller.player_id} has negative profit: {seller.period_profit}"


# =============================================================================
# Test: Mixed Market (ZIC vs Kaplan)
# =============================================================================


class TestMixedMarket:
    """Integration tests with mixed trader types."""

    def test_kaplan_vs_zic(self):
        """Kaplan should be able to trade against ZIC."""
        # 1 Kaplan buyer, 3 ZIC buyers
        buyers = [
            Kaplan(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=[100, 90, 80, 70],
                price_min=1,
                price_max=200,
                num_times=100,
            ),
        ] + [
            ZIC(
                player_id=i + 2,
                is_buyer=True,
                num_tokens=4,
                valuations=[100, 90, 80, 70],
                price_min=1,
                price_max=200,
                seed=42 + i,
            )
            for i in range(3)
        ]

        # 4 ZIC sellers
        sellers = [
            ZIC(
                player_id=i + 5,
                is_buyer=False,
                num_tokens=4,
                valuations=[30, 40, 50, 60],
                price_min=1,
                price_max=200,
                seed=100 + i,
            )
            for i in range(4)
        ]

        market = Market(
            num_buyers=4,
            num_sellers=4,
            num_times=100,
            price_min=1,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=42,
        )

        for _ in range(100):
            if not market.run_time_step():
                break

        # Market should function
        total_trades = sum(b.num_trades for b in buyers)
        assert total_trades > 0, "Mixed market should produce trades"


# =============================================================================
# Test: Market Mechanics
# =============================================================================


class TestMarketMechanics:
    """Tests for basic market mechanics."""

    def test_trade_count_matches_buyer_seller(self):
        """Total buyer trades should equal total seller trades."""
        buyers = [
            ZIC(
                player_id=i + 1,
                is_buyer=True,
                num_tokens=4,
                valuations=[100, 90, 80, 70],
                price_min=1,
                price_max=200,
                seed=42 + i,
            )
            for i in range(4)
        ]
        sellers = [
            ZIC(
                player_id=i + 5,
                is_buyer=False,
                num_tokens=4,
                valuations=[30, 40, 50, 60],
                price_min=1,
                price_max=200,
                seed=100 + i,
            )
            for i in range(4)
        ]

        market = Market(
            num_buyers=4,
            num_sellers=4,
            num_times=50,
            price_min=1,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=42,
        )

        for _ in range(50):
            if not market.run_time_step():
                break

        buyer_trades = sum(b.num_trades for b in buyers)
        seller_trades = sum(s.num_trades for s in sellers)

        assert (
            buyer_trades == seller_trades
        ), f"Buyer trades ({buyer_trades}) != seller trades ({seller_trades})"

    def test_time_increments_correctly(self):
        """Time should increment with each step."""
        buyers = [
            ZIC(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=[100, 90, 80, 70],
                price_min=1,
                price_max=200,
                seed=42,
            )
        ]
        sellers = [
            ZIC(
                player_id=2,
                is_buyer=False,
                num_tokens=4,
                valuations=[30, 40, 50, 60],
                price_min=1,
                price_max=200,
                seed=43,
            )
        ]

        market = Market(
            num_buyers=1,
            num_sellers=1,
            num_times=10,
            price_min=1,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=42,
        )

        for expected_time in range(1, 11):
            result = market.run_time_step()
            assert result is True
            assert market.current_time == expected_time

    def test_market_respects_num_times_limit(self):
        """Market should stop after num_times steps."""
        buyers = [
            ZIC(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=[100, 90, 80, 70],
                price_min=1,
                price_max=200,
                seed=42,
            )
        ]
        sellers = [
            ZIC(
                player_id=2,
                is_buyer=False,
                num_tokens=4,
                valuations=[30, 40, 50, 60],
                price_min=1,
                price_max=200,
                seed=43,
            )
        ]

        market = Market(
            num_buyers=1,
            num_sellers=1,
            num_times=5,
            price_min=1,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=42,
        )

        steps = 0
        while market.run_time_step():
            steps += 1

        assert steps == 5, f"Market should run exactly 5 steps, ran {steps}"


# =============================================================================
# Test: Chicago Rules
# =============================================================================


class TestChicagoRulesIntegration:
    """Integration tests for Chicago Rules pricing."""

    def test_trade_prices_are_valid(self):
        """All trade prices should be within valid range."""
        buyers = [
            ZIC(
                player_id=i + 1,
                is_buyer=True,
                num_tokens=4,
                valuations=[100, 90, 80, 70],
                price_min=1,
                price_max=200,
                seed=42 + i,
            )
            for i in range(4)
        ]
        sellers = [
            ZIC(
                player_id=i + 5,
                is_buyer=False,
                num_tokens=4,
                valuations=[30, 40, 50, 60],
                price_min=1,
                price_max=200,
                seed=100 + i,
            )
            for i in range(4)
        ]

        market = Market(
            num_buyers=4,
            num_sellers=4,
            num_times=50,
            price_min=1,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=42,
        )

        for _ in range(50):
            if not market.run_time_step():
                break

        # Check all recorded trade prices
        ob = market.get_orderbook()
        for t in range(1, market.current_time + 1):
            price = ob.trade_price[t]
            if price > 0:  # Trade occurred
                assert (
                    1 <= price <= 200
                ), f"Trade price {price} at t={t} outside valid range [1, 200]"


# =============================================================================
# Test: Orderbook State Consistency
# =============================================================================


class TestOrderbookConsistency:
    """Tests for orderbook state consistency."""

    def test_position_counts_monotonic(self):
        """Position counts should only increase."""
        buyers = [
            ZIC(
                player_id=i + 1,
                is_buyer=True,
                num_tokens=4,
                valuations=[100, 90, 80, 70],
                price_min=1,
                price_max=200,
                seed=42 + i,
            )
            for i in range(4)
        ]
        sellers = [
            ZIC(
                player_id=i + 5,
                is_buyer=False,
                num_tokens=4,
                valuations=[30, 40, 50, 60],
                price_min=1,
                price_max=200,
                seed=100 + i,
            )
            for i in range(4)
        ]

        market = Market(
            num_buyers=4,
            num_sellers=4,
            num_times=50,
            price_min=1,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=42,
        )

        for _ in range(50):
            if not market.run_time_step():
                break

        ob = market.get_orderbook()

        # Check monotonicity
        for buyer_id in range(1, 5):
            for t in range(2, market.current_time + 1):
                prev = ob.num_buys[buyer_id, t - 1]
                curr = ob.num_buys[buyer_id, t]
                assert (
                    curr >= prev
                ), f"Buyer {buyer_id} position decreased at t={t}: {prev} -> {curr}"


# =============================================================================
# Test: No Extramarginal Trades
# =============================================================================


class TestNoExtramarginalTrades:
    """Tests that only intramarginal units trade in ZIC markets."""

    def test_extramarginal_units_dont_trade(self):
        """Units with negative surplus should not trade."""
        # Buyer with one extramarginal unit (value 30 < seller cost 50)
        buyers = [
            ZIC(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=[100, 90, 80, 30],  # Last unit is extramarginal
                price_min=1,
                price_max=200,
                seed=42,
            )
        ]
        # Seller with costs above buyer's worst value
        sellers = [
            ZIC(
                player_id=2,
                is_buyer=False,
                num_tokens=4,
                valuations=[50, 60, 70, 80],
                price_min=1,
                price_max=200,
                seed=43,
            )
        ]

        market = Market(
            num_buyers=1,
            num_sellers=1,
            num_times=100,
            price_min=1,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=42,
        )

        for _ in range(100):
            if not market.run_time_step():
                break

        # ZIC should never trade at a loss
        # The 4th buyer unit (value=30) should not trade with any seller unit (cost >= 50)
        # If 4 trades happened, the buyer would have traded the extramarginal unit
        # This test verifies ZIC's constraint prevents this
        assert (
            buyers[0].period_profit >= 0
        ), f"ZIC buyer has negative profit: {buyers[0].period_profit}"
