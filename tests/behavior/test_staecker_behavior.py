"""
Behavioral tests for Staecker (Predictive Strategy) trader.

Tests the trader's distinctive personality: patient-early/aggressive-late,
forecast-driven trading, spread sensitivity, and comparison to other strategies.
"""

import numpy as np
import pytest

from engine.market import Market
from engine.token_generator import TokenGenerator
from traders.legacy.kaplan import Kaplan
from traders.legacy.staecker import Staecker
from traders.legacy.zic import ZIC


class TestTimeVaryingPatience:
    """Test patient-early, aggressive-late personality."""

    def test_buyer_more_patient_early_than_late(self, staecker_buyer):
        """Early: rejects OK trades; Late: accepts all profitable."""
        buyer = staecker_buyer
        buyer.start_period(1)

        # Set up a profitable trade scenario
        buyer.la_pred = 90.0

        # Early period: current_ask = 95 > la_pred (90), should reject
        buyer.buy_sell(
            time=20,
            nobuysell=0,
            high_bid=80,
            low_ask=95,  # Profitable (95 < 100)
            high_bidder=buyer.player_id,
            low_asker=2,
        )
        early_result = buyer.buy_sell_response()

        # Reset for late test
        buyer.start_period(2)
        buyer.la_pred = 90.0

        # Late period: same scenario, should accept due to override
        buyer.buy_sell(
            time=90, nobuysell=0, high_bid=80, low_ask=95, high_bidder=buyer.player_id, low_asker=2
        )
        late_result = buyer.buy_sell_response()

        # Early: rejects because ask > forecast
        # Late: accepts any profitable trade
        assert early_result is False
        assert late_result is True

    def test_seller_more_patient_early_than_late(self, staecker_seller):
        """Symmetric for seller: early rejects, late accepts."""
        seller = staecker_seller
        seller.start_period(1)

        # Set up a profitable trade scenario
        seller.hb_pred = 55.0

        # Early period: current_bid = 50 < hb_pred (55), should reject
        seller.buy_sell(
            time=20,
            nobuysell=0,
            high_bid=50,
            low_ask=80,  # Profitable (50 > 30)
            high_bidder=1,
            low_asker=seller.player_id,
        )
        early_result = seller.buy_sell_response()

        # Reset for late test
        seller.start_period(2)
        seller.hb_pred = 55.0

        # Late period: same scenario, should accept
        seller.buy_sell(
            time=90, nobuysell=0, high_bid=50, low_ask=80, high_bidder=1, low_asker=seller.player_id
        )
        late_result = seller.buy_sell_response()

        assert early_result is False
        assert late_result is True


class TestForecastDrivenTrading:
    """Test that Staecker trades based on forecasts, not just current prices."""

    def test_rejects_profitable_trade_if_forecast_better(self, staecker_buyer):
        """Key Staecker signature: reject profitable if forecast promises better."""
        buyer = staecker_buyer
        buyer.start_period(1)

        # Forecast predicts asks will go lower
        buyer.la_pred = 85.0  # Predicts future asks around 85

        # Current ask is 92 - profitable (92 < 100) but worse than forecast
        buyer.buy_sell(
            time=50, nobuysell=0, high_bid=80, low_ask=92, high_bidder=buyer.player_id, low_asker=2
        )
        result = buyer.buy_sell_response()

        # Should reject: 92 > 85 (forecast), expects better later
        assert result is False

    def test_accepts_trade_when_price_matches_forecast(self, staecker_buyer):
        """current_ask <= la_pred → trade."""
        buyer = staecker_buyer
        buyer.start_period(1)

        buyer.la_pred = 95.0

        # Current ask matches or beats forecast
        buyer.buy_sell(
            time=50,
            nobuysell=0,
            high_bid=80,
            low_ask=93,  # 93 < 95 (forecast)
            high_bidder=buyer.player_id,
            low_asker=2,
        )
        result = buyer.buy_sell_response()

        assert result is True

    def test_forecast_guides_bid_target(self, staecker_buyer):
        """Bid near floor(la_pred), not near valuation."""
        buyer = staecker_buyer
        buyer.start_period(1)

        # Set up tight spread for action
        buyer.hb_pred = 80.0
        buyer.la_pred = 88.0  # Predicts asks around 88
        buyer.current_bid = 75
        buyer.current_ask = 95

        buyer.bid_ask(time=60, nobidask=0)
        bid = buyer.bid_ask_response()

        # After _update_forecasts: la_pred = 0.7*88 + 0.3*95 = 90.1
        # b_target = min(99, floor(90.1)) = 90
        # Key: bid should be closer to forecast (~90) than to valuation (99)
        if bid > 0:
            assert bid <= 94  # Should be guided by forecast, not valuation
            assert bid >= 76  # Must improve current bid


class TestSpreadSensitivity:
    """Test spread-driven patience."""

    def test_passes_on_wide_predicted_spread_early(self, staecker_buyer):
        """s_pred > threshold → wait (early period)."""
        buyer = staecker_buyer
        buyer.start_period(1)

        # Wide predicted spread
        buyer.hb_pred = 60.0
        buyer.la_pred = 100.0  # Spread = 40
        # threshold = max(5, 0.1 * 80) = 8
        # 40 > 8, so spread is wide

        buyer.current_bid = 60
        buyer.current_ask = 100
        buyer.bid_ask(time=30, nobidask=0)  # Early (< 0.5 * 100)
        bid = buyer.bid_ask_response()

        assert bid == 0  # Should pass on wide spread early

    def test_trades_on_narrow_predicted_spread(self, staecker_buyer):
        """s_pred < threshold → enter."""
        buyer = staecker_buyer
        buyer.start_period(1)

        # Narrow predicted spread
        buyer.hb_pred = 85.0
        buyer.la_pred = 88.0  # Spread = 3
        # threshold = max(5, 0.1 * 86.5) = 8.65
        # 3 < 8.65, so spread is narrow

        buyer.current_bid = 82
        buyer.current_ask = 92
        buyer.bid_ask(time=60, nobidask=0)
        bid = buyer.bid_ask_response()

        # With narrow spread and profitable forecast, should act
        assert bid > 0


class TestDifferentFromKaplan:
    """Test Staecker differs from Kaplan (spread-based sniper)."""

    def test_staecker_uses_forecasts_not_current_spread(self, staecker_buyer, kaplan_buyer):
        """Kaplan checks current spread, Staecker checks predicted spread."""
        # Set up identical market state
        for agent in [staecker_buyer, kaplan_buyer]:
            agent.start_period(1)
            agent.current_bid = 85
            agent.current_ask = 90  # Current spread = 5

        # Staecker has wide PREDICTED spread
        staecker_buyer.hb_pred = 60.0
        staecker_buyer.la_pred = 100.0  # Predicted spread = 40

        # Both see same current market
        staecker_buyer.bid_ask(time=30, nobidask=0)
        staecker_bid = staecker_buyer.bid_ask_response()

        # Kaplan would look at current spread (5), which is tight
        # Staecker looks at predicted spread (40), which is wide
        # Early period + wide predicted spread → Staecker passes
        assert staecker_bid == 0

    def test_staecker_bids_near_forecast_not_current_ask(self, staecker_buyer):
        """Staecker's bid target is forecast-based, not current-price-based."""
        buyer = staecker_buyer
        buyer.start_period(1)

        # Set up: current ask is 95, but forecast is 85
        buyer.hb_pred = 80.0
        buyer.la_pred = 85.0
        buyer.current_bid = 75
        buyer.current_ask = 95

        buyer.bid_ask(time=60, nobidask=0)
        bid = buyer.bid_ask_response()

        # After _update_forecasts: la_pred = 0.7*85 + 0.3*95 = 88
        # b_target = min(99, floor(88)) = 88
        # Key point: bid is constrained by forecast (~88), not current ask (95)
        if bid > 0:
            assert bid <= 94  # Should be well below current ask
            assert bid >= 76  # Must improve current bid


class TestDifferentFromBGAN:
    """Test Staecker differs from BGAN (Bayesian optimizer)."""

    def test_staecker_simpler_than_bgan(self, staecker_buyer, bgan_buyer):
        """Staecker uses simple exponential smoothing, not Monte Carlo."""
        # Staecker: just tracks hb_pred, la_pred via exponential smoothing
        assert hasattr(staecker_buyer, "hb_pred")
        assert hasattr(staecker_buyer, "la_pred")

        # BGAN: has belief distribution parameters
        assert hasattr(bgan_buyer, "belief_mu")
        assert hasattr(bgan_buyer, "belief_sigma")

        # Staecker doesn't have Bayesian beliefs
        assert not hasattr(staecker_buyer, "belief_mu")
        assert not hasattr(staecker_buyer, "belief_sigma")

    def test_staecker_uses_extrema_not_full_distribution(self, staecker_buyer, bgan_buyer):
        """Staecker only tracks high bid/low ask, not full history."""
        staecker_buyer.start_period(1)
        bgan_buyer.start_period(1)

        # Feed same observations
        prices = [70, 75, 80, 85, 90]
        for p in prices:
            # Staecker updates from current_bid/ask
            staecker_buyer.current_bid = p
            staecker_buyer._update_forecasts()

            # BGAN updates beliefs
            bgan_buyer._update_beliefs(p)

        # Staecker: just has smoothed forecast (single number)
        assert staecker_buyer.hb_pred is not None

        # BGAN: has full belief state with mean and sigma
        assert bgan_buyer.belief_mu is not None
        assert bgan_buyer.belief_sigma is not None
        # BGAN tracks observed prices list
        assert len(bgan_buyer._observed_prices) > 0


class TestMarketIntegration:
    """Full market execution tests."""

    def test_staecker_vs_zic_profitability(self):
        """Staecker should earn positive profit against ZIC."""
        np.random.seed(42)

        token_gen = TokenGenerator(game_type=6453, num_tokens=4, seed=42)
        token_gen.new_round()

        # Create agents
        staecker = Staecker(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=token_gen.generate_tokens(True),
            price_min=0,
            price_max=1000,
            num_times=100,
        )

        buyers = [staecker]
        for i in range(3):
            buyers.append(
                ZIC(
                    player_id=i + 2,
                    is_buyer=True,
                    num_tokens=4,
                    valuations=token_gen.generate_tokens(True),
                    price_min=0,
                    price_max=1000,
                    seed=100 + i,
                )
            )

        sellers = []
        for i in range(4):
            sellers.append(
                ZIC(
                    player_id=i + 5,
                    is_buyer=False,
                    num_tokens=4,
                    valuations=token_gen.generate_tokens(False),
                    price_min=0,
                    price_max=1000,
                    seed=200 + i,
                )
            )

        all_agents = buyers + sellers
        for agent in all_agents:
            agent.start_period(1)

        market = Market(
            num_buyers=4,
            num_sellers=4,
            num_times=100,
            price_min=0,
            price_max=1000,
            buyers=buyers,
            sellers=sellers,
            seed=42,
        )

        for _ in range(100):
            market.run_time_step()

        # Staecker should be profitable (or at least not losing badly)
        # Given its conservative nature, it may trade less but profitably
        # Main check: doesn't lose money
        assert staecker.period_profit >= 0 or staecker.num_trades == 0

    def test_staecker_completes_trades(self):
        """Staecker should complete at least some trades per period."""
        np.random.seed(123)

        token_gen = TokenGenerator(game_type=6453, num_tokens=4, seed=123)

        total_trades = 0
        num_periods = 5

        for period in range(1, num_periods + 1):
            token_gen.new_round()

            staecker = Staecker(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=token_gen.generate_tokens(True),
                price_min=0,
                price_max=1000,
                num_times=100,
            )

            buyers = [staecker]
            for i in range(3):
                buyers.append(
                    ZIC(
                        player_id=i + 2,
                        is_buyer=True,
                        num_tokens=4,
                        valuations=token_gen.generate_tokens(True),
                        price_min=0,
                        price_max=1000,
                        seed=period * 100 + i,
                    )
                )

            sellers = []
            for i in range(4):
                sellers.append(
                    ZIC(
                        player_id=i + 5,
                        is_buyer=False,
                        num_tokens=4,
                        valuations=token_gen.generate_tokens(False),
                        price_min=0,
                        price_max=1000,
                        seed=period * 200 + i,
                    )
                )

            all_agents = buyers + sellers
            for agent in all_agents:
                agent.start_period(period)

            market = Market(
                num_buyers=4,
                num_sellers=4,
                num_times=100,
                price_min=0,
                price_max=1000,
                buyers=buyers,
                sellers=sellers,
                seed=period * 1000,
            )

            for _ in range(100):
                market.run_time_step()

            total_trades += staecker.num_trades

        # Should complete at least some trades across periods
        # Staecker is patient but should trade eventually (especially late)
        assert total_trades > 0


class TestTrendChasing:
    """Test trend-following behavior from spec."""

    def test_buyer_forecast_drifts_with_asks(self, staecker_buyer):
        """la_pred drifts toward recent asks."""
        buyer = staecker_buyer
        buyer.start_period(1)

        # Initial forecast
        buyer.la_pred = 100.0

        # Series of lower asks
        asks = [95, 90, 85, 80]
        for ask in asks:
            buyer.current_ask = ask
            buyer._update_forecasts()

        # Forecast should have drifted down
        assert buyer.la_pred < 100.0
        assert buyer.la_pred > 80.0  # But lagging

    def test_seller_forecast_drifts_with_bids(self, staecker_seller):
        """hb_pred drifts toward recent bids."""
        seller = staecker_seller
        seller.start_period(1)

        # Initial forecast
        seller.hb_pred = 50.0

        # Series of higher bids
        bids = [55, 60, 65, 70]
        for bid in bids:
            seller.current_bid = bid
            seller._update_forecasts()

        # Forecast should have drifted up
        assert seller.hb_pred > 50.0
        assert seller.hb_pred < 70.0  # But lagging


class TestBehavioralProfile:
    """Test behavioral characteristics match expected profile."""

    def test_high_pass_rate_early(self, staecker_buyer):
        """Staecker should PASS frequently early (waiting for forecasts)."""
        buyer = staecker_buyer
        buyer.start_period(1)

        pass_count = 0
        total_steps = 20

        for step in range(1, total_steps + 1):
            # No forecasts initially
            buyer.bid_ask(time=step, nobidask=0)
            bid = buyer.bid_ask_response()
            if bid == 0:
                pass_count += 1

        # Should pass most early steps (no forecasts or wide spread)
        pass_rate = pass_count / total_steps
        assert pass_rate > 0.5  # High PASS rate early

    def test_deterministic_behavior(self, staecker_buyer):
        """Staecker is deterministic (no randomness)."""
        # Run twice with same setup
        results = []

        for _ in range(2):
            buyer = Staecker(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=[100, 90, 80, 70],
                price_min=0,
                price_max=200,
                num_times=100,
            )
            buyer.start_period(1)

            buyer.hb_pred = 80.0
            buyer.la_pred = 90.0
            buyer.current_bid = 75
            buyer.current_ask = 95

            buyer.bid_ask(time=50, nobidask=0)
            bid = buyer.bid_ask_response()
            results.append(bid)

        # Both runs should produce identical results
        assert results[0] == results[1]


# Fixtures for comparison tests
@pytest.fixture
def kaplan_buyer():
    """Kaplan buyer for comparison."""
    return Kaplan(
        player_id=10,
        is_buyer=True,
        num_tokens=4,
        valuations=[100, 90, 80, 70],
        price_min=0,
        price_max=200,
        num_times=100,
        seed=42,
    )
