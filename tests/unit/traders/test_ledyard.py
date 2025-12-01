"""
Unit tests for Ledyard (Easley-Ledyard-Olson) trader.

Tests the price band mechanics, time-progressive reservation prices,
market power adjustments, protocol compliance, and period lifecycle.
"""

import pytest

from traders.legacy.ledyard import (
    Ledyard,
)


@pytest.fixture
def buyer():
    """Standard Ledyard buyer for testing."""
    return Ledyard(
        player_id=1,
        is_buyer=True,
        num_tokens=4,
        valuations=[100, 90, 80, 70],
        price_min=0,
        price_max=200,
        num_times=100,
        num_buyers=4,
        num_sellers=4,
        seed=42,
    )


@pytest.fixture
def seller():
    """Standard Ledyard seller for testing."""
    return Ledyard(
        player_id=2,
        is_buyer=False,
        num_tokens=4,
        valuations=[30, 40, 50, 60],  # These are costs for seller
        price_min=0,
        price_max=200,
        num_times=100,
        num_buyers=4,
        num_sellers=4,
        seed=42,
    )


class TestPriceBandConstruction:
    """Test price band mechanics from previous period."""

    def test_initial_wide_band(self, buyer):
        """First period has wide band [price_min, price_max]."""
        assert buyer.prev_price_low == float(buyer.price_min)
        assert buyer.prev_price_high == float(buyer.price_max)

    def test_band_updates_from_trades(self, buyer):
        """Band updates from traded prices."""
        buyer.start_period(1)

        # Record trades
        buyer.current_traded_prices = [80, 90, 100]

        # Start next period to trigger band update
        buyer.start_period(2)

        # Floor: min(trades + asks) = min(80, 90, 100) = 80
        # Ceiling: max(trades + bids) = max(80, 90, 100) = 100
        assert buyer.prev_price_low == 80.0
        assert buyer.prev_price_high == 100.0

    def test_band_includes_asks_in_floor(self, buyer):
        """P'_n (floor) includes both trades and asks."""
        buyer.start_period(1)

        buyer.current_traded_prices = [90, 100]
        buyer.current_asks = [75, 85]  # Lower asks bring floor down

        buyer.start_period(2)

        # Floor: min(90, 100, 75, 85) = 75
        assert buyer.prev_price_low == 75.0

    def test_band_includes_bids_in_ceiling(self, buyer):
        """P_n (ceiling) includes both trades and bids."""
        buyer.start_period(1)

        buyer.current_traded_prices = [80, 90]
        buyer.current_bids = [95, 105]  # Higher bids raise ceiling

        buyer.start_period(2)

        # Ceiling: max(80, 90, 95, 105) = 105
        assert buyer.prev_price_high == 105.0

    def test_no_update_without_data(self, buyer):
        """Band stays at initial values if no observations."""
        initial_low = buyer.prev_price_low
        initial_high = buyer.prev_price_high

        buyer.start_period(1)
        # No trades, asks, or bids recorded
        buyer.start_period(2)

        assert buyer.prev_price_low == initial_low
        assert buyer.prev_price_high == initial_high


class TestTimePath:
    """Test time progression theta(t) computation."""

    def test_theta_zero_at_start(self, buyer):
        """theta(0) = 0 (conservative)."""
        theta = buyer._compute_theta(0.0)
        assert theta == 0.0

    def test_theta_one_at_end(self, buyer):
        """theta(1) approaches 1 (aggressive) for balanced market."""
        theta = buyer._compute_theta(1.0)
        # For balanced market (power_ratio=1), adjusted_theta = 1.0
        assert abs(theta - 1.0) < 0.01

    def test_theta_monotonic_increasing(self, buyer):
        """theta increases with time."""
        prev_theta = 0.0
        for t in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
            theta = buyer._compute_theta(t)
            assert theta >= prev_theta
            prev_theta = theta

    def test_theta_clamped_to_unit(self, buyer):
        """theta stays in [0, 1]."""
        for t in [0.0, 0.5, 1.0, 1.5]:
            theta = buyer._compute_theta(t)
            assert 0.0 <= theta <= 1.0


class TestMarketPowerAdjustment:
    """Test market power scaling of theta."""

    def test_crowded_buyer_concedes_faster(self):
        """More buyers than sellers -> buyer concedes faster."""
        # Crowded buyer (many buyers, few sellers)
        crowded = Ledyard(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
            num_times=100,
            num_buyers=8,  # Many buyers
            num_sellers=2,  # Few sellers
            seed=42,
        )

        # Balanced buyer
        balanced = Ledyard(
            player_id=2,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
            num_times=100,
            num_buyers=4,
            num_sellers=4,
            seed=42,
        )

        # At t=0.5, crowded buyer should have higher theta (concede faster)
        crowded_theta = crowded._compute_theta(0.5)
        balanced_theta = balanced._compute_theta(0.5)

        assert crowded_theta > balanced_theta

    def test_powerful_buyer_shades_more(self):
        """Fewer buyers than sellers -> buyer shades more (slower theta)."""
        # Powerful buyer (few buyers, many sellers)
        powerful = Ledyard(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
            num_times=100,
            num_buyers=2,  # Few buyers
            num_sellers=8,  # Many sellers
            seed=42,
        )

        # Balanced buyer
        balanced = Ledyard(
            player_id=2,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
            num_times=100,
            num_buyers=4,
            num_sellers=4,
            seed=42,
        )

        # At t=0.5, powerful buyer should have lower theta (shade more)
        powerful_theta = powerful._compute_theta(0.5)
        balanced_theta = balanced._compute_theta(0.5)

        assert powerful_theta < balanced_theta

    def test_seller_power_symmetric(self):
        """Seller market power is symmetric to buyer."""
        # Crowded seller (many sellers, few buyers)
        crowded_seller = Ledyard(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[30, 40, 50, 60],
            price_min=0,
            price_max=200,
            num_times=100,
            num_buyers=2,
            num_sellers=8,
            seed=42,
        )

        # Powerful seller (few sellers, many buyers)
        powerful_seller = Ledyard(
            player_id=2,
            is_buyer=False,
            num_tokens=4,
            valuations=[30, 40, 50, 60],
            price_min=0,
            price_max=200,
            num_times=100,
            num_buyers=8,
            num_sellers=2,
            seed=42,
        )

        # Crowded seller concedes faster
        crowded_theta = crowded_seller._compute_theta(0.5)
        powerful_theta = powerful_seller._compute_theta(0.5)

        assert crowded_theta > powerful_theta


class TestReservationPriceCalculation:
    """Test time-progressive reservation price computation."""

    def test_buyer_reservation_starts_low(self, buyer):
        """Buyer r_B starts at L_B (low end of band)."""
        buyer.prev_price_low = 60.0
        buyer.prev_price_high = 80.0
        buyer.current_time = 0

        value = 100
        r_B = buyer._compute_reservation_price(value)

        # At t=0, theta=0, so r_B = L_B = min(100, 60) = 60
        # Plus small noise
        assert 55 < r_B < 65

    def test_buyer_reservation_walks_up(self, buyer):
        """Buyer r_B increases toward U_B over time."""
        buyer.prev_price_low = 60.0
        buyer.prev_price_high = 80.0

        value = 100
        reservations = []
        for t in [0, 25, 50, 75, 100]:
            buyer.current_time = t
            r_B = buyer._compute_reservation_price(value)
            reservations.append(r_B)

        # Should generally increase (accounting for noise)
        # Check first < last
        assert reservations[0] < reservations[-1]

    def test_seller_reservation_starts_high(self, seller):
        """Seller r_S starts at U_S (high end of band)."""
        seller.prev_price_low = 60.0
        seller.prev_price_high = 80.0
        seller.current_time = 0

        cost = 30
        r_S = seller._compute_reservation_price(cost)

        # At t=0, theta=0, so r_S = U_S = max(30, 80) = 80
        # Plus small noise
        assert 75 < r_S < 85

    def test_seller_reservation_walks_down(self, seller):
        """Seller r_S decreases toward L_S over time."""
        seller.prev_price_low = 60.0
        seller.prev_price_high = 80.0

        cost = 30
        reservations = []
        for t in [0, 25, 50, 75, 100]:
            seller.current_time = t
            r_S = seller._compute_reservation_price(cost)
            reservations.append(r_S)

        # Should generally decrease (accounting for noise)
        # Check first > last
        assert reservations[0] > reservations[-1]

    def test_marginal_buyer_truth_tells(self, buyer):
        """Buyer with collapsed band (L_B >= U_B) truth-tells."""
        # Band where value is below both bounds
        buyer.prev_price_low = 120.0
        buyer.prev_price_high = 140.0
        buyer.current_time = 50

        value = 100  # Below band

        r_B = buyer._compute_reservation_price(value)

        # L_B = min(100, 120) = 100
        # U_B = min(100, 140) = 100
        # L_B >= U_B, so truth-tell: return value
        assert r_B == float(value)

    def test_marginal_seller_truth_tells(self, seller):
        """Seller with collapsed band (L_S >= U_S) truth-tells."""
        # Band where cost is above both bounds
        seller.prev_price_low = 10.0
        seller.prev_price_high = 20.0
        seller.current_time = 50

        cost = 30  # Above band

        r_S = seller._compute_reservation_price(cost)

        # L_S = max(30, 10) = 30
        # U_S = max(30, 20) = 30
        # L_S >= U_S, so truth-tell: return cost
        assert r_S == float(cost)


class TestNeverTradeAtLoss:
    """Test profit protection."""

    def test_buyer_rejects_ask_above_valuation(self, buyer):
        """Buyer never accepts ask >= valuation."""
        buyer.buy_sell(
            time=50,
            nobuysell=0,
            high_bid=80,
            low_ask=110,  # Above valuation of 100
            high_bidder=1,
            low_asker=2,
        )
        result = buyer.buy_sell_response()

        assert result is False

    def test_seller_rejects_bid_below_cost(self, seller):
        """Seller never accepts bid <= cost."""
        seller.buy_sell(
            time=50,
            nobuysell=0,
            high_bid=20,
            low_ask=80,  # Below cost of 30
            high_bidder=1,
            low_asker=2,
        )
        result = seller.buy_sell_response()

        assert result is False

    def test_buyer_rejects_at_value_trade(self, buyer):
        """No profit = no trade for buyer."""
        buyer.buy_sell(
            time=50,
            nobuysell=0,
            high_bid=80,
            low_ask=100,  # Exactly at valuation
            high_bidder=1,
            low_asker=2,
        )
        result = buyer.buy_sell_response()

        assert result is False

    def test_seller_rejects_at_cost_trade(self, seller):
        """No profit = no trade for seller."""
        seller.buy_sell(
            time=50,
            nobuysell=0,
            high_bid=30,
            low_ask=80,  # Exactly at cost
            high_bidder=1,
            low_asker=2,
        )
        result = seller.buy_sell_response()

        assert result is False


class TestProtocolCompliance:
    """Test AURORA protocol adherence."""

    def test_respects_nobidask_flag(self, buyer):
        """Returns 0 when nobidask > 0."""
        buyer.bid_ask(time=50, nobidask=1)
        bid = buyer.bid_ask_response()
        assert bid == 0

    def test_no_bid_ask_after_tokens_exhausted(self, buyer):
        """No bid/ask when all tokens traded."""
        buyer.num_trades = 4  # All tokens gone

        buyer.bid_ask(time=50, nobidask=0)
        bid = buyer.bid_ask_response()
        assert bid == 0

    def test_no_acceptance_after_tokens_exhausted(self, buyer):
        """No acceptance when all tokens traded."""
        buyer.num_trades = 4

        buyer.buy_sell(time=50, nobuysell=0, high_bid=80, low_ask=90, high_bidder=1, low_asker=2)
        result = buyer.buy_sell_response()
        assert result is False

    def test_must_be_winner_to_accept(self, buyer):
        """Buyer must be high_bidder, seller must be low_asker."""
        # Buyer is player 1, but high_bidder is 3
        buyer.buy_sell(time=50, nobuysell=0, high_bid=80, low_ask=90, high_bidder=3, low_asker=2)
        result = buyer.buy_sell_response()
        assert result is False

    def test_seller_must_be_low_asker(self, seller):
        """Seller must be low_asker to accept."""
        # Seller is player 2, but low_asker is 4
        seller.buy_sell(time=50, nobuysell=0, high_bid=80, low_ask=90, high_bidder=1, low_asker=4)
        result = seller.buy_sell_response()
        assert result is False


class TestEarlyPeriodBehavior:
    """Test conservative behavior early in period."""

    def test_buyer_anchors_near_band_floor(self, buyer):
        """Buyer without standing ask posts anchor near L_B."""
        buyer.prev_price_low = 60.0
        buyer.prev_price_high = 80.0
        buyer.current_high_bid = 0
        buyer.current_low_ask = 0

        buyer.bid_ask(time=20, nobidask=0)
        bid = buyer.bid_ask_response()

        # With no ask, buyer should anchor near L_B (60)
        if bid > 0:
            assert bid <= 80  # Should not exceed band ceiling

    def test_seller_anchors_near_band_ceiling(self, seller):
        """Seller without standing bid posts anchor near U_S."""
        seller.prev_price_low = 60.0
        seller.prev_price_high = 80.0
        seller.current_high_bid = 0
        seller.current_low_ask = 0

        seller.bid_ask(time=20, nobidask=0)
        ask = seller.bid_ask_response()

        # With no bid, seller should anchor near U_S (80)
        if ask > 0:
            assert ask >= 60  # Should not go below band floor


class TestLatePeriodOverride:
    """Test aggressive behavior late in period (>= 0.85)."""

    def test_late_buyer_bids_any_profitable_ask(self, buyer):
        """Late: bid any profitable ask, ignore reservation."""
        buyer.current_high_bid = 70
        buyer.current_low_ask = 90  # Below value of 100

        buyer.bid_ask(time=90, nobidask=0)  # >= 0.85 * 100
        bid = buyer.bid_ask_response()

        # Late override: should bid to capture profitable ask
        assert bid > 0

    def test_late_seller_asks_any_profitable_bid(self, seller):
        """Late: ask any profitable bid, ignore reservation."""
        seller.current_high_bid = 50  # Above cost of 30
        seller.current_low_ask = 100

        seller.bid_ask(time=90, nobidask=0)
        ask = seller.bid_ask_response()

        # Late override: should ask to capture profitable bid
        assert ask > 0

    def test_late_buyer_accepts_any_profitable_trade(self, buyer):
        """Late: accept any ask < value."""
        buyer.buy_sell(
            time=90,
            nobuysell=0,
            high_bid=80,
            low_ask=95,  # Profitable (95 < 100)
            high_bidder=1,
            low_asker=2,
        )
        result = buyer.buy_sell_response()

        assert result is True

    def test_late_seller_accepts_any_profitable_trade(self, seller):
        """Late: accept any bid > cost."""
        seller.buy_sell(
            time=90,
            nobuysell=0,
            high_bid=50,
            low_ask=80,  # Profitable (50 > 30)
            high_bidder=1,
            low_asker=2,
        )
        result = seller.buy_sell_response()

        assert result is True


class TestBSStepReservationBased:
    """Test buy/sell acceptance based on reservation price."""

    def test_buyer_accepts_ask_within_reservation(self, buyer):
        """Accept when current_ask <= r_B(t)."""
        buyer.prev_price_low = 60.0
        buyer.prev_price_high = 90.0
        buyer.current_time = 50  # Mid-period

        # At t=0.5, r_B should be around midpoint of band
        # L_B = min(100, 60) = 60
        # U_B = min(100, 90) = 90
        # r_B ~ 60 + 0.5*(90-60) = 75

        buyer.buy_sell(
            time=50,
            nobuysell=0,
            high_bid=80,
            low_ask=70,  # Low ask should be within reservation
            high_bidder=1,
            low_asker=2,
        )
        result = buyer.buy_sell_response()

        # Should accept: 70 <= ~75
        assert result is True

    def test_buyer_rejects_ask_above_reservation(self, buyer):
        """Reject when current_ask > r_B(t)."""
        buyer.prev_price_low = 60.0
        buyer.prev_price_high = 70.0  # Tight band
        buyer.current_time = 10  # Early period (low theta)

        # At t=0.1, r_B should be close to L_B (60)
        buyer.buy_sell(
            time=10,
            nobuysell=0,
            high_bid=80,
            low_ask=90,  # High ask, above reservation
            high_bidder=1,
            low_asker=2,
        )
        result = buyer.buy_sell_response()

        # Should reject: 90 > ~60
        assert result is False

    def test_seller_accepts_bid_within_reservation(self, seller):
        """Accept when current_bid >= r_S(t)."""
        seller.prev_price_low = 60.0
        seller.prev_price_high = 90.0
        seller.current_time = 80  # Late mid-period

        # At t=0.8, r_S should be close to L_S
        # L_S = max(30, 60) = 60
        # U_S = max(30, 90) = 90
        # r_S ~ 90 - 0.8*(90-60) = 66

        seller.buy_sell(
            time=80,
            nobuysell=0,
            high_bid=70,
            low_ask=100,  # Bid above reservation
            high_bidder=1,
            low_asker=2,
        )
        result = seller.buy_sell_response()

        # Should accept: 70 >= ~66
        assert result is True

    def test_seller_rejects_bid_below_reservation(self, seller):
        """Reject when current_bid < r_S(t)."""
        seller.prev_price_low = 60.0
        seller.prev_price_high = 90.0
        seller.current_time = 10  # Early period

        # At t=0.1, r_S should be close to U_S (90)
        seller.buy_sell(
            time=10,
            nobuysell=0,
            high_bid=50,
            low_ask=100,  # Low bid, below reservation
            high_bidder=1,
            low_asker=2,
        )
        result = seller.buy_sell_response()

        # Should reject: 50 < ~90
        assert result is False


class TestPeriodLifecycle:
    """Test period start/reset behavior."""

    def test_band_updates_on_start_period(self, buyer):
        """Price bounds update from previous period data."""
        buyer.current_traded_prices = [85, 90]
        buyer.current_asks = [80]
        buyer.current_bids = [95]

        buyer.start_period(2)

        assert buyer.prev_price_low == 80.0  # min(85, 90, 80)
        assert buyer.prev_price_high == 95.0  # max(85, 90, 95)

    def test_tracking_lists_reset_on_start_period(self, buyer):
        """Price tracking lists clear on new period."""
        buyer.current_traded_prices = [80, 90]
        buyer.current_asks = [75]
        buyer.current_bids = [95]

        buyer.start_period(2)

        assert buyer.current_traded_prices == []
        assert buyer.current_asks == []
        assert buyer.current_bids == []

    def test_market_state_reset_on_start_period(self, buyer):
        """Market state resets on new period."""
        buyer.current_high_bid = 80
        buyer.current_low_ask = 90
        buyer.current_time = 50

        buyer.start_period(2)

        assert buyer.current_high_bid == 0
        assert buyer.current_low_ask == 0
        assert buyer.current_time == 0

    def test_trade_count_reset_on_start_period(self, buyer):
        """Trade count resets on new period."""
        buyer.num_trades = 2

        buyer.start_period(2)

        assert buyer.num_trades == 0


class TestRoundLifecycle:
    """Test round start/reset behavior."""

    def test_wide_prior_on_start_round(self, buyer):
        """New round resets to wide band."""
        buyer.prev_price_low = 80.0
        buyer.prev_price_high = 90.0

        buyer.start_round([100, 90, 80, 70])

        assert buyer.prev_price_low == float(buyer.price_min)
        assert buyer.prev_price_high == float(buyer.price_max)

    def test_tracking_reset_on_start_round(self, buyer):
        """All tracking resets on new round."""
        buyer.current_traded_prices = [80]
        buyer.current_asks = [75]
        buyer.current_bids = [85]

        buyer.start_round([100, 90, 80, 70])

        assert buyer.current_traded_prices == []
        assert buyer.current_asks == []
        assert buyer.current_bids == []


class TestQuoteTracking:
    """Test that quotes are tracked correctly for band computation."""

    def test_tracks_new_bids(self, buyer):
        """New bids are recorded for band computation."""
        buyer.bid_ask_result(
            status=1,
            num_trades=0,
            new_bids=[75, 80],
            new_asks=[],
            high_bid=80,
            high_bidder=1,
            low_ask=0,
            low_asker=0,
        )

        assert 75 in buyer.current_bids
        assert 80 in buyer.current_bids

    def test_tracks_new_asks(self, buyer):
        """New asks are recorded for band computation."""
        buyer.bid_ask_result(
            status=1,
            num_trades=0,
            new_bids=[],
            new_asks=[90, 95],
            high_bid=0,
            high_bidder=0,
            low_ask=90,
            low_asker=2,
        )

        assert 90 in buyer.current_asks
        assert 95 in buyer.current_asks

    def test_tracks_traded_prices(self, buyer):
        """Trade prices are recorded for band computation."""
        buyer.buy_sell_result(
            status=1,
            trade_price=85,
            trade_type=1,  # Trade occurred
            high_bid=0,
            high_bidder=0,
            low_ask=0,
            low_asker=0,
        )

        assert 85 in buyer.current_traded_prices

    def test_ignores_zero_prices_in_tracking(self, buyer):
        """Zero prices are not tracked."""
        buyer.bid_ask_result(
            status=1,
            num_trades=0,
            new_bids=[0, 80],
            new_asks=[0, 90],
            high_bid=80,
            high_bidder=1,
            low_ask=90,
            low_asker=2,
        )

        assert 0 not in buyer.current_bids
        assert 0 not in buyer.current_asks


class TestTimeFraction:
    """Test time fraction calculation."""

    def test_time_fraction_early(self, buyer):
        """Early period time fraction."""
        buyer.current_time = 20
        assert buyer._time_fraction() == 0.2

    def test_time_fraction_late(self, buyer):
        """Late period time fraction."""
        buyer.current_time = 90
        assert buyer._time_fraction() == 0.9

    def test_is_late_threshold(self, buyer):
        """Late threshold at 0.85."""
        buyer.current_time = 84
        assert buyer._is_late() is False

        buyer.current_time = 85
        assert buyer._is_late() is True


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_buyer_with_zero_num_times(self):
        """Handle num_times=0 gracefully."""
        buyer = Ledyard(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
            num_times=0,
            num_buyers=4,
            num_sellers=4,
        )
        # Should be in "late" mode since time_fraction returns 1.0
        assert buyer._time_fraction() == 1.0
        assert buyer._is_late() is True

    def test_empty_market_buyer_response(self, buyer):
        """Buyer in empty market (no standing bid/ask)."""
        buyer.current_high_bid = 0
        buyer.current_low_ask = 0

        buyer.bid_ask(time=30, nobidask=0)
        bid = buyer.bid_ask_response()

        # Should handle gracefully
        assert bid >= 0

    def test_price_bounds_respected(self, buyer):
        """Bids stay within [price_min, price_max]."""
        buyer.current_high_bid = 190
        buyer.current_low_ask = 195

        buyer.bid_ask(time=60, nobidask=0)
        bid = buyer.bid_ask_response()

        if bid > 0:
            assert bid >= buyer.price_min
            assert bid <= buyer.price_max

    def test_zero_sellers_handled(self):
        """Handle edge case of zero sellers."""
        buyer = Ledyard(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
            num_times=100,
            num_buyers=4,
            num_sellers=0,  # Edge case
            seed=42,
        )
        # Should not crash
        theta = buyer._compute_theta(0.5)
        assert theta >= 0

    def test_zero_buyers_handled(self):
        """Handle edge case of zero buyers."""
        seller = Ledyard(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[30, 40, 50, 60],
            price_min=0,
            price_max=200,
            num_times=100,
            num_buyers=0,  # Edge case
            num_sellers=4,
            seed=42,
        )
        # Should not crash
        theta = seller._compute_theta(0.5)
        assert theta >= 0


class TestTradeCountProperty:
    """Test compatibility property."""

    def test_trade_count_returns_num_trades(self, buyer):
        """trade_count property returns num_trades."""
        buyer.num_trades = 3
        assert buyer.trade_count == 3
