# tests/unit/traders/test_bgan.py
"""
Adversarial tests for BGAN (Bayesian Game Against Nature) agent.

BGAN is a belief-based optimizer from the Kennet-Friedman entry in the 1993
Santa Fe tournament. It:
1. Maintains Bayesian beliefs over opponent price distributions (Normal)
2. Computes reservation prices via Monte Carlo simulation
3. Uses time-varying patience (patient early, aggressive late)

These tests verify:
1. Never trades at a loss
2. Belief update mechanics (online mean, outlier filtering)
3. Arrival model (m(t) = m0 * (1-t))
4. Reservation price behavior (early vs late period)
5. BA/BS step logic respects reservation price
"""

import numpy as np
import pytest

from traders.legacy.bgan import BGAN, DEFAULT_M0

# =============================================================================
# Test: Never Trade at a Loss
# =============================================================================


class TestNeverTradeAtLoss:
    """Tests that BGAN never makes unprofitable trades."""

    def test_buyer_rejects_ask_above_valuation(self):
        """Buyer should reject if ask price exceeds valuation."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        # Set up: we're high bidder, but ask exceeds our valuation
        buyer.buy_sell(
            time=50,
            nobuysell=0,
            high_bid=95,
            low_ask=105,  # Above valuation of 100!
            high_bidder=1,
            low_asker=2,
        )

        result = buyer.buy_sell_response()
        assert result is False, "Buyer should reject ask above valuation"

    def test_seller_rejects_bid_below_cost(self):
        """Seller should reject if bid price is below cost."""
        seller = BGAN(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        # Set up: we're low asker, but bid is below our cost
        seller.buy_sell(
            time=50,
            nobuysell=0,
            high_bid=45,  # Below cost of 50!
            low_ask=55,
            high_bidder=2,
            low_asker=1,
        )

        result = seller.buy_sell_response()
        assert result is False, "Seller should reject bid below cost"

    def test_buyer_rejects_at_valuation_exactly(self):
        """Buyer should reject if ask equals valuation (no profit)."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        buyer.buy_sell(
            time=50,
            nobuysell=0,
            high_bid=95,
            low_ask=100,  # Equals valuation - NO PROFIT
            high_bidder=1,
            low_asker=2,
        )

        result = buyer.buy_sell_response()
        assert result is False, "Buyer should reject when ask == valuation (no profit)"

    def test_seller_rejects_at_cost_exactly(self):
        """Seller should reject if bid equals cost (no profit)."""
        seller = BGAN(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        seller.buy_sell(
            time=50,
            nobuysell=0,
            high_bid=50,  # Equals cost - NO PROFIT
            low_ask=55,
            high_bidder=2,
            low_asker=1,
        )

        result = seller.buy_sell_response()
        assert result is False, "Seller should reject when bid == cost (no profit)"


# =============================================================================
# Test: Protocol Compliance (nobidask, nobuysell, tokens)
# =============================================================================


class TestProtocolCompliance:
    """Tests for AURORA protocol compliance."""

    def test_respects_nobidask_flag(self):
        """Should return 0 when nobidask > 0."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        buyer.bid_ask(time=10, nobidask=1)  # nobidask=1 means restricted
        result = buyer.bid_ask_response()
        assert result == 0, "Should return 0 when nobidask > 0"

    def test_no_bid_after_tokens_exhausted(self):
        """Should return 0 when all tokens traded."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        # Exhaust all tokens
        buyer.num_trades = 4

        buyer.bid_ask(time=10, nobidask=0)
        result = buyer.bid_ask_response()
        assert result == 0, "Should return 0 when all tokens traded"

    def test_no_accept_after_tokens_exhausted(self):
        """Should return False when all tokens traded."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        buyer.num_trades = 4

        buyer.buy_sell(
            time=50,
            nobuysell=0,
            high_bid=90,
            low_ask=95,
            high_bidder=1,
            low_asker=2,
        )

        result = buyer.buy_sell_response()
        assert result is False, "Should reject when all tokens traded"


# =============================================================================
# Test: Belief Mechanics
# =============================================================================


class TestBeliefMechanics:
    """Tests for Bayesian belief update."""

    def test_belief_update_online_mean(self):
        """Belief mean should update with observed prices."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        initial_mu = buyer.belief_mu

        # Simulate observing asks (buyer observes asks)
        buyer._update_beliefs(80)
        assert buyer.belief_mu == 80.0

        buyer._update_beliefs(90)
        assert buyer.belief_mu == 85.0  # mean of [80, 90]

        buyer._update_beliefs(100)
        assert buyer.belief_mu == 90.0  # mean of [80, 90, 100]

    def test_belief_outlier_filtering(self):
        """Prices > 3 sigma from mean should be ignored."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=2000,
            num_times=100,
            seed=42,
        )

        # Add prices with some variance to establish mean/std
        # Prices: 95, 100, 105 -> mean=100, std~=4.08
        for price in [95, 100, 105]:
            buyer._update_beliefs(price)

        assert buyer.belief_mu == 100.0
        n_before = len(buyer._observed_prices)

        # Add an extreme outlier (should be filtered)
        # 3 sigma = 3 * 4.08 = 12.24, so 1000 is way more than 3 sigma away
        buyer._update_beliefs(1000)  # Way more than 3 sigma away

        assert len(buyer._observed_prices) == n_before, "Outlier should be filtered"
        assert buyer.belief_mu == 100.0, "Mean should not change from outlier"

    def test_belief_period_reset_sigma(self):
        """Sigma should update from previous period's sample std."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        # Simulate period 1 with some prices
        prices_p1 = [80, 90, 100, 110, 120]
        for p in prices_p1:
            buyer._update_beliefs(p)

        expected_std = np.std(prices_p1)

        # Start period 2 - sigma should update
        buyer.start_period(2)

        assert buyer.belief_sigma == pytest.approx(
            expected_std, rel=0.01
        ), "Sigma should be updated from previous period's sample std"

    def test_m0_updated_from_previous_period(self):
        """m0 should equal count of observations from previous period."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        assert buyer.m0 == DEFAULT_M0  # Initial default

        # Simulate period 1 with some observations
        for p in [80, 90, 100, 110, 120, 130, 140]:  # 7 observations
            buyer._update_beliefs(p)

        # Start period 2
        buyer.start_period(2)

        assert buyer.m0 == 7, "m0 should equal previous period's observation count"


# =============================================================================
# Test: Arrival Model
# =============================================================================


class TestArrivalModel:
    """Tests for arrival intensity model m(t) = m0 * (1 - t)."""

    def test_expected_remaining_quotes_at_start(self):
        """At t=0, expected remaining = m0."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        buyer.current_time = 0
        remaining = buyer._expected_remaining_quotes()
        assert remaining == DEFAULT_M0, "At t=0, should expect m0 quotes"

    def test_expected_remaining_quotes_at_half(self):
        """At t=0.5, expected remaining = m0 * 0.5."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        buyer.current_time = 50  # half of num_times=100
        remaining = buyer._expected_remaining_quotes()
        assert remaining == pytest.approx(DEFAULT_M0 * 0.5), "At t=0.5, should expect m0*0.5 quotes"

    def test_expected_remaining_quotes_at_end(self):
        """At t=1, expected remaining = 0."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        buyer.current_time = 100  # end of period
        remaining = buyer._expected_remaining_quotes()
        assert remaining == 0.0, "At t=1, should expect 0 quotes"


# =============================================================================
# Test: Reservation Price
# =============================================================================


class TestReservationPrice:
    """Tests for reservation price computation."""

    def test_reservation_bounded_by_price_limits(self):
        """Reservation price should always be in [price_min, price_max]."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=10,
            price_max=150,
            num_times=100,
            seed=42,
        )

        for time in [0, 25, 50, 75, 99]:
            buyer.current_time = time
            reservation = buyer._compute_reservation_price(100)
            assert reservation >= 10, f"Reservation should be >= price_min at t={time}"
            assert reservation <= 150, f"Reservation should be <= price_max at t={time}"

    def test_buyer_reservation_below_valuation_early(self):
        """Early in period, buyer reservation should be well below valuation."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        buyer.current_time = 5  # Early in period
        valuation = 100
        reservation = buyer._compute_reservation_price(valuation)

        # Early: high option value of waiting -> reservation << valuation
        assert reservation < valuation, "Early reservation should be below valuation"
        # Should be meaningfully below (not just 1 off)
        assert reservation < valuation - 10, "Early reservation should be well below valuation"

    def test_seller_reservation_above_cost_early(self):
        """Early in period, seller reservation should be well above cost."""
        seller = BGAN(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        seller.current_time = 5  # Early in period
        cost = 50
        reservation = seller._compute_reservation_price(cost)

        # Early: high option value -> reservation >> cost
        assert reservation > cost, "Early reservation should be above cost"
        assert reservation > cost + 10, "Early reservation should be well above cost"

    def test_buyer_reservation_near_valuation_late(self):
        """Late in period, buyer reservation should approach valuation."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        buyer.current_time = 99  # Very late
        valuation = 100
        reservation = buyer._compute_reservation_price(valuation)

        # Late: no option value -> reservation ≈ valuation
        assert reservation == valuation, "Late reservation should equal valuation"

    def test_seller_reservation_near_cost_late(self):
        """Late in period, seller reservation should approach cost."""
        seller = BGAN(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        seller.current_time = 99  # Very late
        cost = 50
        reservation = seller._compute_reservation_price(cost)

        # Late: no option value -> reservation ≈ cost
        assert reservation == cost, "Late reservation should equal cost"

    def test_reservation_decreases_over_time_for_buyer(self):
        """Buyer reservation should increase (get more aggressive) over time."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        reservations = []
        for time in [10, 30, 50, 70, 90]:
            buyer.current_time = time
            r = buyer._compute_reservation_price(100)
            reservations.append(r)

        # Buyer reservation should increase (willing to pay more as time runs out)
        for i in range(len(reservations) - 1):
            assert (
                reservations[i] <= reservations[i + 1]
            ), f"Buyer reservation should increase over time: {reservations}"


# =============================================================================
# Test: BA Step Logic
# =============================================================================


class TestBAStepLogic:
    """Tests for bid-ask phase behavior."""

    def test_aggressive_entry_when_no_standing_bid(self):
        """When no current bid, buyer should enter at reservation."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        buyer.current_bid = 0  # No standing bid
        buyer.current_ask = 0

        buyer.bid_ask(time=50, nobidask=0)
        bid = buyer.bid_ask_response()

        # Should bid at reservation (aggressive entry)
        assert bid > 0, "Should enter market when no standing bid"
        assert bid < 100, "Bid should be less than valuation"

    def test_improvement_within_reservation(self):
        """Should improve bid if still within reservation."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        # Set time late so reservation is close to valuation
        buyer.current_time = 95
        buyer.current_bid = 80  # Existing bid
        buyer.current_ask = 95

        buyer.bid_ask(time=95, nobidask=0)
        bid = buyer.bid_ask_response()

        # Should improve to current_bid + 1 = 81 if within reservation
        assert bid == 81 or bid == 0, "Should either improve by 1 or stay silent"

    def test_no_quote_when_cannot_improve_profitably(self):
        """Should return 0 if improvement crosses reservation."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        # Early time -> low reservation
        buyer.current_time = 5
        # Set current bid very high
        buyer.current_bid = 95  # Already very high
        buyer.current_ask = 98

        buyer.bid_ask(time=5, nobidask=0)
        bid = buyer.bid_ask_response()

        # Cannot improve without crossing reservation
        assert bid == 0, "Should stay silent when cannot improve profitably"

    def test_seller_aggressive_entry_when_no_standing_ask(self):
        """When no current ask, seller should enter at reservation."""
        seller = BGAN(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        seller.current_bid = 0
        seller.current_ask = 0  # No standing ask

        seller.bid_ask(time=50, nobidask=0)
        ask = seller.bid_ask_response()

        # Should ask at reservation (aggressive entry)
        assert ask > 0, "Should enter market when no standing ask"
        assert ask > 50, "Ask should be greater than cost"


# =============================================================================
# Test: BS Step Logic
# =============================================================================


class TestBSStepLogic:
    """Tests for buy-sell phase behavior."""

    def test_accept_when_price_beats_reservation(self):
        """Should accept if standing price beats reservation."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        # Late period -> reservation ≈ valuation
        buyer.buy_sell(
            time=99,
            nobuysell=0,
            high_bid=90,
            low_ask=95,  # Below valuation, beats late reservation
            high_bidder=1,  # We're high bidder
            low_asker=2,
        )

        result = buyer.buy_sell_response()
        assert result is True, "Should accept when ask beats reservation (late period)"

    def test_reject_profitable_trade_early_when_below_reservation(self):
        """Early in period, may reject profitable trade if below reservation."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        # Very early period
        buyer.current_time = 2

        # Simulate BA step to set reservation
        buyer.bid_ask(time=2, nobidask=0)
        buyer.bid_ask_response()

        # Now BS step with a price that is profitable but likely below reservation
        buyer.buy_sell(
            time=2,
            nobuysell=0,
            high_bid=50,
            low_ask=95,  # Below valuation (profitable) but likely above early reservation
            high_bidder=1,
            low_asker=2,
        )

        result = buyer.buy_sell_response()
        # Early period: reservation < 95, so this should be rejected
        assert result is False, "Should reject profitable trade early if above reservation"

    def test_accept_any_profitable_late(self):
        """Near end, should accept any profitable trade."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        # Simulate BA at end of period
        buyer.bid_ask(time=99, nobidask=0)
        buyer.bid_ask_response()

        buyer.buy_sell(
            time=99,
            nobuysell=0,
            high_bid=80,
            low_ask=99,  # Just barely profitable
            high_bidder=1,
            low_asker=2,
        )

        result = buyer.buy_sell_response()
        assert result is True, "Should accept any profitable trade at end of period"

    def test_only_winner_can_accept(self):
        """Only high bidder (buyer) or low asker (seller) can accept."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        buyer.buy_sell(
            time=99,
            nobuysell=0,
            high_bid=80,
            low_ask=90,  # Profitable
            high_bidder=2,  # We're NOT the high bidder
            low_asker=3,
        )

        result = buyer.buy_sell_response()
        assert result is False, "Should not accept if not the winner"


# =============================================================================
# Test: Period Reset
# =============================================================================


class TestPeriodReset:
    """Tests for period lifecycle."""

    def test_observation_history_cleared_on_period_start(self):
        """Observation history should reset each period."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        # Add observations in period 1
        for p in [80, 90, 100]:
            buyer._update_beliefs(p)

        assert len(buyer._observed_prices) == 3

        # Start period 2
        buyer.start_period(2)

        assert len(buyer._observed_prices) == 0, "Observations should be cleared"
        assert len(buyer._prev_period_prices) == 3, "Previous observations should be stored"

    def test_belief_mean_reset_to_prior(self):
        """Belief mean should reset to prior at period start."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        expected_prior = (1 + 200) / 2  # (price_min + price_max) / 2

        buyer._update_beliefs(50)
        assert buyer.belief_mu != expected_prior

        buyer.start_period(2)

        assert buyer.belief_mu == expected_prior, "Mean should reset to prior"

    def test_num_trades_reset(self):
        """num_trades should reset each period."""
        buyer = BGAN(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            seed=42,
        )

        buyer.num_trades = 2

        buyer.start_period(2)

        assert buyer.num_trades == 0, "num_trades should reset"
