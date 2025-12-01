# tests/unit/traders/test_kaplan.py
"""
Adversarial tests for Kaplan (Sniper) agent.

Kaplan won the 1993 Santa Fe Tournament using a "sniper" strategy:
1. Wait in background until conditions are favorable
2. Jump in when spread is small, price is good, or time is running out
3. Accept any profitable trade in last few time steps (sniper behavior)

These tests verify:
1. Never trades at a loss
2. Sniper behavior in final time steps
3. Jump-in logic triggers correctly
4. History learning across periods
"""


from traders.legacy.kaplan import Kaplan, KaplanJava

# =============================================================================
# Test: Never Trade at a Loss
# =============================================================================


class TestNeverTradeAtLoss:
    """Tests that Kaplan never makes unprofitable trades."""

    def test_buyer_rejects_ask_above_valuation(self):
        """Buyer should reject if ask price exceeds valuation."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
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
        seller = Kaplan(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            num_times=100,
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
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
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
        seller = Kaplan(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            num_times=100,
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
# Test: Sniper Behavior
# =============================================================================


class TestSniperBehavior:
    """Tests for sniper behavior in final time steps."""

    def test_buyer_accepts_in_last_steps(self):
        """Buyer should accept any profitable trade in final steps."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            sniper_steps=2,  # Sniper in last 2 steps
        )

        # Time step 99 (1 step from end) - should trigger sniper
        buyer.buy_sell(
            time=99,
            nobuysell=0,
            high_bid=80,
            low_ask=95,  # Below valuation - profitable
            high_bidder=2,  # We're NOT high bidder
            low_asker=3,
        )

        result = buyer.buy_sell_response()
        assert result is True, "Sniper should accept any profitable trade in final steps"

    def test_seller_accepts_in_last_steps(self):
        """Seller should accept any profitable trade in final steps."""
        seller = Kaplan(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            num_times=100,
            sniper_steps=2,
        )

        # Time step 99 - sniper mode
        seller.buy_sell(
            time=99,
            nobuysell=0,
            high_bid=55,  # Above cost - profitable
            low_ask=60,
            high_bidder=2,
            low_asker=3,  # We're NOT low asker
        )

        result = seller.buy_sell_response()
        assert result is True, "Sniper should accept any profitable trade in final steps"

    def test_sniper_still_rejects_unprofitable(self):
        """Even in sniper mode, should reject unprofitable trades."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            sniper_steps=2,
        )

        # Time step 99, but ask exceeds valuation
        buyer.buy_sell(
            time=99,
            nobuysell=0,
            high_bid=80,
            low_ask=105,  # Above valuation - NOT profitable
            high_bidder=2,
            low_asker=3,
        )

        result = buyer.buy_sell_response()
        assert result is False, "Sniper should still reject unprofitable trades"

    def test_no_sniper_outside_final_steps(self):
        """Should not snipe-accept outside the final steps window."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            sniper_steps=2,
        )

        # Time step 50 - far from end
        buyer.buy_sell(
            time=50,
            nobuysell=0,
            high_bid=80,
            low_ask=95,  # Profitable
            high_bidder=2,  # Not high bidder, not our turn
            low_asker=3,
        )

        result = buyer.buy_sell_response()
        assert result is False, "Should not snipe-accept in middle of period"


# =============================================================================
# Test: First Bid/Ask Conservative (Worst-Case Token)
# =============================================================================


class TestFirstBidAskConservative:
    """Tests that first bid/ask uses worst-case token."""

    def test_buyer_first_bid_is_low(self):
        """First bid should be conservative (near min_price)."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],  # Worst is 70
            price_min=1,
            price_max=200,
            num_times=100,
            aggressive_first=False,
        )

        # No current bid/ask
        buyer.current_bid = 0
        buyer.current_ask = 0

        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()

        # First bid should be near price_min (conservative start)
        assert bid <= 10, f"First bid should be conservative (low), got {bid}"

    def test_seller_first_ask_is_high(self):
        """First ask should be conservative (near max_price)."""
        seller = Kaplan(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],  # Worst is 80
            price_min=1,
            price_max=200,
            num_times=100,
            aggressive_first=False,
        )

        seller.current_bid = 0
        seller.current_ask = 0

        seller.bid_ask(time=1, nobidask=0)
        ask = seller.bid_ask_response()

        # First ask should be near price_max (conservative start)
        assert ask >= 190, f"First ask should be conservative (high), got {ask}"


# =============================================================================
# Test: nobuysell/nobidask Flag Handling
# =============================================================================


class TestFlagHandling:
    """Tests for nobidask and nobuysell flag handling."""

    def test_respects_nobidask_flag(self):
        """Should return 0 when nobidask > 0."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
        )

        buyer.bid_ask(time=1, nobidask=1)  # Can't bid
        bid = buyer.bid_ask_response()

        assert bid == 0, f"Should return 0 when nobidask > 0, got {bid}"

    def test_respects_nobuysell_flag(self):
        """Should reject when nobuysell > 0."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
        )

        buyer.buy_sell(
            time=99,  # Sniper mode
            nobuysell=4,  # Not our turn
            high_bid=80,
            low_ask=90,
            high_bidder=2,
            low_asker=3,
        )

        result = buyer.buy_sell_response()
        assert result is False, "Should reject when nobuysell > 0"


# =============================================================================
# Test: Spread Crossing Required
# =============================================================================


class TestSpreadCrossingRequired:
    """Tests that standard acceptance requires crossed spread."""

    def test_buyer_accepts_when_spread_crossed_and_winner(self):
        """Buyer should accept when high bidder and spread is crossed."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
        )

        # Not in sniper mode, but we're high bidder and spread crossed
        buyer.buy_sell(
            time=50,
            nobuysell=0,
            high_bid=95,
            low_ask=90,  # bid >= ask
            high_bidder=1,  # We are high bidder
            low_asker=2,
        )

        result = buyer.buy_sell_response()
        assert result is True, "Should accept when winner and spread crossed"

    def test_buyer_rejects_when_spread_not_crossed(self):
        """Buyer should reject when spread not crossed (outside sniper mode)."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
        )

        buyer.buy_sell(
            time=50,
            nobuysell=0,
            high_bid=85,
            low_ask=90,  # bid < ask - not crossed
            high_bidder=1,
            low_asker=2,
        )

        result = buyer.buy_sell_response()
        assert result is False, "Should reject when spread not crossed (outside sniper)"


# =============================================================================
# Test: Period Reset
# =============================================================================


class TestPeriodReset:
    """Tests for start_period() state reset."""

    def test_start_period_resets_trade_count(self):
        """start_period() should reset trade history."""
        agent = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
        )

        # Simulate some trades
        agent.trade_count = 5
        agent.last_time = 50

        agent.start_period(period_number=2)

        assert agent.trade_count == 0, "trade_count should reset"
        assert agent.last_time == 0, "last_time should reset"

    def test_end_period_updates_price_stats(self):
        """end_period() should compute min/max/avg prices."""
        agent = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
        )

        # Record some trades
        agent.trade_count = 3
        agent.prices[1] = 80
        agent.prices[2] = 90
        agent.prices[3] = 85

        agent.end_period()

        # Should have computed stats
        assert agent.avepr > 0, "avepr should be computed"
        # Note: For buyer, maxpr -= 100 (price_bound_adj), so minpr <= avepr but maxpr can be negative
        # The raw min/max before adjustment are 80 and 90
        # After adjustment for buyer: maxpr = 90 - 100 = -10
        assert agent.minpr == 80, "minpr should be min of prices"
        assert agent.maxpr == 90 - 100, "maxpr should be max - adjustment for buyer"
        assert agent.avepr == (80 + 90 + 85) // 3, "avepr should be average"
        assert agent.has_price_history, "should have price history"


# =============================================================================
# Test: No Tokens Left
# =============================================================================


class TestNoTokensLeft:
    """Tests behavior when all tokens traded."""

    def test_bid_is_zero_when_no_tokens(self):
        """Should return 0 bid when no tokens left."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=2,
            valuations=[100, 90],
            price_min=1,
            price_max=200,
            num_times=100,
        )

        buyer.num_trades = 2  # All traded

        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()

        assert bid == 0, "Should return 0 when no tokens left"

    def test_rejects_when_no_tokens(self):
        """Should reject trade when no tokens left."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=2,
            valuations=[100, 90],
            price_min=1,
            price_max=200,
            num_times=100,
        )

        buyer.num_trades = 2

        buyer.buy_sell(
            time=99,
            nobuysell=0,
            high_bid=85,
            low_ask=80,
            high_bidder=1,
            low_asker=2,
        )

        result = buyer.buy_sell_response()
        assert result is False, "Should reject when no tokens left"


# =============================================================================
# Test: Tunable Parameters
# =============================================================================


class TestTunableParameters:
    """Tests for tunable parameter handling."""

    def test_spread_threshold_respected(self):
        """Spread threshold should affect jump-in behavior."""
        # With default 0.10 threshold
        agent1 = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            spread_threshold=0.10,
        )

        # With very strict 0.01 threshold
        agent2 = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            spread_threshold=0.01,  # Very strict
        )

        assert agent1.spread_threshold == 0.10
        assert agent2.spread_threshold == 0.01

    def test_sniper_steps_configurable(self):
        """sniper_steps should be configurable."""
        agent = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            sniper_steps=5,  # More aggressive
        )

        assert agent.sniper_steps == 5


# =============================================================================
# Test: Price History Learning
# =============================================================================


class TestPriceHistoryLearning:
    """Tests for learning from price history across periods."""

    def test_minpr_maxpr_updated_after_period(self):
        """min/max prices should be tracked for next period."""
        agent = Kaplan(
            player_id=1,
            is_buyer=False,  # Seller
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            num_times=100,
        )

        # Simulate trades at various prices
        agent.trade_count = 4
        agent.prices[1] = 60
        agent.prices[2] = 75
        agent.prices[3] = 70
        agent.prices[4] = 65

        agent.end_period()

        # Seller should track these for next period
        assert agent.minpr > 0, "minpr should be set"
        assert agent.maxpr > 0, "maxpr should be set"
        assert agent.avepr > 0, "avepr should be set"


# =============================================================================
# Test: Jump-In Conditions (RUTHLESS - From traders.md)
# =============================================================================


class TestJumpInConditions:
    """Tests for the three jump-in conditions from traders.md.

    Kaplan's jump-in logic has THREE conditions (any one triggers a jump):
    1. Spread condition: (cask - cbid) / (cask + 1) < 0.10 AND profitable
    2. Price better than last period: cask <= minpr (history-based)
    3. Time running out: various time-based conditions

    These tests verify Kaplan PASSES (returns 0) when conditions NOT met,
    and jumps in when at least one condition IS met.
    """

    def test_buyer_jumps_when_spread_narrow(self):
        """Buyer should jump in when spread is narrow and profitable.

        Condition 1: (cask - cbid) / (cask + 1) < spread_threshold
        AND (token+1)*(1-margin) - cask > 0
        """
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            spread_threshold=0.10,  # 10% threshold
            profit_margin=0.02,  # 2% profit margin
        )

        # Set up state: narrow spread
        buyer.current_bid = 90
        buyer.current_ask = 95
        # Spread = (95-90)/(95+1) = 5/96 = 0.052 < 0.10 ✓
        # Profit check: (100+1)*(1-0.02) - 95 = 101*0.98 - 95 = 98.98 - 95 = 3.98 > 0 ✓

        buyer.bid_ask(time=50, nobidask=0)
        bid = buyer.bid_ask_response()

        # Should jump to current_ask
        assert bid == 95, f"Should jump to ask when spread narrow, got {bid}"

    def test_buyer_no_jump_when_spread_wide(self):
        """Buyer should NOT jump when spread is wide (>10%).

        Note: Must also ensure time conditions don't trigger a jump.
        Set time early in period and last_time recently to avoid time-based jump.
        """
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            spread_threshold=0.10,
        )

        # Set up state: wide spread
        buyer.current_bid = 70
        buyer.current_ask = 90
        # Spread = (90-70)/(90+1) = 20/91 = 0.22 > 0.10 (NOT met)

        # Prevent time-based jump by setting recent last_time
        buyer.last_time = 10  # Recent trade

        buyer.bid_ask(time=15, nobidask=0)  # Early time, won't trigger time condition
        bid = buyer.bid_ask_response()

        # Should NOT jump to ask (but might bid at min_price or pass)
        assert bid != 90, f"Should NOT jump to ask when spread wide, got {bid}"

    def test_buyer_no_jump_when_unprofitable(self):
        """Buyer should NOT jump even if spread narrow, when unprofitable."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            spread_threshold=0.20,  # Wide threshold
            profit_margin=0.02,
        )

        # Narrow spread but ask at/above valuation
        buyer.current_bid = 98
        buyer.current_ask = 101
        # Spread = (101-98)/(101+1) = 3/102 = 0.029 < 0.20 ✓
        # BUT: (100+1)*(1-0.02) - 101 = 98.98 - 101 = -2.02 < 0 ✗ (not profitable)

        buyer.bid_ask(time=50, nobidask=0)
        bid = buyer.bid_ask_response()

        # Should NOT jump to unprofitable ask
        assert bid != 101, f"Should NOT jump when unprofitable, got {bid}"

    def test_buyer_jumps_when_price_below_minpr(self):
        """Buyer should jump when ask is below last period's minpr."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
        )

        # Simulate period 2 with minpr from period 1
        buyer.period = 2
        buyer.minpr = 85  # Last period's min trade price

        # Set up: ask at or below minpr
        buyer.current_bid = 60
        buyer.current_ask = 80  # 80 <= 85 ✓

        buyer.bid_ask(time=50, nobidask=0)
        bid = buyer.bid_ask_response()

        # Should jump to ask (but clamped to most = token_val - 1 = 99)
        # Since 80 < 99, should bid 80
        assert bid == 80, f"Should jump when ask <= minpr, got {bid}"

    def test_buyer_jumps_when_time_running_out(self):
        """Buyer should jump when time is running out.

        Condition 3: (t - last_time) >= (num_times - t) * 0.5
        """
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            time_half_frac=0.5,
        )

        # Set up: time=80, last_time=10, ask available
        buyer.last_time = 10
        buyer.current_bid = 70
        buyer.current_ask = 90

        # Check condition: (80-10) >= (100-80)*0.5 = 70 >= 10 ✓
        buyer.bid_ask(time=80, nobidask=0)
        bid = buyer.bid_ask_response()

        # Should jump to ask
        assert bid == 90, f"Should jump when time running out, got {bid}"

    def test_seller_jumps_when_spread_narrow(self):
        """Seller should jump in when spread is narrow and profitable."""
        seller = Kaplan(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],  # Costs
            price_min=1,
            price_max=200,
            num_times=100,
            spread_threshold=0.10,
            profit_margin=0.02,
        )

        # Set up state: narrow spread
        seller.current_bid = 60
        seller.current_ask = 65

        seller.bid_ask(time=50, nobidask=0)
        ask = seller.bid_ask_response()

        # Should jump to current_bid (clamped to profitability)
        # least = token_val + 1 = 51
        # So should offer at 60 (the bid) if 60 >= 51
        assert ask == 60, f"Should jump to bid when spread narrow, got {ask}"

    def test_spread_threshold_is_configurable(self):
        """Spread threshold parameter should be respected.

        Note: Period 1 has special behavior - spread check includes maxpr check
        that passes in period 1 if profitable. Test in period 2 with maxpr set
        below the ask price to isolate the spread threshold effect.
        """
        # Strict threshold (5%)
        buyer_strict = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            spread_threshold=0.05,  # 5%
        )

        # Loose threshold (15%)
        buyer_loose = Kaplan(
            player_id=2,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
            spread_threshold=0.15,  # 15%
        )

        # Period 2 with maxpr above ask so maxpr check passes
        buyer_strict.period = 2
        buyer_strict.maxpr = 100
        buyer_strict.minpr = 50  # Low minpr so ask > minpr (price check won't trigger)
        buyer_loose.period = 2
        buyer_loose.maxpr = 100
        buyer_loose.minpr = 50

        # Prevent time-based jump - set both last_time and my_last_time recently
        buyer_strict.last_time = 45
        buyer_strict.my_last_time = 45
        buyer_loose.last_time = 45
        buyer_loose.my_last_time = 45

        # 8% spread: meets 15% but not 5%
        buyer_strict.current_bid = 85
        buyer_strict.current_ask = 93
        buyer_loose.current_bid = 85
        buyer_loose.current_ask = 93
        # Spread = (93-85)/(93+1) = 8/94 = 0.085

        buyer_strict.bid_ask(time=50, nobidask=0)
        bid_strict = buyer_strict.bid_ask_response()

        buyer_loose.bid_ask(time=50, nobidask=0)
        bid_loose = buyer_loose.bid_ask_response()

        # Strict should NOT jump (8.5% > 5%)
        assert bid_strict != 93, "Strict threshold should not jump at 8.5% spread"

        # Loose SHOULD jump (8.5% < 15%)
        assert bid_loose == 93, f"Loose threshold should jump at 8.5% spread, got {bid_loose}"


# =============================================================================
# Test: KaplanFixed (minpr/maxpr bug fix)
# =============================================================================


class TestKaplanFixedMinprMaxpr:
    """Tests for the fixed minpr/maxpr computation in Kaplan.

    The original Java SRobotKaplan.java has a bug: when Kaplan trades first,
    prices[1]=-1 (sentinel for own trade), and abs(-1)=1 becomes minpr.
    This breaks the "price better than last period" condition.

    The fixed Kaplan class excludes own trades from minpr/maxpr computation.
    """

    def test_minpr_excludes_own_trades(self):
        """minpr should be computed from OTHER traders' prices only."""
        agent = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
        )

        # Simulate: Kaplan traded first (prices[1]=-1), then others at 178, 263
        agent.trade_count = 3
        agent.prices[1] = -1  # Own trade (sentinel)
        agent.prices[2] = 178  # Other trader
        agent.prices[3] = 263  # Other trader

        agent.end_period()

        # minpr should be 178, NOT 1 (which would happen with abs(-1))
        assert agent.minpr == 178, f"minpr should exclude own trades, got {agent.minpr}"
        # maxpr should be 263 - 100 (buyer adjustment)
        assert agent.maxpr == 263 - 100, f"maxpr should be 263-100=163, got {agent.maxpr}"
        # avepr should be average of non-own prices
        assert agent.avepr == (178 + 263) // 2, "avepr should be avg of other prices"
        assert agent.has_price_history, "should have price history"

    def test_no_history_when_all_own_trades(self):
        """has_price_history should be False when all trades are Kaplan's."""
        agent = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
        )

        # All trades are Kaplan's own
        agent.trade_count = 2
        agent.prices[1] = -1
        agent.prices[2] = -1

        agent.end_period()

        assert not agent.has_price_history, "should have no price history"
        assert agent.minpr == 0, "minpr should be 0 with no history"
        assert agent.maxpr == 0, "maxpr should be 0 with no history"

    def test_no_history_when_zero_trades(self):
        """has_price_history should be False when no trades occurred."""
        agent = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
        )

        agent.trade_count = 0

        agent.end_period()

        assert not agent.has_price_history, "should have no price history"
        assert agent.minpr == 0, "minpr should be 0"
        assert agent.maxpr == 0, "maxpr should be 0"

    def test_price_better_condition_requires_history(self):
        """'Price better than last period' should NOT trigger without history."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
        )

        # Period 2, but no price history (all own trades in period 1)
        buyer.period = 2
        buyer.has_price_history = False
        buyer.minpr = 0

        # Set up a favorable ask
        buyer.current_bid = 60
        buyer.current_ask = 80

        # Prevent time-based jump
        buyer.last_time = 45
        buyer.my_last_time = 45

        buyer.bid_ask(time=50, nobidask=0)
        bid = buyer.bid_ask_response()

        # Should NOT jump to ask because has_price_history=False
        assert bid != 80, f"Should NOT jump without price history, got {bid}"


# =============================================================================
# Test: KaplanJava (bug-for-bug compatibility)
# =============================================================================


class TestKaplanJavaBugCompatibility:
    """Tests that KaplanJava reproduces the original Java bugs.

    KaplanJava should produce minpr=1 when Kaplan trades first,
    matching the original SRobotKaplan.java behavior.
    """

    def test_minpr_bug_when_kaplan_trades_first(self):
        """KaplanJava should have minpr=1 when trading first (the bug)."""
        agent = KaplanJava(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
        )

        # Simulate: Kaplan traded first (prices[1]=-1), then others at 178, 263
        agent.trade_count = 3
        agent.prices[1] = -1  # Own trade (sentinel)
        agent.prices[2] = 178  # Other trader
        agent.prices[3] = 263  # Other trader

        agent.end_period()

        # BUG: minpr should be 1 (abs(-1)), NOT 178
        assert agent.minpr == 1, f"KaplanJava should have minpr bug, got {agent.minpr}"
        # maxpr should be 263 (updated correctly in loop)
        assert agent.maxpr == 263, f"KaplanJava maxpr, got {agent.maxpr}"

    def test_kaplan_vs_kaplanjava_different_minpr(self):
        """Kaplan (fixed) and KaplanJava (buggy) should differ on minpr."""
        fixed = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
        )
        buggy = KaplanJava(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            num_times=100,
        )

        # Same trade history
        for agent in [fixed, buggy]:
            agent.trade_count = 3
            agent.prices[1] = -1
            agent.prices[2] = 150
            agent.prices[3] = 200
            agent.end_period()

        # Fixed: minpr should be 150
        assert fixed.minpr == 150, "Fixed Kaplan minpr wrong"
        # Buggy: minpr should be 1
        assert buggy.minpr == 1, "Buggy KaplanJava minpr wrong"
        # They should differ!
        assert fixed.minpr != buggy.minpr, "Fixed and buggy should differ"
