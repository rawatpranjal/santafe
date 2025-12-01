# tests/behavior/test_kaplan_behavior.py
"""
Behavioral tests for Kaplan (Sniper) agent.

Kaplan won the 1993 Santa Fe Tournament using a "sniper" strategy:
1. Wait in background until conditions are favorable
2. Jump in when spread is small (<10%), price is good, or time is running out
3. Accept any profitable trade in last 2 timesteps (sniper behavior)
4. Learn from history: tracks min/max prices from previous periods

These tests verify the exact Java specification (SRobotKaplan.java):
1. Seeding behavior (minprice+1 for buyer, maxprice-1 for seller)
2. Snipe conditions (10% spread, 2% profit, history comparison)
3. Time pressure conditions
4. Acceptance logic (winner and crossed, or sniper last 2 steps)
"""

import numpy as np

from engine.market import Market
from engine.token_generator import TokenGenerator
from traders.legacy.kaplan import Kaplan
from traders.legacy.zic import ZIC

# =============================================================================
# Test: Seeding Behavior (First Bid/Ask)
# =============================================================================


class TestSeedingBehavior:
    """Tests for first bid/ask when no standing quote exists."""

    def test_buyer_seeds_with_minprice_plus_1(self):
        """Buyer should bid minprice+1 when cbid==0."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
            num_times=100,
        )

        # No current bid
        buyer.current_bid = 0
        buyer.current_ask = 150  # Has ask, but doesn't matter

        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()

        # Java: newbid = minprice + 1
        assert bid == 1, f"First bid should be minprice+1=1, got {bid}"

    def test_seller_seeds_with_maxprice_minus_1(self):
        """Seller should ask maxprice-1 when cask==0."""
        seller = Kaplan(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=0,
            price_max=200,
            num_times=100,
        )

        # No current ask
        seller.current_bid = 50
        seller.current_ask = 0

        seller.bid_ask(time=1, nobidask=0)
        ask = seller.bid_ask_response()

        # Java: newoffer = maxprice - 1
        assert ask == 199, f"First ask should be maxprice-1=199, got {ask}"


# =============================================================================
# Test: Spread Snipe Condition
# =============================================================================


class TestSpreadSnipe:
    """Tests for jump-in when spread < 10% and profit >= 2%."""

    def test_buyer_snipes_on_tight_spread(self):
        """Buyer should bid at cask when spread < 10% and profitable."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
            num_times=100,
        )

        # Set up tight spread: (cask - cbid) / (cask + 1) < 0.10
        # With cask=80, cbid=75: (80-75)/(80+1) = 5/81 ≈ 0.062 < 0.10
        # Profit check: (token+1)*0.98 - cask > 0
        # With token=100: (100+1)*0.98 - 80 = 98.98 - 80 = 18.98 > 0 ✓
        buyer.current_bid = 75
        buyer.current_ask = 80
        buyer.period = 1  # First period, no history check

        buyer.bid_ask(time=50, nobidask=0)
        bid = buyer.bid_ask_response()

        # Should jump in at cask
        assert bid == 80, f"Should snipe at cask=80 when spread tight, got {bid}"

    def test_buyer_no_snipe_on_wide_spread(self):
        """Buyer should NOT snipe when spread >= 10%.

        Must avoid ALL jump-in triggers:
        1. Spread condition: Wide spread (37% > 10%) → No
        2. Price better condition: period 2 + minpr set high → No
        3. Time condition: recent last_time and my_last_time → No
        """
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
            num_times=100,
        )

        # Wide spread: (cask - cbid) / (cask + 1) >= 0.10
        # With cask=80, cbid=50: (80-50)/(80+1) = 30/81 ≈ 0.37 > 0.10
        buyer.current_bid = 50
        buyer.current_ask = 80

        # Period 2 with minpr set LOW (so ask > minpr, price condition won't trigger)
        buyer.period = 2
        buyer.minpr = 50  # Ask 80 > minpr 50, so price condition won't trigger
        buyer.maxpr = 100  # Needed for spread condition to consider maxpr check

        # Set last_time and my_last_time to avoid time pressure trigger
        buyer.last_time = 45
        buyer.my_last_time = 45

        buyer.bid_ask(time=50, nobidask=0)
        bid = buyer.bid_ask_response()

        # Should NOT jump to cask (returns minprice+1 or 0)
        assert bid != 80, f"Should NOT snipe at cask when spread wide, got {bid}"

    def test_seller_uses_cbid_denominator(self):
        """Seller spread check uses cbid+1 denominator (asymmetric)."""
        seller = Kaplan(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=0,
            price_max=200,
            num_times=100,
            symmetric_spread=False,  # Java behavior
        )

        # Test asymmetric: (cask - cbid) / (cbid + 1)
        # With cask=60, cbid=55: (60-55)/(55+1) = 5/56 ≈ 0.089 < 0.10
        # Profit check: (token+1)*1.02 - cbid < 0
        # With token=50: (50+1)*1.02 - 55 = 52.02 - 55 = -2.98 < 0 ✓
        seller.current_bid = 55
        seller.current_ask = 60
        seller.period = 1

        seller.bid_ask(time=50, nobidask=0)
        ask = seller.bid_ask_response()

        # Should jump in at cbid
        assert ask == 55, f"Seller should snipe at cbid=55, got {ask}"


# =============================================================================
# Test: Bargain Hunter / Better Price Conditions
# =============================================================================


class TestBargainHunter:
    """Tests for jump-in based on price history."""

    def test_buyer_jumps_when_ask_below_minpr(self):
        """Buyer should bid at cask when cask <= minpr (better than last period)."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
            num_times=100,
        )

        # Set history from "previous period"
        buyer.minpr = 85
        buyer.maxpr = 95
        buyer.period = 2  # Must be period > 1 for history check

        # Wide spread but cask <= minpr
        buyer.current_bid = 50
        buyer.current_ask = 80  # <= minpr=85

        buyer.bid_ask(time=50, nobidask=0)
        bid = buyer.bid_ask_response()

        # Should jump to cask due to bargain condition
        assert bid == 80, f"Should jump to cask=80 (better than minpr=85), got {bid}"

    def test_seller_jumps_when_bid_above_maxpr(self):
        """Seller should ask at cbid when cbid >= maxpr (better than last period)."""
        seller = Kaplan(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=0,
            price_max=200,
            num_times=100,
        )

        # Set history
        seller.minpr = 55
        seller.maxpr = 65
        seller.period = 2

        # Wide spread but cbid >= maxpr
        seller.current_bid = 70  # >= maxpr=65
        seller.current_ask = 150

        seller.bid_ask(time=50, nobidask=0)
        ask = seller.bid_ask_response()

        # Should jump to cbid due to better price
        assert ask == 70, f"Should jump to cbid=70 (better than maxpr=65), got {ask}"


# =============================================================================
# Test: Time Pressure Conditions
# =============================================================================


class TestTimePressure:
    """Tests for jump-in based on time/inactivity."""

    def test_jump_when_half_remaining_idle(self):
        """Should jump in when (t - lasttime) >= (ntimes - t) / 2."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
            num_times=100,
        )

        # Set up: t=80, remaining=20, need (t - lasttime) >= 10
        buyer.current_bid = 50
        buyer.current_ask = 90  # Wide spread normally
        buyer.last_time = 65  # t - lasttime = 80 - 65 = 15 >= 10
        buyer.my_last_time = 65

        buyer.bid_ask(time=80, nobidask=0)
        bid = buyer.bid_ask_response()

        # Should jump to cask due to time pressure
        assert bid == 90, f"Should jump to cask=90 due to time pressure, got {bid}"

    def test_jump_when_two_thirds_personal_idle(self):
        """Should jump when personal inactivity >= 2/3 remaining AND gap >= 5."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
            num_times=100,
        )

        # Set up: t=70, remaining=30
        # Need: (t - mylasttime) >= 2*(ntimes-t)/3 = 20 AND (t - lasttime) >= 5
        buyer.current_bid = 50
        buyer.current_ask = 90
        buyer.last_time = 60  # t - lasttime = 10 >= 5 ✓
        buyer.my_last_time = 40  # t - mylasttime = 30 >= 20 ✓

        buyer.bid_ask(time=70, nobidask=0)
        bid = buyer.bid_ask_response()

        # Should jump to cask
        assert bid == 90, f"Should jump to cask=90 due to personal inactivity, got {bid}"


# =============================================================================
# Test: Limit Price Protection
# =============================================================================


class TestLimitProtection:
    """Tests that bids/asks are clamped to profitable levels."""

    def test_buyer_bid_capped_at_token_minus_1(self):
        """Buyer bid should never exceed token_val - 1."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
            num_times=100,
        )

        # Force time pressure to make it want to jump
        buyer.current_bid = 50
        buyer.current_ask = 98  # Would want to bid here
        buyer.last_time = 0
        buyer.my_last_time = 0

        buyer.bid_ask(time=90, nobidask=0)  # Time pressure
        bid = buyer.bid_ask_response()

        # most = token[next] - 1 = 100 - 1 = 99
        # cask = 98 < most, so most = cask = 98
        # But if cask > most, it gets capped
        assert bid <= 99, f"Bid should be capped at token-1=99, got {bid}"

    def test_seller_ask_floored_at_token_plus_1(self):
        """Seller ask should never go below token_val + 1."""
        seller = Kaplan(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=0,
            price_max=200,
            num_times=100,
        )

        # Force time pressure
        seller.current_bid = 52  # Would want to ask here
        seller.current_ask = 100
        seller.last_time = 0
        seller.my_last_time = 0

        seller.bid_ask(time=90, nobidask=0)
        ask = seller.bid_ask_response()

        # least = token[next] + 1 = 50 + 1 = 51
        # cbid = 52 > least, so least = 52
        assert ask >= 51, f"Ask should be floored at token+1=51, got {ask}"


# =============================================================================
# Test: Cannot Outbid/Undercut
# =============================================================================


class TestCannotOutbid:
    """Tests for returning 0 when cannot improve."""

    def test_buyer_returns_0_when_cannot_outbid(self):
        """Buyer should return 0 if most <= cbid."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
            num_times=100,
        )

        # token = 100, most = 99
        # cbid = 99 >= most -> cannot outbid
        buyer.current_bid = 99
        buyer.current_ask = 105

        buyer.bid_ask(time=50, nobidask=0)
        bid = buyer.bid_ask_response()

        assert bid == 0, f"Should return 0 when cannot outbid, got {bid}"

    def test_seller_returns_0_when_cannot_undercut(self):
        """Seller should return 0 if least >= cask."""
        seller = Kaplan(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=0,
            price_max=200,
            num_times=100,
        )

        # token = 50, least = 51
        # cask = 51 <= least -> cannot undercut
        seller.current_bid = 45
        seller.current_ask = 51

        seller.bid_ask(time=50, nobidask=0)
        ask = seller.bid_ask_response()

        assert ask == 0, f"Should return 0 when cannot undercut, got {ask}"


# =============================================================================
# Integration Test: Kaplan vs ZIC
# =============================================================================


class TestKaplanVsZIC:
    """Integration tests running Kaplan against ZIC opponents."""

    def test_kaplan_profitable_against_zic(self):
        """Kaplan should be profitable against ZIC opponents."""
        np.random.seed(42)

        token_gen = TokenGenerator(game_type=6453, num_tokens=4, seed=42)
        token_gen.new_round()

        buyers = []
        sellers = []

        # 1 Kaplan buyer
        buyer_tokens = token_gen.generate_tokens(is_buyer=True)
        kaplan = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=buyer_tokens,
            price_min=0,
            price_max=200,
            num_times=100,
        )
        buyers.append(kaplan)

        # 3 ZIC buyers
        for i in range(3):
            tokens = token_gen.generate_tokens(is_buyer=True)
            zic = ZIC(
                player_id=i + 2,
                is_buyer=True,
                num_tokens=4,
                valuations=tokens,
                price_min=0,
                price_max=200,
                seed=100 + i,
            )
            buyers.append(zic)

        # 4 ZIC sellers
        for i in range(4):
            tokens = token_gen.generate_tokens(is_buyer=False)
            zic = ZIC(
                player_id=i + 5,
                is_buyer=False,
                num_tokens=4,
                valuations=tokens,
                price_min=0,
                price_max=200,
                seed=200 + i,
            )
            sellers.append(zic)

        for agent in buyers + sellers:
            agent.start_period(1)

        market = Market(
            num_buyers=4,
            num_sellers=4,
            num_times=100,
            price_min=0,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=42,
        )

        for step in range(1, 101):
            if not market.run_time_step():
                break

        # Kaplan should not lose money
        assert (
            kaplan.period_profit >= 0
        ), f"Kaplan should not lose money, got profit={kaplan.period_profit}"

    def test_kaplan_seller_profitable(self):
        """Kaplan seller should also be profitable against ZIC."""
        np.random.seed(43)

        token_gen = TokenGenerator(game_type=6453, num_tokens=4, seed=43)
        token_gen.new_round()

        buyers = []
        sellers = []

        # 4 ZIC buyers
        for i in range(4):
            tokens = token_gen.generate_tokens(is_buyer=True)
            zic = ZIC(
                player_id=i + 1,
                is_buyer=True,
                num_tokens=4,
                valuations=tokens,
                price_min=0,
                price_max=200,
                seed=100 + i,
            )
            buyers.append(zic)

        # 1 Kaplan seller
        seller_tokens = token_gen.generate_tokens(is_buyer=False)
        kaplan = Kaplan(
            player_id=5,
            is_buyer=False,
            num_tokens=4,
            valuations=seller_tokens,
            price_min=0,
            price_max=200,
            num_times=100,
        )
        sellers.append(kaplan)

        # 3 ZIC sellers
        for i in range(3):
            tokens = token_gen.generate_tokens(is_buyer=False)
            zic = ZIC(
                player_id=i + 6,
                is_buyer=False,
                num_tokens=4,
                valuations=tokens,
                price_min=0,
                price_max=200,
                seed=200 + i,
            )
            sellers.append(zic)

        for agent in buyers + sellers:
            agent.start_period(1)

        market = Market(
            num_buyers=4,
            num_sellers=4,
            num_times=100,
            price_min=0,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=43,
        )

        for step in range(1, 101):
            if not market.run_time_step():
                break

        assert (
            kaplan.period_profit >= 0
        ), f"Kaplan seller should not lose money, got profit={kaplan.period_profit}"


# =============================================================================
# Test: Period History Learning
# =============================================================================


class TestPeriodHistory:
    """Tests for history tracking across periods."""

    def test_minpr_maxpr_computed_correctly(self):
        """end_period should compute minpr/maxpr from trade prices."""
        agent = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
            num_times=100,
        )

        # Simulate some trades
        agent.trade_count = 3
        agent.prices[1] = 75
        agent.prices[2] = 85
        agent.prices[3] = 80

        agent.end_period()

        # Fixed behavior: minpr/maxpr computed from non-own trades only
        # minpr = min(75, 85, 80) = 75
        # maxpr = max(75, 85, 80) - 100 = 85 - 100 = -15 (buyer adjustment)
        # avepr = (75 + 85 + 80) / 3 = 80
        assert agent.minpr == 75, f"minpr should be 75, got {agent.minpr}"
        assert agent.maxpr == 85 - 100, f"maxpr should be 85-100=-15 for buyer, got {agent.maxpr}"
        assert agent.avepr == 80, f"avepr should be 80, got {agent.avepr}"
        assert agent.has_price_history, "should have price history"

    def test_history_affects_next_period_decisions(self):
        """History from previous period should affect snipe decisions."""
        buyer = Kaplan(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
            num_times=100,
        )

        # Set period 2 with history
        buyer.period = 2
        buyer.minpr = 85
        buyer.maxpr = 95

        # Set up a situation where cask <= minpr triggers jump
        buyer.current_bid = 50  # Wide spread normally
        buyer.current_ask = 82  # <= minpr=85

        buyer.bid_ask(time=50, nobidask=0)
        bid = buyer.bid_ask_response()

        # Should jump to cask due to history bargain
        assert bid == 82, f"History should trigger jump to cask=82, got {bid}"
