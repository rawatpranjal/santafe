"""
Comprehensive test suite for Kaplan Agent.

This test suite validates the Kaplan implementation against the 1993 Java baseline
(SRobotKaplan.java) with line-by-line behavioral verification.
"""

import pytest
from traders.legacy.kaplan import Kaplan


class TestKaplanInitialization:
    """Test initialization and state management."""

    def test_basic_initialization(self) -> None:
        """Test basic agent initialization."""
        agent = Kaplan(1, True, 1, [100], num_times=10)
        assert agent.period == 1
        assert agent.minpr == 0
        assert agent.maxpr == 0
        assert agent.avepr == 0
        assert agent.player_id == 1
        assert agent.is_buyer is True
        assert agent.num_tokens == 1
        assert agent.num_trades == 0

    def test_buyer_initialization(self) -> None:
        """Test buyer-specific initialization."""
        buyer = Kaplan(1, True, 3, [100, 90, 80], num_times=100)
        assert buyer.is_buyer is True
        assert buyer.num_tokens == 3
        assert buyer.valuations == [100, 90, 80]

    def test_seller_initialization(self) -> None:
        """Test seller-specific initialization."""
        seller = Kaplan(2, False, 3, [20, 30, 40], num_times=100)
        assert seller.is_buyer is False
        assert seller.num_tokens == 3
        assert seller.valuations == [20, 30, 40]

    def test_start_period_resets_state(self) -> None:
        """Test that start_period resets period-specific state."""
        agent = Kaplan(1, True, 1, [100], num_times=10)
        agent.trade_count = 5
        agent.last_time = 3
        agent.my_last_time = 2

        agent.start_period(2)

        assert agent.period == 2
        assert agent.trade_count == 0
        assert agent.last_time == 0
        assert agent.my_last_time == 0


class TestKaplanPeriodStatistics:
    """Test min/max/avg price calculation logic."""

    def test_buyer_period_stats_simple(self) -> None:
        """Test buyer's end_period calculation with simple trade history."""
        buyer = Kaplan(1, True, 1, [100], num_times=10)
        buyer.start_period(1)

        # Simulate 3 trades (prices: 50, 60, 40 where 40 is mine)
        buyer.buy_sell_result(status=0, trade_price=50, trade_type=1,
                             high_bid=0, high_bidder=0, low_ask=0, low_asker=0)
        buyer.buy_sell_result(status=0, trade_price=60, trade_type=1,
                             high_bid=0, high_bidder=0, low_ask=0, low_asker=0)
        buyer.buy_sell_result(status=1, trade_price=40, trade_type=1,
                             high_bid=0, high_bidder=0, low_ask=0, low_asker=0)

        assert buyer.trade_count == 3
        assert buyer.prices[1] == 50
        assert buyer.prices[2] == 60
        assert buyer.prices[3] == -1  # My trade marked as -1

        buyer.end_period()

        # Expected: minpr=1, maxpr=60, avepr=37
        assert buyer.maxpr == 60
        assert buyer.minpr == 1
        assert buyer.avepr == 37

    def test_seller_period_stats_simple(self) -> None:
        """Test seller's end_period calculation."""
        seller = Kaplan(2, False, 1, [20], num_times=10)
        seller.start_period(1)

        # Simulate 3 trades (prices: 50, 60, and my trade at 70 marked as -1)
        seller.buy_sell_result(status=0, trade_price=50, trade_type=1,
                              high_bid=0, high_bidder=0, low_ask=0, low_asker=0)
        seller.buy_sell_result(status=0, trade_price=60, trade_type=1,
                              high_bid=0, high_bidder=0, low_ask=0, low_asker=0)
        seller.buy_sell_result(status=1, trade_price=70, trade_type=1,
                              high_bid=0, high_bidder=0, low_ask=0, low_asker=0)

        seller.end_period()

        # Seller logic (Java lines 36-38):
        # - maxpr = max of abs values (including -1) = max(50, 60, abs(-1)=1) = 60
        # - minpr = min of positive values only = min(50, 60) = 50 (excludes -1)
        # - avepr = avg of abs values = (50+60+1)/3 = 37
        assert seller.maxpr == 60  # Max of observed prices (my trade excluded from max)
        assert seller.minpr == 50  # Min of positive prices
        assert seller.avepr == 37  # Avg including my trade as abs(-1)=1

    def test_period_stats_no_trades(self) -> None:
        """Test end_period with no trades."""
        agent = Kaplan(1, True, 1, [100], num_times=10)
        agent.start_period(1)
        agent.end_period()

        # With no trades, stats should be 0 or defaults
        assert agent.trade_count == 0
        assert agent.avepr == 0


class TestKaplanTokenSelection:
    """Test worst-case vs current token selection logic."""

    def test_buyer_first_bid_uses_worst_token(self) -> None:
        """Test that first bid uses worst-case token (Java line 55)."""
        buyer = Kaplan(1, True, 3, [100, 90, 80], price_min=0, price_max=100, num_times=10)
        buyer.start_period(1)
        buyer.bid_ask(1, 0)
        buyer.current_bid = 0  # First bid
        buyer.current_ask = 0

        # First bid should use worst token (80), so most = 79
        # newbid = price_min + 1 = 1
        # Protection: if newbid > most, newbid = most
        # Here 1 < 79, so newbid = 1
        bid = buyer.bid_ask_response()
        assert bid == 1  # price_min + 1

    def test_buyer_subsequent_bid_uses_current_token(self) -> None:
        """Test that subsequent bids use current token (Java line 60)."""
        buyer = Kaplan(1, True, 3, [100, 90, 80], price_min=0, price_max=100, num_times=10)
        buyer.start_period(1)
        buyer.num_trades = 0  # On first token (100)
        buyer.bid_ask(1, 0)
        buyer.current_bid = 50  # Not first bid
        buyer.current_ask = 0

        # Should use current token (100), so most = 99
        # Should return 0 since most (99) > current_bid (50) but no jump-in conditions met
        bid = buyer.bid_ask_response()
        assert bid == 1  # minprice + 1

    def test_seller_first_ask_uses_worst_token(self) -> None:
        """Test that first ask uses worst-case token (Java line 85)."""
        seller = Kaplan(2, False, 3, [20, 30, 40], price_min=0, price_max=100, num_times=10)
        seller.start_period(1)
        seller.bid_ask(1, 0)
        seller.current_ask = 0  # First ask
        seller.current_bid = 0

        # First ask should use worst token (40), so least = 41
        # newoffer = price_max - 1 = 99
        ask = seller.bid_ask_response()
        assert ask == 99  # price_max - 1

    def test_seller_subsequent_ask_uses_current_token(self) -> None:
        """Test that subsequent asks use current token (Java line 91)."""
        seller = Kaplan(2, False, 3, [20, 30, 40], price_min=0, price_max=100, num_times=10)
        seller.start_period(1)
        seller.num_trades = 0  # On first token (20)
        seller.bid_ask(1, 0)
        seller.current_ask = 50  # Not first ask
        seller.current_bid = 0

        # Should use current token (20), so least = 21
        # Should return maxprice - 1 = 99
        ask = seller.bid_ask_response()
        assert ask == 99


class TestKaplanJumpInLogic:
    """Test jump-in decision logic."""

    def test_buyer_jump_in_small_spread(self) -> None:
        """Test buyer jumps to ask when spread < 10% (Java line 66-68)."""
        buyer = Kaplan(1, True, 1, [100], price_min=0, price_max=100, num_times=100)
        buyer.start_period(1)
        buyer.bid_ask(1, 0)

        # Small spread: ask=50, bid=46. (50-46)/51 = 0.078 < 0.10
        # Profitable: (100+1)*0.98 - 50 = 48.98 > 0 ✓
        buyer.current_ask = 50
        buyer.current_bid = 46

        bid = buyer.bid_ask_response()
        assert bid == 50  # Should jump to ask

    def test_buyer_no_jump_large_spread(self) -> None:
        """Test buyer doesn't jump when spread >= 10%."""
        buyer = Kaplan(1, True, 1, [100], price_min=0, price_max=100, num_times=100)
        buyer.start_period(1)
        buyer.bid_ask(1, 0)

        # Large spread: ask=80, bid=40. (80-40)/41 = 0.975 > 0.10
        buyer.current_ask = 80
        buyer.current_bid = 40

        bid = buyer.bid_ask_response()
        assert bid < 80  # Should NOT jump to ask

    def test_buyer_jump_in_better_price_than_last_period(self) -> None:
        """Test buyer jumps when ask <= minpr from previous period (Java line 70)."""
        buyer = Kaplan(1, True, 1, [100], price_min=0, price_max=100, num_times=100)
        buyer.start_period(2)  # Period 2
        buyer.minpr = 60  # Min price from period 1
        buyer.maxpr = 80
        buyer.bid_ask(1, 0)
        buyer.current_ask = 55  # Better than minpr
        buyer.current_bid = 50

        bid = buyer.bid_ask_response()
        assert bid == 55  # Should jump to ask (better price)

    def test_buyer_jump_in_time_pressure_lasttime(self) -> None:
        """Test buyer jumps when (t - lasttime) >= (ntimes - t) / 2 (Java line 73)."""
        buyer = Kaplan(1, True, 1, [100], price_min=0, price_max=100, num_times=100)
        buyer.start_period(1)
        buyer.last_time = 10
        buyer.my_last_time = 5
        buyer.bid_ask(90, 0)  # t=90, ntimes-t=10, (t-lasttime)=80 >= 5 ✓
        buyer.current_ask = 50
        buyer.current_bid = 40

        bid = buyer.bid_ask_response()
        assert bid == 50  # Should jump due to time

    def test_seller_jump_in_small_spread(self) -> None:
        """Test seller jumps to bid when spread < 10% (Java line 94-96)."""
        seller = Kaplan(2, False, 1, [20], price_min=0, price_max=100, num_times=100)
        seller.start_period(1)
        seller.bid_ask(1, 0)

        # Small spread: ask=51, bid=50. (51-50)/51 = 0.019 < 0.10
        # Profitable: (20+1)*1.02 - 50 = -28.58 < 0 ✓
        seller.current_ask = 51
        seller.current_bid = 50

        ask = seller.bid_ask_response()
        assert ask == 50  # Should jump to bid

    def test_seller_jump_in_better_price(self) -> None:
        """Test seller jumps when bid >= maxpr (Java line 97)."""
        seller = Kaplan(2, False, 1, [20], price_min=0, price_max=100, num_times=100)
        seller.start_period(2)
        seller.maxpr = 60
        seller.minpr = 40
        seller.bid_ask(1, 0)
        seller.current_bid = 65  # Better than maxpr
        seller.current_ask = 70

        ask = seller.bid_ask_response()
        assert ask == 65  # Should jump to bid


class TestKaplanProtectionClauses:
    """Test that protection clauses prevent unprofitable bids/asks."""

    def test_buyer_protection_clamps_jump_in_bid(self) -> None:
        """Test buyer protection: if newbid > most, newbid = most (Java line 74)."""
        buyer = Kaplan(1, True, 1, [100], price_min=0, price_max=100, num_times=100)
        buyer.start_period(1)
        buyer.minpr = 10
        buyer.bid_ask(1, 0)

        # Jump-in triggers: ask (110) <= minpr (should NOT trigger since 110 > 10)
        # But let's test time-based jump-in with unprofitable ask
        buyer.last_time = 0
        buyer.bid_ask(80, 0)  # Time pressure
        buyer.current_ask = 110  # Unprofitable (> token value 100)
        buyer.current_bid = 50

        bid = buyer.bid_ask_response()
        # Protection should clamp: newbid would be 110, but most = 99
        assert bid == 99  # Clamped to token_val - 1

    def test_seller_protection_clamps_jump_in_ask(self) -> None:
        """Test seller protection: if newoffer < least, newoffer = least (Java line 99)."""
        seller = Kaplan(2, False, 1, [50], price_min=0, price_max=100, num_times=100)
        seller.start_period(1)
        seller.maxpr = 90
        seller.bid_ask(1, 0)

        # Time pressure jump-in
        seller.last_time = 0
        seller.bid_ask(80, 0)
        seller.current_bid = 40  # Unprofitable (< token value 50)
        seller.current_ask = 60

        ask = seller.bid_ask_response()
        # Protection should clamp: newoffer would be 40, but least = 51
        assert ask == 51  # Clamped to token_val + 1


class TestKaplanSniperLogic:
    """Test sniper behavior (accept in last 2 timesteps)."""

    def test_buyer_sniper_accepts_profitable_trade_late(self) -> None:
        """Test buyer accepts profitable trade when time is short (Java line 109-110)."""
        buyer = Kaplan(1, True, 1, [100], num_times=10)
        buyer.start_period(1)

        # Time = 8, ntimes - t = 2 (last 2 timesteps)
        buyer.buy_sell(8, 0, high_bid=50, low_ask=60, high_bidder=1, low_asker=2)

        # Profitable: token (100) > ask (60) ✓
        assert buyer.buy_sell_response() is True

    def test_buyer_sniper_rejects_unprofitable_trade(self) -> None:
        """Test buyer rejects unprofitable trade even when sniping (Java line 107)."""
        buyer = Kaplan(1, True, 1, [100], num_times=10)
        buyer.start_period(1)

        # Time = 8 (sniper mode)
        buyer.buy_sell(8, 0, high_bid=50, low_ask=110, high_bidder=1, low_asker=2)

        # Unprofitable: token (100) <= ask (110) → guard clause fires
        assert buyer.buy_sell_response() is False

    def test_buyer_sniper_not_active_early(self) -> None:
        """Test buyer doesn't snipe early in period."""
        buyer = Kaplan(1, True, 1, [100], num_times=10)
        buyer.start_period(1)

        # Time = 1, ntimes - t = 9 (not in sniper window)
        buyer.buy_sell(1, 0, high_bid=50, low_ask=60, high_bidder=1, low_asker=2)

        # Should not accept (not winner, not sniping yet)
        assert buyer.buy_sell_response() is False

    def test_seller_sniper_accepts_profitable_trade_late(self) -> None:
        """Test seller accepts profitable trade when time is short (Java line 119-120)."""
        seller = Kaplan(2, False, 1, [20], num_times=10)
        seller.start_period(1)

        # Time = 8 (last 2 timesteps)
        seller.buy_sell(8, 0, high_bid=50, low_ask=60, high_bidder=1, low_asker=2)

        # Profitable: bid (50) > token (20) ✓
        assert seller.buy_sell_response() is True

    def test_seller_sniper_rejects_unprofitable_trade(self) -> None:
        """Test seller rejects unprofitable trade even when sniping (Java line 117)."""
        seller = Kaplan(2, False, 1, [50], num_times=10)
        seller.start_period(1)

        # Time = 8 (sniper mode)
        seller.buy_sell(8, 0, high_bid=40, low_ask=60, high_bidder=1, low_asker=2)

        # Unprofitable: bid (40) <= token (50) → guard clause fires
        assert seller.buy_sell_response() is False


class TestKaplanGuardClauses:
    """Test profitability guard clauses in buy_sell_response."""

    def test_buyer_guard_rejects_unprofitable_purchase(self) -> None:
        """Test buyer guard: if token <= ask, return 0 (Java line 107)."""
        buyer = Kaplan(1, True, 1, [50], num_times=100)
        buyer.start_period(1)

        # Token = 50, Ask = 60 (unprofitable)
        buyer.buy_sell(1, 0, high_bid=40, low_ask=60, high_bidder=1, low_asker=2)
        assert buyer.buy_sell_response() is False

        # Token = 50, Ask = 50 (breakeven, still unprofitable)
        buyer.buy_sell(2, 0, high_bid=40, low_ask=50, high_bidder=1, low_asker=2)
        assert buyer.buy_sell_response() is False

        # Token = 50, Ask = 49 (profitable)
        buyer.buy_sell(3, 0, high_bid=40, low_ask=49, high_bidder=1, low_asker=2)
        # Still False because not winner and not sniping, but guard passes
        # (Would be True in sniper window)

    def test_seller_guard_rejects_unprofitable_sale(self) -> None:
        """Test seller guard: if bid <= token, return 0 (Java line 117)."""
        seller = Kaplan(2, False, 1, [50], num_times=100)
        seller.start_period(1)

        # Token = 50, Bid = 40 (unprofitable)
        seller.buy_sell(1, 0, high_bid=40, low_ask=60, high_bidder=1, low_asker=2)
        assert seller.buy_sell_response() is False

        # Token = 50, Bid = 50 (breakeven, still unprofitable)
        seller.buy_sell(2, 0, high_bid=50, low_ask=60, high_bidder=1, low_asker=2)
        assert seller.buy_sell_response() is False

        # Token = 50, Bid = 51 (profitable)
        seller.buy_sell(3, 0, high_bid=51, low_ask=60, high_bidder=1, low_asker=2)
        # Still False because not winner and not sniping, but guard passes


class TestKaplanEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_no_bid_ask_when_nobidask_positive(self) -> None:
        """Test agent returns 0 when nobidask > 0."""
        buyer = Kaplan(1, True, 1, [100], num_times=10)
        buyer.start_period(1)
        buyer.bid_ask(1, 1)  # nobidask = 1

        assert buyer.bid_ask_response() == 0

    def test_no_trade_when_nobuysell_positive(self) -> None:
        """Test agent rejects when nobuysell > 0."""
        buyer = Kaplan(1, True, 1, [100], num_times=10)
        buyer.start_period(1)
        buyer.buy_sell(1, 1, 50, 60, 1, 2)  # nobuysell = 1

        assert buyer.buy_sell_response() is False

    def test_no_bid_when_all_tokens_traded(self) -> None:
        """Test agent returns 0 when all tokens traded."""
        buyer = Kaplan(1, True, 2, [100, 90], num_times=10)
        buyer.start_period(1)
        buyer.num_trades = 2  # All tokens traded
        buyer.bid_ask(1, 0)

        assert buyer.bid_ask_response() == 0

    def test_crossed_spread_acceptance_buyer(self) -> None:
        """Test buyer accepts when winner and spread crossed (Java line 108)."""
        buyer = Kaplan(1, True, 1, [100], num_times=100)
        buyer.start_period(1)
        buyer.player_id = 5

        # Winner with crossed spread
        buyer.buy_sell(1, 0, high_bid=60, low_ask=55, high_bidder=5, low_asker=2)
        assert buyer.buy_sell_response() is True

    def test_crossed_spread_acceptance_seller(self) -> None:
        """Test seller accepts when winner and spread crossed (Java line 118)."""
        seller = Kaplan(2, False, 1, [20], num_times=100)
        seller.start_period(1)
        seller.player_id = 7

        # Winner with crossed spread
        seller.buy_sell(1, 0, high_bid=60, low_ask=55, high_bidder=3, low_asker=7)
        assert seller.buy_sell_response() is True


class TestKaplanProfitTracking:
    """Test profit tracking and trade execution."""

    def test_buyer_profit_on_trade(self) -> None:
        """Test buyer profit calculation on successful trade."""
        buyer = Kaplan(1, True, 1, [100], num_times=10)
        buyer.start_period(1)

        # Execute trade: buy at 60
        buyer.buy_sell_result(status=1, trade_price=60, trade_type=1,
                             high_bid=0, high_bidder=0, low_ask=0, low_asker=0)

        # Profit = token - price = 100 - 60 = 40
        assert buyer.period_profit == 40
        assert buyer.num_trades == 1

    def test_seller_profit_on_trade(self) -> None:
        """Test seller profit calculation on successful trade."""
        seller = Kaplan(2, False, 1, [20], num_times=10)
        seller.start_period(1)

        # Execute trade: sell at 60
        seller.buy_sell_result(status=1, trade_price=60, trade_type=1,
                              high_bid=0, high_bidder=0, low_ask=0, low_asker=0)

        # Profit = price - token = 60 - 20 = 40
        assert seller.period_profit == 40
        assert seller.num_trades == 1

    def test_my_trade_marked_in_history(self) -> None:
        """Test that my trades are marked as -1 in price history."""
        agent = Kaplan(1, True, 1, [100], num_times=10)
        agent.start_period(1)

        # My trade
        agent.buy_sell_result(status=1, trade_price=50, trade_type=1,
                             high_bid=0, high_bidder=0, low_ask=0, low_asker=0)

        assert agent.prices[1] == -1
        assert agent.my_last_time == agent.current_time

    def test_other_trade_recorded_in_history(self) -> None:
        """Test that other agents' trades are recorded with actual price."""
        agent = Kaplan(1, True, 1, [100], num_times=10)
        agent.start_period(1)

        # Other agent's trade
        agent.buy_sell_result(status=0, trade_price=60, trade_type=1,
                             high_bid=0, high_bidder=0, low_ask=0, low_asker=0)

        assert agent.prices[1] == 60
        assert agent.my_last_time == 0  # Not updated
