"""
Comprehensive test suite for GD (Gjerstad-Dickhaut) trading agent.

Tests cover:
1. Initialization and basic properties
2. Belief calculation (p(a) and q(b))
3. Quote calculation (expected surplus maximization)
4. History management and truncation
5. Buy/sell response logic
6. Integration with market simulation
7. Edge cases and error handling

References:
- traders/gd.py (implementation)
- literature/1998_GD.pdf (algorithm specification)
"""

import pytest
from traders.legacy.gd import GD


class TestGDInitialization:
    """Test GD agent initialization and basic properties."""

    def test_basic_initialization(self):
        """Test that GD agent initializes with correct parameters."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=3,
            valuations=[50, 45, 40],
            price_min=0,
            price_max=100,
            memory_length=8,
            seed=42
        )

        assert agent.player_id == 1
        assert agent.is_buyer is True
        assert agent.num_tokens == 3
        assert agent.valuations == [50, 45, 40]
        assert agent.price_min == 0
        assert agent.price_max == 100
        assert agent.memory_length == 8
        assert agent.num_trades == 0
        assert agent.history == []
        assert agent.trade_count == 0
        assert agent.current_time == 0
        assert agent.current_quote == 0
        assert agent.current_high_bid == 0
        assert agent.current_low_ask == 0

    def test_seller_initialization(self):
        """Test GD seller initialization."""
        agent = GD(
            player_id=2,
            is_buyer=False,
            num_tokens=2,
            valuations=[30, 35],
            price_min=0,
            price_max=100
        )

        assert agent.is_buyer is False
        assert agent.num_tokens == 2
        assert agent.valuations == [30, 35]

    def test_custom_price_bounds(self):
        """Test initialization with custom price bounds."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60],
            price_min=10,
            price_max=80
        )

        assert agent.price_min == 10
        assert agent.price_max == 80


class TestGDBeliefCalculation:
    """Test belief calculation for p(a) and q(b)."""

    def test_belief_ask_accepted_no_history(self):
        """Test p(a) with no historical data (should return uniform prior 0.5)."""
        agent = GD(
            player_id=1,
            is_buyer=False,
            num_tokens=1,
            valuations=[40]
        )

        # No history yet
        prob = agent._belief_ask_accepted(50)
        assert prob == 0.5

    def test_belief_bid_accepted_no_history(self):
        """Test q(b) with no historical data (should return uniform prior 0.5)."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60]
        )

        # No history yet
        prob = agent._belief_bid_accepted(50)
        assert prob == 0.5

    def test_belief_ask_accepted_with_history(self):
        """Test p(a) with known history.

        Paper formula (Definition 10): p(a) = [TA(>=a) + B(>=a)] / [TA(>=a) + B(>=a) + RA(<=a)]
        where TA(>=a) = accepted asks >= a
              B(>=a) = ALL bids >= a (both accepted and rejected)
              RA(<=a) = rejected asks <= a
        """
        agent = GD(
            player_id=1,
            is_buyer=False,
            num_tokens=1,
            valuations=[40]
        )

        # Manually create history:
        # (price, is_bid, accepted)
        agent.history = [
            (45, False, True),   # Ask at 45 accepted (< 50, not in TA(>=50))
            (55, False, False),  # Ask at 55 rejected (>= 50, not in RA(<=50))
            (48, True, False),   # Bid at 48 rejected (< 50, not in B(>=50))
            (52, True, False),   # Bid at 52 rejected (>= 50, in B(>=50): +1)
            (40, False, True),   # Ask at 40 accepted (< 50, not in TA(>=50))
        ]

        # p(50) = [TA(>=50) + B(>=50)] / [TA(>=50) + B(>=50) + RA(<=50)]
        # TA(>=50) = accepted asks >= 50 = 0
        # B(>=50) = all bids >= 50 = 1 (bid at 52, rejected)
        # RA(<=50) = rejected asks <= 50 = 0
        # p(50) = [0 + 1] / [0 + 1 + 0] = 1.0
        prob = agent._belief_ask_accepted(50)
        assert prob == pytest.approx(1.0)

    def test_belief_bid_accepted_with_history(self):
        """Test q(b) with known history.

        Paper formula (Definition 11): q(b) = [TB(<=b) + A(<=b)] / [TB(<=b) + A(<=b) + RB(>b)]
        where TB(<=b) = accepted bids <= b
              A(<=b) = ALL asks <= b (both accepted and rejected)
              RB(>b) = rejected bids > b
        """
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60]
        )

        # Manually create history:
        agent.history = [
            (55, True, True),    # Bid at 55 accepted (> 50, not in TB(<=50))
            (45, True, False),   # Bid at 45 rejected (<= 50, not in RB(>50))
            (48, False, False),  # Ask at 48 rejected (<= 50, in A(<=50): +1)
            (52, False, True),   # Ask at 52 accepted (> 50, not in A(<=50))
            (51, True, True),    # Bid at 51 accepted (> 50, not in TB(<=50))
        ]

        # q(50) = [TB(<=50) + A(<=50)] / [TB(<=50) + A(<=50) + RB(>50)]
        # TB(<=50) = accepted bids <= 50 = 0
        # A(<=50) = all asks <= 50 = 1 (ask at 48, rejected)
        # RB(>50) = rejected bids > 50 = 0
        # q(50) = [0 + 1] / [0 + 1 + 0] = 1.0
        prob = agent._belief_bid_accepted(50)
        assert prob == pytest.approx(1.0)

    def test_belief_curves_with_mixed_history(self):
        """Test belief calculation with mixed accepted/rejected history."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60]
        )

        # Create history with both accepted and rejected bids/asks
        agent.history = [
            (45, False, True),   # Ask at 45 accepted
            (55, False, False),  # Ask at 55 rejected
            (48, True, False),   # Bid at 48 rejected
            (52, True, True),    # Bid at 52 accepted
        ]

        # For buyer at price 50:
        # q(50) = [TB(<=50) + A(<=50)] / [TB(<=50) + A(<=50) + RB(>50)]
        # TB(<=50) = 0 (bid at 52 is > 50, bid at 48 rejected not counted)
        # A(<=50) = 1 (ask at 45 accepted)
        # RB(>50) = 0 (bid at 52 accepted not rejected, bid at 48 is <= 50)
        # q(50) = [0 + 1] / [0 + 1 + 0] = 1.0
        prob_bid = agent._belief_bid_accepted(50)
        assert prob_bid == pytest.approx(1.0)

        # Test seller belief on same history
        seller = GD(player_id=2, is_buyer=False, num_tokens=1, valuations=[40])
        seller.history = agent.history

        # For seller at price 50:
        # p(50) = [TA(>=50) + B(>=50)] / [TA(>=50) + B(>=50) + RA(<=50)]
        # TA(>=50) = 0 (ask at 45 < 50, ask at 55 rejected)
        # B(>=50) = 1 (bid at 52 either accepted or rejected)
        # RA(<=50) = 0 (ask at 45 accepted, ask at 55 > 50)
        # p(50) = [0 + 1] / [0 + 1 + 0] = 1.0
        prob_ask = seller._belief_ask_accepted(50)
        assert prob_ask == pytest.approx(1.0)


class TestGDQuoteCalculation:
    """Test optimal quote calculation via expected surplus maximization."""

    def test_buyer_quote_no_history(self):
        """Test buyer quote with no history (should use uniform prior)."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60],
            price_min=0,
            price_max=100
        )

        agent.bid_ask(time=1, nobidask=0)
        quote = agent.bid_ask_response()

        # With uniform prior (q=0.5), optimal bid maximizes q(b) * (60 - b)
        # Boundary conditions: q(price_min=0) = 0, q(price_max=100) = 1
        # Since q(0) = 0, bidding at 0 has 0 expected surplus
        # The implementation finds optimal bid where q(b) * (V - b) is maximized
        # With uniform prior between boundaries, this will be a low bid near price_min
        assert 0 <= quote <= 60, f"Quote should be in profitable range [0, 60], got {quote}"
        assert quote >= 0, f"Quote should be non-negative, got {quote}"

    def test_seller_quote_no_history(self):
        """Test seller quote with no history."""
        agent = GD(
            player_id=1,
            is_buyer=False,
            num_tokens=1,
            valuations=[40],
            price_min=0,
            price_max=100
        )

        agent.bid_ask(time=1, nobidask=0)
        quote = agent.bid_ask_response()

        # Seller should ask in range [cost, price_max]
        assert 40 <= quote <= 100

    def test_buyer_quote_edge_case_valuation_below_min(self):
        """Test buyer with valuation below minimum price."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[5],
            price_min=10,
            price_max=100
        )

        agent.bid_ask(time=1, nobidask=0)
        quote = agent.bid_ask_response()

        # Cannot bid profitably
        assert quote == 0

    def test_seller_quote_edge_case_cost_above_max(self):
        """Test seller with cost above maximum price."""
        agent = GD(
            player_id=1,
            is_buyer=False,
            num_tokens=1,
            valuations=[95],
            price_min=0,
            price_max=90
        )

        agent.bid_ask(time=1, nobidask=0)
        quote = agent.bid_ask_response()

        # Cannot ask profitably
        assert quote == 0

    def test_quote_after_all_trades(self):
        """Test quote when all tokens have been traded."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=2,
            valuations=[60, 55]
        )

        agent.num_trades = 2  # All tokens traded

        agent.bid_ask(time=1, nobidask=1)
        quote = agent.bid_ask_response()

        assert quote == 0


class TestGDHistoryManagement:
    """Test history tracking and truncation."""

    def test_history_truncation(self):
        """Test that history is truncated based on transaction count.

        Per paper (Definition 7), history is last L transactions.
        Truncation happens when history_trade_count > memory_length.
        """
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60],
            memory_length=2  # Keep last 2 transactions
        )

        # Simulate 3 transactions (each creates 2 history entries: bid + ask)
        agent.history = [
            (50, True, True),   # Trade 1: bid accepted
            (50, False, True),  # Trade 1: ask accepted
            (51, True, True),   # Trade 2: bid accepted
            (51, False, True),  # Trade 2: ask accepted
            (52, True, True),   # Trade 3: bid accepted
            (52, False, True),  # Trade 3: ask accepted
        ]
        agent.history_trade_count = 3  # 3 trades total

        agent._truncate_history()

        # Should keep last 2 transactions (memory_length=2)
        # That's 4 history entries (2 entries per transaction)
        assert len(agent.history) == 4
        # Should keep trades 2 and 3
        assert agent.history[0] == (51, True, True)
        assert agent.history[-1] == (52, False, True)

    def test_history_no_truncation_when_small(self):
        """Test that history is not truncated when below threshold."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60],
            memory_length=10
        )

        # Add few observations (< memory_length * 10 = 100)
        for i in range(5):
            agent.history.append((50, True, True))

        agent._truncate_history()

        # Should not truncate
        assert len(agent.history) == 5


class TestGDBidAskStage:
    """Test bid_ask stage behavior."""

    def test_bid_ask_sets_has_responded(self):
        """Test that bid_ask and bid_ask_response set has_responded flag."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60]
        )

        # Initially False
        assert agent.has_responded is False

        # After bid_ask(), should be False
        agent.bid_ask(time=1, nobidask=0)
        assert agent.has_responded is False

        # After bid_ask_response(), should be True
        quote = agent.bid_ask_response()
        assert agent.has_responded is True
        assert isinstance(quote, int)

    def test_bid_ask_resets_has_responded(self):
        """Test that bid_ask resets has_responded flag."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60]
        )

        # Set has_responded to True
        agent.has_responded = True

        # bid_ask should reset it
        agent.bid_ask(time=5, nobidask=0)
        assert agent.has_responded is False

    def test_bid_ask_result_stores_market_state(self):
        """Test that bid_ask_result stores market state."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60]
        )

        agent.bid_ask_result(
            status=2,
            num_trades=0,
            new_bids=[55],
            new_asks=[],
            high_bid=55,
            high_bidder=1,
            low_ask=0,
            low_asker=0
        )

        assert agent.current_high_bid == 55
        assert agent.current_low_ask == 0
        assert agent.num_trades == 0


class TestGDBuySellStage:
    """Test buy_sell stage behavior."""

    def test_buy_sell_sets_has_responded(self):
        """Test that buy_sell and buy_sell_response set has_responded flag."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60]
        )

        # Set up current quote for the acceptance logic
        agent.current_quote = 50

        # After buy_sell(), should be False
        agent.buy_sell(time=1, nobuysell=0, high_bid=50, low_ask=55, high_bidder=1, low_asker=2)
        assert agent.has_responded is False

        # After buy_sell_response(), should be True
        response = agent.buy_sell_response()
        assert agent.has_responded is True
        assert isinstance(response, bool)

    def test_buyer_accepts_profitable_ask(self):
        """Test that buyer accepts ask when certain surplus >= expected surplus."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[70]
        )

        # Set current quote (optimal bid) to a lower expected surplus
        agent.current_quote = 50  # If p(50) = 0.5, expected = 0.5 * (70-50) = 10

        # Offer a very good ask (certain surplus = 70 - 55 = 15 > 10)
        agent.buy_sell(time=1, nobuysell=0, high_bid=0, low_ask=55, high_bidder=0, low_asker=2)
        response = agent.buy_sell_response()

        # Should accept (certain surplus 15 > expected surplus ~10)
        assert response is True

    def test_buyer_rejects_unprofitable_ask(self):
        """Test that buyer rejects ask when not profitable."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60]
        )

        agent.current_quote = 50

        # Offer an unprofitable ask (valuation < ask)
        agent.buy_sell(time=1, nobuysell=0, high_bid=0, low_ask=65, high_bidder=0, low_asker=2)
        response = agent.buy_sell_response()

        # Should reject
        assert response is False

    def test_seller_accepts_profitable_bid(self):
        """Test that seller accepts bid when certain surplus >= expected surplus."""
        agent = GD(
            player_id=2,
            is_buyer=False,
            num_tokens=1,
            valuations=[40]
        )

        # Set current quote (optimal ask)
        agent.current_quote = 60  # If p(60) = 0.5, expected = 0.5 * (60-40) = 10

        # Offer a very good bid (certain surplus = 55 - 40 = 15 > 10)
        agent.buy_sell(time=1, nobuysell=0, high_bid=55, low_ask=0, high_bidder=1, low_asker=0)
        response = agent.buy_sell_response()

        # Should accept
        assert response is True

    def test_seller_rejects_unprofitable_bid(self):
        """Test that seller rejects bid when not profitable."""
        agent = GD(
            player_id=2,
            is_buyer=False,
            num_tokens=1,
            valuations=[50]
        )

        agent.current_quote = 60

        # Offer an unprofitable bid (bid < cost)
        agent.buy_sell(time=1, nobuysell=0, high_bid=45, low_ask=0, high_bidder=1, low_asker=0)
        response = agent.buy_sell_response()

        # Should reject
        assert response is False

    def test_no_trade_when_all_tokens_traded(self):
        """Test that agent rejects when all tokens are traded."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60]
        )

        agent.num_trades = 1  # All tokens traded

        agent.buy_sell(time=1, nobuysell=1, high_bid=0, low_ask=50, high_bidder=0, low_asker=2)
        response = agent.buy_sell_response()

        assert response is False


class TestGDBuySellResult:
    """Test buy_sell_result and history recording."""

    def test_buyer_accepts_ask_history(self):
        """Test history recording when buyer accepts ask (trade_type=1)."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60]
        )

        # Simulate market state before trade
        agent.current_high_bid = 50
        agent.current_low_ask = 55

        # Buyer accepts ask at 55
        agent.buy_sell_result(
            status=1,
            trade_price=55,
            trade_type=1,
            high_bid=0,
            high_bidder=0,
            low_ask=0,
            low_asker=0
        )

        # History records the transaction at trade_price (both bid and ask accepted)
        # This simplifies history to just record trades, not all standing bids/asks
        assert len(agent.history) == 2
        assert (55, True, True) in agent.history   # Bid at trade price accepted
        assert (55, False, True) in agent.history  # Ask at trade price accepted

        assert agent.trade_count == 1

    def test_seller_accepts_bid_history(self):
        """Test history recording when seller accepts bid (trade_type=2)."""
        agent = GD(
            player_id=2,
            is_buyer=False,
            num_tokens=1,
            valuations=[40]
        )

        # Simulate market state before trade
        agent.current_high_bid = 50
        agent.current_low_ask = 55

        # Seller accepts bid at 50
        agent.buy_sell_result(
            status=1,
            trade_price=50,
            trade_type=2,
            high_bid=0,
            high_bidder=0,
            low_ask=0,
            low_asker=0
        )

        # History records the transaction at trade_price (both bid and ask accepted)
        assert len(agent.history) == 2
        assert (50, True, True) in agent.history   # Bid at trade price accepted
        assert (50, False, True) in agent.history  # Ask at trade price accepted

        assert agent.trade_count == 1

    def test_both_accept_history(self):
        """Test history recording when both accept (trade_type=3)."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60]
        )

        # Simulate crossed market
        agent.current_high_bid = 55
        agent.current_low_ask = 50

        # Both accept at 52 (midpoint or some price)
        agent.buy_sell_result(
            status=1,
            trade_price=52,
            trade_type=3,
            high_bid=0,
            high_bidder=0,
            low_ask=0,
            low_asker=0
        )

        # History should record both bid and ask accepted at trade price
        assert len(agent.history) == 2
        assert (52, True, True) in agent.history   # Bid accepted
        assert (52, False, True) in agent.history  # Ask accepted

        assert agent.trade_count == 1

    def test_no_trade_history(self):
        """Test history recording when no trade occurs."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60]
        )

        # Initial market state
        agent.current_high_bid = 50
        agent.current_low_ask = 55

        # No trade, market state unchanged
        agent.buy_sell_result(
            status=0,
            trade_price=0,
            trade_type=0,
            high_bid=50,
            high_bidder=1,
            low_ask=55,
            low_asker=2
        )

        # No history should be added (prices didn't change)
        assert len(agent.history) == 0
        assert agent.trade_count == 0

    def test_bid_beaten_history(self):
        """Test history when a bid is beaten (no trade occurs).

        GD implementation only records actual trades, not beaten bids/asks.
        """
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60]
        )

        # Our bid was 50
        agent.current_high_bid = 50
        agent.current_low_ask = 0

        # No trade, but high bid is now 55 (someone beat us)
        agent.buy_sell_result(
            status=0,
            trade_price=0,
            trade_type=0,
            high_bid=55,
            high_bidder=3,
            low_ask=0,
            low_asker=0
        )

        # No history added (only trades are recorded)
        assert len(agent.history) == 0
        # Market state updated
        assert agent.current_high_bid == 55

    def test_ask_beaten_history(self):
        """Test history when an ask is beaten (no trade occurs).

        GD implementation only records actual trades, not beaten bids/asks.
        """
        agent = GD(
            player_id=2,
            is_buyer=False,
            num_tokens=1,
            valuations=[40]
        )

        # Our ask was 60
        agent.current_high_bid = 0
        agent.current_low_ask = 60

        # No trade, but low ask is now 55 (someone beat us)
        agent.buy_sell_result(
            status=0,
            trade_price=0,
            trade_type=0,
            high_bid=0,
            high_bidder=0,
            low_ask=55,
            low_asker=3
        )

        # No history added (only trades are recorded)
        assert len(agent.history) == 0
        # Market state updated
        assert agent.current_low_ask == 55


class TestGDIntegration:
    """Integration tests with realistic market scenarios."""

    def test_full_trading_cycle(self):
        """Test a complete trading cycle: bid_ask -> buy_sell -> result."""
        buyer = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[70],
            seed=42
        )

        # Stage 1: Bid/Ask
        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()

        assert buyer.has_responded is True
        assert 0 <= bid <= 70

        # Market resolves bid/ask stage
        buyer.bid_ask_result(
            status=2,
            num_trades=0,
            new_bids=[bid],
            new_asks=[],
            high_bid=bid,
            high_bidder=1,
            low_ask=65,
            low_asker=2
        )

        # Stage 2: Buy/Sell
        buyer.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=bid,
            low_ask=65,
            high_bidder=1,
            low_asker=2
        )
        accept = buyer.buy_sell_response()

        assert buyer.has_responded is True
        assert isinstance(accept, bool)

        # If buyer accepts, simulate trade
        if accept:
            buyer.buy_sell_result(
                status=1,
                trade_price=65,
                trade_type=1,
                high_bid=0,
                high_bidder=0,
                low_ask=0,
                low_asker=0
            )

            assert buyer.num_trades == 1
            assert len(buyer.history) > 0

    def test_multiple_trading_periods(self):
        """Test agent behavior over multiple tokens."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=3,
            valuations=[70, 65, 60],
            seed=42
        )

        # First trade
        agent.bid_ask(time=1, nobidask=0)
        bid1 = agent.bid_ask_response()
        assert 0 <= bid1 <= 70

        # Simulate trade
        agent.buy_sell_result(
            status=1,
            trade_price=68,
            trade_type=1,
            high_bid=0,
            high_bidder=0,
            low_ask=0,
            low_asker=0
        )

        assert agent.num_trades == 1

        # Second trade
        agent.bid_ask(time=2, nobidask=0)
        bid2 = agent.bid_ask_response()
        assert 0 <= bid2 <= 65  # Second valuation

        # Agent should have history from first trade
        assert len(agent.history) > 0

    def test_agent_learns_from_history(self):
        """Test that agent's beliefs update based on observed trades."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60]
        )

        # Initially, no history (uniform prior)
        prob_before = agent._belief_bid_accepted(55)
        assert prob_before == 0.5

        # Add positive history (high bids accepted)
        agent.history = [
            (55, True, True),
            (56, True, True),
            (57, True, True),
        ]

        # Now probability should be higher
        prob_after = agent._belief_bid_accepted(55)
        assert prob_after > prob_before

    def test_seller_buyer_symmetry(self):
        """Test that buyer and seller logic is symmetric."""
        buyer = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60],
            seed=42
        )

        seller = GD(
            player_id=2,
            is_buyer=False,
            num_tokens=1,
            valuations=[40],
            seed=42
        )

        # Both should generate valid quotes
        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()

        seller.bid_ask(time=1, nobidask=0)
        ask = seller.bid_ask_response()

        assert 0 <= bid <= 60
        assert 40 <= ask <= 100


class TestGDEdgeCases:
    """Test edge cases and error conditions."""

    def test_agent_with_single_token(self):
        """Test agent with only one token."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[50]
        )

        agent.bid_ask(time=1, nobidask=0)
        quote = agent.bid_ask_response()

        assert isinstance(quote, int)

    def test_agent_with_many_tokens(self):
        """Test agent with many tokens."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=10,
            valuations=list(range(100, 90, -1))
        )

        agent.bid_ask(time=1, nobidask=0)
        quote = agent.bid_ask_response()

        assert isinstance(quote, int)

    def test_extreme_price_bounds(self):
        """Test agent with extreme price bounds."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[50],
            price_min=0,
            price_max=1000
        )

        agent.bid_ask(time=1, nobidask=0)
        quote = agent.bid_ask_response()

        assert 0 <= quote <= 50

    def test_large_memory_length(self):
        """Test agent with large memory length."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60],
            memory_length=1000
        )

        # Add many observations
        for i in range(100):
            agent.history.append((50, True, i % 2 == 0))

        agent._truncate_history()

        # Should not truncate (100 < 1000 * 10)
        assert len(agent.history) == 100

    def test_zero_memory_length(self):
        """Test agent with zero memory length (edge case)."""
        agent = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60],
            memory_length=0
        )

        # Should still work (no truncation)
        agent.bid_ask(time=1, nobidask=0)
        quote = agent.bid_ask_response()

        assert isinstance(quote, int)

    def test_deterministic_with_seed(self):
        """Test that agent is deterministic with same seed."""
        agent1 = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60],
            seed=123
        )

        agent2 = GD(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[60],
            seed=123
        )

        # Note: GD is deterministic in quote calculation (no randomness)
        # So quotes should be identical
        agent1.bid_ask(time=1, nobidask=0)
        quote1 = agent1.bid_ask_response()

        agent2.bid_ask(time=1, nobidask=0)
        quote2 = agent2.bid_ask_response()

        assert quote1 == quote2
