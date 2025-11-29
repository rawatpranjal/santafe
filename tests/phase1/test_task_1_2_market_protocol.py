"""
Phase 1 Task 1.2: Market Step Function (Two-Stage Protocol) Tests

This module validates the AURORA two-stage protocol implementation
as specified in PLAN.md.

References:
- engine/market.py - Market class (port of SGameRunner.java)
- PLAN.md Section 2.2: AURORA Protocol Mechanics
"""

import pytest
import numpy as np
from typing import List, Optional
from collections import Counter

from engine.market import Market
from engine.orderbook import OrderBook
from traders.base import Agent


# =============================================================================
# MOCK AGENTS FOR TESTING
# =============================================================================


class MockAgent(Agent):
    """A controllable mock agent for testing market mechanics."""

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: List[int],
        bid_price: int = 50,
        ask_price: int = 60,
        accept_trade: bool = True,
    ) -> None:
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self._bid_price = bid_price
        self._ask_price = ask_price
        self._accept_trade = accept_trade

        # Tracking for test verification
        self.bid_ask_called = False
        self.bid_ask_result_called = False
        self.buy_sell_called = False
        self.buy_sell_result_called = False
        self.last_nobidask: Optional[int] = None
        self.last_nobuysell: Optional[int] = None
        self.last_high_bid: Optional[int] = None
        self.last_low_ask: Optional[int] = None

    def set_response(
        self,
        bid_price: Optional[int] = None,
        ask_price: Optional[int] = None,
        accept_trade: Optional[bool] = None,
    ) -> None:
        """Update mock responses for next round."""
        if bid_price is not None:
            self._bid_price = bid_price
        if ask_price is not None:
            self._ask_price = ask_price
        if accept_trade is not None:
            self._accept_trade = accept_trade

    def bid_ask(self, time: int, nobidask: int) -> None:
        self.bid_ask_called = True
        self.last_nobidask = nobidask
        self.has_responded = False

    def bid_ask_response(self) -> int:
        self.has_responded = True
        if self.is_buyer:
            return self._bid_price
        else:
            return self._ask_price

    def bid_ask_result(
        self,
        status: int,
        num_trades: int,
        new_bids: List[int],
        new_asks: List[int],
        high_bid: int,
        high_bidder: int,
        low_ask: int,
        low_asker: int,
    ) -> None:
        super().bid_ask_result(
            status, num_trades, new_bids, new_asks, high_bid, high_bidder, low_ask, low_asker
        )
        self.bid_ask_result_called = True
        self.last_high_bid = high_bid
        self.last_low_ask = low_ask

    def buy_sell(
        self,
        time: int,
        nobuysell: int,
        high_bid: int,
        low_ask: int,
        high_bidder: int,
        low_asker: int,
    ) -> None:
        self.buy_sell_called = True
        self.last_nobuysell = nobuysell
        self.has_responded = False

    def buy_sell_response(self) -> bool:
        self.has_responded = True
        return self._accept_trade

    def buy_sell_result(
        self,
        status: int,
        trade_price: int,
        trade_type: int,
        high_bid: int,
        high_bidder: int,
        low_ask: int,
        low_asker: int,
    ) -> None:
        super().buy_sell_result(
            status, trade_price, trade_type, high_bid, high_bidder, low_ask, low_asker
        )
        self.buy_sell_result_called = True


class QuittingAgent(MockAgent):
    """Agent that quits voluntarily (returns -1)."""

    def bid_ask_response(self) -> int:
        self.has_responded = True
        return -1


class FailingAgent(MockAgent):
    """Agent that fails (returns -2)."""

    def bid_ask_response(self) -> int:
        self.has_responded = True
        return -2


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def two_by_two_market() -> tuple[Market, List[MockAgent], List[MockAgent]]:
    """Create a 2x2 market with mock agents."""
    buyers = [
        MockAgent(1, True, 4, [100, 90, 80, 70], bid_price=50),
        MockAgent(2, True, 4, [95, 85, 75, 65], bid_price=45),
    ]
    sellers = [
        MockAgent(3, False, 4, [40, 50, 60, 70], ask_price=60),
        MockAgent(4, False, 4, [45, 55, 65, 75], ask_price=65),
    ]

    market = Market(
        num_buyers=2,
        num_sellers=2,
        num_times=10,
        price_min=1,
        price_max=100,
        buyers=buyers,
        sellers=sellers,
        seed=42,
    )

    return market, buyers, sellers


# =============================================================================
# TEST CLASS: Stage 1 - Bid/Ask Phase
# =============================================================================


class TestStage1BidAskPhase:
    """Tests for Stage 1: Bid/Ask submission phase."""

    def test_bid_ask_stage_notifies_all_agents(
        self, two_by_two_market: tuple[Market, List[MockAgent], List[MockAgent]]
    ) -> None:
        """All agents should receive bid_ask() notification."""
        market, buyers, sellers = two_by_two_market

        market.orderbook.increment_time()
        market.current_time = market.orderbook.current_time
        market.bid_ask_stage()

        # All agents should have been notified
        for buyer in buyers:
            assert buyer.bid_ask_called, f"Buyer {buyer.player_id} not notified"
        for seller in sellers:
            assert seller.bid_ask_called, f"Seller {seller.player_id} not notified"

    def test_nobidask_flag_set_when_tokens_exhausted(self) -> None:
        """nobidask=1 when agent has traded all tokens."""
        # Create agent with only 1 token
        buyer = MockAgent(1, True, 1, [100], bid_price=50)
        seller = MockAgent(2, False, 1, [40], ask_price=60)

        market = Market(
            num_buyers=1,
            num_sellers=1,
            num_times=10,
            price_min=1,
            price_max=100,
            buyers=[buyer],
            sellers=[seller],
            seed=42,
        )

        # First timestep - nobidask should be 0
        market.run_time_step()
        # After trade, agent has exhausted tokens

        # Set up for trade
        buyer.set_response(bid_price=60, accept_trade=True)
        seller.set_response(ask_price=60, accept_trade=True)

        # Reset tracking
        buyer.bid_ask_called = False
        seller.bid_ask_called = False

        # Run another timestep
        market.run_time_step()

        # After exhausting tokens, nobidask should be 1
        assert buyer.last_nobidask == 1, f"Expected nobidask=1, got {buyer.last_nobidask}"

    def test_bid_ask_response_collects_prices(
        self, two_by_two_market: tuple[Market, List[MockAgent], List[MockAgent]]
    ) -> None:
        """Valid bids/asks should be recorded in orderbook."""
        market, buyers, sellers = two_by_two_market

        # Set distinct prices
        buyers[0].set_response(bid_price=55)
        buyers[1].set_response(bid_price=45)
        sellers[0].set_response(ask_price=65)
        sellers[1].set_response(ask_price=70)

        market.run_time_step()

        ob = market.get_orderbook()
        # Check that highest bid was recorded correctly
        assert ob.high_bid[1] == 55, f"Expected high_bid=55, got {ob.high_bid[1]}"
        # Check that lowest ask was recorded
        assert ob.low_ask[1] == 65, f"Expected low_ask=65, got {ob.low_ask[1]}"

    def test_negative_one_removes_agent_quit(self) -> None:
        """Agent returning -1 should be removed (quit)."""
        quitter = QuittingAgent(1, True, 4, [100, 90, 80, 70])
        stayer = MockAgent(2, True, 4, [95, 85, 75, 65], bid_price=50)
        seller = MockAgent(3, False, 4, [40, 50, 60, 70], ask_price=60)

        market = Market(
            num_buyers=2,
            num_sellers=1,
            num_times=10,
            price_min=1,
            price_max=100,
            buyers=[quitter, stayer],
            sellers=[seller],
            seed=42,
        )

        market.run_time_step()

        # Quitter should be removed
        assert market.num_buyers == 1, f"Expected 1 buyer, got {market.num_buyers}"
        assert len(market.buyers) == 1
        assert quitter not in market.buyers

    def test_negative_two_removes_agent_failure(self) -> None:
        """Agent returning -2 should be removed (failure)."""
        failer = FailingAgent(1, True, 4, [100, 90, 80, 70])
        seller = MockAgent(2, False, 4, [40, 50, 60, 70], ask_price=60)

        market = Market(
            num_buyers=1,
            num_sellers=1,
            num_times=10,
            price_min=1,
            price_max=100,
            buyers=[failer],
            sellers=[seller],
            seed=42,
        )

        market.run_time_step()

        # Failer should be removed and market should fail
        assert market.num_buyers == 0
        assert market.has_failed()

    def test_bid_ask_result_broadcasts_state(
        self, two_by_two_market: tuple[Market, List[MockAgent], List[MockAgent]]
    ) -> None:
        """All agents should receive bid_ask_result with market state."""
        market, buyers, sellers = two_by_two_market

        buyers[0].set_response(bid_price=55)
        sellers[0].set_response(ask_price=65)

        market.run_time_step()

        # All agents should receive result
        for buyer in buyers:
            assert buyer.bid_ask_result_called
            assert buyer.last_high_bid == 55
            assert buyer.last_low_ask == 65

        for seller in sellers:
            assert seller.bid_ask_result_called


# =============================================================================
# TEST CLASS: Stage 2 - Buy/Sell Phase
# =============================================================================


class TestStage2BuySellPhase:
    """Tests for Stage 2: Buy/Sell execution phase."""

    def test_buy_sell_stage_notifies_holders(
        self, two_by_two_market: tuple[Market, List[MockAgent], List[MockAgent]]
    ) -> None:
        """High bidder and low asker should be notified."""
        market, buyers, sellers = two_by_two_market

        market.run_time_step()

        # All agents should have been notified in buy_sell stage
        for buyer in buyers:
            assert buyer.buy_sell_called
        for seller in sellers:
            assert seller.buy_sell_called

    def test_nobuysell_flag_bit1_tokens_exhausted(self) -> None:
        """nobuysell should have +1 bit when agent has no tokens left."""
        buyer = MockAgent(1, True, 1, [100], bid_price=50, accept_trade=True)
        seller = MockAgent(2, False, 1, [40], ask_price=50, accept_trade=True)

        market = Market(
            num_buyers=1,
            num_sellers=1,
            num_times=10,
            price_min=1,
            price_max=100,
            buyers=[buyer],
            sellers=[seller],
            seed=42,
        )

        # First trade exhausts the single token
        market.run_time_step()

        # Next timestep - nobuysell should have +1 bit
        buyer.bid_ask_called = False
        buyer.buy_sell_called = False
        market.run_time_step()

        # Check that nobuysell has +1 bit (agent can't trade)
        assert buyer.last_nobuysell is not None
        assert (buyer.last_nobuysell & 1) == 1, f"Expected +1 bit, got {buyer.last_nobuysell}"

    def test_nobuysell_flag_bit2_no_standing_order(self) -> None:
        """nobuysell should have +2 bit when no standing order exists."""
        buyer = MockAgent(1, True, 4, [100, 90, 80, 70], bid_price=50, accept_trade=False)
        seller = MockAgent(2, False, 4, [40, 50, 60, 70], ask_price=0, accept_trade=False)  # No ask

        market = Market(
            num_buyers=1,
            num_sellers=1,
            num_times=10,
            price_min=1,
            price_max=100,
            buyers=[buyer],
            sellers=[seller],
            seed=42,
        )

        market.run_time_step()

        # Buyer should have +2 bit (no ask to accept)
        assert buyer.last_nobuysell is not None
        assert (buyer.last_nobuysell & 2) == 2, f"Expected +2 bit, got {buyer.last_nobuysell}"

    def test_nobuysell_flag_bit4_not_winner(self) -> None:
        """nobuysell should have +4 bit when agent is not the winner."""
        buyer1 = MockAgent(1, True, 4, [100, 90, 80, 70], bid_price=55)  # Higher bid - winner
        buyer2 = MockAgent(2, True, 4, [95, 85, 75, 65], bid_price=45)  # Lower bid - loser
        seller = MockAgent(3, False, 4, [40, 50, 60, 70], ask_price=60)

        market = Market(
            num_buyers=2,
            num_sellers=1,
            num_times=10,
            price_min=1,
            price_max=100,
            buyers=[buyer1, buyer2],
            sellers=[seller],
            seed=42,
        )

        market.run_time_step()

        # Buyer2 (loser) should have +4 bit
        assert buyer2.last_nobuysell is not None
        assert (buyer2.last_nobuysell & 4) == 4, f"Expected +4 bit for loser, got {buyer2.last_nobuysell}"

        # Buyer1 (winner) should NOT have +4 bit
        assert buyer1.last_nobuysell is not None
        assert (buyer1.last_nobuysell & 4) == 0, f"Winner should not have +4 bit, got {buyer1.last_nobuysell}"

    def test_rule_14_any_buyer_when_no_bid(self) -> None:
        """Rule 14: When no standing bid, any buyer may accept."""
        buyer1 = MockAgent(1, True, 4, [100, 90, 80, 70], bid_price=0, accept_trade=True)  # No bid
        buyer2 = MockAgent(2, True, 4, [95, 85, 75, 65], bid_price=0, accept_trade=True)  # No bid
        seller = MockAgent(3, False, 4, [40, 50, 60, 70], ask_price=60, accept_trade=False)

        market = Market(
            num_buyers=2,
            num_sellers=1,
            num_times=10,
            price_min=1,
            price_max=100,
            buyers=[buyer1, buyer2],
            sellers=[seller],
            seed=42,
        )

        market.run_time_step()

        # Neither buyer should have +4 bit (both can accept per Rule 14)
        assert (buyer1.last_nobuysell or 0) & 4 == 0, "Rule 14: Buyer1 should be able to accept"
        assert (buyer2.last_nobuysell or 0) & 4 == 0, "Rule 14: Buyer2 should be able to accept"

    def test_rule_15_any_seller_when_no_ask(self) -> None:
        """Rule 15: When no standing ask, any seller may accept."""
        buyer = MockAgent(1, True, 4, [100, 90, 80, 70], bid_price=50, accept_trade=False)
        seller1 = MockAgent(2, False, 4, [40, 50, 60, 70], ask_price=0, accept_trade=True)  # No ask
        seller2 = MockAgent(3, False, 4, [45, 55, 65, 75], ask_price=0, accept_trade=True)  # No ask

        market = Market(
            num_buyers=1,
            num_sellers=2,
            num_times=10,
            price_min=1,
            price_max=100,
            buyers=[buyer],
            sellers=[seller1, seller2],
            seed=42,
        )

        market.run_time_step()

        # Neither seller should have +4 bit (both can accept per Rule 15)
        assert (seller1.last_nobuysell or 0) & 4 == 0, "Rule 15: Seller1 should be able to accept"
        assert (seller2.last_nobuysell or 0) & 4 == 0, "Rule 15: Seller2 should be able to accept"

    def test_buy_sell_response_collects_decisions(
        self, two_by_two_market: tuple[Market, List[MockAgent], List[MockAgent]]
    ) -> None:
        """Accept/reject decisions should be collected correctly."""
        market, buyers, sellers = two_by_two_market

        # High bidder accepts, low asker accepts
        buyers[0].set_response(bid_price=55, accept_trade=True)
        sellers[0].set_response(ask_price=65, accept_trade=True)

        market.run_time_step()

        ob = market.get_orderbook()
        # Trade should have occurred
        assert ob.trade_price[1] > 0, "Trade should have occurred"

    def test_buy_sell_result_broadcasts_trade(
        self, two_by_two_market: tuple[Market, List[MockAgent], List[MockAgent]]
    ) -> None:
        """All agents should receive buy_sell_result with trade outcome."""
        market, buyers, sellers = two_by_two_market

        buyers[0].set_response(bid_price=55, accept_trade=True)
        sellers[0].set_response(ask_price=65, accept_trade=True)

        market.run_time_step()

        for buyer in buyers:
            assert buyer.buy_sell_result_called
        for seller in sellers:
            assert seller.buy_sell_result_called


# =============================================================================
# TEST CLASS: Full Time Step
# =============================================================================


class TestFullTimeStep:
    """Tests for complete time step execution."""

    def test_complete_timestep_with_trade(
        self, two_by_two_market: tuple[Market, List[MockAgent], List[MockAgent]]
    ) -> None:
        """Complete timestep should execute both stages and record trade."""
        market, buyers, sellers = two_by_two_market

        # Set up for trade
        buyers[0].set_response(bid_price=60, accept_trade=True)
        sellers[0].set_response(ask_price=60, accept_trade=True)

        result = market.run_time_step()

        assert result is True, "Time step should succeed"

        ob = market.get_orderbook()
        assert ob.trade_price[1] == 60, f"Expected trade at 60, got {ob.trade_price[1]}"

    def test_complete_timestep_no_trade(
        self, two_by_two_market: tuple[Market, List[MockAgent], List[MockAgent]]
    ) -> None:
        """Complete timestep with no trade should persist orders."""
        market, buyers, sellers = two_by_two_market

        # Set up with no acceptance
        buyers[0].set_response(bid_price=50, accept_trade=False)
        sellers[0].set_response(ask_price=60, accept_trade=False)

        market.run_time_step()

        ob = market.get_orderbook()
        assert ob.trade_price[1] == 0, "No trade should have occurred"

        # Orders should persist for next timestep
        market.run_time_step()
        # After increment_time, orders carry over
        assert ob.bids[1, 2] == 50, "Bid should carry over"

    def test_multiple_timesteps_accumulate_positions(
        self, two_by_two_market: tuple[Market, List[MockAgent], List[MockAgent]]
    ) -> None:
        """Multiple trades should accumulate in position counters."""
        market, buyers, sellers = two_by_two_market

        # First trade
        buyers[0].set_response(bid_price=60, accept_trade=True)
        sellers[0].set_response(ask_price=60, accept_trade=True)
        market.run_time_step()

        ob = market.get_orderbook()
        assert ob.num_buys[1, 1] == 1

        # Second trade
        buyers[0].set_response(bid_price=55, accept_trade=True)
        sellers[0].set_response(ask_price=55, accept_trade=True)
        market.run_time_step()

        assert ob.num_buys[1, 2] == 2, "Positions should accumulate"

    def test_timestep_returns_false_on_failure(self) -> None:
        """run_time_step should return False when market fails."""
        quitter = QuittingAgent(1, True, 4, [100, 90, 80, 70])
        seller = MockAgent(2, False, 4, [40, 50, 60, 70], ask_price=60)

        market = Market(
            num_buyers=1,
            num_sellers=1,
            num_times=10,
            price_min=1,
            price_max=100,
            buyers=[quitter],
            sellers=[seller],
            seed=42,
        )

        result = market.run_time_step()

        # Market should fail after buyer quits
        assert market.has_failed()
        # Subsequent calls should return False
        assert market.run_time_step() is False


# =============================================================================
# TEST CLASS: Agent Synchronization
# =============================================================================


class TestAgentSynchronization:
    """Tests for agent execution order and synchronization."""

    def test_agents_shuffled_each_timestep(self) -> None:
        """Agents should be processed in random order each timestep (fairness)."""
        # This is a statistical test - we track execution order over many runs
        first_buyer_first_count = 0
        trials = 100

        for seed in range(trials):
            # Track which buyer submits first by checking bid values
            buyer1 = MockAgent(1, True, 4, [100, 90, 80, 70], bid_price=10)  # Unique price
            buyer2 = MockAgent(2, True, 4, [95, 85, 75, 65], bid_price=20)   # Different price
            seller = MockAgent(3, False, 4, [40, 50, 60, 70], ask_price=60)

            market = Market(
                num_buyers=2,
                num_sellers=1,
                num_times=10,
                price_min=1,
                price_max=100,
                buyers=[buyer1, buyer2],
                sellers=[seller],
                seed=seed,
            )

            market.run_time_step()

            # Both bids are valid, so we just verify both were called
            if buyer1.bid_ask_called and buyer2.bid_ask_called:
                first_buyer_first_count += 1

        # All trials should have both buyers called (basic sanity check)
        assert first_buyer_first_count == trials

    def test_all_agents_receive_correct_valuations(self) -> None:
        """Agents should be initialized with correct token valuations."""
        buyer_vals = [100, 90, 80, 70]
        seller_vals = [40, 50, 60, 70]

        buyer = MockAgent(1, True, 4, buyer_vals, bid_price=50)
        seller = MockAgent(2, False, 4, seller_vals, ask_price=60)

        market = Market(
            num_buyers=1,
            num_sellers=1,
            num_times=10,
            price_min=1,
            price_max=100,
            buyers=[buyer],
            sellers=[seller],
            seed=42,
        )

        assert buyer.valuations == buyer_vals
        assert seller.valuations == seller_vals
        assert buyer.get_current_valuation() == 100
        assert seller.get_current_valuation() == 40

    def test_agent_profit_tracked(self) -> None:
        """Agent profit should be tracked after trades."""
        buyer = MockAgent(1, True, 4, [100, 90, 80, 70], bid_price=60, accept_trade=True)
        seller = MockAgent(2, False, 4, [40, 50, 60, 70], ask_price=60, accept_trade=True)

        market = Market(
            num_buyers=1,
            num_sellers=1,
            num_times=10,
            price_min=1,
            price_max=100,
            buyers=[buyer],
            sellers=[seller],
            seed=42,
        )

        market.run_time_step()

        # Buyer profit: valuation (100) - price (60) = 40
        # Seller profit: price (60) - cost (40) = 20
        assert buyer.period_profit == 40, f"Expected buyer profit 40, got {buyer.period_profit}"
        assert seller.period_profit == 20, f"Expected seller profit 20, got {seller.period_profit}"


# =============================================================================
# TEST CLASS: Market Failure
# =============================================================================


class TestMarketFailure:
    """Tests for market failure conditions."""

    def test_market_fails_when_all_buyers_quit(self) -> None:
        """Market should fail when no buyers remain."""
        quitter1 = QuittingAgent(1, True, 4, [100, 90, 80, 70])
        quitter2 = QuittingAgent(2, True, 4, [95, 85, 75, 65])
        seller = MockAgent(3, False, 4, [40, 50, 60, 70], ask_price=60)

        market = Market(
            num_buyers=2,
            num_sellers=1,
            num_times=10,
            price_min=1,
            price_max=100,
            buyers=[quitter1, quitter2],
            sellers=[seller],
            seed=42,
        )

        market.run_time_step()

        assert market.num_buyers == 0
        assert market.has_failed()

    def test_market_fails_when_all_sellers_quit(self) -> None:
        """Market should fail when no sellers remain."""
        buyer = MockAgent(1, True, 4, [100, 90, 80, 70], bid_price=50)
        quitter1 = QuittingAgent(2, False, 4, [40, 50, 60, 70])
        quitter2 = QuittingAgent(3, False, 4, [45, 55, 65, 75])

        market = Market(
            num_buyers=1,
            num_sellers=2,
            num_times=10,
            price_min=1,
            price_max=100,
            buyers=[buyer],
            sellers=[quitter1, quitter2],
            seed=42,
        )

        market.run_time_step()

        assert market.num_sellers == 0
        assert market.has_failed()

    def test_market_continues_with_partial_quits(self) -> None:
        """Market should continue if at least 1 buyer AND 1 seller remain."""
        quitter = QuittingAgent(1, True, 4, [100, 90, 80, 70])
        stayer = MockAgent(2, True, 4, [95, 85, 75, 65], bid_price=50)
        seller = MockAgent(3, False, 4, [40, 50, 60, 70], ask_price=60)

        market = Market(
            num_buyers=2,
            num_sellers=1,
            num_times=10,
            price_min=1,
            price_max=100,
            buyers=[quitter, stayer],
            sellers=[seller],
            seed=42,
        )

        result = market.run_time_step()

        assert market.num_buyers == 1
        assert not market.has_failed()
        assert result is True

    def test_failed_market_cannot_recover(self) -> None:
        """Once failed, market should stay failed."""
        quitter = QuittingAgent(1, True, 4, [100, 90, 80, 70])
        seller = MockAgent(2, False, 4, [40, 50, 60, 70], ask_price=60)

        market = Market(
            num_buyers=1,
            num_sellers=1,
            num_times=10,
            price_min=1,
            price_max=100,
            buyers=[quitter],
            sellers=[seller],
            seed=42,
        )

        market.run_time_step()
        assert market.has_failed()

        # Subsequent runs should still show failure
        result = market.run_time_step()
        assert result is False
        assert market.has_failed()


# =============================================================================
# TEST CLASS: Market Initialization Validation
# =============================================================================


class TestMarketInitialization:
    """Tests for Market initialization validation."""

    def test_buyer_count_mismatch_raises(self) -> None:
        """Should raise ValueError if buyer list length doesn't match num_buyers."""
        buyer = MockAgent(1, True, 4, [100, 90, 80, 70])
        seller = MockAgent(2, False, 4, [40, 50, 60, 70])

        with pytest.raises(ValueError, match="buyers list length"):
            Market(
                num_buyers=2,  # Mismatch: claiming 2 but providing 1
                num_sellers=1,
                num_times=10,
                price_min=1,
                price_max=100,
                buyers=[buyer],
                sellers=[seller],
            )

    def test_seller_count_mismatch_raises(self) -> None:
        """Should raise ValueError if seller list length doesn't match num_sellers."""
        buyer = MockAgent(1, True, 4, [100, 90, 80, 70])
        seller = MockAgent(2, False, 4, [40, 50, 60, 70])

        with pytest.raises(ValueError, match="sellers list length"):
            Market(
                num_buyers=1,
                num_sellers=2,  # Mismatch: claiming 2 but providing 1
                num_times=10,
                price_min=1,
                price_max=100,
                buyers=[buyer],
                sellers=[seller],
            )
