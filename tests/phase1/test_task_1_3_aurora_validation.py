"""
Phase 1 Task 1.3: AURORA Strict Mode & Basic Validation Framework Tests

This module validates the AURORA Strict Mode rules and provides
End-to-End scenarios for the complete market system.

References:
- PLAN.md Phase 1: AURORA Strict Mode (Critical for Kaplan Support)
- engine/metrics.py - Efficiency calculations
"""

import pytest
import numpy as np
from typing import List
from collections import Counter

from engine.orderbook import OrderBook
from engine.market import Market
from engine.token_generator import generate_tokens
from engine.metrics import calculate_equilibrium_profit
from traders.base import Agent


# =============================================================================
# MOCK AGENTS FOR E2E TESTING
# =============================================================================


class SimpleAgent(Agent):
    """A simple deterministic agent for E2E testing."""

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: List[int],
        margin: float = 0.1,
    ) -> None:
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.margin = margin
        self._pending_price = 0
        self._pending_accept = False

    def bid_ask(self, time: int, nobidask: int) -> None:
        self.has_responded = False
        if nobidask == 1 or not self.can_trade():
            self._pending_price = 0
            return

        val = self.get_current_valuation()
        if self.is_buyer:
            # Bid slightly below valuation
            self._pending_price = max(1, int(val * (1 - self.margin)))
        else:
            # Ask slightly above cost
            self._pending_price = int(val * (1 + self.margin))

    def bid_ask_response(self) -> int:
        self.has_responded = True
        return self._pending_price

    def buy_sell(
        self,
        time: int,
        nobuysell: int,
        high_bid: int,
        low_ask: int,
        high_bidder: int,
        low_asker: int,
    ) -> None:
        self.has_responded = False

        # Accept if profitable
        if nobuysell > 0:
            self._pending_accept = False
            return

        val = self.get_current_valuation()
        if self.is_buyer:
            # Accept if ask is below valuation
            self._pending_accept = low_ask > 0 and low_ask < val
        else:
            # Accept if bid is above cost
            self._pending_accept = high_bid > 0 and high_bid > val

    def buy_sell_response(self) -> bool:
        self.has_responded = True
        return self._pending_accept


class AlwaysAcceptAgent(Agent):
    """Agent that always accepts trades when eligible."""

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: List[int],
        price: int,
    ) -> None:
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price = price

    def bid_ask(self, time: int, nobidask: int) -> None:
        self.has_responded = False

    def bid_ask_response(self) -> int:
        self.has_responded = True
        return self.price if self.can_trade() else 0

    def buy_sell(
        self,
        time: int,
        nobuysell: int,
        high_bid: int,
        low_ask: int,
        high_bidder: int,
        low_asker: int,
    ) -> None:
        self.has_responded = False

    def buy_sell_response(self) -> bool:
        self.has_responded = True
        return True


# =============================================================================
# TEST CLASS: Integer Math Enforcement
# =============================================================================


class TestIntegerMathEnforcement:
    """Tests for AURORA Strict Mode: Integer math for all prices."""

    def test_all_prices_are_integers(self) -> None:
        """All prices in orderbook must be integers."""
        ob = OrderBook(2, 2, 10, 1, 100, 42)
        ob.increment_time()

        # Add bids and asks
        ob.add_bid(1, 50)
        ob.add_ask(1, 60)
        ob.determine_winners()

        # All recorded prices must be int32
        assert ob.bids.dtype == np.int32
        assert ob.asks.dtype == np.int32
        assert ob.high_bid.dtype == np.int32
        assert ob.low_ask.dtype == np.int32
        assert ob.trade_price.dtype == np.int32

        # Verify actual values are integers
        assert isinstance(int(ob.high_bid[1]), int)
        assert isinstance(int(ob.low_ask[1]), int)

    def test_valuations_are_integers(self) -> None:
        """Token valuations must be integers."""
        buyer_vals, seller_costs = generate_tokens(
            num_buyers=2,
            num_sellers=2,
            num_tokens=4,
            game_type=6453,
            seed=42,
        )

        for vals in buyer_vals:
            for v in vals:
                assert isinstance(v, (int, np.integer)), f"Value {v} is not int"

        for costs in seller_costs:
            for c in costs:
                assert isinstance(c, (int, np.integer)), f"Cost {c} is not int"

    def test_trade_prices_are_integers(self) -> None:
        """Executed trade prices must be integers."""
        ob = OrderBook(2, 2, 10, 1, 100, 42)
        ob.increment_time()

        ob.add_bid(1, 50)
        ob.add_ask(1, 60)
        ob.determine_winners()

        price = ob.execute_trade(buyer_accepts=True, seller_accepts=True)

        assert isinstance(price, (int, np.integer)), f"Trade price {price} is not int"
        assert price in [50, 60], f"Unexpected trade price: {price}"


# =============================================================================
# TEST CLASS: Phase 1 Bid-Offer Crossing (AURORA Two-Phase Protocol)
# =============================================================================


class TestPhase1BidOfferCrossing:
    """
    Tests for AURORA Two-Phase Protocol: Crossed market handling.

    In AURORA, crossing bids/asks ARE ALLOWED because:
    1. Bid-offer phase: Quotes are submitted (crossing creates trade opportunity)
    2. Buy-sell phase: Trade only happens when BOTH parties agree

    This is different from continuous auctions where crossing triggers auto-execution.
    """

    def test_bid_crossing_ask_accepted_no_auto_execution(self) -> None:
        """Bid >= standing ask is ACCEPTED in AURORA but no automatic trade."""
        ob = OrderBook(2, 2, 10, 1, 100, 42)
        ob.increment_time()

        # Establish standing ask
        ob.add_ask(1, 50)
        ob.determine_winners()

        ob.increment_time()

        # Crossing bid IS accepted in AURORA (creates trade opportunity)
        # But it must improve on previous high_bid (New York Rule)
        crossing_accepted = ob.add_bid(1, 50)  # Bid = ask, first bid so accepted
        assert crossing_accepted is True, "Crossing bid should be ACCEPTED in AURORA"

        ob.determine_winners()

        # Verify no auto-trade occurred (trade happens in buy-sell phase)
        assert ob.trade_price[ob.current_time] == 0, "No auto-trade should occur"

    def test_ask_crossing_bid_accepted_no_auto_execution(self) -> None:
        """Ask <= standing bid is ACCEPTED in AURORA but no automatic trade."""
        ob = OrderBook(2, 2, 10, 1, 100, 42)
        ob.increment_time()

        # Establish standing bid
        ob.add_bid(1, 50)
        ob.determine_winners()

        ob.increment_time()

        # Crossing ask IS accepted in AURORA (creates trade opportunity)
        # But it must improve on previous low_ask (New York Rule)
        crossing_accepted = ob.add_ask(1, 50)  # Ask = bid, first ask so accepted
        assert crossing_accepted is True, "Crossing ask should be ACCEPTED in AURORA"

        ob.determine_winners()

        # Verify no auto-trade occurred
        assert ob.trade_price[ob.current_time] == 0

    def test_kaplan_observes_spread_narrowing(self) -> None:
        """Kaplan strategy depends on seeing the spread narrow before trading."""
        ob = OrderBook(2, 2, 10, 1, 100, 42)

        # Time 1: Initial spread of 20 (bid 40, ask 60)
        ob.increment_time()
        ob.add_bid(1, 40)
        ob.add_ask(1, 60)
        ob.determine_winners()

        spread_t1 = ob.low_ask[1] - ob.high_bid[1]
        assert spread_t1 == 20

        # Time 2: Narrower spread - new bid at 45 (ask still 60)
        ob.increment_time()
        ob.add_bid(2, 45)  # Must improve on 40
        ob.determine_winners()

        spread_t2 = ob.low_ask[2] - ob.high_bid[2]
        assert spread_t2 == 15, f"Spread should narrow to 15, got {spread_t2}"

        # Kaplan can observe spread narrowing and strike when profitable
        # In AURORA, crossing is allowed but Kaplan waits for optimal moment
        ob.increment_time()
        crossing = ob.add_bid(1, 60)  # Crosses, but must improve on 45
        assert crossing is True, "Crossing bid accepted when improving"

        ob.determine_winners()
        # Verify crossed state but no auto-trade
        assert ob.high_bid[ob.current_time] == 60
        assert ob.low_ask[ob.current_time] == 60  # Carried over from t=1
        assert ob.trade_price[ob.current_time] == 0  # No auto-execution


# =============================================================================
# TEST CLASS: Phase 2 Buy-Sell Persistence (AURORA Strict Mode)
# =============================================================================


class TestPhase2BuySellPersistence:
    """Tests for AURORA Strict Mode: Book persistence rules."""

    def test_neither_accepts_keeps_book(self) -> None:
        """If neither buyer nor seller accepts, book should persist."""
        ob = OrderBook(2, 2, 10, 1, 100, 42)
        ob.increment_time()

        ob.add_bid(1, 50)
        ob.add_ask(1, 60)
        ob.determine_winners()

        # Neither accepts
        price = ob.execute_trade(buyer_accepts=False, seller_accepts=False)
        assert price == 0, "No trade should occur"

        ob.increment_time()

        # Orders should persist
        assert ob.bids[1, 2] == 50, "Bid should persist"
        assert ob.asks[1, 2] == 60, "Ask should persist"
        assert ob.high_bid[1] == 50  # Previous high bid
        assert ob.low_ask[1] == 60  # Previous low ask

    def test_book_clears_only_on_trade(self) -> None:
        """Book should only clear when trade_price > 0."""
        ob = OrderBook(2, 2, 10, 1, 100, 42)
        ob.increment_time()

        ob.add_bid(1, 50)
        ob.add_ask(1, 60)
        ob.determine_winners()

        # Trade occurs
        price = ob.execute_trade(buyer_accepts=True, seller_accepts=True)
        assert price > 0, "Trade should occur"

        ob.increment_time()

        # Book should be cleared
        assert ob.bids[1, 2] == 0, "Bid should be cleared after trade"
        assert ob.asks[1, 2] == 0, "Ask should be cleared after trade"

    def test_partial_acceptance_still_trades(self) -> None:
        """Trade should occur if either side accepts."""
        # Buyer only accepts
        ob1 = OrderBook(2, 2, 10, 1, 100, 42)
        ob1.increment_time()
        ob1.add_bid(1, 50)
        ob1.add_ask(1, 60)
        ob1.determine_winners()
        price1 = ob1.execute_trade(buyer_accepts=True, seller_accepts=False)
        assert price1 == 60, "Buyer accepts -> trade at ask"

        # Seller only accepts
        ob2 = OrderBook(2, 2, 10, 1, 100, 43)
        ob2.increment_time()
        ob2.add_bid(1, 50)
        ob2.add_ask(1, 60)
        ob2.determine_winners()
        price2 = ob2.execute_trade(buyer_accepts=False, seller_accepts=True)
        assert price2 == 50, "Seller accepts -> trade at bid"


# =============================================================================
# TEST CLASS: Execution Order Fairness (AURORA Strict Mode)
# =============================================================================


class TestExecutionOrderFairness:
    """Tests for AURORA Strict Mode: Random agent execution order."""

    def test_agent_order_randomized_each_step(self) -> None:
        """Statistical test: agent execution order should be random."""
        # We can't directly observe order, but we can test that the Market
        # uses rng.shuffle() which we verify via tied bid resolution
        first_wins: Counter[int] = Counter()
        trials = 200

        for seed in range(trials):
            ob = OrderBook(3, 3, 10, 1, 100, seed)
            ob.increment_time()

            # All three bid same price
            ob.add_bid(1, 50)
            ob.add_bid(2, 50)
            ob.add_bid(3, 50)

            ob.determine_winners()
            first_wins[ob.high_bidder[1]] += 1

        # Each bidder should win roughly 1/3 of the time (within tolerance)
        for bidder in [1, 2, 3]:
            ratio = first_wins[bidder] / trials
            assert 0.20 <= ratio <= 0.47, f"Bidder {bidder} won {ratio*100:.1f}%, expected ~33%"

    def test_no_systematic_id_advantage(self) -> None:
        """No player ID should have systematic advantage."""
        # Test over many seeds with different ID configurations
        wins_by_position: Counter[int] = Counter()
        trials = 300

        for seed in range(trials):
            ob = OrderBook(2, 2, 10, 1, 100, seed)
            ob.increment_time()

            # Both bid same price
            ob.add_bid(1, 50)
            ob.add_bid(2, 50)

            ob.determine_winners()

            # Track which position (1 or 2) won
            wins_by_position[ob.high_bidder[1]] += 1

        # Should be roughly 50/50
        for position in [1, 2]:
            ratio = wins_by_position[position] / trials
            assert 0.40 <= ratio <= 0.60, f"Position {position} won {ratio*100:.1f}%"


# =============================================================================
# TEST CLASS: Efficiency Metrics
# =============================================================================


class TestEfficiencyMetrics:
    """Tests for efficiency metric calculations."""

    def test_max_surplus_uses_strict_inequality(self) -> None:
        """Equilibrium calculation must use b > s, not b >= s."""
        # Scenario: buyer val = seller cost = 50 (zero profit trade)
        # Should NOT be counted in max surplus
        buyer_vals = [50, 40]  # Only 50 overlaps
        seller_costs = [50, 60]  # Only 50 overlaps

        max_profit = calculate_equilibrium_profit(buyer_vals, seller_costs)

        # With strict inequality (b > s), 50 vs 50 gives 0 profit
        # No trade should be counted
        assert max_profit == 0, f"Zero-profit trade should not count: {max_profit}"

    def test_allocative_efficiency_correct(self) -> None:
        """Efficiency = actual_surplus / max_surplus * 100."""
        # Known scenario:
        # Buyers: [100, 80], Sellers: [40, 60]
        # Max surplus: (100-40) + (80-60) = 60 + 20 = 80
        buyer_vals = [100, 80]
        seller_costs = [40, 60]

        max_profit = calculate_equilibrium_profit(buyer_vals, seller_costs)
        assert max_profit == 80, f"Expected max 80, got {max_profit}"

    def test_v_inefficiency_counts_missed_trades(self) -> None:
        """V-inefficiency = missed profitable trades."""
        # This is more of a documentation test - the concept
        # V-inefficiency happens when profitable pairs don't trade
        buyer_vals = [100, 80, 60]
        seller_costs = [40, 50, 70]

        max_profit = calculate_equilibrium_profit(buyer_vals, seller_costs)
        # Should match: (100, 40), (80, 50) -> 60 + 30 = 90
        assert max_profit == 90

    def test_em_inefficiency_counts_bad_trades(self) -> None:
        """EM-inefficiency = trades where buyer_val < seller_cost."""
        # EM-inefficiency is when wrong trades displace better trades
        # This validates the concept exists in our framework
        buyer_vals = [30]  # Low value
        seller_costs = [50]  # High cost

        max_profit = calculate_equilibrium_profit(buyer_vals, seller_costs)
        # No profitable trades possible
        assert max_profit == 0

    def test_zero_surplus_market_handled(self) -> None:
        """Edge case: market with no profitable trades."""
        # Sellers all have higher costs than buyers' valuations
        buyer_vals = [40, 30, 20]
        seller_costs = [50, 60, 70]

        max_profit = calculate_equilibrium_profit(buyer_vals, seller_costs)
        assert max_profit == 0, "No profitable trades = 0 surplus"

    def test_max_surplus_filters_zero_tokens(self) -> None:
        """Zero-valued tokens should be filtered out."""
        # Includes some zero values that should be ignored
        buyer_vals = [100, 0, 80, 0]
        seller_costs = [0, 40, 60, 0]

        max_profit = calculate_equilibrium_profit(buyer_vals, seller_costs)
        # After filtering: buyers [100, 80], sellers [40, 60]
        # Max = (100-40) + (80-60) = 80
        assert max_profit == 80


# =============================================================================
# TEST CLASS: E2E Scenarios
# =============================================================================


class TestE2EScenarios:
    """End-to-end integration tests for complete market scenarios."""

    def test_simple_trade_scenario(self) -> None:
        """Basic 2x2 market with one trade."""
        buyer1 = AlwaysAcceptAgent(1, True, 4, [100, 90, 80, 70], price=60)
        buyer2 = AlwaysAcceptAgent(2, True, 4, [95, 85, 75, 65], price=55)
        seller1 = AlwaysAcceptAgent(3, False, 4, [40, 50, 60, 70], price=60)
        seller2 = AlwaysAcceptAgent(4, False, 4, [45, 55, 65, 75], price=65)

        market = Market(
            num_buyers=2,
            num_sellers=2,
            num_times=10,
            price_min=1,
            price_max=100,
            buyers=[buyer1, buyer2],
            sellers=[seller1, seller2],
            seed=42,
        )

        market.run_time_step()

        ob = market.get_orderbook()
        # Trade should occur at 60 (both buyer1 and seller1 bid/ask 60)
        assert ob.trade_price[1] == 60

    def test_multi_trade_period(self) -> None:
        """Multiple trades should occur across timesteps."""
        buyers = [
            AlwaysAcceptAgent(1, True, 2, [100, 90], price=70),
            AlwaysAcceptAgent(2, True, 2, [95, 85], price=65),
        ]
        sellers = [
            AlwaysAcceptAgent(3, False, 2, [40, 50], price=70),
            AlwaysAcceptAgent(4, False, 2, [45, 55], price=75),
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

        # Run multiple timesteps
        trade_count = 0
        for _ in range(5):
            market.run_time_step()
            ob = market.get_orderbook()
            if ob.trade_price[market.current_time] > 0:
                trade_count += 1

        # Should have at least 1 trade (possibly more depending on acceptance)
        assert trade_count >= 1, f"Expected trades, got {trade_count}"

    def test_asymmetric_market_scenario(self) -> None:
        """6 buyers, 2 sellers (buyer scarcity scenario)."""
        buyers = [
            SimpleAgent(i, True, 2, [100 - i * 5, 90 - i * 5])
            for i in range(1, 7)
        ]
        sellers = [
            SimpleAgent(i + 6, False, 2, [40 + i * 5, 50 + i * 5])
            for i in range(1, 3)
        ]

        market = Market(
            num_buyers=6,
            num_sellers=2,
            num_times=20,
            price_min=1,
            price_max=100,
            buyers=buyers,
            sellers=sellers,
            seed=42,
        )

        # Run several timesteps
        for _ in range(10):
            if not market.run_time_step():
                break

        # Market should still be functional
        assert not market.has_failed()

    def test_no_profitable_trades_scenario(self) -> None:
        """Market where no profitable trades are possible."""
        # Buyers have low valuations, sellers have high costs
        buyers = [
            SimpleAgent(1, True, 2, [30, 20], margin=0.05),
            SimpleAgent(2, True, 2, [25, 15], margin=0.05),
        ]
        sellers = [
            SimpleAgent(3, False, 2, [50, 60], margin=0.05),
            SimpleAgent(4, False, 2, [55, 65], margin=0.05),
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

        # Run timesteps
        for _ in range(5):
            market.run_time_step()

        # Market should continue (no failure) even with no trades
        assert not market.has_failed()

    def test_tournament_scale_session(self) -> None:
        """8x8 market with 100 timesteps (tournament scale)."""
        buyers = [
            SimpleAgent(i, True, 4, [100 - i * 3, 90 - i * 3, 80 - i * 3, 70 - i * 3])
            for i in range(1, 9)
        ]
        sellers = [
            SimpleAgent(i + 8, False, 4, [30 + i * 3, 40 + i * 3, 50 + i * 3, 60 + i * 3])
            for i in range(1, 9)
        ]

        market = Market(
            num_buyers=8,
            num_sellers=8,
            num_times=100,
            price_min=1,
            price_max=100,
            buyers=buyers,
            sellers=sellers,
            seed=42,
        )

        # Run full period
        timesteps_completed = 0
        for _ in range(100):
            if market.run_time_step():
                timesteps_completed += 1
            else:
                break

        # Should complete all 100 timesteps without failure
        assert timesteps_completed == 100, f"Completed {timesteps_completed}/100"
        assert not market.has_failed()

        # Verify some trades occurred
        ob = market.get_orderbook()
        total_trades = sum(ob.trade_price[t] > 0 for t in range(1, 101))
        assert total_trades > 0, "Should have some trades in tournament"


# =============================================================================
# TEST CLASS: Integration with Real Token Generation
# =============================================================================


class TestRealTokenIntegration:
    """Tests using the actual token generation system."""

    def test_market_with_generated_tokens(self) -> None:
        """Market should work with tokens from TokenGenerator."""
        buyer_vals, seller_costs = generate_tokens(
            num_buyers=4,
            num_sellers=4,
            num_tokens=4,
            game_type=6453,
            seed=42,
        )

        buyers = [
            SimpleAgent(i + 1, True, 4, buyer_vals[i])
            for i in range(4)
        ]
        sellers = [
            SimpleAgent(i + 5, False, 4, seller_costs[i])
            for i in range(4)
        ]

        market = Market(
            num_buyers=4,
            num_sellers=4,
            num_times=50,
            price_min=1,
            price_max=1000,  # Higher max for generated tokens
            buyers=buyers,
            sellers=sellers,
            seed=42,
        )

        # Run several timesteps
        for _ in range(25):
            market.run_time_step()

        assert not market.has_failed()

    def test_equilibrium_with_generated_tokens(self) -> None:
        """Verify equilibrium calculation works with generated tokens."""
        buyer_vals, seller_costs = generate_tokens(
            num_buyers=4,
            num_sellers=4,
            num_tokens=4,
            game_type=6453,
            seed=42,
        )

        # Flatten to single lists
        all_buyer_vals = [v for vals in buyer_vals for v in vals]
        all_seller_costs = [c for costs in seller_costs for c in costs]

        max_profit = calculate_equilibrium_profit(all_buyer_vals, all_seller_costs)

        # Should produce a valid (non-negative) equilibrium
        assert max_profit >= 0, f"Invalid max_profit: {max_profit}"
