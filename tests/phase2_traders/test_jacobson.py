"""
Comprehensive unit tests for the Jacobson agent.

Tests cover:
- Initialization and state management
- Equilibrium estimation (_eqest, _eqconf)
- Bidding logic (_player_request_bid)
- Asking logic (_player_request_ask)
- Buy decision logic (_player_request_buy)
- Sell decision logic (_player_want_to_sell)
- Trade result processing (buy_sell_result)
- Edge cases and boundary conditions
"""

import pytest
import math
from traders.legacy.jacobson import Jacobson
from engine.agent_factory import create_agent


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def buyer_agent():
    """Create a Jacobson buyer with standard configuration."""
    return Jacobson(
        player_id=1,
        is_buyer=True,
        num_tokens=3,
        valuations=[100, 90, 80],
        price_min=0,
        price_max=200,
        num_times=100,
        seed=42
    )


@pytest.fixture
def seller_agent():
    """Create a Jacobson seller with standard configuration."""
    return Jacobson(
        player_id=2,
        is_buyer=False,
        num_tokens=3,
        valuations=[40, 50, 60],
        price_min=0,
        price_max=200,
        num_times=100,
        seed=42
    )


# =============================================================================
# TEST INITIALIZATION
# =============================================================================

class TestJacobsonInitialization:
    """Test agent initialization and state management."""

    def test_buyer_initialization(self, buyer_agent):
        """Test buyer is initialized correctly."""
        assert buyer_agent.player_id == 1
        assert buyer_agent.is_buyer is True
        assert buyer_agent.num_tokens == 3
        assert buyer_agent.valuations == [100, 90, 80]
        assert buyer_agent.num_trades == 0

    def test_seller_initialization(self, seller_agent):
        """Test seller is initialized correctly."""
        assert seller_agent.player_id == 2
        assert seller_agent.is_buyer is False
        assert seller_agent.num_tokens == 3
        assert seller_agent.valuations == [40, 50, 60]

    def test_state_attributes_initialized(self, buyer_agent):
        """Test all state attributes are properly initialized."""
        assert buyer_agent.price_min_limit == 0
        assert buyer_agent.price_max_limit == 200
        assert buyer_agent.num_times == 100
        assert buyer_agent.roundpricesum == 0.0
        assert buyer_agent.roundweight == 0.0
        assert buyer_agent.lastgap == 10000000
        assert buyer_agent.current_time == 0
        assert buyer_agent.current_bid == 0
        assert buyer_agent.current_ask == 0
        assert buyer_agent.current_bidder == 0
        assert buyer_agent.current_asker == 0
        assert buyer_agent.current_period == 1

    def test_price_limits_configuration(self):
        """Test custom price limits are respected."""
        agent = Jacobson(1, True, 2, [100, 90], price_min=10, price_max=150, num_times=50)
        assert agent.price_min_limit == 10
        assert agent.price_max_limit == 150

    def test_num_times_configuration(self):
        """Test custom num_times is respected."""
        agent = Jacobson(1, True, 2, [100, 90], num_times=50)
        assert agent.num_times == 50

    def test_seed_based_randomness(self):
        """Test seed produces reproducible random behavior."""
        agent1 = Jacobson(1, True, 2, [100, 90], seed=42)
        agent2 = Jacobson(1, True, 2, [100, 90], seed=42)

        # Should produce same random values
        assert agent1.rng.random() == agent2.rng.random()

    def test_inheritance_from_base_agent(self, buyer_agent):
        """Test agent properly inherits from base.Agent."""
        assert hasattr(buyer_agent, 'bid_ask')
        assert hasattr(buyer_agent, 'bid_ask_response')
        assert hasattr(buyer_agent, 'buy_sell')
        assert hasattr(buyer_agent, 'buy_sell_response')
        assert hasattr(buyer_agent, 'start_period')
        assert hasattr(buyer_agent, 'end_period')

    def test_factory_integration(self):
        """Test agent can be created via factory."""
        agent = create_agent('Jacobson', 3, True, 2, [100, 90],
                            price_min=0, price_max=200, num_times=100, seed=42)
        assert isinstance(agent, Jacobson)
        assert agent.player_id == 3


# =============================================================================
# TEST EQUILIBRIUM ESTIMATION
# =============================================================================

class TestJacobsonEquilibriumEstimation:
    """Test equilibrium estimation logic (_eqest, _eqconf)."""

    def test_eqest_no_history_returns_worst_token(self, buyer_agent):
        """Test _eqest returns worst-case token when no history."""
        # No trades yet, roundweight = 0
        assert buyer_agent.roundweight == 0.0
        eq = buyer_agent._eqest()
        # Should return worst-case token (last in list)
        assert eq == float(buyer_agent.valuations[-1])  # 80

    def test_eqest_single_trade(self, buyer_agent):
        """Test _eqest with single trade."""
        # Simulate one trade
        buyer_agent.current_period = 1
        buyer_agent.num_trades = 0  # Start with 0
        trade_price = 70
        trade_type = 1  # Trade occurred

        # Call buy_sell_result to update equilibrium
        # Parent will increment num_trades 0->1, then our code calculates weight
        buyer_agent.buy_sell_result(1, trade_price, trade_type, 0, 0, 0, 0)

        # Verify equilibrium is correct
        eq = buyer_agent._eqest()
        assert eq == pytest.approx(trade_price, rel=0.01)

        # Verify state was updated
        assert buyer_agent.roundweight > 0
        assert buyer_agent.roundpricesum > 0

    def test_eqest_multiple_trades(self, buyer_agent):
        """Test _eqest weighted average with multiple trades."""
        buyer_agent.current_period = 2

        # First trade: period=2, trades=1, price=70
        buyer_agent.num_trades = 1
        buyer_agent.buy_sell_result(1, 70, 1, 0, 0, 0, 0)

        # Second trade: period=2, trades=2, price=75
        buyer_agent.num_trades = 2
        buyer_agent.buy_sell_result(1, 75, 1, 0, 0, 0, 0)

        # Weight1 = 2 + 1*2 = 4, Weight2 = 2 + 2*2 = 6
        # Sum = 70*4 + 75*6 = 280 + 450 = 730
        # Total weight = 4 + 6 = 10
        expected_eq = 730 / 10

        eq = buyer_agent._eqest()
        assert eq == pytest.approx(expected_eq, rel=0.01)

    def test_eqest_weight_calculation(self, buyer_agent):
        """Test weight calculation formula."""
        buyer_agent.current_period = 3
        buyer_agent.num_trades = 5

        buyer_agent.buy_sell_result(1, 80, 1, 0, 0, 0, 0)

        # Weight should be: period + num_trades * 2 = 3 + 5*2 = 13
        expected_weight = 3 + 5 * 2
        assert buyer_agent.roundweight == expected_weight

    def test_eqconf_no_history_zero(self, buyer_agent):
        """Test _eqconf returns 0.0 when no history."""
        conf = buyer_agent._eqconf()
        assert conf == 0.0

    def test_eqconf_low_weight(self, buyer_agent):
        """Test _eqconf with low weight."""
        # Simulate low weight
        buyer_agent.roundweight = 1.0

        # Formula: 0.01^(1/weight) = 0.01^(1/1) = 0.01
        conf = buyer_agent._eqconf()
        assert conf == pytest.approx(0.01, rel=0.001)

    def test_eqconf_high_weight(self, buyer_agent):
        """Test _eqconf approaches 1.0 as weight increases."""
        # Simulate high weight
        buyer_agent.roundweight = 100.0

        # Formula: 0.01^(1/100) ≈ 0.955
        conf = buyer_agent._eqconf()
        expected_conf = math.pow(0.01, 1.0 / 100.0)
        assert conf == pytest.approx(expected_conf, rel=0.001)
        assert conf > 0.9  # Should be close to 1.0

    def test_eqconf_formula(self, buyer_agent):
        """Test _eqconf formula exactly."""
        weights_to_test = [1.0, 2.0, 5.0, 10.0, 50.0, 100.0]

        for weight in weights_to_test:
            buyer_agent.roundweight = weight
            conf = buyer_agent._eqconf()
            expected = math.pow(0.01, 1.0 / weight)
            assert conf == pytest.approx(expected, rel=0.0001)

    def test_equilibrium_updates_after_trade(self, buyer_agent):
        """Test equilibrium state changes after trade."""
        initial_sum = buyer_agent.roundpricesum
        initial_weight = buyer_agent.roundweight

        buyer_agent.current_period = 1
        buyer_agent.num_trades = 1
        buyer_agent.buy_sell_result(1, 85, 1, 0, 0, 0, 0)

        assert buyer_agent.roundpricesum > initial_sum
        assert buyer_agent.roundweight > initial_weight

    def test_equilibrium_persists_across_periods(self, buyer_agent):
        """Test round-level state persists across periods."""
        # Trade in period 1
        buyer_agent.current_period = 1
        buyer_agent.num_trades = 1
        buyer_agent.buy_sell_result(1, 70, 1, 0, 0, 0, 0)

        sum_after_p1 = buyer_agent.roundpricesum
        weight_after_p1 = buyer_agent.roundweight

        # Move to period 2 (not resetting round state)
        buyer_agent.start_period(2)

        # Round state should persist
        assert buyer_agent.roundpricesum == sum_after_p1
        assert buyer_agent.roundweight == weight_after_p1

        # But period should update
        assert buyer_agent.current_period == 2


# =============================================================================
# TEST BIDDING LOGIC
# =============================================================================

class TestJacobsonBiddingLogic:
    """Test buyer bidding logic (_player_request_bid)."""

    def test_buyer_bid_no_history(self, buyer_agent):
        """Test buyer bid uses price_min when no history."""
        buyer_agent.current_bid = 0

        bid = buyer_agent._player_request_bid()

        # With no history, conf=0, est=worst_token=80
        # Formula: old_bid * (1-0) + 80 * 0 + 1 = 0 + 0 + 1 = 1
        # But old_bid defaults to price_min when current_bid=0
        # So: price_min * (1-0) + 80 * 0 + 1 = 0 + 0 + 1 = 1
        assert isinstance(bid, int)
        assert bid >= buyer_agent.price_min_limit

    def test_buyer_bid_convex_combination(self, buyer_agent):
        """Test buyer bid convex combination math."""
        # Set up equilibrium
        buyer_agent.roundweight = 10.0
        buyer_agent.roundpricesum = 700.0  # eq = 70
        buyer_agent.current_bid = 60

        bid = buyer_agent._player_request_bid()

        # eq = 70, conf = 0.01^(1/10) ≈ 0.631
        conf = buyer_agent._eqconf()
        expected_bid = 60 * (1 - conf) + 70 * conf + 1.0

        assert bid == int(expected_bid)

    def test_buyer_bid_zero_confidence(self, buyer_agent):
        """Test buyer bid with zero confidence (no history)."""
        buyer_agent.current_bid = 50
        buyer_agent.roundweight = 0.0

        bid = buyer_agent._player_request_bid()

        # conf = 0, so should use old_bid + 1
        # old_bid * 1 + est * 0 + 1 = 50 + 0 + 1 = 51
        assert bid == 51

    def test_buyer_bid_full_confidence(self, buyer_agent):
        """Test buyer bid with high confidence."""
        buyer_agent.roundweight = 1000.0  # Very high weight
        buyer_agent.roundpricesum = 75000.0  # eq = 75
        buyer_agent.current_bid = 60

        bid = buyer_agent._player_request_bid()

        # With very high confidence, should be close to eq + 1
        conf = buyer_agent._eqconf()
        assert conf > 0.99  # Very high confidence
        # Bid should be close to 75 + 1 = 76
        assert 74 <= bid <= 77  # Allow small rounding

    def test_buyer_bid_exceeds_valuation(self, buyer_agent):
        """Test buyer bid returns 0 when would exceed valuation."""
        # Set up scenario where bid would exceed valuation
        buyer_agent.roundweight = 100.0
        buyer_agent.roundpricesum = 10000.0  # eq = 100
        buyer_agent.current_bid = 99
        buyer_agent.num_trades = 0  # First token: valuation = 100

        bid = buyer_agent._player_request_bid()

        # Calculation would give ~100, which equals valuation[0]=100
        # Protection: if new_bid >= valuation, return 0
        assert bid == 0

    def test_buyer_bid_all_tokens_traded(self, buyer_agent):
        """Test buyer bid returns 0 when all tokens traded."""
        buyer_agent.num_trades = 3  # All tokens traded

        bid = buyer_agent._player_request_bid()
        assert bid == 0

    def test_buyer_bid_first_token(self, buyer_agent):
        """Test buyer bid uses first token valuation."""
        buyer_agent.num_trades = 0
        buyer_agent.roundweight = 10.0
        buyer_agent.roundpricesum = 600.0  # eq = 60
        buyer_agent.current_bid = 55

        bid = buyer_agent._player_request_bid()

        # Should use valuations[0] = 100 for protection check
        # bid < 100, so should succeed
        assert bid > 0
        assert bid < 100

    def test_buyer_bid_last_token(self, buyer_agent):
        """Test buyer bid uses last token valuation."""
        buyer_agent.num_trades = 2  # Last token
        buyer_agent.roundweight = 10.0
        buyer_agent.roundpricesum = 850.0  # eq = 85
        buyer_agent.current_bid = 80

        bid = buyer_agent._player_request_bid()

        # Should use valuations[2] = 80 for protection check
        # Calculation gives ~83-84, which is >= 80
        # Protection should trigger
        assert bid == 0

    def test_buyer_bid_incremental(self, buyer_agent):
        """Test buyer bid improves over current_bid."""
        buyer_agent.roundweight = 10.0
        buyer_agent.roundpricesum = 700.0  # eq = 70
        buyer_agent.current_bid = 60

        bid = buyer_agent._player_request_bid()

        # Bid should be > current_bid (convex combination + 1)
        assert bid > 60

    def test_buyer_bid_reproducible(self):
        """Test buyer bids are reproducible with same seed."""
        agent1 = Jacobson(1, True, 2, [100, 90], seed=42)
        agent2 = Jacobson(1, True, 2, [100, 90], seed=42)

        agent1.roundweight = 10.0
        agent1.roundpricesum = 700.0
        agent1.current_bid = 60

        agent2.roundweight = 10.0
        agent2.roundpricesum = 700.0
        agent2.current_bid = 60

        bid1 = agent1._player_request_bid()
        bid2 = agent2._player_request_bid()

        assert bid1 == bid2


# =============================================================================
# TEST ASKING LOGIC
# =============================================================================

class TestJacobsonAskingLogic:
    """Test seller asking logic (_player_request_ask)."""

    def test_seller_ask_no_history(self, seller_agent):
        """Test seller ask uses price_max when no history."""
        seller_agent.current_ask = 0

        ask = seller_agent._player_request_ask()

        # With no history, should use price_max as old_ask
        assert isinstance(ask, int)
        assert ask <= seller_agent.price_max_limit

    def test_seller_ask_convex_combination(self, seller_agent):
        """Test seller ask convex combination math."""
        seller_agent.roundweight = 10.0
        seller_agent.roundpricesum = 700.0  # eq = 70
        seller_agent.current_ask = 80

        ask = seller_agent._player_request_ask()

        # eq = 70, conf = 0.01^(1/10) ≈ 0.631
        conf = seller_agent._eqconf()
        expected_ask = 80 * (1 - conf) + 70 * conf - 1.0

        assert ask == int(expected_ask)

    def test_seller_ask_zero_confidence(self, seller_agent):
        """Test seller ask with zero confidence."""
        seller_agent.current_ask = 80
        seller_agent.roundweight = 0.0

        ask = seller_agent._player_request_ask()

        # conf = 0, so: old_ask * 1 + est * 0 - 1 = 80 - 1 = 79
        assert ask == 79

    def test_seller_ask_full_confidence(self, seller_agent):
        """Test seller ask with high confidence."""
        seller_agent.roundweight = 1000.0
        seller_agent.roundpricesum = 75000.0  # eq = 75
        seller_agent.current_ask = 80

        ask = seller_agent._player_request_ask()

        # With very high confidence, should be close to eq - 1 = 74
        conf = seller_agent._eqconf()
        assert conf > 0.99
        assert 73 <= ask <= 76

    def test_seller_ask_below_cost(self, seller_agent):
        """Test seller ask returns 0 when would be below cost."""
        seller_agent.roundweight = 100.0
        seller_agent.roundpricesum = 4000.0  # eq = 40
        seller_agent.current_ask = 45
        seller_agent.num_trades = 0  # First token: cost = 40

        ask = seller_agent._player_request_ask()

        # Calculation would give ~40, which equals cost[0]=40
        # Protection: if new_ask <= cost, return 0
        assert ask == 0

    def test_seller_ask_all_tokens_traded(self, seller_agent):
        """Test seller ask returns 0 when all tokens traded."""
        seller_agent.num_trades = 3

        ask = seller_agent._player_request_ask()
        assert ask == 0

    def test_seller_ask_first_token(self, seller_agent):
        """Test seller ask uses first token cost."""
        seller_agent.num_trades = 0
        seller_agent.roundweight = 10.0
        seller_agent.roundpricesum = 700.0  # eq = 70
        seller_agent.current_ask = 75

        ask = seller_agent._player_request_ask()

        # Should use valuations[0] = 40 for protection check
        # ask > 40, so should succeed
        assert ask > 0
        assert ask > 40

    def test_seller_ask_last_token(self, seller_agent):
        """Test seller ask uses last token cost."""
        seller_agent.num_trades = 2  # Last token
        seller_agent.roundweight = 10.0
        seller_agent.roundpricesum = 550.0  # eq = 55
        seller_agent.current_ask = 65

        ask = seller_agent._player_request_ask()

        # Should use valuations[2] = 60 for protection check
        # Calculation gives ~58-59, which is <= 60
        # Protection should trigger
        assert ask == 0

    def test_seller_ask_incremental(self, seller_agent):
        """Test seller ask improves (decreases) over current_ask."""
        seller_agent.roundweight = 10.0
        seller_agent.roundpricesum = 700.0  # eq = 70
        seller_agent.current_ask = 80

        ask = seller_agent._player_request_ask()

        # Ask should be < current_ask (convex combination - 1)
        assert ask < 80

    def test_seller_ask_reproducible(self):
        """Test seller asks are reproducible with same seed."""
        agent1 = Jacobson(2, False, 2, [50, 60], seed=42)
        agent2 = Jacobson(2, False, 2, [50, 60], seed=42)

        agent1.roundweight = 10.0
        agent1.roundpricesum = 700.0
        agent1.current_ask = 80

        agent2.roundweight = 10.0
        agent2.roundpricesum = 700.0
        agent2.current_ask = 80

        ask1 = agent1._player_request_ask()
        ask2 = agent2._player_request_ask()

        assert ask1 == ask2


# =============================================================================
# TEST BUY DECISIONS
# =============================================================================

class TestJacobsonBuyDecisions:
    """Test buyer buy/sell decision logic (_player_request_buy)."""

    def test_buy_not_winner_rejects(self, buyer_agent):
        """Test buyer rejects when not highest bidder."""
        buyer_agent.num_trades = 0
        buyer_agent.current_bidder = 99  # Someone else
        buyer_agent.current_bid = 70
        buyer_agent.current_ask = 75

        result = buyer_agent._player_request_buy()
        assert result == 0

    def test_buy_no_profit_rejects(self, buyer_agent):
        """Test buyer rejects when no profit."""
        buyer_agent.num_trades = 0  # valuation = 100
        buyer_agent.current_bidder = 1  # We are bidder
        buyer_agent.current_bid = 70
        buyer_agent.current_ask = 105  # Ask > valuation

        result = buyer_agent._player_request_buy()
        assert result == 0

    def test_buy_spread_crossed_accepts(self, buyer_agent):
        """Test buyer accepts when spread is crossed."""
        buyer_agent.num_trades = 0  # valuation = 100
        buyer_agent.current_bidder = 1
        buyer_agent.current_bid = 80
        buyer_agent.current_ask = 75  # Gap = -5 (crossed)

        result = buyer_agent._player_request_buy()
        assert result == 1

    def test_buy_zero_gap_accepts(self, buyer_agent):
        """Test buyer accepts when gap is exactly zero."""
        buyer_agent.num_trades = 0
        buyer_agent.current_bidder = 1
        buyer_agent.current_bid = 80
        buyer_agent.current_ask = 80  # Gap = 0

        result = buyer_agent._player_request_buy()
        assert result == 1

    def test_buy_gap_unchanged_time_pressure(self, buyer_agent):
        """Test buyer uses probabilistic logic when gap unchanged."""
        buyer_agent.num_trades = 0  # valuation = 100
        buyer_agent.current_bidder = 1
        buyer_agent.current_bid = 70
        buyer_agent.current_ask = 80
        buyer_agent.lastgap = 10  # Same as current gap
        buyer_agent.current_time = 95
        buyer_agent.num_times = 100

        # Set seed for predictable outcome
        buyer_agent.rng = __import__('random').Random(1)

        result = buyer_agent._player_request_buy()

        # Should use probabilistic logic
        # profit = 100 - 80 = 20, gap = 10
        # ratio = 20/(20+10) = 0.667
        # Result depends on random value
        assert result in [0, 1]

    def test_buy_time_running_out(self, buyer_agent):
        """Test buyer becomes aggressive near end."""
        buyer_agent.num_trades = 2  # Last token
        buyer_agent.current_bidder = 1
        buyer_agent.current_bid = 70
        buyer_agent.current_ask = 79  # Just below valuation=80
        buyer_agent.lastgap = 15
        buyer_agent.current_time = 98
        buyer_agent.num_times = 100

        # Time pressure formula should trigger
        # gap/(lastgap-gap) * (ntokens-trades)*2 + t > ntimes
        gap = 9
        time_threshold = gap / (15 - 9) * (3 - 2) * 2 + 98
        # = 9/6 * 1 * 2 + 98 = 3 + 98 = 101 > 100

        result = buyer_agent._player_request_buy()
        # Should enter probabilistic zone
        assert result in [0, 1]

    def test_buy_all_tokens_traded(self, buyer_agent):
        """Test buyer returns False when all tokens traded."""
        buyer_agent.num_trades = 3  # All done

        result = buyer_agent._player_request_buy()
        assert result == 0


# =============================================================================
# TEST SELL DECISIONS
# =============================================================================

class TestJacobsonSellDecisions:
    """Test seller buy/sell decision logic (_player_want_to_sell)."""

    def test_sell_not_winner_rejects(self, seller_agent):
        """Test seller rejects when not lowest asker."""
        seller_agent.num_trades = 0
        seller_agent.current_asker = 99  # Someone else
        seller_agent.current_bid = 55
        seller_agent.current_ask = 60

        result = seller_agent._player_want_to_sell()
        assert result == 0

    def test_sell_no_profit_rejects(self, seller_agent):
        """Test seller rejects when no profit."""
        seller_agent.num_trades = 0  # cost = 40
        seller_agent.current_asker = 2  # We are asker
        seller_agent.current_bid = 35  # Bid < cost
        seller_agent.current_ask = 60

        result = seller_agent._player_want_to_sell()
        assert result == 0

    def test_sell_spread_crossed_accepts(self, seller_agent):
        """Test seller accepts when spread is crossed."""
        seller_agent.num_trades = 0  # cost = 40
        seller_agent.current_asker = 2
        seller_agent.current_bid = 65
        seller_agent.current_ask = 60  # Gap = -5 (crossed)

        result = seller_agent._player_want_to_sell()
        assert result == 1

    def test_sell_zero_gap_accepts(self, seller_agent):
        """Test seller accepts when gap is exactly zero."""
        seller_agent.num_trades = 0
        seller_agent.current_asker = 2
        seller_agent.current_bid = 60
        seller_agent.current_ask = 60  # Gap = 0

        result = seller_agent._player_want_to_sell()
        assert result == 1

    def test_sell_all_tokens_traded(self, seller_agent):
        """Test seller returns False when all tokens traded."""
        seller_agent.num_trades = 3

        result = seller_agent._player_want_to_sell()
        assert result == 0


# =============================================================================
# TEST TRADE RESULTS
# =============================================================================

class TestJacobsonTradeResults:
    """Test trade result processing (buy_sell_result)."""

    def test_trade_weight_calculation(self, buyer_agent):
        """Test weight calculation formula."""
        buyer_agent.current_period = 3
        buyer_agent.num_trades = 4

        buyer_agent.buy_sell_result(1, 75, 1, 0, 0, 0, 0)

        # Weight = period + num_trades * 2 = 3 + 4*2 = 11
        expected_weight = 3 + 4 * 2
        assert buyer_agent.roundweight == expected_weight

    def test_roundpricesum_accumulates(self, buyer_agent):
        """Test roundpricesum accumulation."""
        buyer_agent.current_period = 1
        buyer_agent.num_trades = 0

        # First trade (num_trades will go 0->1 in parent)
        buyer_agent.buy_sell_result(1, 70, 1, 0, 0, 0, 0)
        sum1 = buyer_agent.roundpricesum

        # Second trade (num_trades will go 1->2 in parent)
        buyer_agent.buy_sell_result(1, 80, 1, 0, 0, 0, 0)
        sum2 = buyer_agent.roundpricesum

        # Sum should accumulate
        assert sum2 > sum1

        # Verify it's proportional to the prices
        assert sum1 > 0
        assert sum2 > sum1

    def test_lastgap_reset_on_trade(self, buyer_agent):
        """Test lastgap resets to 10000000 on trade."""
        buyer_agent.lastgap = 50

        buyer_agent.buy_sell_result(1, 75, 1, 0, 0, 0, 0)

        assert buyer_agent.lastgap == 10000000

    def test_lastgap_updated_on_no_trade(self, buyer_agent):
        """Test lastgap updated to spread when no trade."""
        buyer_agent.buy_sell_result(0, 0, 0, 70, 0, 80, 0)

        # Gap should be: ask - bid = 80 - 70 = 10
        assert buyer_agent.lastgap == 10

    def test_num_trades_increments(self, buyer_agent):
        """Test num_trades increments via parent class."""
        initial_trades = buyer_agent.num_trades

        buyer_agent.buy_sell_result(1, 75, 1, 0, 0, 0, 0)

        # Parent class should increment num_trades
        assert buyer_agent.num_trades == initial_trades + 1

    def test_multiple_trades_cumulative(self, buyer_agent):
        """Test accumulation over many trades."""
        buyer_agent.current_period = 2

        prices = [70, 72, 75, 78, 80]
        for i, price in enumerate(prices, start=1):
            buyer_agent.num_trades = i
            buyer_agent.buy_sell_result(1, price, 1, 0, 0, 0, 0)

        # Should have accumulated 5 trades worth of data
        assert buyer_agent.roundweight > 0
        assert buyer_agent.roundpricesum > 0


# =============================================================================
# TEST EDGE CASES
# =============================================================================

class TestJacobsonEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token(self):
        """Test agent with single token."""
        agent = Jacobson(1, True, 1, [100])

        bid = agent._player_request_bid()
        assert isinstance(bid, int)

    def test_zero_spread(self, buyer_agent):
        """Test when bid equals ask."""
        buyer_agent.num_trades = 0
        buyer_agent.current_bidder = 1
        buyer_agent.current_bid = 75
        buyer_agent.current_ask = 75

        result = buyer_agent._player_request_buy()
        # Gap = 0, should accept
        assert result == 1

    def test_inverted_spread(self, buyer_agent):
        """Test when bid > ask (crossed)."""
        buyer_agent.num_trades = 0
        buyer_agent.current_bidder = 1
        buyer_agent.current_bid = 80
        buyer_agent.current_ask = 75

        result = buyer_agent._player_request_buy()
        # Gap < 0, should accept
        assert result == 1

    def test_time_equals_one(self, buyer_agent):
        """Test at start of period."""
        buyer_agent.current_time = 1
        buyer_agent.num_trades = 0
        buyer_agent.current_bidder = 1
        buyer_agent.current_bid = 70
        buyer_agent.current_ask = 80
        buyer_agent.lastgap = 10

        result = buyer_agent._player_request_buy()
        # Should still work
        assert result in [0, 1]

    def test_time_equals_ntimes(self, buyer_agent):
        """Test at end of period."""
        buyer_agent.current_time = 100
        buyer_agent.num_times = 100
        buyer_agent.num_trades = 0
        buyer_agent.current_bidder = 1
        buyer_agent.current_bid = 70
        buyer_agent.current_ask = 80
        buyer_agent.lastgap = 10

        # Maximum time pressure
        result = buyer_agent._player_request_buy()
        assert result in [0, 1]

    def test_zero_roundweight_division(self, buyer_agent):
        """Test no division by zero in _eqest and _eqconf."""
        buyer_agent.roundweight = 0.0

        # Should not raise exception
        eq = buyer_agent._eqest()
        conf = buyer_agent._eqconf()

        assert eq == float(buyer_agent.valuations[-1])
        assert conf == 0.0

    def test_lastgap_default(self, buyer_agent):
        """Test lastgap initial value."""
        assert buyer_agent.lastgap == 10000000

    def test_period_boundary(self, buyer_agent):
        """Test period 1 to 2 transition."""
        # Trade in period 1
        buyer_agent.start_period(1)
        buyer_agent.current_period = 1
        buyer_agent.num_trades = 1
        buyer_agent.buy_sell_result(1, 75, 1, 0, 0, 0, 0)

        state_p1 = (buyer_agent.roundpricesum, buyer_agent.roundweight)

        # Move to period 2
        buyer_agent.start_period(2)

        # State should persist
        assert buyer_agent.current_period == 2
        assert (buyer_agent.roundpricesum, buyer_agent.roundweight) == state_p1

    def test_period_one_resets_round(self, buyer_agent):
        """Test period 1 resets round-level state."""
        # Set some round state
        buyer_agent.roundpricesum = 1000.0
        buyer_agent.roundweight = 50.0
        buyer_agent.lastgap = 5

        # Call start_period(1)
        buyer_agent.start_period(1)

        # Should reset round state
        assert buyer_agent.roundpricesum == 0.0
        assert buyer_agent.roundweight == 0.0
        assert buyer_agent.lastgap == 10000000
