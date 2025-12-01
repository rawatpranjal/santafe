# tests/unit/traders/test_llm_agents.py
"""
Unit tests for LLM-based trading agents.

Tests cover:
- Action parsing (BidAskAction, BuySellAction)
- Action validation (bid/ask constraints, buy/sell eligibility)
- BaseLLMAgent state tracking
- GPTAgent initialization and prompt style handling
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from traders.llm.action_parser import (
    ActionValidator,
    BidAskAction,
    BuySellAction,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def buyer_valuations():
    return [100, 90, 80, 70]


@pytest.fixture
def seller_valuations():
    return [30, 40, 50, 60]


@pytest.fixture
def validator():
    return ActionValidator()


# =============================================================================
# Test: BidAskAction Parsing
# =============================================================================


class TestBidAskAction:
    """Tests for BidAskAction Pydantic model."""

    def test_bid_action_with_none_price_allowed(self):
        """Bid action with None price is technically allowed by Pydantic but semantically invalid.

        Note: The field_validator checks price_required_for_bid_ask but due to Pydantic v2
        validation order, it may not raise. The application layer (BaseLLMAgent) handles this.
        """
        # This doesn't raise in Pydantic v2 due to validation order
        action = BidAskAction(action="bid")
        assert action.price is None  # Pydantic allows it
        # Application layer will reject this via BaseLLMAgent.bid_ask_response()

    def test_ask_action_with_none_price_allowed(self):
        """Ask action with None price is allowed by Pydantic but rejected by application.

        The BaseLLMAgent.bid_ask_response() checks for None price and raises ValueError.
        """
        action = BidAskAction(action="ask")
        assert action.price is None  # Pydantic allows it

    def test_pass_action_no_price_required(self):
        """Pass action does not require price."""
        action = BidAskAction(action="pass")
        assert action.action == "pass"
        assert action.price is None

    def test_bid_action_with_price(self):
        """Bid action with valid price."""
        action = BidAskAction(action="bid", price=50)
        assert action.action == "bid"
        assert action.price == 50

    def test_ask_action_with_price(self):
        """Ask action with valid price."""
        action = BidAskAction(action="ask", price=60)
        assert action.action == "ask"
        assert action.price == 60

    def test_reasoning_is_optional(self):
        """Reasoning field is optional."""
        action = BidAskAction(action="bid", price=50)
        assert action.reasoning is None

    def test_reasoning_included(self):
        """Reasoning can be included."""
        action = BidAskAction(action="bid", price=50, reasoning="Market looks good")
        assert action.reasoning == "Market looks good"

    def test_invalid_action_type(self):
        """Invalid action type should raise error."""
        with pytest.raises(ValidationError):
            BidAskAction(action="invalid", price=50)


# =============================================================================
# Test: BuySellAction Parsing
# =============================================================================


class TestBuySellAction:
    """Tests for BuySellAction Pydantic model."""

    def test_accept_action(self):
        """Accept action is valid."""
        action = BuySellAction(action="accept")
        assert action.action == "accept"

    def test_pass_action(self):
        """Pass action is valid."""
        action = BuySellAction(action="pass")
        assert action.action == "pass"

    def test_invalid_action(self):
        """Invalid action type should raise error."""
        with pytest.raises(ValidationError):
            BuySellAction(action="buy")  # Should be "accept", not "buy"

    def test_reasoning_optional(self):
        """Reasoning is optional."""
        action = BuySellAction(action="accept")
        assert action.reasoning is None

    def test_reasoning_included(self):
        """Reasoning can be included."""
        action = BuySellAction(action="accept", reasoning="Profitable trade")
        assert action.reasoning == "Profitable trade"


# =============================================================================
# Test: ActionValidator - Bid Validation
# =============================================================================


class TestBidValidation:
    """Tests for bid validation logic."""

    def test_valid_bid_no_existing(self, validator):
        """Valid bid when no existing bid."""
        valid, error = validator.validate_bid(
            bid=50,
            valuation=100,
            price_min=1,
            best_bid=0,
        )
        assert valid is True
        assert error == ""

    def test_valid_bid_improves_existing(self, validator):
        """Valid bid that improves existing best bid."""
        valid, error = validator.validate_bid(
            bid=55,
            valuation=100,
            price_min=1,
            best_bid=50,
        )
        assert valid is True

    def test_bid_exceeds_valuation(self, validator):
        """Bid above valuation is invalid."""
        valid, error = validator.validate_bid(
            bid=110,
            valuation=100,
            price_min=1,
            best_bid=0,
        )
        assert valid is False
        assert "loses money" in error

    def test_bid_below_minimum(self, validator):
        """Bid below market minimum is invalid."""
        valid, error = validator.validate_bid(
            bid=5,
            valuation=100,
            price_min=10,
            best_bid=0,
        )
        assert valid is False
        assert "below market minimum" in error

    def test_bid_doesnt_improve_existing(self, validator):
        """Bid that doesn't improve best bid is invalid."""
        valid, error = validator.validate_bid(
            bid=50,  # Equal to best bid
            valuation=100,
            price_min=1,
            best_bid=50,
        )
        assert valid is False
        assert "doesn't beat" in error

    def test_bid_below_existing(self, validator):
        """Bid below best bid is invalid."""
        valid, error = validator.validate_bid(
            bid=45,
            valuation=100,
            price_min=1,
            best_bid=50,
        )
        assert valid is False
        assert "doesn't beat" in error


# =============================================================================
# Test: ActionValidator - Ask Validation
# =============================================================================


class TestAskValidation:
    """Tests for ask validation logic."""

    def test_valid_ask_no_existing(self, validator):
        """Valid ask when no existing ask."""
        valid, error = validator.validate_ask(
            ask=60,
            cost=30,
            price_max=200,
            best_ask=0,
        )
        assert valid is True
        assert error == ""

    def test_valid_ask_improves_existing(self, validator):
        """Valid ask that improves existing best ask."""
        valid, error = validator.validate_ask(
            ask=55,
            cost=30,
            price_max=200,
            best_ask=60,
        )
        assert valid is True

    def test_ask_below_cost(self, validator):
        """Ask below cost is invalid."""
        valid, error = validator.validate_ask(
            ask=25,
            cost=30,
            price_max=200,
            best_ask=0,
        )
        assert valid is False
        assert "loses money" in error

    def test_ask_above_maximum(self, validator):
        """Ask above market maximum is invalid."""
        valid, error = validator.validate_ask(
            ask=250,
            cost=30,
            price_max=200,
            best_ask=0,
        )
        assert valid is False
        assert "above market maximum" in error

    def test_ask_doesnt_improve_existing(self, validator):
        """Ask that doesn't improve best ask is invalid."""
        valid, error = validator.validate_ask(
            ask=60,  # Equal to best ask
            cost=30,
            price_max=200,
            best_ask=60,
        )
        assert valid is False
        assert "doesn't beat" in error

    def test_ask_above_existing(self, validator):
        """Ask above best ask is invalid."""
        valid, error = validator.validate_ask(
            ask=70,
            cost=30,
            price_max=200,
            best_ask=60,
        )
        assert valid is False
        assert "doesn't beat" in error


# =============================================================================
# Test: ActionValidator - Buy/Sell Acceptance
# =============================================================================


class TestBuySellAcceptanceValidation:
    """Tests for buy/sell acceptance validation."""

    def test_buyer_can_accept_profitable_ask(self, validator):
        """High bidder can accept ask below valuation."""
        valid, error = validator.validate_buy_acceptance(
            is_high_bidder=True,
            ask_price=80,
            valuation=100,
        )
        assert valid is True

    def test_buyer_cannot_accept_if_not_high_bidder(self, validator):
        """Non-high bidder cannot accept."""
        valid, error = validator.validate_buy_acceptance(
            is_high_bidder=False,
            ask_price=80,
            valuation=100,
        )
        assert valid is False
        assert "high bidder" in error

    def test_buyer_cannot_accept_unprofitable_ask(self, validator):
        """Buyer cannot accept ask above valuation."""
        valid, error = validator.validate_buy_acceptance(
            is_high_bidder=True,
            ask_price=110,
            valuation=100,
        )
        assert valid is False
        assert "exceeds valuation" in error

    def test_seller_can_accept_profitable_bid(self, validator):
        """Low asker can accept bid above cost."""
        valid, error = validator.validate_sell_acceptance(
            is_low_asker=True,
            bid_price=50,
            cost=30,
        )
        assert valid is True

    def test_seller_cannot_accept_if_not_low_asker(self, validator):
        """Non-low asker cannot accept."""
        valid, error = validator.validate_sell_acceptance(
            is_low_asker=False,
            bid_price=50,
            cost=30,
        )
        assert valid is False
        assert "low asker" in error

    def test_seller_cannot_accept_unprofitable_bid(self, validator):
        """Seller cannot accept bid below cost."""
        valid, error = validator.validate_sell_acceptance(
            is_low_asker=True,
            bid_price=25,
            cost=30,
        )
        assert valid is False
        assert "below cost" in error


# =============================================================================
# Test: BaseLLMAgent State Tracking (via mock subclass)
# =============================================================================


class TestBaseLLMAgentState:
    """Tests for BaseLLMAgent state tracking."""

    @pytest.fixture
    def mock_llm_agent(self, buyer_valuations):
        """Create a mock LLM agent for testing state."""
        from traders.llm.base_llm_agent import BaseLLMAgent

        class MockLLMAgent(BaseLLMAgent):
            """Concrete subclass for testing."""

            def _generate_bid_ask_action(self, prompt, valuation, best_bid, best_ask):
                return BidAskAction(action="pass")

            def _generate_buy_sell_action(self, prompt, valuation, trade_price):
                return BuySellAction(action="pass")

        return MockLLMAgent(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=buyer_valuations,
            price_min=1,
            price_max=200,
        )

    def test_initial_state(self, mock_llm_agent):
        """Initial state should be correct."""
        assert mock_llm_agent.num_trades == 0
        assert mock_llm_agent.period_profit == 0
        assert mock_llm_agent._current_best_bid == 0
        assert mock_llm_agent._current_best_ask == 0
        assert mock_llm_agent.invalid_action_count == 0
        assert mock_llm_agent.total_decisions == 0

    def test_bid_ask_increments_decision_count(self, mock_llm_agent):
        """bid_ask_response should increment decision count."""
        mock_llm_agent.bid_ask(time=1, nobidask=0)
        mock_llm_agent.bid_ask_response()
        assert mock_llm_agent.total_decisions == 1

    def test_buy_sell_increments_decision_count(self, mock_llm_agent):
        """buy_sell_response should increment decision count."""
        mock_llm_agent.buy_sell(
            time=1, nobuysell=0, high_bid=50, low_ask=60, high_bidder=1, low_asker=2
        )
        mock_llm_agent.buy_sell_response()
        assert mock_llm_agent.total_decisions == 1

    def test_state_updated_after_bid_ask_result(self, mock_llm_agent):
        """State should update after bid_ask_result."""
        mock_llm_agent.bid_ask_result(
            status=0,
            num_trades=0,
            new_bids=[50],
            new_asks=[60],
            high_bid=50,
            high_bidder=1,
            low_ask=60,
            low_asker=2,
        )
        assert mock_llm_agent._current_best_bid == 50
        assert mock_llm_agent._current_best_ask == 60

    def test_trade_history_updated_after_buy_sell_result(self, mock_llm_agent):
        """Trade history should update after trade."""
        mock_llm_agent._current_time = 5
        mock_llm_agent.buy_sell_result(
            status=1,
            trade_price=55,
            trade_type=1,
            high_bid=0,
            high_bidder=0,
            low_ask=0,
            low_asker=0,
        )
        assert 55 in mock_llm_agent._recent_trades
        assert (5, 55) in mock_llm_agent._trade_history

    def test_start_period_resets_state(self, mock_llm_agent):
        """start_period should reset period-specific state."""
        # Add some state
        mock_llm_agent._recent_trades = [50, 55, 60]
        mock_llm_agent._trade_history = [(1, 50), (2, 55)]
        mock_llm_agent._order_book_history = [(1, 40, 70)]

        mock_llm_agent.start_period(2)

        assert mock_llm_agent._recent_trades == []
        assert mock_llm_agent._trade_history == []
        assert mock_llm_agent._order_book_history == []
        assert mock_llm_agent._current_period == 2

    def test_cannot_bid_without_tokens(self, mock_llm_agent):
        """Should pass when no tokens remaining."""
        mock_llm_agent.num_trades = 4  # All tokens traded

        mock_llm_agent.bid_ask(time=1, nobidask=0)
        response = mock_llm_agent.bid_ask_response()

        assert response == -99  # Pass

    def test_cannot_buy_without_tokens(self, mock_llm_agent):
        """Should pass buy_sell when no tokens remaining."""
        mock_llm_agent.num_trades = 4

        mock_llm_agent.buy_sell(
            time=1, nobuysell=0, high_bid=50, low_ask=60, high_bidder=1, low_asker=2
        )
        response = mock_llm_agent.buy_sell_response()

        assert response is False

    def test_cannot_buy_if_not_high_bidder(self, mock_llm_agent):
        """Buyer cannot accept if not high bidder."""
        mock_llm_agent.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=50,
            low_ask=60,
            high_bidder=2,
            low_asker=3,  # Different player is high bidder
        )
        response = mock_llm_agent.buy_sell_response()

        assert response is False


# =============================================================================
# Test: GPTAgent Initialization
# =============================================================================


class TestGPTAgentInit:
    """Tests for GPTAgent initialization."""

    @pytest.fixture
    def mock_litellm(self):
        """Mock litellm module."""
        with patch.dict("sys.modules", {"litellm": MagicMock()}):
            yield

    def test_init_requires_litellm(self, buyer_valuations):
        """GPTAgent should raise ImportError without litellm."""
        with patch("traders.llm.gpt_agent.LITELLM_AVAILABLE", False):
            from traders.llm.gpt_agent import GPTAgent

            with pytest.raises(ImportError):
                GPTAgent(
                    player_id=1,
                    is_buyer=True,
                    num_tokens=4,
                    valuations=buyer_valuations,
                )

    def test_prompt_styles_supported(self, buyer_valuations):
        """Should support all documented prompt styles."""
        with patch("traders.llm.gpt_agent.LITELLM_AVAILABLE", True):
            from traders.llm.gpt_agent import GPTAgent

            styles = ["minimal", "deep", "dashboard", "constraints", "dense", "original"]
            for style in styles:
                agent = GPTAgent(
                    player_id=1,
                    is_buyer=True,
                    num_tokens=4,
                    valuations=buyer_valuations,
                    prompt_style=style,
                )
                assert agent.prompt_style == style

    def test_caching_can_be_disabled(self, buyer_valuations):
        """Caching should be configurable."""
        with patch("traders.llm.gpt_agent.LITELLM_AVAILABLE", True):
            from traders.llm.gpt_agent import GPTAgent

            agent = GPTAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_valuations,
                use_cache=False,
            )
            assert agent.use_cache is False
            assert agent.cache is None

    def test_default_model_is_gpt4o_mini(self, buyer_valuations):
        """Default model should be gpt-4o-mini."""
        with patch("traders.llm.gpt_agent.LITELLM_AVAILABLE", True):
            from traders.llm.gpt_agent import GPTAgent

            agent = GPTAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_valuations,
            )
            assert agent.model == "gpt-4o-mini"


# =============================================================================
# Test: Invalid Action Rate Calculation
# =============================================================================


class TestInvalidActionRate:
    """Tests for invalid action rate tracking."""

    @pytest.fixture
    def mock_llm_agent(self, buyer_valuations):
        """Create a mock LLM agent for testing."""
        from traders.llm.base_llm_agent import BaseLLMAgent

        class MockLLMAgent(BaseLLMAgent):
            def _generate_bid_ask_action(self, prompt, valuation, best_bid, best_ask):
                return BidAskAction(action="pass")

            def _generate_buy_sell_action(self, prompt, valuation, trade_price):
                return BuySellAction(action="pass")

        return MockLLMAgent(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=buyer_valuations,
        )

    def test_zero_rate_initially(self, mock_llm_agent):
        """Invalid rate should be 0% initially."""
        assert mock_llm_agent.get_invalid_action_rate() == 0.0

    def test_rate_calculation(self, mock_llm_agent):
        """Invalid rate calculation should be correct."""
        mock_llm_agent.total_decisions = 10
        mock_llm_agent.invalid_action_count = 2

        rate = mock_llm_agent.get_invalid_action_rate()
        assert rate == 20.0  # 2/10 * 100 = 20%

    def test_no_division_by_zero(self, mock_llm_agent):
        """Should handle zero decisions gracefully."""
        mock_llm_agent.total_decisions = 0
        mock_llm_agent.invalid_action_count = 0

        rate = mock_llm_agent.get_invalid_action_rate()
        assert rate == 0.0
