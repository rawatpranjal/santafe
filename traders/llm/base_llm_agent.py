"""
Abstract base class for LLM-based trading agents.

This module defines the common interface and logic for agents that use
Large Language Models to make trading decisions in the AURORA protocol.
"""

import logging
from abc import abstractmethod
from typing import Any, Optional
from traders.base import Agent
from traders.llm.prompt_builder import PromptBuilder
from traders.llm.action_parser import (
    BidAskAction,
    BuySellAction,
    ActionValidator
)

logger = logging.getLogger(__name__)


class BaseLLMAgent(Agent):
    """
    Abstract base class for LLM-powered trading agents.

    Handles the AURORA protocol callbacks and converts them to natural language
    prompts. Subclasses implement the actual LLM inference logic.
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        price_min: int = 0,
        price_max: int = 1000,
        max_retries: int = 3,
        num_times: int = 100,
        prompt_style: str = "minimal",
        **kwargs: Any
    ) -> None:
        """
        Initialize LLM agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations
            price_min: Minimum market price (default 0)
            price_max: Maximum market price (default 1000)
            max_retries: Maximum retry attempts for invalid actions (default 3)
            num_times: Maximum time steps per period (default 100)
            prompt_style: "minimal" for pure facts, "original" for verbose prompts
            **kwargs: Additional agent-specific parameters
        """
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min = price_min
        self.price_max = price_max
        self.max_retries = max_retries
        self.max_time = num_times
        self.prompt_style = prompt_style

        # Prompt builder for converting state to text
        self.prompt_builder = PromptBuilder()

        # Action validator
        self.validator = ActionValidator()

        # State tracking for current decision
        self._current_time = 0
        self._current_best_bid = 0
        self._current_best_ask = 0
        self._current_high_bid = 0
        self._current_low_ask = 0
        self._is_high_bidder = False
        self._is_low_asker = False

        # Trade history tracking (for context)
        self._recent_trades: list[int] = []
        self._max_history = 10

        # Statistics
        self.invalid_action_count = 0
        self.total_decisions = 0

    # =========================================================================
    # ABSTRACT METHODS (Subclasses must implement)
    # =========================================================================

    @abstractmethod
    def _generate_bid_ask_action(
        self,
        prompt: str,
        valuation: int,
        best_bid: int,
        best_ask: int
    ) -> BidAskAction:
        """
        Generate a bid/ask action using the LLM or rule-based logic.

        Args:
            prompt: Natural language prompt describing situation
            valuation: Current token valuation
            best_bid: Current best bid
            best_ask: Current best ask

        Returns:
            BidAskAction with action type and optional price
        """
        pass

    @abstractmethod
    def _generate_buy_sell_action(
        self,
        prompt: str,
        valuation: int,
        trade_price: int
    ) -> BuySellAction:
        """
        Generate a buy/sell decision using the LLM or rule-based logic.

        Args:
            prompt: Natural language prompt describing situation
            valuation: Current token valuation
            trade_price: Price if trade is accepted

        Returns:
            BuySellAction with accept/pass decision
        """
        pass

    # =========================================================================
    # AURORA PROTOCOL IMPLEMENTATION
    # =========================================================================

    def bid_ask(self, time: int, nobidask: int) -> None:
        """Prepare for bid/ask decision."""
        self.has_responded = False
        self._current_time = time

    def bid_ask_response(self) -> int:
        """
        Return bid/ask decision with retry logic for invalid actions.

        Returns:
            Price for bid/ask, or -99 to pass
        """
        self.has_responded = True
        self.total_decisions += 1

        # If no tokens left, pass
        if self.num_trades >= self.num_tokens:
            return -99

        valuation = self.valuations[self.num_trades]

        # Build prompt based on style
        if self.prompt_style == "minimal":
            prompt = self.prompt_builder.build_minimal_bid_ask_prompt(
                is_buyer=self.is_buyer,
                valuation=valuation,
                tokens_remaining=self.num_tokens - self.num_trades,
                tokens_total=self.num_tokens,
                time_step=self._current_time,
                max_time=self.max_time,
                best_bid=self._current_best_bid,
                best_ask=self._current_best_ask
            )
        else:
            # Original verbose prompt
            spread = 0
            if self._current_best_bid > 0 and self._current_best_ask > 0:
                spread = self._current_best_ask - self._current_best_bid

            prompt = self.prompt_builder.build_bid_ask_prompt(
                is_buyer=self.is_buyer,
                valuation=valuation,
                tokens_remaining=self.num_tokens - self.num_trades,
                tokens_total=self.num_tokens,
                time_step=self._current_time,
                max_time=self.max_time,
                best_bid=self._current_best_bid,
                best_ask=self._current_best_ask,
                spread=spread,
                price_min=self.price_min,
                price_max=self.price_max,
                recent_trades=self._recent_trades
            )

        # Retry loop for invalid actions
        for attempt in range(1, self.max_retries + 1):
            try:
                # Generate action
                action = self._generate_bid_ask_action(
                    prompt=prompt,
                    valuation=valuation,
                    best_bid=self._current_best_bid,
                    best_ask=self._current_best_ask
                )

                # If passing, return immediately
                if action.action == "pass":
                    return -99

                # Validate action
                if action.price is None:
                    raise ValueError("Price is required for bid/ask action")

                # Validate constraints
                if self.is_buyer:
                    valid, error = self.validator.validate_bid(
                        bid=action.price,
                        valuation=valuation,
                        price_min=self.price_min,
                        best_bid=self._current_best_bid
                    )
                else:
                    valid, error = self.validator.validate_ask(
                        ask=action.price,
                        cost=valuation,
                        price_max=self.price_max,
                        best_ask=self._current_best_ask
                    )

                if not valid:
                    raise ValueError(error)

                # Valid action!
                return action.price

            except Exception as e:
                # Invalid action - log and retry
                logger.warning(
                    f"Invalid action (attempt {attempt}/{self.max_retries}): {e} | "
                    f"agent={self.player_id} role={'buyer' if self.is_buyer else 'seller'} "
                    f"valuation={valuation} best_bid={self._current_best_bid} "
                    f"best_ask={self._current_best_ask} time={self._current_time}"
                )
                self.invalid_action_count += 1

                if attempt < self.max_retries:
                    # Add error feedback to prompt for retry
                    if self.prompt_style == "minimal":
                        error_feedback = self.prompt_builder.format_minimal_error_feedback(str(e))
                    else:
                        error_feedback = self.prompt_builder.format_error_feedback(str(e), attempt)
                    prompt = prompt + "\n" + error_feedback
                else:
                    # Max retries exceeded - pass
                    return -99

        # Should never reach here, but safety fallback
        return -99

    def buy_sell(
        self,
        time: int,
        nobuysell: int,
        high_bid: int,
        low_ask: int,
        high_bidder: int,
        low_asker: int,
    ) -> None:
        """Prepare for buy/sell decision."""
        self.has_responded = False
        self._current_time = time
        self._current_high_bid = high_bid
        self._current_low_ask = low_ask
        self._is_high_bidder = (high_bidder == self.player_id)
        self._is_low_asker = (low_asker == self.player_id)

    def buy_sell_response(self) -> bool:
        """
        Return buy/sell decision.

        Returns:
            True to accept trade, False to pass
        """
        self.has_responded = True
        self.total_decisions += 1

        # If cannot trade, must pass
        if self.num_trades >= self.num_tokens:
            return False

        # If not the high bidder/low asker, cannot accept
        can_accept = self._is_high_bidder if self.is_buyer else self._is_low_asker
        if not can_accept:
            return False

        valuation = self.valuations[self.num_trades]
        trade_price = self._current_low_ask if self.is_buyer else self._current_high_bid

        # Build prompt based on style
        if self.prompt_style == "minimal":
            prompt = self.prompt_builder.build_minimal_buy_sell_prompt(
                is_buyer=self.is_buyer,
                valuation=valuation,
                trade_price=trade_price,
                can_accept=True  # Already checked above
            )
        else:
            prompt = self.prompt_builder.build_buy_sell_prompt(
                is_buyer=self.is_buyer,
                valuation=valuation,
                tokens_remaining=self.num_tokens - self.num_trades,
                time_step=self._current_time,
                max_time=self.max_time,
                high_bid=self._current_high_bid,
                low_ask=self._current_low_ask,
                is_high_bidder=self._is_high_bidder,
                is_low_asker=self._is_low_asker
            )

        # Generate action (no retry needed - only accept/pass, always valid)
        action = self._generate_buy_sell_action(
            prompt=prompt,
            valuation=valuation,
            trade_price=trade_price
        )

        return action.action == "accept"

    # =========================================================================
    # RESULT NOTIFICATION CALLBACKS
    # =========================================================================

    def bid_ask_result(
        self,
        status: int,
        num_trades: int,
        new_bids: list[int],
        new_asks: list[int],
        high_bid: int,
        high_bidder: int,
        low_ask: int,
        low_asker: int,
    ) -> None:
        """Update state after bid/ask stage."""
        super().bid_ask_result(
            status, num_trades, new_bids, new_asks,
            high_bid, high_bidder, low_ask, low_asker
        )
        self._current_best_bid = high_bid
        self._current_best_ask = low_ask

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
        """Update state after buy/sell stage."""
        super().buy_sell_result(
            status, trade_price, trade_type,
            high_bid, high_bidder, low_ask, low_asker
        )

        # Track trades for history
        if trade_price > 0:
            self._recent_trades.append(trade_price)
            if len(self._recent_trades) > self._max_history:
                self._recent_trades = self._recent_trades[-self._max_history:]

        # Update best bid/ask
        self._current_best_bid = high_bid
        self._current_best_ask = low_ask

    def start_period(self, period_number: int) -> None:
        """Reset period-specific state."""
        super().start_period(period_number)
        self._recent_trades = []

    def get_invalid_action_rate(self) -> float:
        """Calculate percentage of invalid actions."""
        if self.total_decisions == 0:
            return 0.0
        return (self.invalid_action_count / self.total_decisions) * 100
