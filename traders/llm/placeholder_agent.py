"""
Placeholder LLM agent for infrastructure testing.

This agent uses simple rule-based logic but has the same interface
as real LLM agents. It's used to validate the infrastructure before
integrating actual LLM API calls.

Strategy:
- Bid at 90-95% of valuation (buyers)
- Ask at 105-110% of cost (sellers)
- Accept trades if profitable
"""

import random
from typing import Any
from traders.llm.base_llm_agent import BaseLLMAgent
from traders.llm.action_parser import BidAskAction, BuySellAction


class PlaceholderLLM(BaseLLMAgent):
    """
    Placeholder LLM agent using simple rules.

    This agent mimics the interface of a real LLM agent but uses
    deterministic/simple random logic for decisions. It's used to:
    1. Test the LLM infrastructure end-to-end
    2. Benchmark baseline performance
    3. Validate prompt building and action parsing

    Strategy:
    - Conservative bidding/asking (maintain profit margin)
    - Accept profitable trades
    - Simple time-based urgency (more aggressive near deadline)
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
        margin: float = 0.05,
        seed: int | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize placeholder agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations
            price_min: Minimum market price
            price_max: Maximum market price
            max_retries: Max retry attempts (inherited)
            margin: Profit margin target (default 5%)
            seed: Random seed for reproducibility
            **kwargs: Additional parameters
        """
        super().__init__(
            player_id, is_buyer, num_tokens, valuations,
            price_min, price_max, max_retries, **kwargs
        )
        self.margin = margin
        self.rng = random.Random(seed)

    def _generate_bid_ask_action(
        self,
        prompt: str,
        valuation: int,
        best_bid: int,
        best_ask: int
    ) -> BidAskAction:
        """
        Generate bid/ask using simple rules.

        Strategy:
        - Buyers: Bid at 90-95% of valuation (leave profit margin)
        - Sellers: Ask at 105-110% of cost (ensure profit)
        - Pass if valuation is at boundary

        Args:
            prompt: Ignored (rule-based, doesn't use prompt)
            valuation: Current token valuation
            best_bid: Current best bid
            best_ask: Current best ask

        Returns:
            BidAskAction
        """
        if self.is_buyer:
            # Buyer: Bid conservatively below valuation
            # Random bid in [90%, 95%] of valuation
            bid_pct = 0.90 + self.rng.random() * 0.05
            bid_price = int(valuation * bid_pct)

            # Ensure bid is valid
            if bid_price <= self.price_min:
                return BidAskAction(action="pass")

            # Must improve best bid (if exists)
            if best_bid > 0 and bid_price <= best_bid:
                # Try to improve by small amount
                bid_price = best_bid + 1
                if bid_price > valuation:
                    return BidAskAction(action="pass")

            return BidAskAction(action="bid", price=bid_price)

        else:
            # Seller: Ask conservatively above cost
            # Random ask in [105%, 110%] of cost
            ask_pct = 1.05 + self.rng.random() * 0.05
            ask_price = int(valuation * ask_pct)

            # Ensure ask is valid
            if ask_price >= self.price_max:
                return BidAskAction(action="pass")

            # Must improve best ask (if exists)
            if best_ask > 0 and ask_price >= best_ask:
                # Try to improve by small amount
                ask_price = best_ask - 1
                if ask_price < valuation:
                    return BidAskAction(action="pass")

            return BidAskAction(action="ask", price=ask_price)

    def _generate_buy_sell_action(
        self,
        prompt: str,
        valuation: int,
        trade_price: int
    ) -> BuySellAction:
        """
        Generate buy/sell decision using simple profitability check.

        Strategy:
        - Accept if profit > margin threshold
        - Add slight randomness (90% accept if profitable)

        Args:
            prompt: Ignored (rule-based, doesn't use prompt)
            valuation: Current token valuation
            trade_price: Price if trade is accepted

        Returns:
            BuySellAction
        """
        # Calculate profit
        if self.is_buyer:
            profit = valuation - trade_price
        else:
            profit = trade_price - valuation

        # Accept if profitable (with high probability)
        if profit > 0:
            # 90% chance to accept profitable trade
            # (10% pass to simulate strategic waiting)
            if self.rng.random() < 0.9:
                return BuySellAction(action="accept")

        return BuySellAction(action="pass")

    def __repr__(self) -> str:
        """String representation."""
        agent_type = "Buyer" if self.is_buyer else "Seller"
        invalid_rate = self.get_invalid_action_rate()
        return (
            f"PlaceholderLLM(id={self.player_id}, type={agent_type}, "
            f"trades={self.num_trades}/{self.num_tokens}, "
            f"invalid_rate={invalid_rate:.1f}%)"
        )
