"""
Action parsing and validation for LLM agents.

This module defines Pydantic models for structured LLM outputs
and validates that actions respect AURORA market constraints.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class BidAskAction(BaseModel):
    """
    Structured output for bid/ask stage decision with optional reasoning.

    Buyers submit bids (prices they're willing to pay).
    Sellers submit asks (prices they're willing to accept).
    """
    reasoning: Optional[str] = Field(
        default=None,
        description="Step-by-step explanation of the decision (2-3 sentences)"
    )
    action: Literal["bid", "ask", "pass"] = Field(
        description="Action to take: 'bid' (buyers), 'ask' (sellers), or 'pass' (no action)"
    )
    price: Optional[int] = Field(
        default=None,
        description="Price for bid/ask (required if action != 'pass')"
    )

    @field_validator("price")
    @classmethod
    def price_required_for_bid_ask(cls, v, info):
        """Validate that price is provided when action is bid/ask."""
        if info.data.get("action") in ["bid", "ask"] and v is None:
            raise ValueError(f"price is required when action={info.data.get('action')}")
        return v


class BuySellAction(BaseModel):
    """
    Structured output for buy/sell stage decision with optional reasoning.

    Only the high bidder can buy (accept low ask).
    Only the low asker can sell (accept high bid).
    """
    reasoning: Optional[str] = Field(
        default=None,
        description="Step-by-step explanation of the decision (2-3 sentences)"
    )
    action: Literal["accept", "pass"] = Field(
        description="Action to take: 'accept' (trade) or 'pass' (wait)"
    )


class ActionValidator:
    """
    Validates LLM actions against AURORA market constraints.
    """

    @staticmethod
    def validate_bid(
        bid: int,
        valuation: int,
        price_min: int,
        best_bid: int = 0
    ) -> tuple[bool, str]:
        """
        Validate a buyer's bid.

        Constraints:
        1. Bid <= valuation (cannot bid above willingness to pay)
        2. Bid >= price_min (market minimum)
        3. Bid > best_bid (must improve current best)

        Args:
            bid: Proposed bid price
            valuation: Buyer's private valuation
            price_min: Market minimum price
            best_bid: Current best bid (0 if none)

        Returns:
            (valid, error_message) tuple
        """
        if bid > valuation:
            return False, f"Bid {bid} loses money. Your value is {valuation}."
        if bid < price_min:
            return False, f"Bid {bid} below market minimum {price_min}."
        if best_bid > 0 and bid <= best_bid:
            return False, f"Bid {bid} doesn't beat current best of {best_bid}."
        return True, ""

    @staticmethod
    def validate_ask(
        ask: int,
        cost: int,
        price_max: int,
        best_ask: int = 0
    ) -> tuple[bool, str]:
        """
        Validate a seller's ask.

        Constraints:
        1. Ask >= cost (cannot sell below cost)
        2. Ask <= price_max (market maximum)
        3. Ask < best_ask (must improve current best)

        Args:
            ask: Proposed ask price
            cost: Seller's private cost
            price_max: Market maximum price
            best_ask: Current best ask (0 if none)

        Returns:
            (valid, error_message) tuple
        """
        if ask < cost:
            return False, f"Ask {ask} loses money. Your cost is {cost}."
        if ask > price_max:
            return False, f"Ask {ask} above market maximum {price_max}."
        if best_ask > 0 and ask >= best_ask:
            return False, f"Ask {ask} doesn't beat current best of {best_ask}."
        return True, ""

    @staticmethod
    def validate_buy_acceptance(
        is_high_bidder: bool,
        ask_price: int,
        valuation: int
    ) -> tuple[bool, str]:
        """
        Validate a buyer's decision to accept an ask.

        Constraints:
        1. Must be the high bidder
        2. Ask <= valuation (cannot buy above willingness to pay)

        Args:
            is_high_bidder: Whether this agent is the high bidder
            ask_price: The ask price to accept
            valuation: Buyer's private valuation

        Returns:
            (valid, error_message) tuple
        """
        if not is_high_bidder:
            return False, "Only the high bidder can accept the ask"
        if ask_price > valuation:
            return False, f"Ask {ask_price} exceeds valuation {valuation}"
        return True, ""

    @staticmethod
    def validate_sell_acceptance(
        is_low_asker: bool,
        bid_price: int,
        cost: int
    ) -> tuple[bool, str]:
        """
        Validate a seller's decision to accept a bid.

        Constraints:
        1. Must be the low asker
        2. Bid >= cost (cannot sell below cost)

        Args:
            is_low_asker: Whether this agent is the low asker
            bid_price: The bid price to accept
            cost: Seller's private cost

        Returns:
            (valid, error_message) tuple
        """
        if not is_low_asker:
            return False, "Only the low asker can accept the bid"
        if bid_price < cost:
            return False, f"Bid {bid_price} below cost {cost}"
        return True, ""
