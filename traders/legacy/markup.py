"""
Markup Agent - Fixed percentage markup strategy.

From Chen et al. (2010): A non-adaptive baseline that applies a fixed
percentage markup to valuations/costs. Simpler than ZIC as it doesn't
even randomize - just applies the same markup every time.
"""

from typing import Any, Optional
from traders.base import Agent


class Markup(Agent):
    """
    Fixed percentage markup trader - non-adaptive baseline.

    Strategy:
    - Buyers: Bid at valuation * (1 - markup_pct)
    - Sellers: Ask at cost * (1 + markup_pct)
    - Trade acceptance: Accept if profitable (like ZIC)

    This is simpler than ZIC because it's deterministic - no randomization.
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        price_min: int = 0,
        price_max: int = 100,
        markup_pct: float = 0.10,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize Markup agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations
            price_min: Minimum allowed price (default 0)
            price_max: Maximum allowed price (default 100)
            markup_pct: Markup percentage (default 0.10 = 10%)
            seed: Ignored (kept for interface consistency)
            **kwargs: Ignored extra arguments
        """
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min = price_min
        self.price_max = price_max
        self.markup_pct = markup_pct

        # State tracking for buy/sell phase
        self.current_bid = 0
        self.current_ask = 0
        self.current_bidder = 0
        self.current_asker = 0

    def bid_ask(self, time: int, nobidask: int) -> None:
        """Prepare for bid/ask phase."""
        self.has_responded = False

    def bid_ask_response(self) -> int:
        """
        Return a bid/ask with fixed markup applied.

        Buyers bid below valuation, sellers ask above cost.
        """
        self.has_responded = True

        if self.num_trades >= self.num_tokens:
            return 0

        valuation = self.valuations[self.num_trades]

        if self.is_buyer:
            # Bid at valuation * (1 - markup%)
            bid = int(valuation * (1 - self.markup_pct))
            return max(self.price_min, bid)
        else:
            # Ask at cost * (1 + markup%)
            ask = int(valuation * (1 + self.markup_pct))
            return min(self.price_max, ask)

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
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker

    def buy_sell_response(self) -> bool:
        """
        Accept trade if profitable and we are the winner.

        Same logic as ZIC - only accept if we're the winner and it's profitable.
        """
        self.has_responded = True

        if self.num_trades >= self.num_tokens:
            return False

        valuation = self.valuations[self.num_trades]

        if self.is_buyer:
            # Don't buy above valuation
            if self.current_ask > 0 and valuation <= self.current_ask:
                return False
            # Accept if we're high bidder and spread is crossed
            if (self.player_id == self.current_bidder and
                self.current_bid > 0 and
                self.current_bid >= self.current_ask):
                return True
        else:
            # Don't sell below cost
            if self.current_bid > 0 and self.current_bid <= valuation:
                return False
            # Accept if we're low asker and spread is crossed
            if (self.player_id == self.current_asker and
                self.current_ask > 0 and
                self.current_ask <= self.current_bid):
                return True

        return False
