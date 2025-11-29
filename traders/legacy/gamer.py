"""
Gamer Agent - Fixed margin strategy.

From the 1993 Santa Fe Tournament (24th Place).
A simple baseline that ignores market state and bids/asks at fixed margins.
"""

from typing import Any, Optional
from traders.base import Agent


class Gamer(Agent):
    """
    Gamer trader - fixed profit margin, ignores market state.

    Strategy:
    - Buyers: Bid at valuation * (1 - margin)
    - Sellers: Ask at cost * (1 + margin)
    - Does NOT adjust based on current bid/ask spread
    - Trade acceptance: Accept if profitable

    This is a poor performer (24th place) but useful as a fixed-margin baseline.
    Unlike Markup, Gamer was an actual Santa Fe tournament entrant.
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        price_min: int = 0,
        price_max: int = 100,
        margin: float = 0.10,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize Gamer agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations
            price_min: Minimum allowed price (default 0)
            price_max: Maximum allowed price (default 100)
            margin: Profit margin (default 0.10 = 10%)
            seed: Ignored (kept for interface consistency)
            **kwargs: Ignored extra arguments
        """
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min = price_min
        self.price_max = price_max
        self.margin = margin

        # State tracking
        self.current_bid = 0
        self.current_ask = 0
        self.current_bidder = 0
        self.current_asker = 0

    def bid_ask(self, time: int, nobidask: int) -> None:
        """Prepare for bid/ask phase."""
        self.has_responded = False

    def bid_ask_response(self) -> int:
        """
        Return bid/ask at fixed margin from valuation.

        Does NOT consider current market state for price calculation.
        Only checks if our fixed price would improve the market.
        """
        self.has_responded = True

        if self.num_trades >= self.num_tokens:
            return 0

        valuation = self.valuations[self.num_trades]

        if self.is_buyer:
            # Bid 10% below valuation
            target = int(valuation * (1 - self.margin))

            # Clamp to price range
            target = max(self.price_min, min(target, self.price_max))

            # Only submit if it would improve current bid
            if self.current_bid > 0 and target <= self.current_bid:
                return 0

            return target
        else:
            # Ask 10% above cost
            target = int(valuation * (1 + self.margin))

            # Clamp to price range
            target = max(self.price_min, min(target, self.price_max))

            # Only submit if it would improve current ask
            if self.current_ask > 0 and target >= self.current_ask:
                return 0

            return target

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
        """Update market state after bid/ask phase."""
        super().bid_ask_result(status, num_trades, new_bids, new_asks,
                              high_bid, high_bidder, low_ask, low_asker)
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker

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
        """Accept if profitable."""
        self.has_responded = True

        if self.num_trades >= self.num_tokens:
            return False

        valuation = self.valuations[self.num_trades]

        if self.is_buyer:
            # Accept if current ask is below our valuation
            if self.current_ask > 0 and self.current_ask < valuation:
                return True
        else:
            # Accept if current bid is above our cost
            if self.current_bid > 0 and self.current_bid > valuation:
                return True

        return False

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
        """Update state after trade."""
        super().buy_sell_result(status, trade_price, trade_type,
                                high_bid, high_bidder, low_ask, low_asker)
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker
