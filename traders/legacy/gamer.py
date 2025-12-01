"""
Gamer Agent - Fixed margin strategy.

From the 1993 Santa Fe Tournament (24th Place).
A level-1, nonadaptive strategy that:
- Bids/asks at fixed 10% margin from token value
- Only accepts trades if holding the current bid/ask
- No time pressure, no history, no re-pricing
"""

import math
from typing import Any

from traders.base import Agent


class Gamer(Agent):
    """
    Gamer trader - fixed profit margin, only accepts when holding quote.

    Strategy:
    - Buyers: Bid at floor(valuation * 0.9) - 10% below value
    - Sellers: Ask at ceil(cost * 1.1) - 10% above cost
    - Only submits quote if it would improve the current best
    - Accepts trades ONLY if holding the current bid/ask (not just any profitable trade)

    This is a level-1, nonadaptive strategy from the 1993 Santa Fe Tournament.
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
        seed: int | None = None,
        **kwargs: Any,
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
        self.nobidask = 0
        self.nobuysell = 0

    def bid_ask(self, time: int, nobidask: int) -> None:
        """Prepare for bid/ask phase."""
        self.nobidask = nobidask
        self.has_responded = False

    def bid_ask_response(self) -> int:
        """
        Return bid/ask at fixed margin from valuation.

        Buyer: floor(valuation * 0.9) - 10% below value
        Seller: ceil(cost * 1.1) - 10% above cost
        Only submits if it would improve current best quote.
        """
        self.has_responded = True

        # Check nobidask flag - no more tokens to commit
        if self.nobidask > 0:
            return 0

        if self.num_trades >= self.num_tokens:
            return 0

        valuation = self.valuations[self.num_trades]

        if self.is_buyer:
            # Buyer: floor(0.9 * v) - int() truncates = floor for positive
            target = int(valuation * (1 - self.margin))

            # Clamp to price range
            target = max(self.price_min, min(target, self.price_max))

            # Only submit if it would improve current bid
            if self.current_bid > 0 and target <= self.current_bid:
                return 0

            return target
        else:
            # Seller: ceil(1.1 * c)
            target = math.ceil(valuation * (1 + self.margin))

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
        super().bid_ask_result(
            status, num_trades, new_bids, new_asks, high_bid, high_bidder, low_ask, low_asker
        )
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
        self.nobuysell = nobuysell
        self.has_responded = False
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker

    def buy_sell_response(self) -> bool:
        """
        Accept trade only if:
        1. We hold the current bid/ask (standing quote holder)
        2. The spread is crossed (cbid >= cask)
        3. The trade is profitable
        """
        self.has_responded = True

        # Check nobuysell flag - no more tokens to trade
        if self.nobuysell > 0:
            return False

        if self.num_trades >= self.num_tokens:
            return False

        valuation = self.valuations[self.num_trades]

        if self.is_buyer:
            # Must have a valid ask
            if self.current_ask == 0:
                return False
            # Never buy at or above valuation (loss check)
            if valuation <= self.current_ask:
                return False
            # Accept only if we are the standing bidder AND spread is crossed
            if self.player_id == self.current_bidder and self.current_bid >= self.current_ask:
                return True
            return False
        else:
            # Must have a valid bid
            if self.current_bid == 0:
                return False
            # Never sell at or below cost (loss check)
            if self.current_bid <= valuation:
                return False
            # Accept only if we are the standing asker AND spread is crossed
            if self.player_id == self.current_asker and self.current_ask <= self.current_bid:
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
        super().buy_sell_result(
            status, trade_price, trade_type, high_bid, high_bidder, low_ask, low_asker
        )
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker
