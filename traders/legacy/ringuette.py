"""
Ringuette Agent.

Based on Chen et al. (2010) description:
"The Ringuette strategy was designed by computer scientist Marc Ringuette.
It is also a background trader, whose strategy is to wait until the first
time when the current bid exceeds the current ask less a profit margin.
The Ringuette strategy is a simple rule of thumb, and it won the second
place in the SFDA tournament."

Key behavioral characteristics:
1. Background trader: Stays silent most of the time
2. Waits for profitable spread: bid > ask - profit_margin
3. When spread is favorable, jumps in to capture it
4. Simpler than Kaplan - no complex timing or history tracking
"""

from typing import Any, Optional
from traders.base import Agent


class Ringuette(Agent):
    """
    Ringuette trading agent (Background Trader).

    Strategy:
    - Wait until bid > ask - profit_margin (favorable spread)
    - When spread is favorable, submit a competitive bid/ask
    - Accept trades when profitable
    - 2nd place in original SFDA tournament (similar to Kaplan)
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        price_min: int = 0,
        price_max: int = 100,
        profit_margin: int = 5,
        deadlock_threshold: int = 10,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize Ringuette agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations
            price_min: Minimum allowed price
            price_max: Maximum allowed price
            profit_margin: Minimum spread to trigger jump-in (default 5)
            deadlock_threshold: Steps without trade before fallback to Skeleton behavior (default 10)
            seed: Ignored (kept for interface consistency)
            **kwargs: Ignored extra arguments
        """
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min = price_min
        self.price_max = price_max
        self.profit_margin = profit_margin
        self.deadlock_threshold = deadlock_threshold

        # State tracking
        self.current_bid = 0
        self.current_ask = 0
        self.current_bidder = 0
        self.current_asker = 0
        self.nobidask = 0
        self.steps_without_trade = 0

    def bid_ask(self, time: int, nobidask: int) -> None:
        """Prepare for bid/ask phase."""
        self.has_responded = False
        self.nobidask = nobidask

    def bid_ask_response(self) -> int:
        """
        Return a bid/ask only when spread is favorable.

        Core logic from Chen description:
        - Wait until bid > ask - profit_margin
        - If favorable, submit competitive price
        - Otherwise, stay silent (return 0)
        """
        self.has_responded = True

        if self.num_trades >= self.num_tokens:
            return 0

        valuation = self.valuations[self.num_trades]

        # Check if spread is favorable (bid close to or above ask)
        spread_favorable = False
        deadlock_active = self.steps_without_trade >= self.deadlock_threshold

        if self.current_bid > 0 and self.current_ask > 0:
            spread_favorable = self.current_bid >= (self.current_ask - self.profit_margin)

        if self.is_buyer:
            if spread_favorable and self.current_ask > 0:
                # Spread is tight - jump in with competitive bid
                # Bid at the ask price (to guarantee execution) but cap at valuation-1
                bid = min(self.current_ask, valuation - 1)
                return max(self.price_min, bid)
            elif deadlock_active:
                # DEADLOCK BREAKER: Fall back to Skeleton-like behavior
                # Submit a conservative bid to stimulate market
                bid = self.price_min + (valuation - self.price_min) // 2
                if self.current_bid > 0:
                    bid = max(self.current_bid + 1, bid)
                return max(self.price_min, min(bid, valuation - 1))
            else:
                # Stay silent - no bid
                return 0
        else:
            if spread_favorable and self.current_bid > 0:
                # Spread is tight - jump in with competitive ask
                # Ask at the bid price (to guarantee execution) but floor at cost+1
                ask = max(self.current_bid, valuation + 1)
                return min(self.price_max, ask)
            elif deadlock_active:
                # DEADLOCK BREAKER: Fall back to Skeleton-like behavior
                # Submit a conservative ask to stimulate market
                ask = valuation + (self.price_max - valuation) // 2
                if self.current_ask > 0:
                    ask = min(self.current_ask - 1, ask)
                return min(self.price_max, max(ask, valuation + 1))
            else:
                # Stay silent - no ask
                return 0

    def buy_sell(
        self,
        time: int,
        nobuysell: int,
        high_bid: int,
        low_ask: int,
        high_bidder: int,
        low_asker: int,
    ) -> None:
        """Prepare for buy/sell decision - update market state."""
        self.has_responded = False
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker

    def buy_sell_response(self) -> bool:
        """
        Accept trade if profitable and we are the winner.

        Ringuette accepts any profitable trade when selected.
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
        """Update market state after trade result."""
        super().buy_sell_result(status, trade_price, trade_type, high_bid,
                                high_bidder, low_ask, low_asker)
        # Update current bid/ask for next round
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker

        # Track deadlock - reset counter if trade occurred, increment otherwise
        if trade_price > 0:
            self.steps_without_trade = 0
        else:
            self.steps_without_trade += 1
