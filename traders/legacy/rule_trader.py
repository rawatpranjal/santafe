"""
RuleTrader Agent - Simplified Genetic Programming.

Instead of evolving rules via GP, this uses pre-defined heuristic rules
that represent strategies evolution might discover.
"""

from typing import Any, Optional
from traders.base import Agent


class RuleTrader(Agent):
    """
    Simplified GP - rule-based trader with pre-defined heuristics.

    Strategy:
    - Multiple behavioral rules that select based on market state
    - Aggression parameter controls how aggressive vs conservative
    - Rules mimic what genetic programming might evolve

    Rules:
    1. If no trades yet in period, be conservative
    2. If have reference price, bid/ask relative to it
    3. Default fallback: simple markup
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        price_min: int = 0,
        price_max: int = 100,
        aggression: float = 0.5,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize RuleTrader agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations
            price_min: Minimum allowed price
            price_max: Maximum allowed price
            aggression: 0.0=passive, 1.0=aggressive (default 0.5)
            seed: Ignored (kept for interface consistency)
            **kwargs: Ignored extra arguments
        """
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min = price_min
        self.price_max = price_max
        self.aggression = max(0.0, min(1.0, aggression))  # Clamp to [0,1]

        # Tracking state
        self.last_trade_price: Optional[int] = None
        self.trades_this_period = 0

        # State tracking for buy/sell phase
        self.current_bid = 0
        self.current_ask = 0
        self.current_bidder = 0
        self.current_asker = 0

    def start_period(self, period_number: int) -> None:
        """Reset per-period tracking."""
        super().start_period(period_number)
        self.trades_this_period = 0
        # Keep last_trade_price for cross-period reference

    def bid_ask(self, time: int, nobidask: int) -> None:
        """Prepare for bid/ask phase."""
        self.has_responded = False

    def bid_ask_response(self) -> int:
        """
        Return a bid/ask based on rule selection.

        Rules are applied in priority order:
        1. Conservative start if no reference
        2. Relative to last trade price
        3. Simple markup fallback
        """
        self.has_responded = True

        if self.num_trades >= self.num_tokens:
            return 0

        valuation = self.valuations[self.num_trades]

        # Rule 1: If no trades yet and no reference, be conservative
        if self.trades_this_period == 0 and self.last_trade_price is None:
            if self.is_buyer:
                # Conservative: bid at 60-80% of valuation based on aggression
                factor = 0.6 + 0.2 * self.aggression
                return max(self.price_min, int(valuation * factor))
            else:
                # Conservative: ask at 120-140% of cost based on aggression
                factor = 1.4 - 0.2 * self.aggression
                return min(self.price_max, int(valuation * factor))

        # Rule 2: If have reference price, bid relative to it
        if self.last_trade_price is not None:
            if self.is_buyer:
                # Bid near last trade, offset by aggression
                # Passive: bid 5 below last, Aggressive: bid 5 above last
                offset = -5 + 10 * self.aggression
                target = self.last_trade_price + offset
                bid = min(int(target), valuation - 1)
                return max(self.price_min, bid)
            else:
                # Ask near last trade, offset by aggression
                # Passive: ask 5 above last, Aggressive: ask 5 below last
                offset = 5 - 10 * self.aggression
                target = self.last_trade_price + offset
                ask = max(int(target), valuation + 1)
                return min(self.price_max, ask)

        # Rule 3: Default fallback - simple markup
        if self.is_buyer:
            return max(self.price_min, int(valuation * 0.85))
        else:
            return min(self.price_max, int(valuation * 1.15))

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
        """Accept trade if profitable and we are the winner."""
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
        """Track trades for rule selection."""
        super().buy_sell_result(status, trade_price, trade_type, high_bid,
                                high_bidder, low_ask, low_asker)
        # Record trade price if a trade occurred
        if trade_type != 0 and trade_price > 0:
            self.last_trade_price = trade_price
            self.trades_this_period += 1
