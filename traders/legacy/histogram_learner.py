"""
HistogramLearner Agent - Simplified Empirical Bayesian.

Based on Easley-Ledyard concept but simplified:
- Tracks recent transaction prices in a running window
- Uses mean and std dev to set bid/ask prices
- No full histogram bins or market structure awareness
"""

from typing import Any, Optional
import numpy as np
from traders.base import Agent


class HistogramLearner(Agent):
    """
    Simplified Empirical Bayesian - learns from price history.

    Strategy:
    - Track recent transaction prices in a sliding window
    - Buyers: Bid below mean price (capturing buyer-favorable trades)
    - Sellers: Ask above mean price (capturing seller-favorable trades)
    - Falls back to markup strategy when insufficient data

    This captures the core insight that observing market prices
    helps predict where future trades will occur.
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        price_min: int = 0,
        price_max: int = 100,
        window_size: int = 50,
        margin_factor: float = 0.5,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize HistogramLearner agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations
            price_min: Minimum allowed price
            price_max: Maximum allowed price
            window_size: Number of recent prices to track
            margin_factor: How many std devs from mean to bid/ask
            seed: Ignored (kept for interface consistency)
            **kwargs: Ignored extra arguments
        """
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min = price_min
        self.price_max = price_max
        self.window_size = window_size
        self.margin_factor = margin_factor

        # Price history (persists across periods for learning)
        self.price_history: list[int] = []

        # State tracking for buy/sell phase
        self.current_bid = 0
        self.current_ask = 0
        self.current_bidder = 0
        self.current_asker = 0

    def _update_history(self, trade_price: int) -> None:
        """Add a trade price to history, maintaining window size."""
        self.price_history.append(trade_price)
        if len(self.price_history) > self.window_size:
            self.price_history.pop(0)

    def bid_ask(self, time: int, nobidask: int) -> None:
        """Prepare for bid/ask phase."""
        self.has_responded = False

    def bid_ask_response(self) -> int:
        """
        Return a bid/ask based on learned price distribution.

        Uses mean and std dev of recent trades to set price.
        Falls back to simple markup when insufficient data.
        """
        self.has_responded = True

        if self.num_trades >= self.num_tokens:
            return 0

        valuation = self.valuations[self.num_trades]

        # Not enough data - use simple markup fallback
        if len(self.price_history) < 5:
            if self.is_buyer:
                return max(self.price_min, int(valuation * 0.8))
            else:
                return min(self.price_max, int(valuation * 1.2))

        # Calculate statistics from price history
        mean_price = np.mean(self.price_history)
        std_price = max(1, np.std(self.price_history))

        if self.is_buyer:
            # Bid below mean, capped by valuation
            target = mean_price - self.margin_factor * std_price
            bid = min(int(target), valuation - 1)
            return max(self.price_min, bid)
        else:
            # Ask above mean, floored by cost
            target = mean_price + self.margin_factor * std_price
            ask = max(int(target), valuation + 1)
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
        """Track trade prices for learning."""
        super().buy_sell_result(status, trade_price, trade_type, high_bid,
                                high_bidder, low_ask, low_asker)
        # Record trade price if a trade occurred
        if trade_type != 0 and trade_price > 0:
            self._update_history(trade_price)
