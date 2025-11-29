"""
Breton Agent - Stochastic weighted average strategy.

From the 1993 Santa Fe Tournament.
Calculates a weighted average between market state and private value,
then adds random noise to prevent deadlock.
"""

from typing import Any, Optional
import numpy as np
from traders.base import Agent


class Breton(Agent):
    """
    Breton trader - weighted average with noise injection.

    Strategy:
    - Calculate target as weighted average of current price and private value
    - Add random noise (shake) to prevent predictability and deadlock
    - Trade acceptance: Accept if profitable

    The noise injection is key - it prevents the strategy from getting
    stuck in equilibrium deadlocks where no one wants to move first.
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        price_min: int = 0,
        price_max: int = 100,
        weight: float = 0.5,
        noise_range: int = 2,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize Breton agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations
            price_min: Minimum allowed price (default 0)
            price_max: Maximum allowed price (default 100)
            weight: Weight for market price vs private value (default 0.5)
            noise_range: Range for random noise [-n, +n] (default 2)
            seed: Random seed for reproducibility
            **kwargs: Ignored extra arguments
        """
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min = price_min
        self.price_max = price_max
        self.weight = weight
        self.noise_range = noise_range
        self.rng = np.random.default_rng(seed)

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
        Return bid/ask using weighted average with noise.

        Target = weight * market_price + (1-weight) * private_value + noise
        """
        self.has_responded = True

        if self.num_trades >= self.num_tokens:
            return 0

        valuation = self.valuations[self.num_trades]

        # Generate noise
        noise = self.rng.integers(-self.noise_range, self.noise_range + 1)

        if self.is_buyer:
            # Get market reference price (current bid or min_price)
            market_price = self.current_bid if self.current_bid > 0 else self.price_min

            # Weighted average
            target = self.weight * market_price + (1 - self.weight) * valuation

            # Add noise
            proposed = int(target + noise)

            # Validity checks
            # Must improve current bid
            if self.current_bid > 0 and proposed <= self.current_bid:
                return 0

            # Must be below valuation (profitable)
            if proposed >= valuation:
                return 0

            # Clamp to price range
            proposed = max(self.price_min, min(proposed, self.price_max))

            return proposed
        else:
            # Get market reference price (current ask or max_price)
            market_price = self.current_ask if self.current_ask > 0 else self.price_max

            # Weighted average
            target = self.weight * market_price + (1 - self.weight) * valuation

            # Add noise
            proposed = int(target + noise)

            # Validity checks
            # Must improve current ask
            if self.current_ask > 0 and proposed >= self.current_ask:
                return 0

            # Must be above cost (profitable)
            if proposed <= valuation:
                return 0

            # Clamp to price range
            proposed = max(self.price_min, min(proposed, self.price_max))

            return proposed

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
