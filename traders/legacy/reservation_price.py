"""
ReservationPrice Agent - Simplified BGAN (Bayesian Game Against Nature).

Based on Friedman (1991) concept but simplified:
- Uses time-decaying reservation price instead of full Bayesian updating
- Starts conservative, becomes aggressive as time runs out
"""

from typing import Any, Optional
from traders.base import Agent


class ReservationPrice(Agent):
    """
    Simplified BGAN - time-decaying reservation price.

    Strategy:
    - Buyers: Start bidding at 50% of valuation, approach 95% as time runs out
    - Sellers: Start asking at 150% of cost, approach 105% as time runs out
    - Trade acceptance: Accept if profitable

    This captures BGAN's core insight that traders should become more
    aggressive as trading opportunities diminish, without the full
    Bayesian machinery.
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        price_min: int = 0,
        price_max: int = 100,
        num_times: int = 100,
        urgency_rate: float = 0.02,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize ReservationPrice agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations
            price_min: Minimum allowed price
            price_max: Maximum allowed price
            num_times: Total time steps per period
            urgency_rate: Rate at which reservation price becomes aggressive
            seed: Ignored (kept for interface consistency)
            **kwargs: Ignored extra arguments
        """
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min = price_min
        self.price_max = price_max
        self.num_times = num_times
        self.urgency_rate = urgency_rate
        self.current_time = 0

        # State tracking for buy/sell phase
        self.current_bid = 0
        self.current_ask = 0
        self.current_bidder = 0
        self.current_asker = 0

    def bid_ask(self, time: int, nobidask: int) -> None:
        """Prepare for bid/ask phase - track current time."""
        self.has_responded = False
        self.current_time = time

    def bid_ask_response(self) -> int:
        """
        Return a bid/ask with time-decaying reservation price.

        Early: Conservative (far from valuation/cost)
        Late: Aggressive (close to valuation/cost)
        """
        self.has_responded = True

        if self.num_trades >= self.num_tokens:
            return 0

        valuation = self.valuations[self.num_trades]

        # Urgency factor: 0 at start, approaches 1 at end
        urgency = min(1.0, self.current_time * self.urgency_rate)

        if self.is_buyer:
            # Start at 50% of valuation, approach 95% as time runs out
            reserve = valuation * (0.5 + 0.45 * urgency)
            return max(self.price_min, int(reserve))
        else:
            # Start at 150% of cost, approach 105% as time runs out
            reserve = valuation * (1.5 - 0.45 * urgency)
            return min(self.price_max, int(reserve))

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
