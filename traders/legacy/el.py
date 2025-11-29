"""
Easley-Ledyard (EL) trading agent.

Based on:
Easley, D., & Ledyard, J. (1992). "Theories of Price Formation and Exchange
in Double Oral Auctions" in The Double Auction Market: Institutions,
Theories, and Evidence, edited by Friedman and Rust, Addison-Wesley.

The EL agent uses reservation prices derived from previous period's price bounds.
Key behavioral rules (Assumptions 1, 1', 2, 2' from the paper):

1. Inframarginal traders (v > P̄ for buyers, c < P for sellers):
   - Early in period: use conservative reservation prices based on observed bounds
   - Late in period: become more aggressive, approach true valuation/cost

2. Marginal traders (v ≤ P̄ for buyers, c ≥ P for sellers):
   - Truth-tell: reservation price equals true valuation/cost

The theory shows this behavior leads to convergence to competitive equilibrium.
"""

from typing import Any, Optional
import numpy as np
from traders.base import Agent


class EL(Agent):
    """
    Easley-Ledyard trader using reservation price theory.

    Strategy:
    - Track previous period's price bounds [P_min, P_max]
    - Classify self as "inframarginal" (outside bounds) or "marginal" (at bounds)
    - Inframarginal: interpolate between bounds and true value based on time
    - Marginal: truth-tell

    The key insight is that traders learn from price history:
    - If buyer's value > max observed price → can afford to wait early, aggressive late
    - If buyer's value ≤ max observed price → must bid true value (marginal)
    - Mirror logic for sellers

    This creates the observed convergence pattern in double auctions.
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
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize EL agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations/costs
            price_min: Minimum allowed price
            price_max: Maximum allowed price
            num_times: Number of time steps per period
            seed: Random seed
            **kwargs: Ignored extra arguments
        """
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min = price_min
        self.price_max = price_max
        self.num_times = num_times
        self.rng = np.random.default_rng(seed)

        # Price bounds from previous period
        # P_d = min traded price, P̄_d = max traded price
        self.prev_price_low: float = (price_min + price_max) / 2  # Start with midpoint
        self.prev_price_high: float = (price_min + price_max) / 2

        # Current period's observed prices (to update bounds)
        self.current_traded_prices: list[int] = []

        # Current time step (for time-based strategy adjustment)
        self.current_time = 0

        # Current market state
        self.current_high_bid = 0
        self.current_low_ask = 0

        # Small randomization to prevent perfect ties
        self.noise_scale = 0.5

    def _get_reservation_price(self) -> float:
        """
        Calculate reservation price based on EL theory.

        For buyers:
        - If v > P̄ (inframarginal): r(t) = P̄ early, approaches v late
        - If v ≤ P̄ (marginal): r(t) = v

        For sellers:
        - If c < P (inframarginal): s(t) = P early, approaches c late
        - If c ≥ P (marginal): s(t) = c

        Returns:
            Reservation price for current time step
        """
        if self.num_trades >= self.num_tokens:
            return 0.0

        true_value = self.valuations[self.num_trades]

        # Time progression factor: 0 at start, 1 at end
        time_progress = self.current_time / max(1, self.num_times)

        if self.is_buyer:
            # Buyer reservation price
            if true_value > self.prev_price_high:
                # Inframarginal buyer: interpolate from P̄ to v
                # Early: bid at/below P̄ (conservative)
                # Late: bid up to v (aggressive)
                reservation = (1 - time_progress) * self.prev_price_high + time_progress * true_value
            else:
                # Marginal buyer: truth-tell
                reservation = true_value
        else:
            # Seller reservation price
            if true_value < self.prev_price_low:
                # Inframarginal seller: interpolate from P to c
                # Early: ask at/above P (conservative)
                # Late: ask down to c (aggressive)
                reservation = (1 - time_progress) * self.prev_price_low + time_progress * true_value
            else:
                # Marginal seller: truth-tell
                reservation = true_value

        # Add small noise to prevent ties
        noise = self.rng.normal(0, self.noise_scale)
        reservation += noise

        return reservation

    def bid_ask(self, time: int, nobidask: int) -> None:
        """Notification: Time to submit bid/ask."""
        self.has_responded = False
        self.current_time = time

    def bid_ask_response(self) -> int:
        """
        Return bid or ask based on reservation price.

        Buyers: Bid at reservation price (capped by improving current bid)
        Sellers: Ask at reservation price (must improve current ask)
        """
        self.has_responded = True

        if self.num_trades >= self.num_tokens:
            return 0

        reservation = self._get_reservation_price()
        true_value = self.valuations[self.num_trades]

        if self.is_buyer:
            # Bid at reservation price, but must improve current high bid
            bid = int(reservation)

            # Constraint: must improve high bid (or match if no bid exists)
            if self.current_high_bid > 0 and bid <= self.current_high_bid:
                bid = self.current_high_bid + 1

            # Constraint: cannot exceed valuation (would lose money)
            if bid > true_value:
                bid = true_value

            # Constraint: cannot exceed valuation even after adjustments
            if bid > true_value:
                return 0  # Cannot make a profitable bid

            # Enforce price bounds
            bid = max(self.price_min, min(bid, self.price_max))

            return bid
        else:
            # Ask at reservation price, but must improve current low ask
            ask = int(reservation)

            # Constraint: must improve low ask (or match if no ask exists)
            if self.current_low_ask > 0 and ask >= self.current_low_ask:
                ask = self.current_low_ask - 1

            # Constraint: cannot go below cost (would lose money)
            if ask < true_value:
                ask = true_value

            # If we can't improve and can't profit, pass
            if self.current_low_ask > 0 and ask >= self.current_low_ask:
                return 0

            # Enforce price bounds
            ask = max(self.price_min, min(ask, self.price_max))

            return ask

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
        """Process bid/ask results and update market state."""
        super().bid_ask_result(status, num_trades, new_bids, new_asks,
                               high_bid, high_bidder, low_ask, low_asker)
        self.current_high_bid = high_bid
        self.current_low_ask = low_ask

    def buy_sell(
        self,
        time: int,
        nobuysell: int,
        high_bid: int,
        low_ask: int,
        high_bidder: int,
        low_asker: int,
    ) -> None:
        """Notification: Time for buy/sell decision."""
        self.has_responded = False
        self.current_high_bid = high_bid
        self.current_low_ask = low_ask

    def buy_sell_response(self) -> bool:
        """
        Decide whether to accept current offer.

        Accept if the current market price is within our reservation price:
        - Buyers: Accept low_ask if it's below our reservation price
        - Sellers: Accept high_bid if it's above our reservation price
        """
        self.has_responded = True

        if self.num_trades >= self.num_tokens:
            return False

        reservation = self._get_reservation_price()
        true_value = self.valuations[self.num_trades]

        if self.is_buyer:
            # Accept if low_ask is at or below reservation price AND profitable
            if self.current_low_ask > 0 and self.current_low_ask <= reservation:
                if self.current_low_ask < true_value:  # Profitable
                    return True
            return False
        else:
            # Accept if high_bid is at or above reservation price AND profitable
            if self.current_high_bid > 0 and self.current_high_bid >= reservation:
                if self.current_high_bid > true_value:  # Profitable
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
        """Process trade results and track prices for bound updates."""
        super().buy_sell_result(status, trade_price, trade_type,
                                high_bid, high_bidder, low_ask, low_asker)

        # Track traded prices for end-of-period bound update
        if trade_type != 0 and trade_price > 0:
            self.current_traded_prices.append(trade_price)

        self.current_high_bid = high_bid
        self.current_low_ask = low_ask

    def start_period(self, period_number: int) -> None:
        """
        Start new period.

        Updates price bounds from previous period's trades.
        Per EL theory, [P_d, P̄_d] are the bounds from the previous day (period).
        """
        super().start_period(period_number)

        # Update bounds from previous period's trades
        if self.current_traded_prices:
            self.prev_price_low = min(self.current_traded_prices)
            self.prev_price_high = max(self.current_traded_prices)
        # If no trades occurred, keep previous bounds (or initial midpoint)

        # Reset for new period
        self.current_traded_prices = []
        self.current_high_bid = 0
        self.current_low_ask = 0

    def start_round(self, valuations: list[int]) -> None:
        """
        Start new round (new equilibrium).

        Resets price bounds to midpoint since we have no history for new equilibrium.
        """
        super().start_round(valuations)

        # Reset to uninformed prior
        self.prev_price_low = (self.price_min + self.price_max) / 2
        self.prev_price_high = (self.price_min + self.price_max) / 2
        self.current_traded_prices = []
        self.current_high_bid = 0
        self.current_low_ask = 0
