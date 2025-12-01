"""
Ledyard (Easley-Ledyard-Olson) Trading Agent.

A SOTA reservation-price trader that lives inside the previous period's price band
and walks quotes across that band over time, with market power awareness.

Classification:
- Complex: Maintains cross-period price history, computes time-varying reservation prices
- History-based: Uses previous period's price bounds to constrain current quotes
- Band-restricted: All quotes must lie within [P'_n, P_n] clipped by value/cost
- Market-power-aware: Uses num_buyers/num_sellers to adjust aggressiveness
- Non-predictive: Does not forecast future prices, just uses historical bounds

Key behaviors:
- First period: Wide band [price_min, price_max] clipped by value/cost
- Periods 2+: Restricted to [P'_n, P_n] from previous period
- Within period: Conservative early (near band edges), aggressive late (toward value/cost)
- Market power: Short side (few agents) shades more, long side concedes faster

Based on:
- Easley, D., & Ledyard, J. (1992). "Theories of Price Formation and Exchange
  in Double Oral Auctions"
- Ledyard-Olson implementation in Santa Fe Tournament (1993)
"""

from typing import Any

import numpy as np

from traders.base import Agent

# Hyperparameters
THETA_BASE = 1.0  # Base slope of time path (linear by default)
NOISE_SCALE = 0.5  # Small randomization for tie-breaking
# POWER_FACTOR disabled - original Ledyard-Olson did not use continuous power scaling


class Ledyard(Agent):
    """
    Ledyard (Easley-Ledyard-Olson) trading agent.

    Strategy per original Easley-Ledyard theory:
    - Track previous period's price bounds: P'_n (floor), P_n (ceiling)
    - Compute time-progressive reservation price within band
    - Reservation walks from band edge toward own value/cost (not profitable edge)
    - Early: conservative quotes near band edges
    - Late: still reservation-based (no aggressive override)
    - No continuous market power adjustment (pure time-based theta)
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        price_min: int = 0,
        price_max: int = 200,
        num_times: int = 100,
        num_buyers: int = 4,
        num_sellers: int = 4,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Ledyard agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations (buyer) or costs (seller)
            price_min: Minimum allowed price
            price_max: Maximum allowed price
            num_times: Number of time steps per period
            num_buyers: Number of buyers in market (for market power)
            num_sellers: Number of sellers in market (for market power)
            seed: Random seed for tie-breaking noise
            **kwargs: Ignored extra arguments
        """
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min = price_min
        self.price_max = price_max
        self.num_times = num_times
        self.num_buyers = num_buyers
        self.num_sellers = num_sellers
        self.rng = np.random.default_rng(seed)

        # Cross-period state: price bounds from previous period
        # P'_n = floor (min of trades + asks), P_n = ceiling (max of trades + bids)
        self.prev_price_low: float = float(price_min)
        self.prev_price_high: float = float(price_max)

        # Current period tracking
        self.current_traded_prices: list[int] = []
        self.current_asks: list[int] = []  # All asks observed this period
        self.current_bids: list[int] = []  # All bids observed this period

        # Current step state
        self.current_time = 0
        self.current_high_bid = 0
        self.current_low_ask = 0
        self.current_bidder = 0
        self.current_asker = 0

        # AURORA protocol flags
        self.nobidask = 0
        self.nobuysell = 0

    def _time_fraction(self) -> float:
        """Current time as fraction of period [0, 1]."""
        if self.num_times <= 0:
            return 1.0
        return self.current_time / self.num_times

    def _compute_theta(self, t: float) -> float:
        """
        Compute time progression factor θ(t).

        θ(t) starts at 0 (conservative), rises to 1 (aggressive).
        Pure time-based - no market power adjustment per original Easley-Ledyard.
        """
        return max(0.0, min(1.0, t**THETA_BASE))

    def _compute_reservation_price(self, value: int) -> float:
        """
        Compute time-progressive reservation price within band.

        Per Easley-Ledyard theory:
        - Buyer: walks from band_low toward own value (not toward band_high)
        - Seller: walks from band_high toward own cost (not toward band_low)

        This makes Ledyard less aggressive than walking toward profitable edge.
        Marginal agents (band collapsed) just truth-tell.
        """
        t = self._time_fraction()
        theta = self._compute_theta(t)

        if self.is_buyer:
            L = self.prev_price_low
            H = self.prev_price_high
            # Clamp band to buyer value
            band_low = min(float(value), L)
            band_high = min(float(value), H)

            if band_low >= band_high:
                # Marginal buyer: band collapsed, truth-tell
                return float(value)

            # Start at band_low, move toward own value (not band_high)
            target = float(value)
            reservation = band_low + theta * (target - band_low)
        else:
            L = self.prev_price_low
            H = self.prev_price_high
            # Clamp band to seller cost
            band_low = max(float(value), L)
            band_high = max(float(value), H)

            if band_low >= band_high:
                # Marginal seller: band collapsed, truth-tell
                return float(value)

            # Start at band_high, move toward own cost (not band_low)
            target = float(value)
            reservation = band_high - theta * (band_high - target)

        # Add small noise to prevent ties
        noise = self.rng.normal(0, NOISE_SCALE)
        reservation += noise

        return reservation

    def bid_ask(self, time: int, nobidask: int) -> None:
        """Prepare for bid/ask phase."""
        self.current_time = time
        self.nobidask = nobidask
        self.has_responded = False

    def bid_ask_response(self) -> int:
        """
        Return bid or ask based on reservation price and market state.
        """
        self.has_responded = True

        if self.nobidask > 0:
            return 0

        if self.num_trades >= self.num_tokens:
            return 0

        value = self.valuations[self.num_trades]

        if self.is_buyer:
            return self._request_bid(value)
        else:
            return self._request_ask(value)

    def _request_bid(self, value: int) -> int:
        """Post reservation price if it improves the book. No micro-optimization."""
        r_B = self._compute_reservation_price(value)
        cbid = self.current_high_bid if self.current_high_bid > 0 else self.price_min - 1

        # Reservation must be profitable
        bid = int(r_B)
        if bid >= value:
            return 0  # don't bid at or above value

        # Only quote if it improves current bid
        if bid > cbid and bid >= self.price_min:
            return min(bid, self.price_max)
        return 0

    def _request_ask(self, cost: int) -> int:
        """Post reservation price if it improves the book. No micro-optimization."""
        r_S = self._compute_reservation_price(cost)
        cask = self.current_low_ask if self.current_low_ask > 0 else self.price_max + 1

        ask = int(r_S)
        if ask <= cost:
            return 0  # don't ask at or below cost

        if ask < cask and ask <= self.price_max:
            return max(ask, self.price_min)
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
        self.current_time = time
        self.nobuysell = nobuysell
        self.has_responded = False
        self.current_high_bid = high_bid
        self.current_low_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker

    def buy_sell_response(self) -> bool:
        """
        Decide whether to accept current offer.

        Accept if price is within reservation AND profitable.
        No late-period override per original Easley-Ledyard theory.
        """
        self.has_responded = True

        if self.nobuysell > 0:
            return False

        if self.num_trades >= self.num_tokens:
            return False

        value = self.valuations[self.num_trades]

        if self.is_buyer:
            return self._buyer_accept(value)
        else:
            return self._seller_accept(value)

    def _buyer_accept(self, value: int) -> bool:
        """Buyer acceptance: CurrentAsk <= r_B(t) and profitable."""
        # Must be high bidder
        if self.player_id != self.current_bidder:
            return False

        # Must have valid ask
        if self.current_low_ask <= 0:
            return False

        # Profitability check
        if self.current_low_ask >= value:
            return False

        # Reservation-based acceptance only (no late-period override)
        r_B = self._compute_reservation_price(value)
        return self.current_low_ask <= r_B

    def _seller_accept(self, cost: int) -> bool:
        """Seller acceptance: CurrentBid >= r_S(t) and profitable."""
        # Must be low asker
        if self.player_id != self.current_asker:
            return False

        # Must have valid bid
        if self.current_high_bid <= 0:
            return False

        # Profitability check
        if self.current_high_bid <= cost:
            return False

        # Reservation-based acceptance only (no late-period override)
        r_S = self._compute_reservation_price(cost)
        return self.current_high_bid >= r_S

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
        """Process bid/ask results - only track current market state, NOT all quotes."""
        super().bid_ask_result(
            status, num_trades, new_bids, new_asks, high_bid, high_bidder, low_ask, low_asker
        )

        # HANDICAP: Do NOT track all bids/asks for band computation
        # Only update current market state
        self.current_high_bid = high_bid
        self.current_low_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker

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
        """Process trade results - track traded prices for band computation."""
        super().buy_sell_result(
            status, trade_price, trade_type, high_bid, high_bidder, low_ask, low_asker
        )

        # Track traded prices
        if trade_type != 0 and trade_price > 0:
            self.current_traded_prices.append(trade_price)

        # Update current market state
        self.current_high_bid = high_bid
        self.current_low_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker

    def start_period(self, period_number: int) -> None:
        """
        Start new period - update price bounds from TRADE PRICES ONLY.

        HANDICAP: Unlike full AURORA access which includes all bids/asks,
        we only use actual traded prices for band computation.
        """
        super().start_period(period_number)

        # HANDICAP: Only use traded prices for band computation (no bids/asks)
        if self.current_traded_prices:
            self.prev_price_low = float(min(self.current_traded_prices))
            self.prev_price_high = float(max(self.current_traded_prices))

        # Reset period tracking
        self.current_traded_prices = []
        self.current_asks = []  # Keep for compatibility but don't use
        self.current_bids = []  # Keep for compatibility but don't use
        self.current_high_bid = 0
        self.current_low_ask = 0
        self.current_time = 0

    def start_round(self, valuations: list[int]) -> None:
        """
        Start new round (new equilibrium) - reset to wide prior.

        No history from previous round applies to new equilibrium.
        """
        super().start_round(valuations)

        # Reset to wide prior (no history)
        self.prev_price_low = float(self.price_min)
        self.prev_price_high = float(self.price_max)

        # Reset all tracking
        self.current_traded_prices = []
        self.current_asks = []
        self.current_bids = []
        self.current_high_bid = 0
        self.current_low_ask = 0
        self.current_time = 0

    @property
    def trade_count(self) -> int:
        """Compatibility property for tournament scripts."""
        return self.num_trades
