"""
BGAN (Bayesian Game Against Nature) Trader.

Implementation of the Kennet-Friedman entry from the 1993 Santa Fe tournament.
A sophisticated belief-based trader that:
1. Maintains Bayesian beliefs over opponent price distributions (Normal)
2. Computes reservation prices via Monte Carlo simulation
3. Uses aggressive reservation-price strategy for bidding/accepting

Classification:
- Complex: Maintains beliefs, computes optimal strategies
- Predictive: Uses Bayesian updating on price distributions
- Optimizing: Computes reservation prices to maximize expected profit
- Belief-Based: Normal parametric model for opponent prices
- Stochastic: Monte Carlo reservation price computation
"""

from typing import Any

import numpy as np

from traders.base import Agent

# Fixed hyperparameters
MC_SAMPLES = 100  # Monte Carlo samples for reservation price
DEFAULT_M0 = 10  # expected quotes per period
DEFAULT_SIGMA = 200  # initial std dev assumption
OUTLIER_SIGMA = 3.0  # ignore quotes > 3σ from mean


class BGAN(Agent):
    """
    BGAN (Bayesian Game Against Nature) trading agent.

    Strategy:
    - Maintains Normal beliefs over opponent prices (μ, σ)
    - Computes reservation price via Monte Carlo simulation
    - BA step: Seize market if can improve quote while staying within reservation
    - BS step: Accept if standing price beats reservation price
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        price_min: int = 1,
        price_max: int = 2000,
        num_times: int = 100,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize BGAN agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations (sorted descending for buyers, ascending for sellers)
            price_min: Minimum allowed price
            price_max: Maximum allowed price
            num_times: Total time steps in period (NTIMES)
            seed: Random seed for reproducibility
            **kwargs: Ignored extra arguments
        """
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min = price_min
        self.price_max = price_max
        self.num_times = num_times
        self.rng = np.random.default_rng(seed)

        # Belief state (Normal model)
        # For sellers: track bid distribution Normal(μ_b, σ_b)
        # For buyers: track ask distribution Normal(μ_a, σ_a)
        self.belief_mu: float = (price_min + price_max) / 2  # prior mean
        self.belief_sigma: float = DEFAULT_SIGMA  # prior std

        # Arrival model: m(t) = m0 * (1 - t)
        self.m0: int = DEFAULT_M0  # expected quotes per period

        # Observation history (for updating beliefs)
        self._observed_prices: list[int] = []  # prices from opposite side
        self._prev_period_prices: list[int] = []  # from previous period
        self._prev_period_count: int = DEFAULT_M0  # quote count from previous period

        # Current market state
        self.current_time = 0
        self.current_bid = 0
        self.current_ask = 0
        self.current_bidder = 0
        self.current_asker = 0
        self.nobidask = 0
        self._reservation_price: int | None = None

    def _time_fraction(self) -> float:
        """Current time as fraction of period [0, 1]."""
        if self.num_times <= 0:
            return 1.0
        return self.current_time / self.num_times

    def _expected_remaining_quotes(self) -> float:
        """
        Expected remaining quotes from opposite side.
        m(t) = m0 * (1 - t)
        """
        return max(0.0, self.m0 * (1.0 - self._time_fraction()))

    def _compute_reservation_price(self, valuation: int) -> int:
        """
        Compute reservation price via Monte Carlo simulation.

        For buyer: rb = value - E[(value - min_future_ask)^+]
        For seller: rs = cost + E[(max_future_bid - cost)^+]

        Uses current beliefs (μ, σ) and arrival model to sample future prices.
        """
        m_remaining = self._expected_remaining_quotes()

        if m_remaining < 0.5:
            # No more expected quotes - reservation = valuation
            return valuation

        # Draw number of remaining quotes from Poisson(m_remaining)
        n_expected = int(round(m_remaining))
        if n_expected < 1:
            n_expected = 1

        profits = []
        for _ in range(MC_SAMPLES):
            # Sample n future prices from Normal(μ, σ)
            n_draws = max(1, self.rng.poisson(n_expected))
            future_prices = self.rng.normal(self.belief_mu, self.belief_sigma, n_draws)

            if self.is_buyer:
                # Buyer: profit from min future ask
                min_ask = np.min(future_prices)
                profit = max(0, valuation - min_ask)
            else:
                # Seller: profit from max future bid
                max_bid = np.max(future_prices)
                profit = max(0, max_bid - valuation)

            profits.append(profit)

        expected_option_value = np.mean(profits)

        if self.is_buyer:
            # rb = value - E[(value - min_future_ask)^+]
            reservation = int(valuation - expected_option_value)
        else:
            # rs = cost + E[(max_future_bid - cost)^+]
            reservation = int(valuation + expected_option_value)

        # Clamp to valid range
        return max(self.price_min, min(self.price_max, reservation))

    def _update_beliefs(self, new_price: int) -> None:
        """
        Update beliefs with new observed price.

        Uses outlier filtering: ignore prices > 3σ from mean.
        Online mean update, σ fixed from previous period.
        """
        # Outlier filtering (PA4)
        if len(self._observed_prices) > 2:
            current_mean = np.mean(self._observed_prices)
            current_std = np.std(self._observed_prices)
            if current_std > 0:
                z_score = abs(new_price - current_mean) / current_std
                if z_score > OUTLIER_SIGMA:
                    return  # Ignore outlier

        self._observed_prices.append(new_price)

        # Update belief mean (online)
        self.belief_mu = np.mean(self._observed_prices)

    def bid_ask(self, time: int, nobidask: int) -> None:
        """Prepare for bid/ask phase."""
        self.current_time = time
        self.nobidask = nobidask
        self.has_responded = False
        self._reservation_price = None  # Reset for this step

    def bid_ask_response(self) -> int:
        """
        BA step: Seize market if can improve quote while staying within reservation price.
        """
        self.has_responded = True

        if self.nobidask > 0:
            return 0

        if self.num_trades >= self.num_tokens:
            return 0

        valuation = self.valuations[self.num_trades]
        reservation = self._compute_reservation_price(valuation)
        self._reservation_price = reservation

        if self.is_buyer:
            return self._request_bid(valuation, reservation)
        else:
            return self._request_ask(valuation, reservation)

    def _request_bid(self, valuation: int, reservation: int) -> int:
        """
        Buyer bidding logic.

        Seize market if can improve quote while staying within reservation price.
        - If no current bid: bid at reservation (aggressive entry)
        - If current bid exists: bid current_bid + 1 if still profitable
        """
        if self.current_bid == 0:
            # No current bid - enter market aggressively at reservation
            new_bid = min(reservation, valuation - 1)
            return max(self.price_min, new_bid)

        # Current bid exists - can we improve and still stay within reservation?
        new_bid = self.current_bid + 1

        if new_bid <= reservation and new_bid < valuation:
            return new_bid

        # Cannot profitably improve - stay silent
        return 0

    def _request_ask(self, valuation: int, reservation: int) -> int:
        """
        Seller asking logic.

        Seize market if can improve quote while staying within reservation price.
        - If no current ask: ask at reservation (aggressive entry)
        - If current ask exists: ask current_ask - 1 if still profitable
        """
        if self.current_ask == 0:
            # No current ask - enter market aggressively at reservation
            new_ask = max(reservation, valuation + 1)
            return min(self.price_max, new_ask)

        # Current ask exists - can we improve and still stay within reservation?
        new_ask = self.current_ask - 1

        if new_ask >= reservation and new_ask > valuation:
            return new_ask

        # Cannot profitably improve - stay silent
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
        self.has_responded = False
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker

    def buy_sell_response(self) -> bool:
        """
        BS step: Accept if standing price beats reservation price.
        """
        self.has_responded = True

        if self.num_trades >= self.num_tokens:
            return False

        valuation = self.valuations[self.num_trades]

        # Recompute reservation if not cached
        if self._reservation_price is None:
            self._reservation_price = self._compute_reservation_price(valuation)

        reservation = self._reservation_price

        if self.is_buyer:
            # Accept if we're high bidder and ask beats reservation
            if (
                self.player_id == self.current_bidder
                and self.current_ask > 0
                and self.current_ask <= reservation
                and valuation > self.current_ask
            ):
                return True
        else:
            # Accept if we're low asker and bid beats reservation
            if (
                self.player_id == self.current_asker
                and self.current_bid > 0
                and self.current_bid >= reservation
                and self.current_bid > valuation
            ):
                return True

        return False

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
        """
        Update beliefs from observed quotes.

        Buyers observe asks, sellers observe bids.
        """
        super().bid_ask_result(
            status, num_trades, new_bids, new_asks, high_bid, high_bidder, low_ask, low_asker
        )

        # Update current market state
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker

        # Update beliefs from new opposite-side quotes
        if self.is_buyer:
            # Buyer observes asks
            for ask in new_asks:
                if ask > 0:
                    self._update_beliefs(ask)
        else:
            # Seller observes bids
            for bid in new_bids:
                if bid > 0:
                    self._update_beliefs(bid)

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
        """Update state after trade result."""
        super().buy_sell_result(
            status, trade_price, trade_type, high_bid, high_bidder, low_ask, low_asker
        )

        # Update market state
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker

        # Trade prices are also informative
        if trade_price > 0:
            self._update_beliefs(trade_price)

    def start_period(self, period_number: int) -> None:
        """
        Reset period state and update beliefs from previous period.
        """
        super().start_period(period_number)

        # Update m0 from previous period's quote count
        if len(self._observed_prices) > 0:
            self._prev_period_count = len(self._observed_prices)
            self.m0 = self._prev_period_count

            # Update sigma from previous period's sample std
            if len(self._observed_prices) > 1:
                self.belief_sigma = max(10.0, np.std(self._observed_prices))

            # Store for reference
            self._prev_period_prices = self._observed_prices.copy()

        # Reset observation history for new period
        self._observed_prices = []

        # Reset belief mean to prior (will update with new observations)
        self.belief_mu = (self.price_min + self.price_max) / 2

        # Reset market state
        self.current_time = 0
        self.current_bid = 0
        self.current_ask = 0
        self.current_bidder = 0
        self.current_asker = 0
        self._reservation_price = None

    @property
    def trade_count(self) -> int:
        """Compatibility property for tournament scripts."""
        return self.num_trades
