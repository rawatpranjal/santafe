"""
Staecker (Predictive Strategy) Trader.

A price-forecasting trader from the JEDC paper that predicts future high bids
and low asks using exponential smoothing, then trades when predicted prices
make deals attractive.

Classification:
- Complex: Maintains forecast state, uses prediction-driven decisions
- Predictive: Exponentially smoothed forecasts of next extrema
- Non-Adaptive: Uses current-game history only, no cross-game learning

Key behaviors:
- Tracks recent bids/asks and forms smoothed forecasts
- Trades when PREDICTED spread is tight (not just current spread)
- Patient early (waits for forecast to settle), aggressive late
- Intermediate complexity between Skeleton/Kaplan and full Bayesian models
"""

import math
from typing import Any

from traders.base import Agent

# Fixed hyperparameters
LAMBDA = 0.3  # Exponential smoothing factor for forecasts
SPREAD_THRESH_FACTOR = 0.10  # 10% of mid_pred
SPREAD_THRESH_MIN = 5  # minimum absolute spread threshold
LATE_TIME_FRACTION = 0.85  # give up on prediction, trade if profitable


class Staecker(Agent):
    """
    Staecker (Predictive Strategy) trading agent.

    Strategy:
    - Maintains exponentially smoothed forecasts of high bid and low ask
    - BA step: Quote based on predicted prices, not just current
    - BS step: Accept when current price beats forecast
    - Patient early, aggressive late
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
        Initialize Staecker agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations
            price_min: Minimum allowed price
            price_max: Maximum allowed price
            num_times: Total time steps in period (NTIMES)
            seed: Random seed (unused, Staecker is deterministic)
            **kwargs: Ignored extra arguments
        """
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min = price_min
        self.price_max = price_max
        self.num_times = num_times

        # Forecast state
        self.hb_pred: float | None = None  # predicted next high bid
        self.la_pred: float | None = None  # predicted next low ask

        # Current market state
        self.current_time = 0
        self.current_bid = 0
        self.current_ask = 0
        self.current_bidder = 0
        self.current_asker = 0
        self.nobidask = 0

    def _time_fraction(self) -> float:
        """Current time as fraction of period [0, 1]."""
        if self.num_times <= 0:
            return 1.0
        return self.current_time / self.num_times

    def _is_late(self) -> bool:
        """Check if we're in late-time override mode."""
        return self._time_fraction() >= LATE_TIME_FRACTION

    def _update_forecasts(self) -> None:
        """
        Update forecasts using exponential smoothing.

        pred ← (1-λ)*pred + λ*current
        """
        if self.current_bid > 0:
            if self.hb_pred is None:
                self.hb_pred = float(self.current_bid)
            else:
                self.hb_pred = (1 - LAMBDA) * self.hb_pred + LAMBDA * self.current_bid

        if self.current_ask > 0:
            if self.la_pred is None:
                self.la_pred = float(self.current_ask)
            else:
                self.la_pred = (1 - LAMBDA) * self.la_pred + LAMBDA * self.current_ask

    def _compute_spread_threshold(self) -> float:
        """
        Compute spread threshold.

        spread_thresh = max(SPREAD_THRESH_MIN, SPREAD_THRESH_FACTOR * mid_pred)
        """
        if self.hb_pred is not None and self.la_pred is not None:
            mid_pred = 0.5 * (self.hb_pred + self.la_pred)
            return max(SPREAD_THRESH_MIN, SPREAD_THRESH_FACTOR * mid_pred)
        return SPREAD_THRESH_MIN

    def _predicted_spread(self) -> float | None:
        """Compute predicted spread if both forecasts exist."""
        if self.hb_pred is not None and self.la_pred is not None:
            return self.la_pred - self.hb_pred
        return None

    def bid_ask(self, time: int, nobidask: int) -> None:
        """Prepare for bid/ask phase."""
        self.current_time = time
        self.nobidask = nobidask
        self.has_responded = False

    def bid_ask_response(self) -> int:
        """
        BA step: Prediction-driven quoting.
        """
        self.has_responded = True

        if self.nobidask > 0:
            return 0

        if self.num_trades >= self.num_tokens:
            return 0

        # Update forecasts from current market state
        self._update_forecasts()

        valuation = self.valuations[self.num_trades]

        if self.is_buyer:
            return self._request_bid(valuation)
        else:
            return self._request_ask(valuation)

    def _request_bid(self, value: int) -> int:
        """
        Buyer bidding logic (prediction-driven).

        1. If la_pred >= value → PASS (even predicted asks unprofitable)
        2. If s_pred > spread_thresh and time < 0.5 → PASS (wait for market)
        3. Target bid = min(value-1, floor(la_pred))
        4. Apply AURORA constraints
        5. Late-time override
        """
        # Get current prices (use boundary defaults if missing)
        cbid = self.current_bid if self.current_bid > 0 else (self.price_min - 1)
        cask = self.current_ask if self.current_ask > 0 else (self.price_max + 1)

        # Late-time override: trade if profitable, ignore spread
        if self._is_late():
            if cask < value:
                # Submit bid to win the ask
                new_bid = min(value - 1, cask)
                if new_bid > cbid:
                    return max(self.price_min, min(new_bid, self.price_max))
            return 0

        # Need forecasts to proceed with prediction-based logic
        if self.la_pred is None:
            # No forecast yet - bootstrap by submitting a reasonable bid
            # This prevents deadlock in self-play where all agents wait for forecasts
            # Use a fraction of valuation as initial bid (like Skeleton/ZIC would)
            target_bid = int(value * 0.7)  # Bid 70% of valuation to leave profit margin
            b_min = cbid + 1  # AURORA: must beat current bid
            b_max = min(value - 1, cask - 1 if cask > 0 else self.price_max)

            # Clamp target to valid range
            new_bid = max(b_min, min(target_bid, b_max))
            if b_min <= b_max:
                return max(self.price_min, new_bid)
            return 0

        # Profitability check (predicted ask)
        if self.la_pred >= value:
            return 0  # Even predicted future asks are not profitable

        # Spread tightness check (early game)
        s_pred = self._predicted_spread()
        spread_thresh = self._compute_spread_threshold()

        if s_pred is not None and s_pred > spread_thresh and self._time_fraction() < 0.5:
            return 0  # Wait for market to settle

        # Target bid from forecast
        b_target = min(value - 1, int(math.floor(self.la_pred)))

        # AURORA constraints
        b_min = cbid + 1
        b_max = min(b_target, cask - 1, self.price_max)

        if b_min > b_max:
            return 0  # Cannot place valid bid

        # Choose highest compatible with prediction & constraints
        new_bid = b_max
        return max(self.price_min, new_bid)

    def _request_ask(self, cost: int) -> int:
        """
        Seller asking logic (prediction-driven).

        1. If hb_pred <= cost → PASS (even predicted bids unprofitable)
        2. If s_pred > spread_thresh and time < 0.5 → PASS (wait for market)
        3. Target ask = max(cost+1, ceil(hb_pred))
        4. Apply AURORA constraints
        5. Late-time override
        """
        # Get current prices (use boundary defaults if missing)
        cbid = self.current_bid if self.current_bid > 0 else (self.price_min - 1)
        cask = self.current_ask if self.current_ask > 0 else (self.price_max + 1)

        # Late-time override: trade if profitable, ignore spread
        if self._is_late():
            if cbid > cost:
                # Submit ask to win the bid
                new_ask = max(cost + 1, cbid)
                if new_ask < cask:
                    return max(self.price_min, min(new_ask, self.price_max))
            return 0

        # Need forecasts to proceed with prediction-based logic
        if self.hb_pred is None:
            # No forecast yet - bootstrap by submitting a reasonable ask
            # This prevents deadlock in self-play where all agents wait for forecasts
            # Use cost + 30% markup as initial ask (like Skeleton/ZIC would)
            target_ask = int(cost * 1.3)  # Ask 130% of cost to leave profit margin
            a_min = max(cost + 1, cbid + 1 if cbid > 0 else self.price_min)
            a_max = cask - 1 if cask > 0 else self.price_max

            # Clamp target to valid range
            new_ask = max(a_min, min(target_ask, a_max))
            if a_min <= a_max:
                return min(self.price_max, new_ask)
            return 0

        # Profitability check (predicted bid)
        if self.hb_pred <= cost:
            return 0  # Even predicted future bids are not profitable

        # Spread tightness check (early game)
        s_pred = self._predicted_spread()
        spread_thresh = self._compute_spread_threshold()

        if s_pred is not None and s_pred > spread_thresh and self._time_fraction() < 0.5:
            return 0  # Wait for market to settle

        # Target ask from forecast
        a_target = max(cost + 1, int(math.ceil(self.hb_pred)))

        # AURORA constraints
        a_max = cask - 1
        a_min = max(a_target, cbid + 1, self.price_min)

        if a_min > a_max:
            return 0  # Cannot place valid ask

        # Choose lowest compatible with prediction & constraints
        new_ask = a_min
        return min(self.price_max, new_ask)

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

        # Update forecasts from this new market state
        self._update_forecasts()

    def buy_sell_response(self) -> bool:
        """
        BS step: Accept when current price beats forecast.

        Buyer: Accept if current_ask <= la_pred (predicted future not better)
        Seller: Accept if current_bid >= hb_pred
        Late-time override: Accept any profitable trade
        """
        self.has_responded = True

        if self.num_trades >= self.num_tokens:
            return False

        valuation = self.valuations[self.num_trades]

        if self.is_buyer:
            return self._buyer_accept(valuation)
        else:
            return self._seller_accept(valuation)

    def _buyer_accept(self, value: int) -> bool:
        """
        Buyer acceptance logic.

        - If current_ask >= value → PASS
        - If la_pred defined and current_ask <= la_pred → BUY
        - If late → BUY if profitable
        - Otherwise → PASS
        """
        # Must be high bidder to accept
        if self.player_id != self.current_bidder:
            return False

        # Must have valid ask
        if self.current_ask <= 0:
            return False

        # Profitability check
        if self.current_ask >= value:
            return False

        # Late-time override
        if self._is_late():
            return True  # Accept any profitable trade

        # Prediction-based take rule
        if self.la_pred is not None and self.current_ask <= self.la_pred:
            return True  # Current ask is as good as or better than predicted

        return False

    def _seller_accept(self, cost: int) -> bool:
        """
        Seller acceptance logic.

        - If current_bid <= cost → PASS
        - If hb_pred defined and current_bid >= hb_pred → SELL
        - If late → SELL if profitable
        - Otherwise → PASS
        """
        # Must be low asker to accept
        if self.player_id != self.current_asker:
            return False

        # Must have valid bid
        if self.current_bid <= 0:
            return False

        # Profitability check
        if self.current_bid <= cost:
            return False

        # Late-time override
        if self._is_late():
            return True  # Accept any profitable trade

        # Prediction-based take rule
        if self.hb_pred is not None and self.current_bid >= self.hb_pred:
            return True  # Current bid is as good as or better than predicted

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
        """Update state after bid/ask phase."""
        super().bid_ask_result(
            status, num_trades, new_bids, new_asks, high_bid, high_bidder, low_ask, low_asker
        )

        # Update market state
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker

        # Update forecasts
        self._update_forecasts()

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
        """Update state after buy/sell phase."""
        super().buy_sell_result(
            status, trade_price, trade_type, high_bid, high_bidder, low_ask, low_asker
        )

        # Update market state
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker

    def start_period(self, period_number: int) -> None:
        """Reset period state."""
        super().start_period(period_number)

        # Reset forecasts for new period
        self.hb_pred = None
        self.la_pred = None

        # Reset market state
        self.current_time = 0
        self.current_bid = 0
        self.current_ask = 0
        self.current_bidder = 0
        self.current_asker = 0

    @property
    def trade_count(self) -> int:
        """Compatibility property for tournament scripts."""
        return self.num_trades
