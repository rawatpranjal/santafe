"""
Lin (Truth-Teller) agent for the Santa Fe double auction.

Based on SRobotLin.java from the 1993 Santa Fe tournament.
Uses statistical price prediction with confidence weighting.
"""

import math
from typing import Any, Optional
import numpy as np
from traders.base import Agent


class Lin(Agent):
    """
    Lin agent - statistical price prediction with normal distribution sampling.

    Strategy:
    - Maintains mean prices per period
    - Estimates target price from historical prices
    - Uses Box-Muller transform for normal distribution sampling
    - Weighted combination of conservative and target prices

    Java source: SRobotLin.java

    Performance: 26th place in 1993 tournament (<70% efficiency)
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        price_min: int = 0,
        price_max: int = 1000,
        num_buyers: int = 1,
        num_sellers: int = 1,
        num_times: int = 100,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize Lin agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations
            price_min: Minimum allowed price (default 0)
            price_max: Maximum allowed price (default 1000)
            num_buyers: Number of buyers in market (for weighting formula)
            num_sellers: Number of sellers in market (for weighting formula)
            num_times: Number of time steps per period (for weighting formula)
            seed: Random seed for reproducibility
            **kwargs: Ignored extra arguments
        """
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min = price_min
        self.price_max = price_max
        self.rng = np.random.default_rng(seed)

        # Price history tracking (for statistical prediction)
        # Dynamically grows to support >100 periods (Chen et al. uses 7000)
        self.mean_price: list[float] = [0.0] * 100  # 1-indexed, starts with 100
        self.traded_prices: list[int] = []  # Current period's trade prices
        self.current_period = 0
        self.num_periods = 100  # Default max periods (grows as needed)

        # Market state (for weighting formula in Java line 55, 82-84)
        self.num_buyers = num_buyers
        self.num_sellers = num_sellers
        self.num_times = num_times
        self.current_time = 0  # Updated in bid_ask()

        # State tracking for buy/sell phase
        self.current_bid = 0
        self.current_ask = 0
        self.current_bidder = 0
        self.current_asker = 0
        self.last_nobuysell = 0

    def bid_ask(self, time: int, nobidask: int) -> None:
        """
        Prepare for bid/ask phase.
        Updates current_time for use in weighting formula.
        Java equivalent: SRobotSkeleton.java line 171
        """
        self.has_responded = False
        self.current_time = time  # Needed for weighting formula

    def bid_ask_response(self) -> int:
        """
        Return a bid or ask using statistical price prediction.
        Routes to Lin's statistical methods.
        """
        self.has_responded = True

        if self.is_buyer:
            return self._player_request_bid()
        else:
            return self._player_request_ask()

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
        Capture current bid/ask state after bid/ask phase completes.
        This is needed because Lin reads current_bid/current_ask in bid_ask_response().
        Java equivalent: SRobotSkeleton.java lines 207-208
        """
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
        """
        Prepare for buy/sell decision.
        Store market state to use in response.
        """
        self.has_responded = False
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker
        self.last_nobuysell = nobuysell

    def buy_sell_response(self) -> bool:
        """
        Return buy/sell decision using statistical threshold.
        Routes to Lin's statistical decision methods.
        """
        self.has_responded = True

        if self.is_buyer:
            result = self._player_request_buy()
        else:
            result = self._player_want_to_sell()

        return bool(result)

    def start_period(self, period_number: int) -> None:
        """Start new period - reset traded prices and update period tracking."""
        super().start_period(period_number)
        self.current_period = period_number
        self.traded_prices = []  # Reset for new period

        # Dynamically grow mean_price if needed (for Chen et al. experiments with 7000+ days)
        while len(self.mean_price) <= period_number:
            self.mean_price.extend([0.0] * 100)  # Grow in chunks of 100

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
        """Track trade prices for statistical calculations."""
        super().buy_sell_result(status, trade_price, trade_type, high_bid,
                                high_bidder, low_ask, low_asker)
        # Record trade price if a trade occurred
        if trade_type != 0 and trade_price > 0:
            self.traded_prices.append(trade_price)

    def end_period(self) -> None:
        """End period - store mean price for statistical tracking."""
        super().end_period()
        # Calculate and store mean price for this period
        if 0 < self.current_period < len(self.mean_price):
            mean = self._get_mean_price()
            self.mean_price[self.current_period] = mean

    def _get_mean_price(self) -> float:
        """
        Compute mean price in current period.
        Java lines 121-129
        """
        if len(self.traded_prices) == 0:
            return 0.0

        total = sum(abs(p) for p in self.traded_prices)
        return total / len(self.traded_prices)

    def _get_target_price(self) -> float:
        """
        Compute target price from historical means.
        Java lines 131-145
        """
        target = self._get_mean_price()  # Current period mean

        # Add previous period means
        period = max(1, self.current_period)  # Avoid division by zero
        for i in range(1, min(period, len(self.mean_price))):  # Bounds check
            target += self.mean_price[i]

        # Average across periods
        if len(self.traded_prices) != 0:
            target = target / period
        elif period > 1:
            target = target / (period - 1)

        return target

    def _get_stderr_price(self) -> float:
        """
        Compute standard error of prices in current period.
        Java lines 147-159
        """
        mean = self._get_mean_price()

        if mean <= 0:
            return 1.0

        if len(self.traded_prices) <= 1:
            return 1.0

        # Calculate variance
        sum_sq = sum((abs(p) - mean) ** 2 for p in self.traded_prices)
        return math.sqrt(sum_sq) / (len(self.traded_prices) - 1)

    def _norm(self, mean: float, err: float) -> float:
        """
        Generate normally distributed random value using Box-Muller transform.
        Java lines 161-172
        """
        while True:
            r1 = 2.0 * self.rng.random() - 1.0
            r2 = 2.0 * self.rng.random() - 1.0
            s = r1 * r1 + r2 * r2
            if s < 1.0 and s > 0:
                break

        rn = r1 * math.sqrt(-2 * math.log(s) / s)
        return mean + rn * err

    def _player_request_bid(self) -> int:
        """
        Submit a bid using statistical price prediction.
        Java lines 38-63
        """
        if self.num_trades >= self.num_tokens:
            return 0

        most = self.valuations[self.num_trades] - 1  # Upper limit
        if self.current_bid >= most:
            return 0

        # Get target price from statistics
        mean = self._get_target_price()
        err = self._get_stderr_price()
        target = self._norm(mean, err)

        # Adjust target based on current bid
        if self.current_bid > 0 and self.current_bid > target:
            target = self.current_bid + 1
        if target <= 0.0 or target > most:
            target = most

        # Calculate weight based on time, tokens, and market composition
        time_factor = (self.num_times - self.current_time + 1) / self.num_times
        token_factor = (self.num_tokens - self.num_trades) / self.num_tokens
        market_factor = self.num_sellers / (self.num_buyers + self.num_sellers)
        weight = time_factor * token_factor * market_factor

        # Weighted combination of conservative and target price
        if self.current_bid > 0:
            newbid = int(weight * (self.current_bid + 1) + (1.0 - weight) * target)
        else:
            # Use worst-case token value when no current bid
            worst_token = self.valuations[self.num_tokens - 1]
            newbid = int(weight * (worst_token - 1) + (1.0 - weight) * target)

        return max(self.price_min, newbid)

    def _player_request_ask(self) -> int:
        """
        Submit an ask using statistical price prediction.
        Java lines 65-92
        """
        if self.num_trades >= self.num_tokens:
            return 0

        least = self.valuations[self.num_trades] + 1  # Lower limit
        if self.current_ask > 0 and self.current_ask <= least:
            return 0

        # Get target price from statistics
        mean = self._get_target_price()
        err = self._get_stderr_price()
        target = self._norm(mean, err)

        # Adjust target based on current ask
        if self.current_ask > 0 and self.current_ask < target:
            target = self.current_ask - 1
        if target <= 0.0 or target < least:
            target = least

        # Calculate weight
        time_factor = (self.num_times - self.current_time + 1) / self.num_times
        token_factor = (self.num_tokens - self.num_trades) / self.num_tokens
        market_factor = self.num_buyers / (self.num_buyers + self.num_sellers)
        weight = time_factor * token_factor * market_factor

        # Weighted combination
        if self.current_ask > 0:
            newoffer = int(weight * (self.current_ask - 1) + (1.0 - weight) * target)
        else:
            # Use worst-case token value when no current ask
            worst_token = self.valuations[self.num_tokens - 1]
            newoffer = int(weight * (worst_token + 1) + (1.0 - weight) * target)

        return min(self.price_max, newoffer)

    def _player_request_buy(self) -> int:
        """
        Decide whether to buy using statistical threshold.
        Java lines 94-105
        """
        if self.num_trades >= self.num_tokens:
            return 0

        token_val = self.valuations[self.num_trades]

        # Don't buy at a loss
        if token_val <= self.current_ask:
            return 0

        # Accept if we're current bidder
        if self.player_id == self.current_bidder and self.current_bid >= self.current_ask:
            return 1

        # Use statistical threshold
        target = self._get_target_price() + self._get_stderr_price()

        if target < self.price_max and self.current_ask < int(target):
            return 1

        return 0

    def _player_want_to_sell(self) -> int:
        """
        Decide whether to sell using statistical threshold.
        Java lines 107-119
        """
        if self.num_trades >= self.num_tokens:
            return 0

        token_val = self.valuations[self.num_trades]

        # Don't sell at a loss
        if self.current_bid <= token_val:
            return 0

        # Accept if we're current asker
        if self.player_id == self.current_asker and self.current_ask <= self.current_bid:
            return 1

        # Use statistical threshold
        target = self._get_target_price() - self._get_stderr_price()

        if target > self.price_min and self.current_bid > int(target):
            return 1

        return 0