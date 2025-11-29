"""
Perry Original agent for the Santa Fe double auction.

Based on SRobotPerryOriginal.java from the 1993 Santa Fe tournament.
Uses adaptive learning with parameter tuning based on efficiency.
"""

from typing import Any, Optional
import random
import math
from traders.base import Agent


class Perry(Agent):
    """
    Perry Original agent - adaptive learning with efficiency-based tuning.

    Strategy:
    - Maintains round and period-level price statistics
    - Uses adaptive parameter a0 that adjusts based on performance
    - Conservative strategy for first 3 trades, then statistical
    - Complex acceptance thresholds with time pressure
    - Evaluates efficiency each period and tunes parameters

    Java source: SRobotPerryOriginal.java
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        price_min: int = 0,
        price_max: int = 100,
        num_buyers: int = 1,
        num_sellers: int = 1,
        num_times: int = 100,
        seed: Optional[int] = None,
        a0_initial: float = 2.0,
        desperate_threshold: float = 0.20,
        desperate_margin: int = 2,
        **kwargs: Any
    ) -> None:
        """
        Initialize Perry agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations
            price_min: Minimum allowed price (default 0)
            price_max: Maximum allowed price (default 100)
            num_buyers: Number of buyers in market
            num_sellers: Number of sellers in market
            num_times: Number of time steps per period
            seed: Random seed for reproducibility
            a0_initial: Initial adaptive parameter (default 2.0, Java default)
            desperate_threshold: Time remaining fraction for desperate acceptance (default 0.20)
            desperate_margin: Units above cost for desperate acceptance (default 2)
            **kwargs: Ignored extra arguments
        """
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min = price_min
        self.price_max = price_max
        self.rng = random.Random(seed)

        # Market state (for weighting formulas)
        self.num_buyers = num_buyers
        self.num_sellers = num_sellers
        self.num_times = num_times
        self.current_time = 0

        # Hyperparameters (tunable)
        self.a0_initial = a0_initial
        self.desperate_threshold = desperate_threshold
        self.desperate_margin = desperate_margin

        # Round-level tracking (Java lines 24-34)
        self.r_price_sum: int = 0  # Sum of prices within a round
        self.r_price_ss: int = 0   # Price sum of squares in the round
        self.rtrades: int = 0      # Number of trades in the round
        self.ctrades: int = 0      # Trade counter
        self.r_price_ave: int = 0  # Average trading price in the round
        self.r_price_std: int = 50 # Price standard deviation within the round
        self.round_count: int = 0  # Round number
        self.a0: float = a0_initial  # Adaptive parameter (initialized from a0_initial)
        self.p_ave_price: int = 0  # Average price in the period
        self.price_std: int = 0    # Price standard deviation in the round
        self.mprice: int = 0       # Maximum trade price so far this game

        # Period tracking
        self.current_period = 0
        self.current_round = 0
        self.traded_prices: list[int] = []  # Current period's trade prices

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
        """
        self.has_responded = False
        self.current_time = time

    def bid_ask_response(self) -> int:
        """
        Return a bid or ask using adaptive learning strategy.
        Routes to Perry's statistical methods.
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
        """Capture current bid/ask state after bid/ask phase completes."""
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
        Return buy/sell decision using adaptive threshold.
        Routes to Perry's decision methods.
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
        # Java lines 41-48 (playerRoundBegin)
        self.p_ave_price = 0
        self.price_std = 0

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
        """End period - evaluate performance and adjust parameters."""
        # Java line 55 (playerPeriodEnd)
        self._evaluate()
        super().end_period()

    def _ave_price(self) -> int:
        """
        Returns the average price within the current period.
        Java lines 382-390
        """
        if len(self.traded_prices) == 0:
            return 0

        total = sum(abs(p) for p in self.traded_prices)
        return int(total / len(self.traded_prices))

    def _round_average_price(self) -> None:
        """
        Computes the average price within the round and standard deviation.
        Java lines 396-428
        """
        # Check if we're in a new round
        if self.current_round != self.round_count:
            self.round_count += 1
            self.r_price_sum = 0
            self.r_price_ss = 0
            self.rtrades = 0
            self.ctrades = 0
            self.r_price_ave = 0
            self.r_price_std = 30  # Default std dev

        # Update statistics with new trades
        ntrades = len(self.traded_prices)
        if ntrades > 0 and ntrades != self.ctrades:
            if ntrades == 1 and self.ctrades != 0:
                self.ctrades = 0
            self.ctrades += 1
            self.rtrades += 1

            # Update with last price
            last_price = abs(self.traded_prices[-1]) if self.traded_prices else 0
            self.r_price_sum += last_price
            self.r_price_ss += last_price * last_price
            self.r_price_ave = int(self.r_price_sum / self.rtrades)

            if self.rtrades > 1:
                r_price_var = int((self.r_price_ss - self.r_price_sum * self.r_price_sum / self.rtrades) / self.rtrades)
            else:
                r_price_var = 900

            self.r_price_std = int(math.sqrt(abs(r_price_var)))

    def _evaluate(self) -> None:
        """
        Adjust parameter a0 based on efficiency.
        Java lines 317-377
        """
        x = 1
        feasible_trades = 0

        self.p_ave_price = self._ave_price()

        # Calculate price statistics
        price_sum = sum(abs(p) for p in self.traded_prices)
        price_ss = sum(p * p for p in self.traded_prices)

        if len(self.traded_prices) > 1:
            price_var = int((price_ss - price_sum * price_sum / len(self.traded_prices)) / len(self.traded_prices))
            self.price_std = int(math.sqrt(abs(price_var)))

        # Calculate feasible trades and potential profit
        # Java lines 339-357 (role==1 for buyer, role==2 for seller)
        if self.is_buyer:
            for i in range(self.num_tokens):
                if self.valuations[i] - self.p_ave_price >= 0:
                    feasible_trades += 1
                    x += self.valuations[i] - self.p_ave_price
        else:  # seller
            for i in range(self.num_tokens):
                if self.p_ave_price - self.valuations[i] >= 0:
                    feasible_trades += 1
                    x += self.p_ave_price - self.valuations[i]

        # Calculate efficiency
        e = float(self.period_profit) / x if x > 0 else 0.0

        # Adjust a0 based on efficiency
        if e < 1.0 and self.num_trades < feasible_trades:
            if e == 0:
                self.a0 = self.a0 / 3
            else:
                self.a0 = self.a0 * e

        if e <= 0.8 and self.num_trades < feasible_trades and self.price_std < 10:
            self.price_std = 30

        if e <= 0.9 and self.num_trades >= feasible_trades:
            self.a0 = self.a0 * (2 - e)

        if e <= 0.8 and self.num_trades >= feasible_trades and self.price_std < 10:
            self.a0 = self.a0 * (2 - e)
            self.price_std = 30

    def _player_request_bid(self) -> int:
        """
        Submit a bid using adaptive learning strategy.
        Java lines 62-156
        """
        if self.num_trades >= self.num_tokens:
            return 0

        # Calculate adaptive parameter a1
        a1 = (self.a0 * (self.num_times - self.current_time) / self.num_times *
              (self.num_buyers + self.num_sellers - 1) / (self.num_buyers + self.num_sellers) *
              self.num_sellers / self.num_buyers)

        self._round_average_price()
        s = 1 if self.rng.random() > 0.5 else -1

        # Weight previous period statistics
        if self.p_ave_price != 0:
            self.r_price_ave = self.p_ave_price
            self.r_price_std = (3 * self.price_std + self.r_price_std) // 3

        # Conservative strategy for first 3 trades of first period
        if len(self.traded_prices) < 3 and self.current_period == 1:
            if self.current_ask == 0:
                # First bid is random, less than smallest token
                newbid = int(self.valuations[self.num_tokens - 1] * self.rng.random())
                return newbid
            else:
                # Gradual increase
                newbid = int(self.current_bid * (1 + (self.current_ask - self.current_bid) /
                            ((1.2 + self.rng.random()) * (self.current_bid + self.current_ask + 1))))
                return newbid if newbid < self.valuations[self.num_trades] else 0

        # Statistical approach after first 3 trades
        else:
            low = self.r_price_ave - 3 * self.r_price_std

            if self.current_ask == 0 and self.current_bid == 0:
                # Opening bid
                most = min(self.r_price_ave, self.valuations[self.num_trades] - 1)
                newbid = int(self.r_price_ave - a1 * self.r_price_std)
                newbid = max(newbid, low)
                newbid = max(newbid, self.price_min) if newbid > self.price_min else 0
                return min(newbid, most)
            else:
                # Subsequent bids
                most = self.valuations[self.num_trades] - 1
                if most <= self.current_bid:
                    return 0
                if self.current_ask > 0:
                    most = min(most, self.current_ask)

                # Calculate new bid with time pressure
                newbid = int(self.r_price_ave + 0.2 * self.r_price_std - a1 * self.r_price_std +
                            self.rng.random() * 4 * s)

                t1 = self.current_time
                while newbid <= self.current_bid and t1 < self.num_times:
                    t1 += 1
                    a1 = (self.a0 * (self.num_times - t1) / self.num_times *
                          (self.num_buyers + self.num_sellers - 1) / (self.num_buyers + self.num_sellers) *
                          self.num_sellers / self.num_buyers * self.num_sellers / self.num_buyers *
                          self.num_sellers / self.num_buyers)
                    newbid = int(self.r_price_ave + 0.2 * self.r_price_std - a1 * self.r_price_std)
                    if t1 == self.num_times - 1:
                        newbid = self.current_bid + 1

                newbid = newbid if newbid > low else 0
                return min(newbid, most)

    def _player_request_ask(self) -> int:
        """
        Submit an ask using adaptive learning strategy.
        Java lines 159-242
        """
        if self.num_trades >= self.num_tokens:
            return 0

        if self.current_ask != 0 and self.current_ask < self.valuations[self.num_trades]:
            return 0

        self._round_average_price()

        # Update max price
        for price in self.traded_prices:
            self.mprice = max(self.mprice, abs(price))

        if self.p_ave_price != 0:
            self.r_price_ave = self.p_ave_price
            self.r_price_std = (3 * self.price_std + self.r_price_std) // 4

        a1 = (self.a0 * (self.num_times - self.current_time) / self.num_times *
              self.num_buyers / self.num_sellers *
              (self.num_buyers + self.num_sellers - 1) / (self.num_buyers + self.num_sellers))

        # Conservative strategy for first 3 trades
        if self.current_period == 1 and len(self.traded_prices) <= 3:
            if self.current_bid == 0 and self.current_ask == 0:
                newoffer = int(self.price_max * self.rng.random())
                while newoffer <= self.valuations[self.num_tokens - 1]:
                    newoffer = int(self.price_max * self.rng.random())
                return newoffer
            else:
                if self.current_ask > self.current_bid:
                    # Linear decrease
                    newoffer = int(self.current_ask * (0.97 + 0.05 * self.rng.random()))
                    return newoffer if newoffer < self.current_ask else int(self.current_ask * 0.99)

        # Statistical adjustment after first 3 trades
        if self.current_bid == 0:
            # First ask
            newoffer = int(self.r_price_ave + a1 * self.r_price_std + 20 * self.rng.random())
            newoffer = max(newoffer, int(self.valuations[self.num_trades] * (1.1 + self.rng.random())))

            if newoffer >= self.price_max:
                while newoffer >= self.price_max:
                    newoffer = int(self.valuations[self.num_tokens - 1] * (1 + self.rng.random()))
            return newoffer
        else:
            # Subsequent asks
            least = self.valuations[self.num_trades] + 1
            if self.current_bid > least:
                least = self.current_bid

            if self.r_price_ave > 0:
                newoffer = int(self.r_price_ave + a1 * self.r_price_std + 20 * self.rng.random())
                if newoffer > self.current_ask:
                    newoffer = int(self.current_ask - 5 * self.rng.random())
                return max(newoffer, least)
            else:
                newoffer = int(self.current_ask * 0.99)
                return max(newoffer, least)

    def _player_request_buy(self) -> int:
        """
        Decide whether to buy using adaptive threshold.
        Java lines 244-269
        """
        if self.num_trades >= self.num_tokens:
            return 0

        # Weighting factor
        a1 = 2 * (self.num_times - self.current_time) / self.num_times * \
             (self.num_buyers + self.num_sellers - 1) / (self.num_buyers + self.num_sellers) + \
             self.rng.random()

        # Don't buy at a loss
        if self.valuations[self.num_trades] <= self.current_ask:
            return 0

        # Accept if we're current bidder and our bid >= ask
        if self.player_id == self.current_bidder and self.current_bid >= self.current_ask:
            return 1

        # Buy if current offer is in acceptance region
        if self.player_id == self.current_bidder:
            threshold = self.r_price_ave + 0.2 * self.r_price_std - a1 * self.r_price_std
            return 1 if self.current_ask <= threshold else 0

        return 0

    def _player_want_to_sell(self) -> int:
        """
        Decide whether to sell using adaptive threshold.
        Java lines 271-302
        """
        if self.num_trades >= self.num_tokens:
            return 0

        a1 = 2 * (self.num_times - self.current_time) / self.num_times * \
             (self.num_buyers + self.num_sellers - 1) / (self.num_buyers + self.num_sellers)

        # Don't sell at a loss
        if self.valuations[self.num_trades] - 1 >= self.current_bid:
            return 0

        # Accept if we're current asker and bid >= ask
        if self.player_id == self.current_asker and self.current_bid >= self.current_ask:
            return 1

        if len(self.traded_prices) > 0:
            self._round_average_price()

            if self.p_ave_price != 0:
                self.r_price_ave = self.p_ave_price
                self.r_price_std = (3 * self.price_std + self.r_price_std) // 3

            # Check threshold
            answer = 1 if self.current_bid >= (self.r_price_ave + a1 * self.r_price_std) else 0

            # Desperate acceptance near end of period
            if answer == 0 and (self.num_times - self.current_time) / self.num_times <= 0.20:
                least = self.valuations[self.num_trades] + 2
                answer = 1 if self.current_bid > least else 0

            return answer

        return 0