"""
GradualBidder Agent.

A specialized agent that narrows the bid-ask spread through gradual bidding
but NEVER accepts trades. This is used to demonstrate Kaplan's "deal stealing"
mechanic where Kaplan waits for other buyers to narrow the spread, then
jumps in with bid = ask to "steal" the trade.

Strategy:
1. Submit bids/asks using Skeleton's weighted average approach
2. NEVER accept trades in buy_sell phase (always return False)
3. This forces the spread to narrow without trades executing
4. Allows testing of Kaplan's spread < 10% jump-in logic
"""

from traders.base import Agent
import numpy as np


class GradualBidder(Agent):
    """
    Agent that only narrows the spread, never accepts trades.

    Used to demonstrate Kaplan's deal-stealing mechanic where:
    1. GradualBidders narrow the spread through iterative bidding
    2. Kaplan waits until spread < 10%
    3. Kaplan bids = CurrentAsk and "steals" the deal
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
        seed: int | None = None,
    ) -> None:
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min_limit = price_min
        self.price_max_limit = price_max
        self.num_times = num_times
        self.rng = np.random.default_rng(seed)

        # State
        self.current_time = 0
        self.current_bid = 0
        self.current_ask = 0
        self.current_bidder = 0
        self.current_asker = 0
        self.nobidask = 0
        self.nobuysell = 0

    def bid_ask(self, time: int, nobidask: int) -> None:
        self.current_time = time
        self.nobidask = nobidask
        self.has_responded = False

    def bid_ask_response(self) -> int:
        """Use Skeleton's weighted average approach to narrow the spread."""
        self.has_responded = True
        if self.is_buyer:
            return self._request_bid()
        else:
            return self._request_ask()

    def _request_bid(self) -> int:
        """
        Gradual bid improvement - start conservative, improve slowly.

        Key difference from Skeleton: First bid is VERY conservative (price_min + small amount)
        to ensure the spread starts wide and narrows over time.
        """
        if self.nobidask > 0:
            return 0
        if self.num_trades >= self.num_tokens:
            return 0

        # Weighted average strategy
        alpha = 0.25 + 0.1 * self.rng.random()

        token_val = self.valuations[self.num_trades]
        first_token = self.valuations[0]
        last_token = self.valuations[self.num_tokens - 1]

        cbid = self.current_bid
        coffer = self.current_ask

        if cbid == 0:
            # CONSERVATIVE FIRST BID: Start at price_min + 10% of valuation range
            # This ensures spread starts WIDE
            spread_val = first_token - last_token
            conservative_start = self.price_min_limit + int(0.10 * spread_val)
            newbid = conservative_start
        else:
            most = token_val - 1  # Current token
            if coffer > 0 and coffer < most:
                most = coffer

            if most <= cbid:
                return 0

            # Weighted average: slowly approach 'most'
            newbid = int((1.0 - alpha) * (cbid + 1) + alpha * most + 0.001)

        return max(newbid, self.price_min_limit)

    def _request_ask(self) -> int:
        """
        Gradual ask improvement - start high, decrease slowly.

        Key difference from Skeleton: First ask is VERY high (price_max - small amount)
        to ensure the spread starts wide and narrows over time.
        """
        if self.nobidask > 0:
            return 0
        if self.num_trades >= self.num_tokens:
            return 0

        alpha = 0.25 + 0.1 * self.rng.random()

        token_val = self.valuations[self.num_trades]
        first_token = self.valuations[0]  # Lowest cost (Best)
        last_token = self.valuations[self.num_tokens - 1]  # Highest cost (Worst)

        cbid = self.current_bid
        coffer = self.current_ask

        if coffer == 0:
            # CONSERVATIVE FIRST ASK: Start at price_max - 10% of cost range
            # This ensures spread starts WIDE
            spread_val = last_token - first_token
            conservative_start = self.price_max_limit - int(0.10 * spread_val)
            newask = conservative_start
        else:
            least = token_val + 1  # Current token
            if cbid > 0 and cbid > least:
                least = cbid

            if least >= coffer:
                return 0

            # Weighted average: slowly approach 'least'
            newask = int((1.0 - alpha) * (coffer - 1) + alpha * least + 0.001)

        return min(newask, self.price_max_limit)

    def buy_sell(
        self,
        time: int,
        nobuysell: int,
        high_bid: int,
        low_ask: int,
        high_bidder: int,
        low_asker: int,
    ) -> None:
        self.current_time = time
        self.nobuysell = nobuysell
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker
        self.has_responded = False

    def buy_sell_response(self) -> bool:
        """
        NEVER accept trades - let Kaplan steal them.

        This is the key difference from Skeleton: we narrow the spread
        through bidding but never actually trade, allowing Kaplan to
        jump in when spread < 10% and steal the deal.
        """
        self.has_responded = True
        return False  # Never accept - we're just here to narrow the spread

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
        super().buy_sell_result(
            status, trade_price, trade_type, high_bid, high_bidder, low_ask, low_asker
        )
        # Update state
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker
