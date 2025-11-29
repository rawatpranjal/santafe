"""
Zero Intelligence 2 (Ringuette) agent for the Santa Fe double auction.

Based on SRobotZI2.java from the 1993 Santa Fe tournament.
Improves on ZIC by considering current market bid/ask when submitting orders.
"""

import numpy as np
from typing import Any, Optional
from traders.base import Agent


class ZI2(Agent):
    """
    Zero Intelligence 2 agent - market-aware random bidding.

    Key differences from ZIC:
    - Considers current bid (cbid) when making new bids
    - Considers current ask (cask) when making new asks
    - Still zero intelligence (no learning or strategy)

    Java source: SRobotZI2.java
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        price_min: int = 0,
        price_max: int = 100,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize ZI2 agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations
            price_min: Minimum allowed price (default 0)
            price_max: Maximum allowed price (default 100)
            seed: Random seed for reproducibility
            **kwargs: Ignored extra arguments
        """
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min = price_min
        self.price_max = price_max
        self.rng = np.random.default_rng(seed)

        # State tracking for buy/sell phase
        self.current_bid = 0
        self.current_ask = 0
        self.current_bidder = 0
        self.current_asker = 0
        self.last_nobuysell = 0

    def bid_ask(self, time: int, nobidask: int) -> None:
        """
        Prepare for bid/ask phase.
        ZI2 doesn't need special preparation, just reset response flag.
        """
        self.has_responded = False

    def bid_ask_response(self) -> int:
        """
        Return a bid or ask based on current market state.
        Routes to market-aware private methods.
        """
        self.has_responded = True

        if self.is_buyer:
            return self._player_request_bid()
        else:
            return self._player_request_ask()

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
        Accept trade if profitable and we are the winner.
        Routes to private methods and converts int to bool.
        """
        self.has_responded = True

        # Check if we are allowed to trade
        if self.last_nobuysell > 0:
            return False

        if self.num_trades >= self.num_tokens:
            return False

        if self.is_buyer:
            result = self._player_request_buy()
        else:
            result = self._player_want_to_sell()

        return bool(result)

    def _player_request_bid(self) -> int:
        """
        Submit a bid considering current market bid.

        Java lines 31-49: Market-aware bidding
        - If cbid exists and <= token value: random between [cbid, token]
        - If cbid exists and > token value: return min_price
        - If no cbid: random between [min_price, token]
        """
        if self.num_trades >= self.num_tokens:
            return 0  # Nothing left to trade

        token_val = self.valuations[self.num_trades]

        if self.current_bid > 0 and self.current_bid <= token_val:
            # Market-aware: Java formula token - floor(random * (token - cbid))
            range_size = token_val - self.current_bid
            random_offset = int(self.rng.random() * range_size)
            newbid = token_val - random_offset
        elif self.current_bid > 0 and self.current_bid > token_val:
            # Current bid already exceeds our valuation
            newbid = self.price_min
        else:
            # No current bid: Java formula token - floor(random * (token - min))
            range_size = token_val - self.price_min
            random_offset = int(self.rng.random() * range_size)
            newbid = token_val - random_offset

        # Clamp to valid range
        newbid = max(self.price_min, min(newbid, self.price_max))
        return newbid

    def _player_request_ask(self) -> int:
        """
        Submit an ask considering current market ask.

        Java lines 51-68: Market-aware asking
        - If cask exists and >= token cost: random between [token, cask]
        - If cask exists and < token cost: return max_price
        - If no cask: random between [token, max_price]
        """
        if self.num_trades >= self.num_tokens:
            return 0  # Nothing left to trade

        token_val = self.valuations[self.num_trades]

        if self.current_ask > 0 and self.current_ask >= token_val:
            # Market-aware: Java formula token + floor(random * (max - token))
            # NOTE: Java uses maxprice even when cask exists (line 58)
            range_size = self.price_max - token_val
            random_offset = int(self.rng.random() * range_size)
            newoffer = token_val + random_offset
        elif self.current_ask > 0 and self.current_ask < token_val:
            # Current ask already below our cost
            newoffer = self.price_max
        else:
            # No current ask: Java formula token + floor(random * (max - token))
            range_size = self.price_max - token_val
            random_offset = int(self.rng.random() * range_size)
            newoffer = token_val + random_offset

        # Clamp to valid range
        newoffer = max(self.price_min, min(newoffer, self.price_max))
        return newoffer

    def _player_request_buy(self) -> int:
        """
        Decide whether to buy at current ask price.

        Java lines 70-75: Same as ZIC - avoid losses, accept if we're current bidder
        """
        if self.num_trades >= self.num_tokens:
            return 0

        token_val = self.valuations[self.num_trades]

        # Don't buy at a loss (strict inequality per Java)
        if token_val <= self.current_ask:
            return 0

        # Accept if we're current bidder and our bid >= ask
        if self.player_id == self.current_bidder and self.current_bid >= self.current_ask:
            return 1

        return 0

    def _player_want_to_sell(self) -> int:
        """
        Decide whether to sell at current bid price.

        Java lines 79-85: Same as ZIC - avoid losses, accept if we're current asker
        """
        if self.num_trades >= self.num_tokens:
            return 0

        token_val = self.valuations[self.num_trades]

        # Don't sell at a loss (strict inequality per Java)
        if self.current_bid <= token_val:
            return 0

        # Accept if we're current asker and our ask <= bid
        if self.player_id == self.current_asker and self.current_ask <= self.current_bid:
            return 1

        return 0