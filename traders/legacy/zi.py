"""
Zero Intelligence (ZI) Agent - UNCONSTRAINED.

This is the CONTROL CONDITION from Gode & Sunder (1993).
Unlike ZIC, this agent has NO budget constraints and can trade at a loss.

This agent provides the baseline showing that randomness alone (without constraints)
achieves poor efficiency (~60-70%), proving that the budget constraint in ZIC is
the critical feature that creates market efficiency.

Reference: Gode & Sunder (1993), "Allocative Efficiency of Markets with
Zero-Intelligence Traders", Journal of Political Economy, Vol. 101, No. 1
Table 1 reports ZI (unconstrained) efficiency of 60-70% vs ZIC efficiency of 98.7%.
"""

import numpy as np
from typing import Any, Optional
from traders.base import Agent

class ZI(Agent):
    """
    Zero Intelligence (ZI) trader - UNCONSTRAINED.

    Strategy:
    - Bid/Ask: Random value in [price_min, price_max] regardless of valuation/cost
    - Buy/Sell: Accept trades even if unprofitable (no budget constraint)

    This is the control group that proves budget constraints matter.
    Expected efficiency: 60-70% (Gode & Sunder 1993 Table 1)
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
        Initialize ZI agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations (used for tracking but not bidding)
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
        Prepare for bid/ask.
        ZI doesn't need to do anything here as it doesn't track state.
        """
        self.has_responded = False

    def bid_ask_response(self) -> int:
        """
        Return a random bid/ask from FULL price range [price_min, price_max].

        KEY DIFFERENCE FROM ZIC:
        - ZIC constrains to [price_min, valuation] for buyers
        - ZI uses [price_min, price_max] regardless of valuation
        - This allows unprofitable bids/asks
        """
        self.has_responded = True

        # If no tokens left, return 0
        if self.num_trades >= self.num_tokens:
            return 0

        # Random bid/ask from FULL range (no budget constraint)
        # Use integer formula similar to ZIC for consistency
        range_size = self.price_max - self.price_min

        if range_size <= 0:
            return self.price_min

        random_offset = int(self.rng.random() * range_size)
        return self.price_min + random_offset

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
        Store market state to decide in response.
        """
        self.has_responded = False
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker
        self.last_nobuysell = nobuysell

    def buy_sell_response(self) -> bool:
        """
        Accept trade if we are the winner, regardless of profitability.

        KEY DIFFERENCE FROM ZIC:
        - ZIC rejects unprofitable trades (valuation <= ask for buyers)
        - ZI accepts ALL trades when winner (no budget constraint)
        - This allows trades at a loss
        """
        self.has_responded = True

        # Check if we are allowed to trade
        if self.last_nobuysell > 0:
            return False

        if self.num_trades >= self.num_tokens:
            return False

        if self.is_buyer:
            # Accept if we are the high bidder AND spread is crossed/met
            # NO profitability check (unlike ZIC)
            if self.player_id == self.current_bidder and self.current_bid >= self.current_ask:
                return True

        else:
            # Accept if we are the low asker AND spread is crossed/met
            # NO profitability check (unlike ZIC)
            if self.player_id == self.current_asker and self.current_ask <= self.current_bid:
                return True

        return False
