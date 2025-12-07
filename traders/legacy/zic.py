"""
Zero Intelligence Constrained 1 (ZIC1) Agent.

Implements the ZIC strategy from Gode & Sunder (1993).
This agent submits RANDOM bids/asks within budget constraints only.
It does NOT observe or react to market state - purely random within profitable range.

KEY IMPLEMENTATION:
===================
1. BID-ASK STEP: Random price within budget constraint
   - Buyer: Random in [min_price, valuation]
   - Seller: Random in [cost, max_price]
   - No market observation, no New York Rule awareness

2. BUY-SELL STEP: Accept if profitable
   - Buyer: Accept if CurrentAsk < TokenRedemptionValue
   - Seller: Accept if CurrentBid > TokenCost

The New York Rule (must improve current bid/ask) is enforced by the MARKET,
not by ZIC1. ZIC1 just submits random prices; invalid ones get rejected.

Hierarchy: ZI → ZIC1 (budget) → ZIC2 (budget + market) → ZIP1 → ZIP2

Reference: Gode & Sunder (1993), "Allocative Efficiency of Markets with
Zero-Intelligence Traders", Journal of Political Economy, Vol. 101, No. 1
"""

from typing import Any

import numpy as np

from traders.base import Agent


class ZIC1(Agent):
    """
    Zero Intelligence Constrained 1 (ZIC1) trader - budget constraint only.

    Strategy:
    - Bid: Random value between [min_price, valuation]
    - Ask: Random value between [cost, max_price]
    - Buy/Sell: Only accept if we are the current winner and spread is crossed.

    This agent provides a baseline for market efficiency without intelligence.
    Does NOT consider current market state (bid/ask) when placing orders.

    Hierarchy: ZI → ZIC1 (budget) → ZIC2 (budget + market) → ZIP1 → ZIP2
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        price_min: int = 0,
        price_max: int = 100,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize ZIC agent.

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
        Prepare for bid/ask.
        ZIC doesn't need to do anything here as it doesn't track state.
        """
        self.has_responded = False

    def bid_ask_response(self) -> int:
        """
        Return a random bid/ask within profitable bounds.

        Note: This is a faithful port of SRobotZI1.java logic.
        Java uses 1-indexed arrays (token[mytrades+1] where mytrades starts at 0).
        Python uses 0-indexed arrays (valuations[num_trades] where num_trades starts at 0).
        Both access the first element when no trades have occurred yet.
        """
        self.has_responded = True

        # If no tokens left, return 0 (should be handled by market, but safe check)
        if self.num_trades >= self.num_tokens:
            return 0

        valuation = self.valuations[self.num_trades]

        if self.is_buyer:
            # Bid: Random in [min_price, valuation]
            # Java: newbid=token-(int)(drand()*(token-minprice));
            # Clamped to minprice.

            # If valuation <= min_price, can only bid min_price (or 0?)
            # SRobotZI1 logic implies it bids min_price.
            if valuation <= self.price_min:
                return self.price_min

            # Match Java formula exactly: V - floor(random * (V - min))
            # This creates a bias toward higher bids due to subtraction + truncation
            range_size = valuation - self.price_min
            random_offset = int(self.rng.random() * range_size)
            newbid = valuation - random_offset
            return max(self.price_min, newbid)  # Clamp to min

        else:
            # Ask: Random in [cost, max_price]
            # Java: newask=token+(int)(drand()*(maxprice-token));
            # Clamped to maxprice.

            if valuation >= self.price_max:
                return self.price_max

            # Match Java formula exactly: C + floor(random * (max - C))
            # This creates a bias toward lower asks due to truncation
            range_size = self.price_max - valuation
            random_offset = int(self.rng.random() * range_size)
            newask = valuation + random_offset
            return min(self.price_max, newask)  # Clamp to max

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
        Accept trade if profitable and we are the winner.

        DEFENSIVE IMPLEMENTATION:
        - Extra validation to prevent irrational trades
        - Guards against state corruption across rounds
        - Ensures we never trade at a loss
        """
        self.has_responded = True

        # Check if we are allowed to trade
        if self.last_nobuysell > 0:
            return False

        # Defensive: Check we have tokens left
        if self.num_trades >= self.num_tokens:
            return False

        # Defensive: Validate num_trades is within bounds
        if self.num_trades < 0 or self.num_trades >= len(self.valuations):
            return False

        valuation = self.valuations[self.num_trades]

        # Defensive: Validate market prices are reasonable
        if self.current_ask < 0 or self.current_bid < 0:
            return False

        if self.is_buyer:
            # Per checklist: Accept if CurrentAsk < TokenRedemptionValue
            # This means accept if profitable (ask strictly less than valuation)
            if self.current_ask > 0 and self.current_ask >= valuation:
                return False  # Not profitable

            # Only accept if we are the high bidder AND spread is crossed/met
            if (
                self.player_id == self.current_bidder
                and self.current_bid > 0
                and self.current_bid >= self.current_ask
            ):
                return True

        else:
            # Per checklist: Accept if CurrentBid > TokenCost
            # This means accept if profitable (bid strictly greater than cost)
            if self.current_bid > 0 and self.current_bid <= valuation:
                return False  # Not profitable

            # Only accept if we are the low asker AND spread is crossed/met
            if (
                self.player_id == self.current_asker
                and self.current_ask > 0
                and self.current_ask <= self.current_bid
            ):
                return True

        return False


# ZIC is a proper subclass for correct __name__ attribute
class ZIC(ZIC1):
    """ZIC alias with proper class name for tournament tracking."""

    pass
