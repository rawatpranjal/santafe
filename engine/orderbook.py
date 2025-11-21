"""
engine/orderbook.py - AURORA Double Auction Order Book

Port of PeriodHistory.java from the 1993 Santa Fe Tournament.
Reference: /oldcode/extracted/double_auction/java/PeriodHistory.java (612 lines)

This implementation maintains 1:1 fidelity with the Java code, including:
- 2D array structure indexed by [player][time]
- Integer-only prices
- Bid/ask improvement rules
- Random tie-breaking for winner selection
- Chicago Rules for trade pricing
"""

from typing import Tuple
import numpy as np
from numpy.random import Generator, default_rng


class OrderBook:
    """
    AURORA double auction order book matching PeriodHistory.java.

    Maintains bid/ask state for a single period of trading.
    Arrays are 1-indexed (player 0 is sentinel, time 0 is initialization).
    """

    def __init__(
        self,
        num_buyers: int,
        num_sellers: int,
        num_times: int,
        min_price: int,
        max_price: int,
        rng_seed: int,
    ) -> None:
        """
        Initialize order book for a trading period.

        Args:
            num_buyers: Number of buyers (players will be 1..num_buyers)
            num_sellers: Number of sellers (players will be 1..num_sellers)
            num_times: Number of time steps in the period
            min_price: Minimum allowed price (inclusive)
            max_price: Maximum allowed price (inclusive)
            rng_seed: Random seed for reproducible tie-breaking
        """
        self.num_buyers = num_buyers
        self.num_sellers = num_sellers
        self.num_times = num_times
        self.min_price = min_price
        self.max_price = max_price

        # Random number generator for tie-breaking (Java lines 442-476)
        self.rng: Generator = default_rng(rng_seed)

        # Current time step (0 = initialization)
        self.current_time: int = 0

        # === ORDER BOOK STATE ===
        # bid[x][y] = bid of buyer x at time y (0 if none)
        # Shape: (num_buyers+1, num_times+1) - 1-indexed
        self.bids: np.ndarray = np.zeros((num_buyers + 1, num_times + 1), dtype=np.int32)

        # ask[x][y] = ask of seller x at time y (0 if none)
        self.asks: np.ndarray = np.zeros(
            (num_sellers + 1, num_times + 1), dtype=np.int32
        )

        # === MARKET STATE ===
        # High bidder/asker IDs and prices at each time
        self.high_bidder: np.ndarray = np.zeros(num_times + 1, dtype=np.int32)
        self.low_asker: np.ndarray = np.zeros(num_times + 1, dtype=np.int32)
        self.high_bid: np.ndarray = np.zeros(num_times + 1, dtype=np.int32)
        self.low_ask: np.ndarray = np.zeros(num_times + 1, dtype=np.int32)

        # === TRADE EXECUTION ===
        self.trade_price: np.ndarray = np.zeros(num_times + 1, dtype=np.int32)
        self.buyer_accepted: np.ndarray = np.zeros(num_times + 1, dtype=bool)
        self.seller_accepted: np.ndarray = np.zeros(num_times + 1, dtype=bool)

        # === POSITION TRACKING ===
        # Cumulative trades through each time step
        self.num_buys: np.ndarray = np.zeros(
            (num_buyers + 1, num_times + 1), dtype=np.int32
        )
        self.num_sells: np.ndarray = np.zeros(
            (num_sellers + 1, num_times + 1), dtype=np.int32
        )

        # === STATUS FLAGS ===
        # Track bid/ask status for each player (Java lines 54-67)
        # 0 = No new bid/ask, not current bidder/asker
        # 1 = No new bid/ask, still current bidder/asker
        # 2 = New bid/ask, now current bidder/asker
        # 3 = New bid/ask, beaten by another
        # 4 = New bid/ask, tied and lost random tie-break
        self.bid_status: np.ndarray = np.zeros(num_buyers + 1, dtype=np.int32)
        self.ask_status: np.ndarray = np.zeros(num_sellers + 1, dtype=np.int32)

        # === NEW ORDERS TRACKING ===
        self.num_bids: np.ndarray = np.zeros(num_times + 1, dtype=np.int32)
        self.num_asks: np.ndarray = np.zeros(num_times + 1, dtype=np.int32)

    def add_bid(self, bidder: int, price: int) -> bool:
        """
        Submit a bid from a buyer (PeriodHistory.java lines 152-159).

        Args:
            bidder: Buyer ID (1..num_buyers)
            price: Bid price (must be integer)

        Returns:
            True if bid was accepted, False if rejected
        """
        assert 1 <= bidder <= self.num_buyers, f"Invalid bidder ID: {bidder}"

        if not self._is_bid_valid(price):
            return False

        self.num_bids[self.current_time] += 1
        self.bid_status[bidder] = 3  # Mark as "new bid, beaten by another" initially
        self.bids[bidder, self.current_time] = price
        return True

    def add_ask(self, asker: int, price: int) -> bool:
        """
        Submit an ask from a seller (PeriodHistory.java lines 173-180).

        Args:
            asker: Seller ID (1..num_sellers)
            price: Ask price (must be integer)

        Returns:
            True if ask was accepted, False if rejected
        """
        assert 1 <= asker <= self.num_sellers, f"Invalid asker ID: {asker}"

        if not self._is_ask_valid(price):
            return False

        self.num_asks[self.current_time] += 1
        self.ask_status[asker] = 3  # Mark as "new ask, beaten by another" initially
        self.asks[asker, self.current_time] = price
        return True

    def _is_bid_valid(self, price: int) -> bool:
        """
        Validate bid according to AURORA rules (Java lines 162-170).

        Rules:
        1. Price must be in range [min_price, max_price]
        2. If previous time had a trade: any in-range bid is valid (book cleared)
        3. If no previous trade: bid must strictly improve high bid

        Args:
            price: Proposed bid price

        Returns:
            True if bid is valid
        """
        if not self._price_is_in_range(price):
            return False

        # If there was a trade last time, book is cleared -> any bid valid
        if self.current_time > 0 and self.trade_price[self.current_time - 1] > 0:
            return True

        # Must improve on current high bid
        if self.current_time > 0:
            return bool(price > self.high_bid[self.current_time - 1])

        # First time: any in-range bid is valid
        return True

    def _is_ask_valid(self, price: int) -> bool:
        """
        Validate ask according to AURORA rules (Java lines 183-195).

        Symmetric to bid validation: ask must beat current low ask.

        Args:
            price: Proposed ask price

        Returns:
            True if ask is valid
        """
        if not self._price_is_in_range(price):
            return False

        # If there was a trade last time, book is cleared -> any ask valid
        if self.current_time > 0 and self.trade_price[self.current_time - 1] > 0:
            return True

        # Must improve on current low ask
        if self.current_time > 0 and self.low_ask[self.current_time - 1] > 0:
            return bool(price < self.low_ask[self.current_time - 1])

        # First time or no previous ask: any in-range ask is valid
        return True

    def _price_is_in_range(self, price: int) -> bool:
        """
        Check if price is within allowed bounds (Java lines 198-204).

        Args:
            price: Price to check

        Returns:
            True if min_price <= price <= max_price
        """
        return self.min_price <= price <= self.max_price

    def determine_winners(self) -> Tuple[int, int]:
        """
        Determine high bidder and low asker (Java lines 420-512).

        Three-step process:
        1. Find best bid and best ask prices
        2. Select high bidder (with random tie-breaking)
        3. Select low asker (with random tie-breaking)

        Returns:
            Tuple of (high_bidder_id, low_asker_id). IDs are 0 if none.
        """
        self._determine_high_bid_low_ask()
        self._determine_high_bidder()
        self._determine_low_asker()

        return self.high_bidder[self.current_time], self.low_asker[self.current_time]

    def _determine_high_bid_low_ask(self) -> None:
        """
        Find best bid and ask prices across all players (Java lines 426-439).

        Sets:
            self.high_bid[current_time]: Highest bid price (0 if none)
            self.low_ask[current_time]: Lowest ask price (0 if none)
        """
        t = self.current_time

        # Find highest bid
        self.high_bid[t] = 0
        for buyer in range(1, self.num_buyers + 1):
            if self.bids[buyer, t] > self.high_bid[t]:
                self.high_bid[t] = self.bids[buyer, t]

        # Find lowest ask (ignore 0 = no ask)
        self.low_ask[t] = 0
        for seller in range(1, self.num_sellers + 1):
            ask_price = self.asks[seller, t]
            if ask_price > 0:
                if self.low_ask[t] == 0 or ask_price < self.low_ask[t]:
                    self.low_ask[t] = ask_price

    def _determine_high_bidder(self) -> None:
        """
        Select high bidder with random tie-breaking (Java lines 442-476).

        If multiple buyers have the high bid, randomly select one.

        Sets:
            self.high_bidder[current_time]: ID of winning buyer (0 if none)
            self.bid_status[]: Status codes for all buyers
        """
        t = self.current_time
        high_bid_price = self.high_bid[t]

        if high_bid_price == 0:
            self.high_bidder[t] = 0
            return

        # Find all bidders at the high bid price
        tied_bidders = []
        for buyer in range(1, self.num_buyers + 1):
            if self.bids[buyer, t] == high_bid_price:
                tied_bidders.append(buyer)
                if self.bid_status[buyer] == 0:
                    self.bid_status[buyer] = 1  # Still current bidder
                else:
                    self.bid_status[buyer] = 4  # Tied, might lose tie-break

        # Random selection if multiple tied bidders
        if len(tied_bidders) > 1:
            chosen_index = self.rng.integers(0, len(tied_bidders))
            winner = tied_bidders[chosen_index]
        else:
            winner = tied_bidders[0]

        self.high_bidder[t] = winner

        # Winner gets status=2 (new bid, now current)
        if self.bid_status[winner] == 4:
            self.bid_status[winner] = 2

    def _determine_low_asker(self) -> None:
        """
        Select low asker with random tie-breaking (Java lines 478-512).

        Symmetric to high bidder selection.

        Sets:
            self.low_asker[current_time]: ID of winning seller (0 if none)
            self.ask_status[]: Status codes for all sellers
        """
        t = self.current_time
        low_ask_price = self.low_ask[t]

        if low_ask_price == 0:
            self.low_asker[t] = 0
            return

        # Find all askers at the low ask price
        tied_askers = []
        for seller in range(1, self.num_sellers + 1):
            if self.asks[seller, t] == low_ask_price:
                tied_askers.append(seller)
                if self.ask_status[seller] == 0:
                    self.ask_status[seller] = 1  # Still current asker
                else:
                    self.ask_status[seller] = 4  # Tied, might lose tie-break

        # Random selection if multiple tied askers
        if len(tied_askers) > 1:
            chosen_index = self.rng.integers(0, len(tied_askers))
            winner = tied_askers[chosen_index]
        else:
            winner = tied_askers[0]

        self.low_asker[t] = winner

        # Winner gets status=2 (new ask, now current)
        if self.ask_status[winner] == 4:
            self.ask_status[winner] = 2

    def execute_trade(self, buyer_accepts: bool, seller_accepts: bool) -> int:
        """
        Execute trade with Chicago Rules pricing (Java lines 259-282).

        Pricing rules:
        - Only buyer accepts → trade at ask price (seller's price)
        - Only seller accepts → trade at bid price (buyer's price)
        - Both accept → 50/50 random between bid and ask
        - Neither accepts → no trade (price = 0)

        Args:
            buyer_accepts: True if high bidder accepts the low ask
            seller_accepts: True if low asker accepts the high bid

        Returns:
            Trade price (0 if no trade occurred)
        """
        t = self.current_time
        self.buyer_accepted[t] = buyer_accepts
        self.seller_accepted[t] = seller_accepts

        price = 0

        if buyer_accepts:
            if seller_accepts:
                # BOTH ACCEPT: Random 50/50 between bid and ask
                r = self.rng.integers(0, 2)
                if r == 1:
                    price = self.low_ask[t]
                else:
                    price = self.high_bid[t]
            else:
                # Only buyer accepts: trade at ask price
                price = self.low_ask[t]
        elif seller_accepts:
            # Only seller accepts: trade at bid price
            price = self.high_bid[t]

        # Record trade
        if buyer_accepts or seller_accepts:
            self._add_trade(price)

        return price

    def _add_trade(self, price: int) -> None:
        """
        Record a completed trade (Java lines 283-300).

        Updates position tracking for buyer and seller.

        Args:
            price: Trade price
        """
        t = self.current_time
        self.trade_price[t] = price

        # Update position counters
        buyer_id = self.high_bidder[t]
        seller_id = self.low_asker[t]

        if buyer_id > 0:
            # Copy previous position
            if t > 0:
                self.num_buys[buyer_id, t] = self.num_buys[buyer_id, t - 1]
            self.num_buys[buyer_id, t] += 1

        if seller_id > 0:
            # Copy previous position
            if t > 0:
                self.num_sells[seller_id, t] = self.num_sells[seller_id, t - 1]
            self.num_sells[seller_id, t] += 1

    def increment_time(self) -> bool:
        """
        Advance to next time step (Java lines 129-149).

        If no trade occurred, standing orders carry over.
        If trade occurred, order book is cleared.

        Returns:
            True if advanced successfully, False if at end of period
        """
        self.current_time += 1

        if self.current_time > self.num_times:
            return False

        t = self.current_time

        # IF NO TRADE: bids and asks carry over
        if t > 0 and self.trade_price[t - 1] == 0:
            for buyer in range(1, self.num_buyers + 1):
                self.bids[buyer, t] = self.bids[buyer, t - 1]
            for seller in range(1, self.num_sellers + 1):
                self.asks[seller, t] = self.asks[seller, t - 1]

            # Carry over position counters
            for buyer in range(1, self.num_buyers + 1):
                self.num_buys[buyer, t] = self.num_buys[buyer, t - 1]
            for seller in range(1, self.num_sellers + 1):
                self.num_sells[seller, t] = self.num_sells[seller, t - 1]
        else:
            # Trade occurred: book is cleared (arrays already initialized to 0)
            # Position counters still carry over
            if t > 0:
                for buyer in range(1, self.num_buyers + 1):
                    self.num_buys[buyer, t] = self.num_buys[buyer, t - 1]
                for seller in range(1, self.num_sellers + 1):
                    self.num_sells[seller, t] = self.num_sells[seller, t - 1]

        # Reset status flags
        self.bid_status[:] = 0
        self.ask_status[:] = 0

        return True
