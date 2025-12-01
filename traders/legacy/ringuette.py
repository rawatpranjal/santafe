"""
Ringuette Agent.

A "wait in the weeds and snipe" strategy for the double auction. It stays quiet
while others negotiate, and only jumps in when there's a very good deal, slightly
overbidding everyone else to "steal the deal."

Key behavioral characteristics:
1. Computes SPAN = max_token - min_token + 10 (profit margin = SPAN/5)
2. Waits until bid and ask are within margin AND can profit at least margin
3. Jumps in with randomized overbid: CASK + 1 + 0.05 * U[0,1] * SPAN
4. Falls back to Skeleton logic if time running out or inactive too long
5. Early in period, may do simple incremental bidding

Tournament Performance: 2nd place in 1993 Santa Fe Double Auction tournament.
"""

from typing import Any

import numpy as np

from traders.base import Agent


class Ringuette(Agent):
    """
    Ringuette trading agent (Background Sniper).

    Strategy:
    - Wait until spread is tight (CBID >= CASK - SPAN/5)
    - AND can profit at least SPAN/5 (token > CASK + SPAN/5)
    - Then jump in with randomized overbid to steal the deal
    - Falls back to Skeleton behavior if time pressure or inactive
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
        **kwargs: Any,
    ) -> None:
        """
        Initialize Ringuette agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations
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

        # Compute SPAN once at initialization
        # SPAN = max_token - min_token + 10 (keeps it nonzero for single token)
        self.span = max(valuations) - min(valuations) + 10
        self.margin = self.span / 5.0

        # State tracking
        self.current_time = 0
        self.current_bid = 0
        self.current_ask = 0
        self.current_bidder = 0
        self.current_asker = 0
        self.nobidask = 0
        self.last_trade_time = 0
        self.my_last_trade_time = 0

    def _calculate_skeleton_weight(self) -> float:
        """
        Calculate weight for Skeleton subroutine.
        Weight on CBID+1 increases from ~0.3-0.4 early to ~0.7-0.8 late.
        """
        if self.num_times <= 0:
            return 0.5
        time_fraction = self.current_time / self.num_times
        # Linear interpolation: 0.35 at t=0, 0.75 at t=num_times
        return 0.35 + 0.40 * time_fraction

    def _should_fallback(self) -> bool:
        """
        Determine if we should fall back to Skeleton behavior.
        Triggers: time running out OR been inactive too long.
        """
        remaining = self.num_times - self.current_time
        time_since_trade = self.current_time - self.last_trade_time
        time_since_my_trade = self.current_time - self.my_last_trade_time

        # Time running out: less than 20% of period remaining
        if remaining < self.num_times * 0.2:
            return True

        # Inactive too long: no trade for 30% of remaining time
        if time_since_trade > remaining * 0.3:
            return True

        # Haven't traded in a while personally
        if time_since_my_trade > self.num_times * 0.5:
            return True

        return False

    def _skeleton_bid(self, token_val: int) -> int:
        """
        Skeleton subroutine for buyer fallback.

        If no current bid: bid = min(CASK, least_token) - 0.3 * SPAN
        If current bid: weighted average of (CBID+1) and MOST
        """
        least_token = min(self.valuations)

        if self.current_bid == 0:
            # No current bid
            if self.current_ask > 0:
                base = min(self.current_ask, least_token)
            else:
                base = least_token
            bid = int(base - 0.3 * self.span)
            return max(self.price_min, bid)
        else:
            # Current bid exists - weighted average
            most = token_val - 1
            if self.current_ask > 0 and self.current_ask < most:
                most = self.current_ask

            if most <= self.current_bid:
                return 0  # Can't improve

            weight = self._calculate_skeleton_weight()
            bid = int(weight * (self.current_bid + 1) + (1 - weight) * most)
            return max(self.price_min, min(bid, token_val - 1))

    def _skeleton_ask(self, token_val: int) -> int:
        """
        Skeleton subroutine for seller fallback.

        Symmetric to buyer version.
        """
        highest_token = max(self.valuations)

        if self.current_ask == 0:
            # No current ask
            if self.current_bid > 0:
                base = max(self.current_bid, highest_token)
            else:
                base = highest_token
            ask = int(base + 0.3 * self.span)
            return min(self.price_max, ask)
        else:
            # Current ask exists - weighted average
            least = token_val + 1
            if self.current_bid > 0 and self.current_bid > least:
                least = self.current_bid

            if least >= self.current_ask:
                return 0  # Can't improve

            weight = self._calculate_skeleton_weight()
            ask = int(weight * (self.current_ask - 1) + (1 - weight) * least)
            return min(self.price_max, max(ask, token_val + 1))

    def bid_ask(self, time: int, nobidask: int) -> None:
        """Prepare for bid/ask phase."""
        self.current_time = time
        self.nobidask = nobidask
        self.has_responded = False

    def bid_ask_response(self) -> int:
        """
        Main bidding logic implementing Ringuette strategy.
        """
        self.has_responded = True

        if self.nobidask > 0:
            return 0

        if self.num_trades >= self.num_tokens:
            return 0

        if self.is_buyer:
            return self._player_request_bid()
        else:
            return self._player_request_ask()

    def _player_request_bid(self) -> int:
        """
        Buyer bidding logic.

        1. Main entry: tight spread AND profitable -> overbid
        2. Early incremental: CBID < NTIMES/4 -> bid CBID+1
        3. Time pressure -> Skeleton fallback
        4. Otherwise -> stay silent
        """
        token_val = self.valuations[self.num_trades]

        # === MAIN ENTRY RULE ===
        # Jump in when spread is tight AND we can profit at least margin
        if (
            self.current_bid > 0
            and self.current_ask > 0
            and self.current_bid >= self.current_ask - self.margin
            and token_val > self.current_ask + self.margin
        ):
            # Overbid the ask: CASK + 1 + 0.05 * U[0,1] * SPAN
            random_component = 0.05 * self.rng.random() * self.span
            overbid = int(self.current_ask + 1 + random_component)
            # Cap at profitable level
            return max(self.price_min, min(overbid, token_val - 1))

        # === EARLY INCREMENTAL ===
        # If bid is still small relative to period length, just increment
        if self.current_bid > 0 and self.current_bid < self.num_times / 4 and self.current_ask > 0:
            new_bid = self.current_bid + 1
            if new_bid < token_val:
                return new_bid

        # === TIME PRESSURE FALLBACK ===
        if self._should_fallback():
            return self._skeleton_bid(token_val)

        # === STAY SILENT ===
        return 0

    def _player_request_ask(self) -> int:
        """
        Seller asking logic (symmetric to buyer).

        1. Main entry: tight spread AND profitable -> underbid
        2. Early incremental: CASK > NTIMES*3/4 -> ask CASK-1
        3. Time pressure -> Skeleton fallback
        4. Otherwise -> stay silent
        """
        token_val = self.valuations[self.num_trades]

        # === MAIN ENTRY RULE ===
        # Jump in when spread is tight AND we can profit at least margin
        if (
            self.current_bid > 0
            and self.current_ask > 0
            and self.current_bid >= self.current_ask - self.margin
            and self.current_bid > token_val + self.margin
        ):
            # Underbid the bid: CBID - 1 - 0.05 * U[0,1] * SPAN
            random_component = 0.05 * self.rng.random() * self.span
            underbid = int(self.current_bid - 1 - random_component)
            # Floor at profitable level
            return min(self.price_max, max(underbid, token_val + 1))

        # === EARLY INCREMENTAL ===
        # If ask is still high relative to period length, just decrement
        if (
            self.current_ask > 0
            and self.current_ask > self.price_max - self.num_times / 4
            and self.current_bid > 0
        ):
            new_ask = self.current_ask - 1
            if new_ask > token_val:
                return new_ask

        # === TIME PRESSURE FALLBACK ===
        if self._should_fallback():
            return self._skeleton_ask(token_val)

        # === STAY SILENT ===
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
        Accept trade if profitable and we are the winner.
        Same logic as Kaplan.
        """
        self.has_responded = True

        if self.num_trades >= self.num_tokens:
            return False

        token_val = self.valuations[self.num_trades]

        if self.is_buyer:
            # Don't buy at a loss
            if self.current_ask > 0 and token_val <= self.current_ask:
                return False
            # Accept if we're high bidder and spread is crossed
            if (
                self.player_id == self.current_bidder
                and self.current_bid > 0
                and self.current_bid >= self.current_ask
            ):
                return True
        else:
            # Don't sell at a loss
            if self.current_bid > 0 and self.current_bid <= token_val:
                return False
            # Accept if we're low asker and spread is crossed
            if (
                self.player_id == self.current_asker
                and self.current_ask > 0
                and self.current_ask <= self.current_bid
            ):
                return True

        return False

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

        # Track trade times for fallback logic
        if trade_type != 0:  # A trade occurred
            self.last_trade_time = self.current_time
            if status == 1:  # I traded
                self.my_last_trade_time = self.current_time

    def start_period(self, period_number: int) -> None:
        """Reset period state."""
        super().start_period(period_number)
        self.last_trade_time = 0
        self.my_last_trade_time = 0
        self.current_time = 0
