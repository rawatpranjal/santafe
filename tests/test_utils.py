"""
Test utilities for integration testing.

Provides programmed agent strategies and helper functions for setting up
and running market sessions.
"""

from typing import Any

import numpy as np

from traders.base import Agent


class ProgrammedAgent(Agent):
    """
    Test agent with programmable strategy.

    Supports different trading strategies:
    - "aggressive": Always bid/ask aggressively, always accept
    - "patient": Wait until late in session, then bid/ask conservatively
    - "rational": Only make profitable bids/asks, accept profitable trades
    - "random": Random bids/asks within range
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        strategy: str = "rational",
        seed: int | None = None,
    ) -> None:
        """
        Initialize a programmed agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations (buyer values or seller costs)
            strategy: Trading strategy ("aggressive", "patient", "rational", "random")
            seed: Random seed for "random" strategy
        """
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)

        # State tracking
        self.current_time = 0
        self.num_times_total = 100  # Will be set when market starts
        self.last_nobidask = 0
        self.last_high_bid = 0
        self.last_low_ask = 0

    def bid_ask(self, time: int, nobidask: int) -> None:
        """Record current time and nobidask flag."""
        self.has_responded = False
        self.current_time = time
        self.last_nobidask = nobidask

    def bid_ask_response(self) -> int:
        """Return bid/ask based on strategy."""
        self.has_responded = True

        # Can't bid/ask if nobidask flag is set (token exhausted)
        if self.last_nobidask == 1:
            return 0

        # Can't bid/ask if no tokens left (redundant check)
        if not self.can_trade():
            return 0

        current_val = self.get_current_valuation()

        if self.strategy == "aggressive":
            return self._aggressive_bid_ask(current_val)
        elif self.strategy == "patient":
            return self._patient_bid_ask(current_val)
        elif self.strategy == "rational":
            return self._rational_bid_ask(current_val)
        elif self.strategy == "random":
            return self._random_bid_ask(current_val)
        else:
            # Default: bid at valuation
            return current_val

    def buy_sell(
        self,
        time: int,
        nobuysell: int,
        high_bid: int,
        low_ask: int,
        high_bidder: int,
        low_asker: int,
    ) -> None:
        """Record market state."""
        self.has_responded = False
        self.current_time = time
        self.last_high_bid = high_bid
        self.last_low_ask = low_ask

    def buy_sell_response(self) -> bool:
        """Return accept/reject based on strategy."""
        self.has_responded = True

        current_val = self.get_current_valuation()

        if self.strategy == "aggressive":
            return self._aggressive_accept(current_val)
        elif self.strategy == "patient":
            return self._patient_accept(current_val)
        elif self.strategy == "rational":
            return self._rational_accept(current_val)
        elif self.strategy == "random":
            return self._random_accept()
        else:
            # Default: accept if profitable
            return self._rational_accept(current_val)

    # =========================================================================
    # STRATEGY IMPLEMENTATIONS
    # =========================================================================

    def _aggressive_bid_ask(self, valuation: int) -> int:
        """Aggressive: Bid/ask close to valuation."""
        if self.is_buyer:
            # Bid high (within 5% of valuation)
            margin = int(valuation * 0.05)
            return valuation - self.rng.integers(0, max(1, margin))
        else:
            # Ask low (within 5% of cost)
            margin = int(valuation * 0.05)
            return valuation + self.rng.integers(0, max(1, margin))

    def _patient_bid_ask(self, valuation: int) -> int:
        """Patient: Wait until late, then bid conservatively."""
        # Only bid/ask in last 30% of timesteps
        progress = self.current_time / max(1, self.num_times_total)

        if progress < 0.7:
            # Wait - don't submit
            return 0

        # Bid/ask conservatively (within 20% of valuation)
        if self.is_buyer:
            margin = int(valuation * 0.2)
            return valuation - self.rng.integers(0, max(1, margin))
        else:
            margin = int(valuation * 0.2)
            return valuation + self.rng.integers(0, max(1, margin))

    def _rational_bid_ask(self, valuation: int) -> int:
        """Rational: Bid/ask at exact valuation."""
        return valuation

    def _random_bid_ask(self, valuation: int) -> int:
        """Random: Bid/ask randomly within ±20% of valuation."""
        margin = int(valuation * 0.2)
        offset = self.rng.integers(-margin, margin + 1)
        return max(0, valuation + offset)

    def _aggressive_accept(self, valuation: int) -> bool:
        """Aggressive: Always accept."""
        return True

    def _patient_accept(self, valuation: int) -> bool:
        """Patient: Only accept late in session if profitable."""
        progress = self.current_time / max(1, self.num_times_total)

        if progress < 0.8:
            return False

        # Accept if profitable
        return self._rational_accept(valuation)

    def _rational_accept(self, valuation: int) -> bool:
        """Rational: Accept only if profitable."""
        if self.is_buyer:
            # Accept if low ask <= valuation
            return self.last_low_ask > 0 and self.last_low_ask <= valuation
        else:
            # Accept if high bid >= cost
            return self.last_high_bid > 0 and self.last_high_bid >= valuation

    def _random_accept(self) -> bool:
        """Random: 50% chance to accept."""
        return bool(self.rng.integers(0, 2))


def create_agent(
    player_id: int,
    is_buyer: bool,
    valuations: list[int],
    strategy: str = "rational",
    seed: int | None = None,
) -> ProgrammedAgent:
    """
    Convenience function to create a programmed agent.

    Args:
        player_id: Agent ID (1-indexed)
        is_buyer: True for buyer, False for seller
        valuations: Private valuations
        strategy: Trading strategy ("aggressive", "patient", "rational", "random")
        seed: Random seed for random strategy

    Returns:
        Configured ProgrammedAgent
    """
    return ProgrammedAgent(
        player_id=player_id,
        is_buyer=is_buyer,
        num_tokens=len(valuations),
        valuations=valuations,
        strategy=strategy,
        seed=seed,
    )
