"""
Truth Teller Strategy.

From the 1993 Santa Fe Tournament (control program).
"A 'truthteller' program which always places bids and asks equal to its true token valuations."

Level-1, nonadaptive, deterministic strategy:
- Bids/asks at exactly the true valuation (no markup/markdown)
- Accepts only strictly profitable trades (no breakeven)
- No learning, no prediction, no randomization
"""

from traders.base import Agent


class TruthTeller(Agent):
    """
    Truth Teller - bids/asks exactly at reservation price.

    Buyers bid their valuation, sellers ask their cost.
    This is the theoretical baseline of zero strategic behavior.

    Accepts only STRICTLY profitable trades (no breakeven).
    Must be holding the current bid/ask to accept.
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
        **kwargs,
    ) -> None:
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min = price_min
        self.price_max = price_max

        # State tracking for buy/sell phase
        self.current_bid = 0
        self.current_ask = 0
        self.current_bidder = 0
        self.current_asker = 0
        self.last_nobuysell = 0
        self.nobidask = 0

    def bid_ask(self, time: int, nobidask: int) -> None:
        """Prepare for bid/ask phase."""
        self.nobidask = nobidask
        self.has_responded = False

    def bid_ask_response(self) -> int:
        """
        Return the reservation price for the current token.

        For buyers: bid = valuation (maximum willingness to pay)
        For sellers: ask = cost (minimum willingness to accept)

        Returns:
            The reservation price, or 0 if no tokens left
        """
        self.has_responded = True

        # Check nobidask flag
        if self.nobidask > 0:
            return 0

        if self.num_trades >= self.num_tokens:
            return 0

        # Simply return the reservation price - no markup, no strategy
        reservation = self.valuations[self.num_trades]

        # Clamp to valid price range
        return max(self.price_min, min(self.price_max, reservation))

    def buy_sell(
        self,
        time: int,
        nobuysell: int,
        high_bid: int,
        low_ask: int,
        high_bidder: int,
        low_asker: int,
    ) -> None:
        """Store market state for buy/sell decision."""
        self.has_responded = False
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker
        self.last_nobuysell = nobuysell

    def buy_sell_response(self) -> bool:
        """
        Accept trade only if STRICTLY profitable.

        Truth Teller accepts only trades where profit > 0 (no breakeven).
        Must be holding the current bid/ask to accept.
        """
        self.has_responded = True

        # Check if we are allowed to trade
        if self.last_nobuysell > 0:
            return False

        # Check we have tokens left
        if self.num_trades >= self.num_tokens:
            return False

        valuation = self.valuations[self.num_trades]

        if self.is_buyer:
            # Accept if we're the high bidder and ask < valuation (STRICTLY profitable)
            if self.player_id == self.current_bidder:
                if self.current_ask > 0 and self.current_ask < valuation:
                    if self.current_bid >= self.current_ask:
                        return True
        else:
            # Accept if we're the low asker and bid > cost (STRICTLY profitable)
            if self.player_id == self.current_asker:
                if self.current_bid > 0 and self.current_bid > valuation:
                    if self.current_ask <= self.current_bid:
                        return True

        return False

    def auction_history(
        self, best_bid: int, best_ask: int, trade_occurred: bool, trade_price: int
    ) -> None:
        """Truth Teller doesn't learn from history - it always tells the truth."""
        pass

    def end_period(self) -> None:
        """Reset for new period."""
        pass
