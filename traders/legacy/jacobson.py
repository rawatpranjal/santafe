"""
Jacobson agent for the Santa Fe double auction.

Based on SRobotJacobson.java from the 1993 Santa Fe tournament.
Uses weighted equilibrium estimation with exponential confidence.
"""

from typing import Any, Optional
import random
import math
from traders.base import Agent


class Jacobson(Agent):
    """
    Jacobson agent - equilibrium estimation with confidence weighting.

    Strategy:
    - Tracks round-level price statistics (weighted sum and weights)
    - Estimates equilibrium from weighted price history
    - Uses exponential confidence function (0.01^(1/weight))
    - Makes convex combination of current prices and equilibrium estimate
    - Complex probabilistic acceptance with gap analysis

    Java source: SRobotJacobson.java
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        price_min: int = 0,
        price_max: int = 1000,
        num_times: int = 100,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize Jacobson agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations
            price_min: Minimum allowed price (default 0)
            price_max: Maximum allowed price (default 1000)
            num_times: Number of time steps per period (default 100)
            seed: Random seed for reproducibility
            **kwargs: Hyperparameter overrides (see below)

        Hyperparameters (passed via kwargs):
            bid_ask_offset (float): Price adjustment magnitude for bid/ask updates
                Default: 1.0, Range: [0.5, 3.0]
                Higher = more aggressive price improvement
            trade_weight_multiplier (float): Scaling factor for trade weight in equilibrium
                Default: 2.0, Range: [1.0, 4.0]
                Formula: weight = period + num_trades * multiplier
                Higher = faster learning from recent trades
            confidence_base (float): Base for exponential confidence function
                Default: 0.01, Range: [0.001, 0.1]
                Formula: confidence = base^(1/weight)
                Lower = slower confidence growth (more conservative)
            time_pressure_multiplier (float): Urgency scaling for end-of-period decisions
                Default: 2.0, Range: [1.0, 4.0]
                Formula: urgency = gap_rate * remaining_tokens * multiplier
                Higher = more aggressive near deadline

        Note:
            Original defaults are from SRobotJacobson.java (1993 Santa Fe Tournament).
            Calibrated defaults may be different after hyperparameter search.
        """
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min_limit = price_min
        self.price_max_limit = price_max
        self.num_times = num_times
        self.rng = random.Random(seed)

        # Hyperparameter 1: Bid/Ask Offset
        # Controls price aggressiveness (±1 in original)
        self.bid_ask_offset = kwargs.get('bid_ask_offset', 1.0)

        # Hyperparameter 2: Trade Weight Multiplier
        # Controls learning speed (×2 in original: weight = period + trades * 2)
        self.trade_weight_multiplier = kwargs.get('trade_weight_multiplier', 2.0)

        # Hyperparameter 3: Confidence Base
        # Controls confidence growth rate (0.01 in original: conf = 0.01^(1/weight))
        self.confidence_base = kwargs.get('confidence_base', 0.01)

        # Hyperparameter 4: Time Pressure Multiplier
        # Controls urgency scaling (×2 in original: urgency ∝ remaining_tokens * 2)
        self.time_pressure_multiplier = kwargs.get('time_pressure_multiplier', 2.0)

        # Round-level equilibrium tracking
        self.roundpricesum: float = 0.0
        self.roundweight: float = 0.0
        self.lastgap: int = 10000000

        # Current step state (set by bid_ask/buy_sell notifications)
        self.current_time = 0
        self.current_bid = 0
        self.current_ask = 0
        self.current_bidder = 0
        self.current_asker = 0
        self.current_period = 1  # Track period number for weighting

    # =========================================================================
    # REQUIRED ABSTRACT METHODS (Base.Agent Interface)
    # =========================================================================

    def bid_ask(self, time: int, nobidask: int) -> None:
        """
        Notification: Time to submit a bid (buyers) or ask (sellers).

        Args:
            time: Current time step (1-indexed)
            nobidask: Flag indicating trading restrictions
        """
        self.current_time = time
        self.has_responded = False

    def bid_ask_response(self) -> int:
        """
        Return the bid (buyers) or ask (sellers) for this time step.

        Returns:
            Bid price (buyers) or ask price (sellers)
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
        Notification: Time to accept or reject a trade opportunity.

        Args:
            time: Current time step (1-indexed)
            nobuysell: Bit flags indicating restrictions
            high_bid: Current highest bid price (0 if none)
            low_ask: Current lowest ask price (0 if none)
            high_bidder: ID of high bidder (0 if none)
            low_asker: ID of low asker (0 if none)
        """
        self.current_time = time
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker
        self.has_responded = False

    def buy_sell_response(self) -> bool:
        """
        Return whether to accept the trade opportunity.

        Returns:
            True: Accept the trade (buy at low_ask or sell at high_bid)
            False: Reject (wait for better opportunity)
        """
        self.has_responded = True

        if self.is_buyer:
            return bool(self._player_request_buy())
        else:
            return bool(self._player_want_to_sell())

    # =========================================================================
    # LIFECYCLE METHODS (Override from Base.Agent)
    # =========================================================================

    def start_period(self, period_number: int) -> None:
        """
        Called at the start of a trading period.

        Note: In Santa Fe rules, equilibrium changes between ROUNDS,
        not PERIODS. Round-level state (roundpricesum, roundweight)
        persists across periods within a round.

        Args:
            period_number: Current period number (1-indexed)
        """
        super().start_period(period_number)
        self.current_period = period_number

        # If this is the first period, reset round-level tracking
        # (In practice, tournaments call this at the start of each round)
        if period_number == 1:
            self.lastgap = 10000000
            self.roundpricesum = 0.0
            self.roundweight = 0.0

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
        """
        Notification: Results of the buy/sell stage.

        Update equilibrium estimate based on trade outcome.
        Java source: SRobotJacobson.java lines 39-50 (playerBuySellEnd)

        Args:
            status: This agent's trade status
            trade_price: Execution price (0 if no trade)
            trade_type: How trade was determined (0 = no trade)
            high_bid: Current highest bid after trade
            high_bidder: ID of high bidder after trade
            low_ask: Current lowest ask after trade
            low_asker: ID of low asker after trade
        """
        super().buy_sell_result(status, trade_price, trade_type,
                               high_bid, high_bidder, low_ask, low_asker)

        # Update current bid/ask for next iteration
        self.current_bid = high_bid
        self.current_ask = low_ask

        if trade_type != 0:  # Trade occurred (bstype != 0 in Java)
            # Weight increases with period number and number of trades
            # Java: weight = p + ntrades * 2 (now parameterized)
            weight = self.current_period + self.num_trades * self.trade_weight_multiplier
            self.roundpricesum += trade_price * weight
            self.roundweight += weight
            self.lastgap = 10000000
        else:
            # No trade - track the spread for gap analysis
            self.lastgap = (low_ask - high_bid) if (low_ask > 0 and high_bid > 0) else 10000000

    # =========================================================================
    # PRIVATE HELPER METHODS (Equilibrium Estimation)
    # =========================================================================

    def _eqest(self) -> float:
        """
        Estimate equilibrium price from weighted history.
        Java lines 141-146

        Returns:
            Estimated equilibrium price
        """
        if self.roundweight == 0.0:
            # Use worst-case token as default
            return float(self.valuations[self.num_tokens - 1])
        else:
            return self.roundpricesum / self.roundweight

    def _eqconf(self) -> float:
        """
        Calculate confidence in equilibrium estimate (0.0 to 1.0).
        Java lines 148-153

        Returns:
            Confidence level (higher weight → higher confidence)
        """
        if self.roundweight == 0.0:
            return 0.0
        # Exponential confidence: approaches 1.0 as weight increases (now parameterized)
        return math.pow(self.confidence_base, 1.0 / self.roundweight)

    # =========================================================================
    # PRIVATE HELPER METHODS (Bidding and Asking Logic)
    # =========================================================================

    def _player_request_bid(self) -> int:
        """
        Submit a bid using equilibrium estimation.
        Java lines 52-69

        Returns:
            Bid price (0 if unable to bid)
        """
        if self.num_trades >= self.num_tokens:
            return 0

        # Get current or default bid
        old_bid = float(self.current_bid if self.current_bid > 0 else self.price_min_limit)

        # Get equilibrium estimate and confidence
        est = self._eqest()
        conf = self._eqconf()

        # Convex combination of old bid and equilibrium estimate (now parameterized)
        new_bid = old_bid * (1.0 - conf) + est * conf + self.bid_ask_offset

        # Don't bid above our valuation
        if new_bid >= self.valuations[self.num_trades]:
            return 0

        return int(new_bid)

    def _player_request_ask(self) -> int:
        """
        Submit an ask using equilibrium estimation.
        Java lines 71-88

        Returns:
            Ask price (0 if unable to ask)
        """
        if self.num_trades >= self.num_tokens:
            return 0

        # Get current or default ask
        old_ask = float(self.current_ask if self.current_ask > 0 else self.price_max_limit)

        # Get equilibrium estimate and confidence
        est = self._eqest()
        conf = self._eqconf()

        # Convex combination of old ask and equilibrium estimate (now parameterized)
        new_ask = old_ask * (1.0 - conf) + est * conf - self.bid_ask_offset

        # Don't ask below our cost
        if new_ask <= self.valuations[self.num_trades]:
            return 0

        return int(new_ask)

    # =========================================================================
    # PRIVATE HELPER METHODS (Buy/Sell Decision Logic)
    # =========================================================================

    def _player_request_buy(self) -> int:
        """
        Decide whether to buy with complex gap analysis.
        Java lines 90-113

        Returns:
            1 to accept, 0 to reject
        """
        if self.num_trades >= self.num_tokens:
            return 0

        # Must be current bidder to buy
        if self.player_id != self.current_bidder:
            return 0

        profit = self.valuations[self.num_trades] - self.current_ask
        gap = self.current_ask - self.current_bid if self.current_bid > 0 else self.current_ask

        # No profit, don't buy
        if profit <= 0.0:
            return 0

        # Spread has closed, accept
        if gap <= 0.0:
            return 1

        # Complex time pressure and gap closing analysis (now parameterized)
        # Check if gap isn't closing or if time is running out
        if int(gap) == self.lastgap or (
            gap / max(self.lastgap - gap, 0.01) * (self.num_tokens - self.num_trades) * self.time_pressure_multiplier
            + self.current_time > self.num_times
        ):
            # Probabilistic acceptance based on profit/gap ratio
            if profit / (profit + gap) > self.rng.random():
                return 1

        return 0

    def _player_want_to_sell(self) -> int:
        """
        Decide whether to sell with complex gap analysis.
        Java lines 115-139

        Returns:
            1 to accept, 0 to reject
        """
        if self.num_trades >= self.num_tokens:
            return 0

        # Must be current asker to sell
        if self.player_id != self.current_asker:
            return 0

        profit = self.current_bid - self.valuations[self.num_trades]
        gap = self.current_ask - self.current_bid if self.current_ask > 0 else -self.current_bid

        # No profit, don't sell
        if profit <= 0.0:
            return 0

        # Spread has closed, accept
        if gap <= 0.0:
            return 1

        # Complex time pressure and gap closing analysis (now parameterized)
        if int(gap) == self.lastgap or (
            gap / max(self.lastgap - gap, 0.01) * (self.num_tokens - self.num_trades) * self.time_pressure_multiplier
            + self.current_time > self.num_times
        ):
            # Probabilistic acceptance based on profit/gap ratio
            if profit / (profit + gap) > self.rng.random():
                return 1

        return 0
