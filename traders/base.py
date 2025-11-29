"""
Abstract base class for trading agents in the Santa Fe Double Auction.

This module defines the Agent interface that all trading strategies must implement.
It ports the core logic from SPlayer.java in the original 1993 tournament code.

The AURORA protocol requires agents to respond to two types of notifications:
1. BID/ASK Stage: Submit a bid (buyers) or ask (sellers)
2. BUY/SELL Stage: Accept or reject the current market opportunity

All responses are synchronized - the market waits for all agents before proceeding.
"""

from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    """
    Abstract base class for all trading agents.

    Ports the interface from SPlayer.java (lines 24-99).
    All subclasses must implement the decision-making methods.

    Attributes:
        player_id: Unique identifier (1-indexed)
        is_buyer: True if buyer, False if seller
        num_tokens: Total number of units to trade
        valuations: Array of private valuations (buyers) or costs (sellers)
        num_trades: Number of successful trades completed
        has_responded: Synchronization flag (True when agent is ready)
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
    ) -> None:
        """
        Initialize a trading agent.

        Args:
            player_id: Unique identifier (must be >= 1)
            is_buyer: True for buyers, False for sellers
            num_tokens: Number of units allocated to this agent
            valuations: Private values (willingness to pay for buyers,
                       costs for sellers). Length must equal num_tokens.

        Raises:
            ValueError: If player_id < 1 or len(valuations) != num_tokens
        """
        if player_id < 1:
            raise ValueError(f"player_id must be >= 1, got {player_id}")
        if len(valuations) != num_tokens:
            raise ValueError(
                f"valuations length ({len(valuations)}) must equal "
                f"num_tokens ({num_tokens})"
            )

        self.player_id = player_id
        self.is_buyer = is_buyer
        self.num_tokens = num_tokens
        self.valuations = valuations
        self.num_trades = 0
        self.has_responded = False
        self.period_profit = 0  # Initialize to avoid AttributeError when using Market directly
        self.total_profit = 0  # Track cumulative profit across all periods

    def get_current_valuation(self) -> int:
        """
        Get the valuation for the next unit to trade.

        Returns:
            The valuation for unit at index num_trades.
            Returns 0 if all tokens have been traded.
        """
        if self.num_trades >= self.num_tokens:
            return 0
        return self.valuations[self.num_trades]

    def can_trade(self) -> bool:
        """Check if agent has remaining tokens to trade."""
        return self.num_trades < self.num_tokens

    # =========================================================================
    # BID/ASK STAGE METHODS (Port from SPlayer.java lines 63-77)
    # =========================================================================

    @abstractmethod
    def bid_ask(self, time: int, nobidask: int) -> None:
        """
        Notification: Time to submit a bid (buyers) or ask (sellers).

        This is the first call in Stage 1 of the AURORA protocol.
        The agent should prepare to respond with a price.

        Args:
            time: Current time step (1-indexed)
            nobidask: Flag indicating trading restrictions:
                0 = Agent can submit a bid/ask
                1 = Agent has traded all tokens (cannot submit)

        Implementation Note:
            Subclasses MUST set self.has_responded = False when called,
            then set it to True when ready to return a value via bid_ask_response().
        """
        pass

    @abstractmethod
    def bid_ask_response(self) -> int:
        """
        Return the bid (buyers) or ask (sellers) for this time step.

        This is called repeatedly by the market until the agent signals readiness
        by setting self.has_responded = True.

        Returns:
            Bid price (buyers) or ask price (sellers)
            Special values:
                -3: Still thinking (market will poll again)
                -1: Quit voluntarily
                -2: Failed (error condition)

        Implementation Note:
            When returning a valid price (>=0), MUST set self.has_responded = True.
            The market will not proceed until ALL agents have responded.
        """
        pass

    # =========================================================================
    # BUY/SELL STAGE METHODS (Port from SPlayer.java lines 79-93)
    # =========================================================================

    @abstractmethod
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

        This is the first call in Stage 2 of the AURORA protocol.
        Only the high bidder can accept the low ask.
        Only the low asker can accept the high bid.

        Args:
            time: Current time step (1-indexed)
            nobuysell: Bit flags indicating restrictions:
                +1: Agent has traded all tokens
                +2: No standing bid or ask (nothing to accept)
                +4: Agent is not the high bidder/low asker (cannot accept)
            high_bid: Current highest bid price (0 if none)
            low_ask: Current lowest ask price (0 if none)
            high_bidder: ID of high bidder (0 if none)
            low_asker: ID of low asker (0 if none)

        Implementation Note:
            Subclasses MUST set self.has_responded = False when called,
            then set it to True when ready to return via buy_sell_response().
        """
        pass

    @abstractmethod
    def buy_sell_response(self) -> bool:
        """
        Return whether to accept the trade opportunity.

        This is called repeatedly by the market until the agent signals readiness
        by setting self.has_responded = True.

        Returns:
            True: Accept the trade (buy at low_ask or sell at high_bid)
            False: Reject (wait for better opportunity)

        Implementation Note:
            MUST set self.has_responded = True when returning.
            For buyers: True means "buy at the low ask price"
            For sellers: True means "sell at the high bid price"
        """
        pass

    # =========================================================================
    # RESULT NOTIFICATION METHODS (Informational only)
    # =========================================================================

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
        """
        Notification: Results of the bid/ask stage.

        This is called after all agents have submitted bids/asks.
        The market broadcasts the current state to all participants.

        Args:
            status: This agent's bid/ask status:
                0 = No new bid/ask, not current winner
                1 = No new bid/ask, still current winner (carried over)
                2 = New bid/ask, now current winner
                3 = New bid/ask, beaten by another
                4 = New bid/ask, tied and lost random tie-break
            num_trades: Number of trades this agent has completed
            new_bids: Array of NEW bids submitted this time step
            new_asks: Array of NEW asks submitted this time step
            high_bid: Current highest bid (0 if none)
            high_bidder: ID of high bidder (0 if none)
            low_ask: Current lowest ask (0 if none)
            low_asker: ID of low asker (0 if none)

        Note:
            Default implementation updates num_trades. Subclasses may override
            to learn from market state (e.g., update beliefs, adjust strategy).
        """
        self.num_trades = num_trades

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

        This is called after the trade execution decision.

        Args:
            status: This agent's trade status:
                0 = No trade occurred
                1 = Agent bought (if buyer) or sold (if seller)
                2 = Agent's order was cleared by opponent's acceptance
            trade_price: Execution price (0 if no trade)
            trade_type: How trade was determined:
                0 = No trade
                1 = Buyer accepted (trade at ask)
                2 = Seller accepted (trade at bid)
                3 = Both accepted (50/50 random)
            high_bid: Current highest bid after trade (0 if cleared)
            high_bidder: ID of high bidder (0 if cleared)
            low_ask: Current lowest ask after trade (0 if cleared)
            low_asker: ID of low asker (0 if cleared)

        Note:
            Default implementation does nothing. Subclasses may override
            to update internal state (e.g., profit tracking, learning).
        """
        if status == 1:
            # Agent traded
            if self.num_trades < self.num_tokens:
                val = self.valuations[self.num_trades]
                if self.is_buyer:
                    self.period_profit += val - trade_price
                else:
                    self.period_profit += trade_price - val
                self.num_trades += 1

    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================

    def start_period(self, period_number: int) -> None:
        """
        Called at the start of a trading period.
        Agents should reset period-specific state here.

        Per AURORA protocol:
        - Each period, agents get N fresh tokens to trade
        - Valuations stay the same across periods in a round
        - So num_trades resets but valuations don't
        """
        self.period_profit = 0
        self.num_trades = 0


    def end_period(self) -> None:
        """
        Called at the end of a trading period.
        Agents can perform end-of-period learning/updates here.
        """
        # Accumulate period profit into total profit
        self.total_profit += self.period_profit

    def start_round(self, valuations: list[int]) -> None:
        """
        Called at the start of a new round (multiple periods).
        Resets all agent state and assigns new valuations.

        Args:
            valuations: New valuations for this round
        """
        # Reset all state
        self.num_trades = 0
        self.period_profit = 0
        self.total_profit = 0
        self.has_responded = False

        # Update valuations
        if len(valuations) != self.num_tokens:
            raise ValueError(
                f"valuations length ({len(valuations)}) must equal "
                f"num_tokens ({self.num_tokens})"
            )
        self.valuations = valuations

    def __repr__(self) -> str:
        """String representation for debugging."""
        agent_type = "Buyer" if self.is_buyer else "Seller"
        return (
            f"{self.__class__.__name__}(id={self.player_id}, "
            f"type={agent_type}, trades={self.num_trades}/{self.num_tokens})"
        )
