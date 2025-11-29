"""
Market orchestrator for the Santa Fe Double Auction (AURORA protocol).

This module implements the synchronized two-stage auction mechanism from the 1993
tournament. It ports the logic from SGameRunner.java (da2.7.2).

The AURORA protocol has two stages per time step:
1. BID/ASK STAGE: All agents simultaneously submit prices
2. BUY/SELL STAGE: High bidder and low asker decide whether to accept

All agent interactions are synchronized - the market waits for all responses
before proceeding to the next stage.
"""

from typing import Any, Sequence, TYPE_CHECKING

import numpy as np

from engine.orderbook import OrderBook
from traders.base import Agent

if TYPE_CHECKING:
    from engine.event_logger import EventLogger


class Market:
    """
    The AURORA double auction orchestrator.

    Ports the game loop logic from SGameRunner.java (lines 20-360).
    Manages the synchronized two-stage protocol and coordinates with the OrderBook.

    Attributes:
        num_buyers: Number of buying agents
        num_sellers: Number of selling agents
        num_times: Number of time steps per period
        orderbook: The order matching engine (from Task 1.1)
        buyers: List of buyer agents
        sellers: List of seller agents
        current_time: Current time step (1-indexed)
        fail_state: True if market has failed (all buyers or sellers quit)
        rng: Random number generator for reproducible randomness
    """

    def __init__(
        self,
        num_buyers: int,
        num_sellers: int,
        num_times: int,
        price_min: int,
        price_max: int,
        buyers: Sequence[Agent],
        sellers: Sequence[Agent],
        seed: int | None = None,
        deadsteps: int = 0,
        event_logger: "EventLogger | None" = None,
    ) -> None:
        """
        Initialize the market.

        Args:
            num_buyers: Number of buying agents
            num_sellers: Number of selling agents
            num_times: Number of time steps in a period
            price_min: Minimum valid price
            price_max: Maximum valid price
            buyers: List of buyer agents (must have length num_buyers)
            sellers: List of seller agents (must have length num_sellers)
            seed: Random seed for reproducibility (default: None)
            deadsteps: Consecutive no-trade steps before early termination (0 = disabled)
            event_logger: Optional EventLogger for bid-level event logging

        Raises:
            ValueError: If agent list lengths don't match counts
        """
        if len(buyers) != num_buyers:
            raise ValueError(
                f"buyers list length ({len(buyers)}) must equal "
                f"num_buyers ({num_buyers})"
            )
        if len(sellers) != num_sellers:
            raise ValueError(
                f"sellers list length ({len(sellers)}) must equal "
                f"num_sellers ({num_sellers})"
            )

        self.num_buyers = num_buyers
        self.num_sellers = num_sellers
        self.num_times = num_times

        # Create the order book (Task 1.1)
        # Note: OrderBook uses min_price/max_price/rng_seed parameter names
        self.orderbook = OrderBook(
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            num_times=num_times,
            min_price=price_min,
            max_price=price_max,
            rng_seed=seed,
            deadsteps=deadsteps,
        )

        # Agent lists (convert Sequence to list for mutability)
        self.buyers = list(buyers)
        self.sellers = list(sellers)

        # State tracking
        self.current_time = 0
        self.fail_state = False

        # Random number generator (for tie-breaking, trade pricing)
        self.rng = np.random.default_rng(seed)

        # Event logger for Market Heartbeat visualization (optional)
        self.event_logger = event_logger
        self._current_round = 0
        self._current_period = 0

        # Inject orderbook into agents that need it (e.g. RL agents)
        for agent in self.buyers + self.sellers:
            if hasattr(agent, "set_orderbook"):
                agent.set_orderbook(self.orderbook)

    # =========================================================================
    # THE MASTER LOOP (Port from SGameRunner.java lines 145-155)
    # =========================================================================

    def run_time_step(self) -> bool:
        """
        Execute one time step of the AURORA protocol.

        This is the master loop that orchestrates both stages:
        1. Bid/Ask Stage: Collect orders from all agents
        2. Buy/Sell Stage: Execute trade if high bidder/low asker accept

        Returns:
            True if time step completed successfully, False if market failed

        Note:
            Implements runTime() from SGameRunner.java (lines 145-155).
            Market fails if all buyers OR all sellers have quit/failed.
        """
        if self.fail_state:
            return False

        # Increment time and get current time from orderbook
        if not self.orderbook.increment_time():
            return False

        self.current_time = self.orderbook.current_time

        # Stage 1: Bid/Ask
        self.bid_ask_stage()
        self.bid_ask_response()
        self.bid_ask_result()
        self._log_bid_ask_events()

        # Stage 2: Buy/Sell
        self.buy_sell_stage()
        self.buy_sell_response()
        self.buy_sell_result()
        self._log_trade_event()

        # Check if market has failed
        self.check_fail_state()

        return not self.fail_state

    # =========================================================================
    # STAGE 1: BID/ASK (Order Submission)
    # =========================================================================

    def bid_ask_stage(self) -> None:
        """
        Notify all agents to submit bids or asks.

        Implements bidAskStage() from SGameRunner.java (lines 157-168).
        Sends nobidask flag: 1 if agent has traded all tokens, 0 otherwise.

        Note:
            We check the OrderBook's position directly (from previous timestep)
            rather than the agent's self.num_trades, which hasn't been updated
            yet in this timestep's bid_ask_result().

        Fairness: Agents are notified in random order each time step.
        """
        t = self.current_time

        # Create randomized order for fairness
        buyer_indices = list(range(len(self.buyers)))
        seller_indices = list(range(len(self.sellers)))
        self.rng.shuffle(buyer_indices)
        self.rng.shuffle(seller_indices)

        # Notify buyers in random order
        for idx in buyer_indices:
            buyer = self.buyers[idx]
            local_id = idx + 1  # OrderBook uses 1-indexed positions
            # Check actual position from OrderBook (from current state, which includes previous trades)
            # At t=1, check index 0 (initialized to 0); at t>1, check t-1
            num_buys = int(self.orderbook.num_buys[local_id, t]) if t >= 1 else 0
            nobidask = 1 if num_buys >= buyer.num_tokens else 0
            buyer.bid_ask(t, nobidask)

        # Notify sellers in random order
        for idx in seller_indices:
            seller = self.sellers[idx]
            local_id = idx + 1  # OrderBook uses 1-indexed positions
            # Check actual position from OrderBook (from current state, which includes previous trades)
            num_sells = int(self.orderbook.num_sells[local_id, t]) if t >= 1 else 0
            nobidask = 1 if num_sells >= seller.num_tokens else 0
            seller.bid_ask(t, nobidask)

    def bid_ask_response(self) -> None:
        """
        Collect bids and asks from all agents (synchronized).

        Implements bidAskResponse() from SGameRunner.java (lines 170-205).

        This is a synchronization point - in the Java code, this polls agents
        until all have responded. In Python, we call each agent once since
        there's no network delay.

        Handles special responses:
            -1: Agent quit voluntarily (remove from market)
            -2: Agent failed (remove from market)
            >= 0: Valid bid/ask (add to orderbook)

        Fairness: Agents are polled in random order each time step.
        """
        # Save player_id mapping before any removals (for later reference)
        buyer_local_to_global = {}
        for i, buyer in enumerate(self.buyers):
            buyer_local_to_global[i + 1] = buyer.player_id

        seller_local_to_global = {}
        for i, seller in enumerate(self.sellers):
            seller_local_to_global[i + 1] = seller.player_id

        # Save the mapping for use in bid_ask_result
        self._buyer_local_to_global = buyer_local_to_global
        self._seller_local_to_global = seller_local_to_global

        # Create randomized order for fairness
        buyer_indices = list(range(len(self.buyers)))
        seller_indices = list(range(len(self.sellers)))
        self.rng.shuffle(buyer_indices)
        self.rng.shuffle(seller_indices)

        # Collect bids from buyers in random order
        buyers_to_remove: list[Agent] = []
        for idx in buyer_indices:
            buyer = self.buyers[idx]
            local_id = idx + 1  # OrderBook uses 1-indexed positions
            bid_value = buyer.bid_ask_response()

            if bid_value == -1 or bid_value == -2:
                # Agent quit or failed - mark for removal
                buyers_to_remove.append(buyer)
            elif bid_value >= 0:
                # Valid bid - add to orderbook
                # OrderBook validates price and improvement rules
                self.orderbook.add_bid(local_id, bid_value)

        # Collect asks from sellers in random order
        sellers_to_remove: list[Agent] = []
        for idx in seller_indices:
            seller = self.sellers[idx]
            local_id = idx + 1  # OrderBook uses 1-indexed positions
            ask_value = seller.bid_ask_response()

            if ask_value == -1 or ask_value == -2:
                # Agent quit or failed - mark for removal
                sellers_to_remove.append(seller)
            elif ask_value >= 0:
                # Valid ask - add to orderbook
                self.orderbook.add_ask(local_id, ask_value)

        # Remove failed/quit agents
        for buyer in buyers_to_remove:
            self.buyers.remove(buyer)
            self.num_buyers -= 1

        for seller in sellers_to_remove:
            self.sellers.remove(seller)
            self.num_sellers -= 1

    def bid_ask_result(self) -> None:
        """
        Broadcast bid/ask stage results to all agents.

        Implements bidAskResult() from SGameRunner.java (lines 207-246).

        Determines winners (high bidder, low asker) and broadcasts:
        - Agent's status (0-4)
        - Number of trades completed
        - All new bids/asks submitted this time step
        - Current high bid/bidder and low ask/asker
        """
        t = self.current_time

        # Determine winners (high bidder and low asker)
        self.orderbook.determine_winners()

        # Get current market state
        high_bid = int(self.orderbook.high_bid[t])
        high_bidder_local = int(self.orderbook.high_bidder[t])
        low_ask = int(self.orderbook.low_ask[t])
        low_asker_local = int(self.orderbook.low_asker[t])
        
        # Use saved mapping from bid_ask_response (before agents were removed)
        high_bidder_global = 0
        if high_bidder_local > 0 and hasattr(self, '_buyer_local_to_global'):
            high_bidder_global = self._buyer_local_to_global.get(high_bidder_local, 0)
        elif high_bidder_local > 0 and high_bidder_local <= len(self.buyers):
            high_bidder_global = self.buyers[high_bidder_local - 1].player_id

        low_asker_global = 0
        if low_asker_local > 0 and hasattr(self, '_seller_local_to_global'):
            low_asker_global = self._seller_local_to_global.get(low_asker_local, 0)
        elif low_asker_local > 0 and low_asker_local <= len(self.sellers):
            low_asker_global = self.sellers[low_asker_local - 1].player_id

        # Get arrays of NEW bids/asks (status > 1 means new offer this time step)
        new_bids: list[int] = []
        for buyer_id in range(1, self.num_buyers + 1):
            if self.orderbook.bid_status[buyer_id] > 1:
                new_bids.append(int(self.orderbook.bids[buyer_id, t]))

        new_asks: list[int] = []
        for seller_id in range(1, self.num_sellers + 1):
            if self.orderbook.ask_status[seller_id] > 1:
                new_asks.append(int(self.orderbook.asks[seller_id, t]))

        # Broadcast to all buyers
        # Note: At time t, after increment_time(), positions contain cumulative trades through t-1
        for i, buyer in enumerate(self.buyers):
            local_id = i + 1
            status = int(self.orderbook.bid_status[local_id])
            num_trades = int(self.orderbook.num_buys[local_id, t])
            buyer.bid_ask_result(
                status=status,
                num_trades=num_trades,
                new_bids=new_bids,
                new_asks=new_asks,
                high_bid=high_bid,
                high_bidder=high_bidder_global,
                low_ask=low_ask,
                low_asker=low_asker_global,
            )

        # Broadcast to all sellers
        # Note: At time t, after increment_time(), positions contain cumulative trades through t-1
        for i, seller in enumerate(self.sellers):
            local_id = i + 1
            status = int(self.orderbook.ask_status[local_id])
            num_trades = int(self.orderbook.num_sells[local_id, t])
            seller.bid_ask_result(
                status=status,
                num_trades=num_trades,
                new_bids=new_bids,
                new_asks=new_asks,
                high_bid=high_bid,
                high_bidder=high_bidder_global,
                low_ask=low_ask,
                low_asker=low_asker_global,
            )

    # =========================================================================
    # STAGE 2: BUY/SELL (Trade Execution)
    # =========================================================================

    def buy_sell_stage(self) -> None:
        """
        Notify agents to accept or reject trade opportunities.

        Implements buySellStage() from SGameRunner.java (lines 248-271).

        Sends nobuysell flags (bit flags):
            +1: Agent has traded all tokens
            +2: No standing bid or ask (nothing to accept)
            +4: Agent is not the high bidder/low asker (cannot accept)

        Note:
            We check the OrderBook's position directly rather than the agent's
            self.num_trades. After bid_ask_result(), the agent's num_trades is
            up-to-date from the previous timestep's trades. We read from t (current)
            since bid_ask_result() has already been called and updated agent state.
        """
        t = self.current_time

        # Get current high bid/low ask from OrderBook
        high_bid = int(self.orderbook.high_bid[t])
        high_bidder_local = int(self.orderbook.high_bidder[t])
        low_ask = int(self.orderbook.low_ask[t])
        low_asker_local = int(self.orderbook.low_asker[t])

        # Save bid/ask for trade type determination in buy_sell_result()
        # (Java uses local variables currentBid/currentAsk for this purpose)
        self._saved_high_bid = high_bid
        self._saved_low_ask = low_ask
        
        # Map to global IDs
        # Use saved mapping from bid_ask_response (before agents were removed)
        high_bidder_global = 0
        if high_bidder_local > 0 and hasattr(self, '_buyer_local_to_global'):
            high_bidder_global = self._buyer_local_to_global.get(high_bidder_local, 0)
        elif high_bidder_local > 0 and high_bidder_local <= len(self.buyers):
            high_bidder_global = self.buyers[high_bidder_local - 1].player_id

        low_asker_global = 0
        if low_asker_local > 0 and hasattr(self, '_seller_local_to_global'):
            low_asker_global = self._seller_local_to_global.get(low_asker_local, 0)
        elif low_asker_local > 0 and low_asker_local <= len(self.sellers):
            low_asker_global = self.sellers[low_asker_local - 1].player_id

        # Create randomized order for fairness
        buyer_indices = list(range(len(self.buyers)))
        seller_indices = list(range(len(self.sellers)))
        self.rng.shuffle(buyer_indices)
        self.rng.shuffle(seller_indices)

        # Notify buyers in random order
        for idx in buyer_indices:
            buyer = self.buyers[idx]
            nobuysell = 0

            # +1: Has traded all tokens
            # After bid_ask_result, agent's num_trades is current, so can_trade() is accurate
            if not buyer.can_trade():
                nobuysell += 1

            # +2: No standing ask to accept
            # NOTE: Java implementation uses OR logic: if (currentAsker==0)||(currentBidder==0)
            # giving +2 to ALL agents if EITHER is missing. We use more precise logic:
            # buyers only get +2 if the ask they need is missing.
            # This is a justified deviation - more logically correct per AURORA semantics.
            if low_ask == 0:
                nobuysell += 2

            # +4: Not the high bidder (cannot accept)
            # Rule 14: "If there is no current bid, then any buyer... may make a buy request."
            # Only block if there IS a high bidder and this buyer is not it.
            if high_bidder_global > 0 and buyer.player_id != high_bidder_global:
                nobuysell += 4

            buyer.buy_sell(
                time=t,
                nobuysell=nobuysell,
                high_bid=high_bid,
                low_ask=low_ask,
                high_bidder=high_bidder_global,
                low_asker=low_asker_global,
            )

        # Notify sellers in random order
        for idx in seller_indices:
            seller = self.sellers[idx]
            nobuysell = 0

            # +1: Has traded all tokens
            # After bid_ask_result, agent's num_trades is current, so can_trade() is accurate
            if not seller.can_trade():
                nobuysell += 1

            # +2: No standing bid to accept
            # NOTE: See comment above for buyers - we use precise logic (sellers check bid)
            # rather than Java's blanket OR logic. This is a justified deviation.
            if high_bid == 0:
                nobuysell += 2

            # +4: Not the low asker (cannot accept)
            # Rule 15: "If there is no current offer, then any seller... may make a sell request."
            # Only block if there IS a low asker and this seller is not it.
            if low_asker_global > 0 and seller.player_id != low_asker_global:
                nobuysell += 4

            seller.buy_sell(
                time=t,
                nobuysell=nobuysell,
                high_bid=high_bid,
                low_ask=low_ask,
                high_bidder=high_bidder_global,
                low_asker=low_asker_global,
            )

    def buy_sell_response(self) -> None:
        """
        Collect accept/reject decisions from agents (synchronized).

        Implements buySellResponse() from SGameRunner.java (lines 273-311).

        Only the high bidder can accept the low ask (buy=True).
        Only the low asker can accept the high bid (sell=True).

        Trade execution uses Chicago Rules (implemented in OrderBook):
        - Both accept: 50/50 random (bid vs ask price)
        - Only buyer accepts: trade at ask
        - Only seller accepts: trade at bid
        """
        t = self.current_time

        # Get current high bid/low ask from OrderBook
        high_bid = int(self.orderbook.high_bid[t])
        low_ask = int(self.orderbook.low_ask[t])
        high_bidder_local = int(self.orderbook.high_bidder[t])
        low_asker_local = int(self.orderbook.low_asker[t])
        
        # Map to global IDs for agent notification
        # Use saved mapping from bid_ask_response (before agents were removed)
        high_bidder_global = 0
        if high_bidder_local > 0 and hasattr(self, '_buyer_local_to_global'):
            high_bidder_global = self._buyer_local_to_global.get(high_bidder_local, 0)
        elif high_bidder_local > 0 and high_bidder_local <= len(self.buyers):
            high_bidder_global = self.buyers[high_bidder_local - 1].player_id

        low_asker_global = 0
        if low_asker_local > 0 and hasattr(self, '_seller_local_to_global'):
            low_asker_global = self._seller_local_to_global.get(low_asker_local, 0)
        elif low_asker_local > 0 and low_asker_local <= len(self.sellers):
            low_asker_global = self.sellers[low_asker_local - 1].player_id

        # Create randomized order for fairness
        buyer_indices = list(range(len(self.buyers)))
        seller_indices = list(range(len(self.sellers)))
        self.rng.shuffle(buyer_indices)
        self.rng.shuffle(seller_indices)

        # Collect buyer's decision
        buyer_accepts = False
        accepted_buyer_local_id = 0

        if high_bidder_global > 0:
            # Case 1: Current bidder exists - only they can accept
            for idx in buyer_indices:
                buyer = self.buyers[idx]
                if buyer.player_id == high_bidder_global:
                    decision = buyer.buy_sell_response()
                    if decision and (low_ask > 0):
                        buyer_accepts = True
                        accepted_buyer_local_id = high_bidder_local
                    break
        else:
            # Case 2: No current bidder - ANY buyer can accept (Rule 14)
            potential_buyers = [] # List of (global_id, local_id)
            for idx in buyer_indices:
                buyer = self.buyers[idx]
                local_id = idx + 1  # OrderBook uses 1-indexed positions
                # Only check buyers who were eligible (nobuysell check passed)
                # We re-check basic eligibility here to be safe
                if buyer.can_trade():
                    decision = buyer.buy_sell_response()
                    if decision and (low_ask > 0):
                        potential_buyers.append((buyer.player_id, local_id))

            if potential_buyers:
                # Rule 13: "If more than one request occurs, one is selected randomly"
                if len(potential_buyers) > 1:
                    idx = self.rng.integers(0, len(potential_buyers))
                    # Unpack the tuple (global_id, local_id)
                    _, accepted_buyer_local_id = potential_buyers[idx]
                else:
                    _, accepted_buyer_local_id = potential_buyers[0]
                buyer_accepts = True

        # Collect seller's decision
        seller_accepts = False
        accepted_seller_local_id = 0

        if low_asker_global > 0:
            # Case 1: Current offerer exists - only they can accept
            for idx in seller_indices:
                seller = self.sellers[idx]
                if seller.player_id == low_asker_global:
                    decision = seller.buy_sell_response()
                    if decision and (high_bid > 0):
                        seller_accepts = True
                        accepted_seller_local_id = low_asker_local
                    break
        else:
            # Case 2: No current offerer - ANY seller can accept (Rule 15)
            potential_sellers = [] # List of (global_id, local_id)
            for idx in seller_indices:
                seller = self.sellers[idx]
                local_id = idx + 1  # OrderBook uses 1-indexed positions
                if seller.can_trade():
                    decision = seller.buy_sell_response()
                    if decision and (high_bid > 0):
                        potential_sellers.append((seller.player_id, local_id))

            if potential_sellers:
                # Rule 13: "If more than one request occurs, one is selected randomly"
                if len(potential_sellers) > 1:
                    idx = self.rng.integers(0, len(potential_sellers))
                    _, accepted_seller_local_id = potential_sellers[idx]
                else:
                    _, accepted_seller_local_id = potential_sellers[0]
                seller_accepts = True

        # Execute trade via OrderBook (implements Chicago Rules)
        # Pass explicit IDs if we found them (needed for Case 2)
        # If IDs are 0, OrderBook defaults to high_bidder/low_asker (Case 1)
        self.orderbook.execute_trade(
            buyer_accepts=buyer_accepts, 
            seller_accepts=seller_accepts,
            buyer_id=accepted_buyer_local_id if accepted_buyer_local_id > 0 else None,
            seller_id=accepted_seller_local_id if accepted_seller_local_id > 0 else None
        )

    def buy_sell_result(self) -> None:
        """
        Broadcast trade execution results to all agents.

        Implements buySellResult() from SGameRunner.java (lines 313-352).

        Broadcasts:
        - Trade price (0 if no trade)
        - Trade type (how price was determined)
        - Agent's status (0 = no trade, 1/2 = traded)
        - Updated current bid/ask state (cleared if trade occurred)
        """
        t = self.current_time

        # Get trade outcome
        trade_price = int(self.orderbook.trade_price[t])

        # Determine trade type based on accept pattern
        # Type 0: No trade
        # Type 1: Buyer accepted (trade at ask)
        # Type 2: Seller accepted (trade at bid)
        # Type 3: Both accepted (50/50 random) - not distinguishable from 1 or 2
        #
        # NOTE: Use saved bid/ask from buy_sell_stage(), not t-1 indices.
        # Reading from t-1 fails when trades occur in consecutive time steps
        # (book is cleared after trade, so t-1 values would be 0).
        # This matches Java's approach of using local variables currentBid/currentAsk.
        if trade_price == 0:
            trade_type = 0
        elif trade_price == self._saved_low_ask:
            # Could be buyer-only OR both-accepted-random
            # We don't track this info, so default to type 1
            trade_type = 1
        elif trade_price == self._saved_high_bid:
            # Could be seller-only OR both-accepted-random
            trade_type = 2
        else:
            # Neither exact match - shouldn't happen, default to 0
            trade_type = 0

        # Get current state (cleared if trade occurred)
        high_bid = int(self.orderbook.high_bid[t])
        high_bidder_local = int(self.orderbook.high_bidder[t])
        low_ask = int(self.orderbook.low_ask[t])
        low_asker_local = int(self.orderbook.low_asker[t])
        
        # Map to global IDs
        # Use saved mapping from bid_ask_response (before agents were removed)
        high_bidder_global = 0
        if high_bidder_local > 0 and hasattr(self, '_buyer_local_to_global'):
            high_bidder_global = self._buyer_local_to_global.get(high_bidder_local, 0)
        elif high_bidder_local > 0 and high_bidder_local <= len(self.buyers):
            high_bidder_global = self.buyers[high_bidder_local - 1].player_id

        low_asker_global = 0
        if low_asker_local > 0 and hasattr(self, '_seller_local_to_global'):
            low_asker_global = self._seller_local_to_global.get(low_asker_local, 0)
        elif low_asker_local > 0 and low_asker_local <= len(self.sellers):
            low_asker_global = self.sellers[low_asker_local - 1].player_id

        # Determine who traded
        buyer_traded_id = 0
        seller_traded_id = 0
        if trade_price > 0:
            # Find who traded by checking position change
            for i, buyer in enumerate(self.buyers):
                local_id = i + 1
                if t > 0:
                    prev_buys = int(self.orderbook.num_buys[local_id, t - 1])
                    curr_buys = int(self.orderbook.num_buys[local_id, t])
                    if curr_buys > prev_buys:
                        buyer_traded_id = buyer.player_id
                        break

            for i, seller in enumerate(self.sellers):
                local_id = i + 1
                if t > 0:
                    prev_sells = int(
                        self.orderbook.num_sells[local_id, t - 1]
                    )
                    curr_sells = int(self.orderbook.num_sells[local_id, t])
                    if curr_sells > prev_sells:
                        seller_traded_id = seller.player_id
                        break

        # Broadcast to all buyers
        for buyer in self.buyers:
            if buyer.player_id == buyer_traded_id:
                status = 1  # Bought
            else:
                status = 0  # No trade

            buyer.buy_sell_result(
                status=status,
                trade_price=trade_price,
                trade_type=trade_type,
                high_bid=high_bid,
                high_bidder=high_bidder_global,
                low_ask=low_ask,
                low_asker=low_asker_global,
            )

        # Broadcast to all sellers
        for seller in self.sellers:
            if seller.player_id == seller_traded_id:
                status = 1  # Sold
            else:
                status = 0  # No trade

            seller.buy_sell_result(
                status=status,
                trade_price=trade_price,
                trade_type=trade_type,
                high_bid=high_bid,
                high_bidder=high_bidder_global,
                low_ask=low_ask,
                low_asker=low_asker_global,
            )

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def check_fail_state(self) -> None:
        """
        Check if market has failed.

        Implements checkFailState() from SGameRunner.java (lines 354-360).
        Market fails if all buyers OR all sellers have quit/failed.
        """
        if self.num_buyers == 0 or self.num_sellers == 0:
            self.fail_state = True

    def get_current_time(self) -> int:
        """Get current time step (1-indexed)."""
        return self.current_time

    def has_failed(self) -> bool:
        """Check if market is in fail state."""
        return self.fail_state

    def get_orderbook(self) -> OrderBook:
        """Get the orderbook instance for inspection/testing."""
        return self.orderbook

    def set_period(self, round_num: int, period_num: int) -> None:
        """Set current round and period for event logging."""
        self._current_round = round_num
        self._current_period = period_num

    def _log_bid_ask_events(self) -> None:
        """Log all bid/ask events from the current time step."""
        if self.event_logger is None:
            return

        t = self.current_time

        # Map bid_status codes to readable names
        status_names = {
            0: "pass",      # No bid submitted
            1: "standing",  # Currently holding best price
            2: "winner",    # New best price this step
            3: "beaten",    # Submitted but beaten
            4: "tie_lost",  # Tied but lost random draw
        }

        # Log buyer bids
        for i, buyer in enumerate(self.buyers):
            local_id = i + 1
            bid_price = int(self.orderbook.bids[local_id, t])
            status_code = int(self.orderbook.bid_status[local_id])
            status = status_names.get(status_code, "unknown")

            # Only log if agent submitted a bid (price > 0) or we want to track passes
            if bid_price > 0 or status_code > 0:
                self.event_logger.log_bid(
                    round=self._current_round,
                    period=self._current_period,
                    step=t,
                    agent_id=buyer.player_id,
                    agent_type=buyer.__class__.__name__,
                    price=bid_price,
                    status=status,
                )

        # Log seller asks
        for i, seller in enumerate(self.sellers):
            local_id = i + 1
            ask_price = int(self.orderbook.asks[local_id, t])
            status_code = int(self.orderbook.ask_status[local_id])
            status = status_names.get(status_code, "unknown")

            if ask_price > 0 or status_code > 0:
                self.event_logger.log_ask(
                    round=self._current_round,
                    period=self._current_period,
                    step=t,
                    agent_id=seller.player_id,
                    agent_type=seller.__class__.__name__,
                    price=ask_price,
                    status=status,
                )

    def _log_trade_event(self) -> None:
        """Log trade event if one occurred this time step."""
        if self.event_logger is None:
            return

        t = self.current_time
        trade_price = int(self.orderbook.trade_price[t])

        if trade_price > 0:
            # Find who traded by checking position change
            buyer_id = 0
            seller_id = 0

            for i, buyer in enumerate(self.buyers):
                local_id = i + 1
                if t > 0:
                    prev_buys = int(self.orderbook.num_buys[local_id, t - 1])
                    curr_buys = int(self.orderbook.num_buys[local_id, t])
                    if curr_buys > prev_buys:
                        buyer_id = buyer.player_id
                        break

            for i, seller in enumerate(self.sellers):
                local_id = i + 1
                if t > 0:
                    prev_sells = int(self.orderbook.num_sells[local_id, t - 1])
                    curr_sells = int(self.orderbook.num_sells[local_id, t])
                    if curr_sells > prev_sells:
                        seller_id = seller.player_id
                        break

            self.event_logger.log_trade(
                round=self._current_round,
                period=self._current_period,
                step=t,
                buyer_id=buyer_id,
                seller_id=seller_id,
                price=trade_price,
            )
