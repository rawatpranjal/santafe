"""
Skeleton Agent.

A simplified version of the Kaplan agent, stripping away the complex price history and trend analysis.
It retains the core "sniper" and "jump-in" logic which Rust et al. (1994) identified as the effective components.

Strategy:
1. Wait in the background (submit conservative bids/asks).
2. "Jump in" to the best price if:
   - The spread is small (< 10%).
   - Time is running out.
3. Accept any profitable trade in the final moments (Sniper).
"""

from traders.base import Agent
import logging
import numpy as np

class Skeleton(Agent):
    """
    Skeleton Agent (SRobotExample).
    
    Based on SRobotExample.java and Figure 4 of Rust et al. (1994).
    Uses a weighted average strategy with a random parameter alpha.
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
    ) -> None:
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min_limit = price_min
        self.price_max_limit = price_max
        self.num_times = num_times
        self.rng = np.random.default_rng(seed)
        
        # State
        self.current_time = 0
        self.current_bid = 0
        self.current_ask = 0
        self.current_bidder = 0
        self.current_asker = 0
        self.nobidask = 0
        self.nobuysell = 0
        self.last_time = 0

    def bid_ask(self, time: int, nobidask: int) -> None:
        self.current_time = time
        self.nobidask = nobidask
        self.has_responded = False

    def bid_ask_response(self) -> int:
        self.has_responded = True
        if self.is_buyer:
            return self._request_bid()
        else:
            return self._request_ask()

    def _request_bid(self) -> int:
        """
        Determine bid price.
        
        Corresponds to the BID routine in DATManual (Section 3.4).
        Uses the strategy from Figure 4 of Rust et al. (1994).
        """
        # DATManual: nobidoff - 0 if allowed, 1 if no tokens, etc.
        # We use self.nobidask (from Agent) which maps to nobidoff.
        if self.nobidask > 0: return 0
        
        # DATManual: mytrades - number of trades made so far
        # DATManual: ntokens - total tokens
        if self.num_trades >= self.num_tokens: return 0
        
        # Strategy Logic (Figure 4 / SRobotExample.java)
        # alpha = 0.25 + 0.1 * U[0,1]
        alpha = 0.25 + 0.1 * self.rng.random()
        
        # DATManual: token[mytrades+1] - next available token value
        # In our 0-indexed list, this is self.valuations[self.num_trades]
        token_val = self.valuations[self.num_trades]
        first_token = self.valuations[0]
        last_token = self.valuations[self.num_tokens - 1]
        
        newbid = 0
        most = 0
        
        # DATManual: cbid - current bid
        # DATManual: coffer - current offer
        cbid = self.current_bid
        coffer = self.current_ask
        
        if cbid == 0:
            most = last_token - 1 # Worst case
            if coffer > 0 and coffer < most:
                most = coffer
            # newbid = most - alpha * (token[1] - token[ntokens])
            # Note: Java uses token[1] (best) - token[ntokens] (worst).
            # In our list, best is index 0, worst is index -1.
            spread_val = first_token - last_token
            newbid = int(most - (alpha * spread_val))
        else:
            most = token_val - 1 # Current token
            if coffer > 0 and coffer < most:
                most = coffer
                
            if most <= cbid:
                return 0
                
            # newbid = (1-alpha)*(cbid+1) + alpha*most
            newbid = int((1.0 - alpha) * (cbid + 1) + alpha * most + 0.001)
            
        return max(newbid, self.price_min_limit)

    def _request_ask(self) -> int:
        """
        Determine ask price.
        
        Corresponds to the OFFER routine in DATManual (Section 3.4).
        """
        if self.nobidask > 0: return 0
        if self.num_trades >= self.num_tokens: return 0
        
        alpha = 0.25 + 0.1 * self.rng.random()
        
        token_val = self.valuations[self.num_trades]
        first_token = self.valuations[0] # Lowest cost (Best)
        last_token = self.valuations[self.num_tokens - 1] # Highest cost (Worst)
        
        newask = 0
        least = 0
        
        cbid = self.current_bid
        coffer = self.current_ask
        
        if coffer == 0:
            least = last_token + 1 # Worst case
            if cbid > 0 and cbid > least:
                least = cbid
            # newask = least + alpha * (token[ntokens] - token[1])
            # Spread is worst - best (high cost - low cost)
            spread_val = last_token - first_token
            newask = int(least + (alpha * spread_val))
        else:
            least = token_val + 1 # Current token
            if cbid > 0 and cbid > least:
                least = cbid
                
            if least >= coffer:
                return 0
                
            # newask = (1-alpha)*(coffer-1) + alpha*least
            newask = int((1.0 - alpha) * (coffer - 1) + alpha * least + 0.001)
            
        return min(newask, self.price_max_limit)

    def buy_sell(self, time: int, nobuysell: int, high_bid: int, low_ask: int, high_bidder: int, low_asker: int) -> None:
        self.current_time = time
        self.nobuysell = nobuysell
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker
        self.has_responded = False

    def buy_sell_response(self) -> bool:
        """
        Decide whether to accept a trade.
        
        Corresponds to the BUY/SELL routines in DATManual (Section 3.4).
        """
        # DATManual: nobuysell - 0 if allowed
        # We assume the engine calls this only if allowed, or we check self.can_trade()
        if not self.can_trade(): return False
        
        # DATManual: cbid, coffer
        cbid = self.current_bid
        coffer = self.current_ask
        
        # Logic from SRobotExample.java
        # If buyer: accept if coffer <= limit
        # If seller: accept if cbid >= limit
        
        # DATManual: token[mytrades+1]
        token_val = self.valuations[self.num_trades]
        
        if self.is_buyer:
            if token_val <= coffer: return False # Loss
            
            # Check if we are the current bidder (Chicago rules)
            if self.player_id == self.current_bidder and cbid >= coffer: return True
            
            # target = 1.3*token[ntokens] - 0.3*token[1]
            # alpha = 1.0 / (t - lasttime)
            # DATManual: lasttime - time of most recent trade in this period
            # DATManual: mylasttime - time of YOUR most recent trade
            # Java uses 'lasttime' (global).
            # We track self.last_time which is updated on ANY trade in buy_sell_result.
            
            target = 1.3 * self.valuations[self.num_tokens - 1] - 0.3 * self.valuations[0]
            
            denom = self.current_time - self.last_time
            if denom <= 0: denom = 1 # Safety
            alpha = 1.0 / float(denom)
            
            threshold = int(alpha * target + (1.0 - alpha) * token_val)
            if coffer < threshold:
                return True
                
        else:
            if cbid <= token_val: return False # Loss
            if self.player_id == self.current_asker and coffer <= cbid: return True
            
            # target = 1.3*token[ntokens] - 0.3*token[1]
            target = 1.3 * self.valuations[self.num_tokens - 1] - 0.3 * self.valuations[0]
            
            denom = self.current_time - self.last_time
            if denom <= 0: denom = 1
            alpha = 1.0 / float(denom)
            
            threshold = int(alpha * target + (1.0 - alpha) * token_val)
            if cbid > threshold:
                return True
                
        return False
        
    def buy_sell_result(self, status, trade_price, trade_type, high_bid, high_bidder, low_ask, low_asker):
        super().buy_sell_result(status, trade_price, trade_type, high_bid, high_bidder, low_ask, low_asker)
        if trade_type != 0:
            # Trade occurred (any trade in the market updates last_time for the strategy)
            # DATManual: lasttime "The time (value of t) at which the most recent trade occurred in this period"
            # We update our local view of last_time.
            self.last_time = self.current_time
