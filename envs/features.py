import numpy as np
from typing import Any, Dict, List, Optional
from engine.orderbook import OrderBook

class ObservationGenerator:
    """
    Generates normalized observation vectors for RL agents.
    
    Features (Normalized 0-1):
    1. Private State:
       - valuation: Agent's private value for the next token (norm by max_price)
       - held_tokens: Inventory level (norm by max_tokens)
       - time_progress: current_step / max_steps
       
    2. Market State:
       - best_bid: Current highest buy order (norm by max_price)
       - best_ask: Current lowest sell order (norm by max_price)
       - spread: best_ask - best_bid (norm by max_price)
       - mid_price: (best_bid + best_ask) / 2 (norm by max_price)
       
    3. Derived / Strategic:
       - immediate_surplus: Potential profit if Accept is chosen (norm by max_price)
       - inventory_pressure: held_tokens / (1 - time_progress) (clipped 0-1)
       
    4. Market Dynamics (History):
       - price_trend: Rolling average of last 5 trade prices (norm by max_price)
       - volatility: Std dev of last 5 trade prices (norm by max_price)
       - imbalance: (BidVol - AskVol) / (BidVol + AskVol) (shifted to 0-1)
    """
    
    def __init__(self, 
                 max_price: int = 1000, 
                 max_tokens: int = 4, 
                 max_steps: int = 100,
                 history_len: int = 5):
        self.max_price = max_price
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.history_len = history_len
        
        # Feature vector size
        # Private (3) + Market (4) + Derived (2) + Dynamics (3) = 12
        self.feature_dim = 12
        
        # History buffer for price trend/volatility
        self.trade_history: List[int] = []

    def reset(self) -> None:
        """Reset history buffer."""
        self.trade_history = []

    def update_history(self, trade_price: int) -> None:
        """Update trade history with new trade price."""
        self.trade_history.append(trade_price)
        if len(self.trade_history) > self.history_len:
            self.trade_history.pop(0)

    def generate(self, 
                 agent: Any, 
                 orderbook: OrderBook, 
                 current_step: int) -> np.ndarray:
        """
        Generate observation vector for a specific agent.
        
        Args:
            agent: The Agent instance (must have valuations, num_tokens, etc.)
            orderbook: Current OrderBook state
            current_step: Current time step in the period
            
        Returns:
            np.ndarray: Normalized feature vector (shape=(12,), dtype=float32)
        """
        obs = np.zeros(self.feature_dim, dtype=np.float32)
        
        # --- 1. Private State ---
        
        # Valuation for NEXT token
        # If agent has traded all tokens, valuation is 0
        current_holding = agent.num_trades
        if current_holding < agent.num_tokens:
            valuation = agent.valuations[current_holding]
        else:
            valuation = 0
            
        obs[0] = valuation / self.max_price
        
        # Held Tokens (Inventory)
        # Note: agent.num_trades tracks how many they have bought/sold
        # For a buyer, held = num_trades. For seller, held = num_tokens - num_trades?
        # Actually, let's stick to "Inventory Progress" -> num_trades / max_tokens
        obs[1] = current_holding / self.max_tokens
        
        # Time Progress
        obs[2] = current_step / self.max_steps
        
        # --- 2. Market State ---
        
        # Get Best Bid/Ask from OrderBook
        # OrderBook maintains high_bid and low_ask arrays indexed by time
        # We want the current standing best bid/ask
        # If no bid/ask, use defaults (0 for bid, max_price for ask)
        
        # Note: OrderBook.high_bid[t] is populated AFTER determine_winners
        # But we need the state BEFORE the agent acts.
        # We should look at the current standing bests.
        # In AURORA, bids/asks carry over.
        # Let's assume we can access the current bests from the orderbook state.
        # OrderBook doesn't explicitly store "current best" in a simple variable, 
        # it uses the arrays. We need to be careful here.
        # Actually, OrderBook.high_bid[current_step-1] might be the previous step's best.
        # But wait, if we are in the middle of a step, new bids might have come in?
        # No, in AURORA, it's synchronized. We act based on PREVIOUS step's outcome 
        # or the carry-over.
        
        # Let's use the helper method we'll assume exists or implement logic to find it.
        # OrderBook has `_determine_high_bid_low_ask` which runs at end of step.
        # So `high_bid[current_step-1]` is valid.
        
        t_prev = max(0, current_step - 1)
        best_bid = orderbook.high_bid[t_prev]
        best_ask = orderbook.low_ask[t_prev]
        
        # If best_ask is 0 (no ask), treat as max_price for normalization context
        # But wait, 0 means "no ask".
        display_ask = best_ask if best_ask > 0 else self.max_price
        
        obs[3] = best_bid / self.max_price
        obs[4] = display_ask / self.max_price
        
        # Spread
        spread = display_ask - best_bid
        obs[5] = spread / self.max_price
        
        # Mid Price
        mid_price = (best_bid + display_ask) / 2
        obs[6] = mid_price / self.max_price
        
        # --- 3. Derived / Strategic ---
        
        # Immediate Surplus
        # If Buyer: Val - BestAsk
        # If Seller: BestBid - Cost
        immediate_surplus = 0.0
        if agent.is_buyer:
            if best_ask > 0:
                immediate_surplus = valuation - best_ask
        else:
            if best_bid > 0:
                immediate_surplus = best_bid - valuation
                
        # Normalize (can be negative, so offset? or just clip?)
        # Surplus is usually within [-max_price, max_price]. 
        # Let's normalize to [-1, 1] then shift to [0, 1]? 
        # Or just simple division. Let's keep it simple: / max_price.
        # It can be negative. RL nets handle negative inputs fine usually, 
        # but 0-1 is preferred. Let's clip to [-1, 1] and not shift for now, 
        # or just leave as is / max_price.
        obs[7] = np.clip(immediate_surplus / self.max_price, -1.0, 1.0)
        
        # Inventory Pressure
        # held / (1 - time_progress)
        # If time is near end (1.0), pressure explodes.
        remaining_time = 1.0 - obs[2]
        if remaining_time < 0.01:
            pressure = 1.0 # Max pressure
        else:
            # How much of the goal is left vs time left
            # Goal: trade all tokens. 
            # Remaining trades needed: (max_tokens - current_holding)
            # Wait, "held_tokens" usually means "tokens I still need to trade" for a buyer?
            # No, num_trades is "tokens traded".
            remaining_trades = self.max_tokens - current_holding
            pressure = (remaining_trades / self.max_tokens) / remaining_time
            
        obs[8] = np.clip(pressure, 0.0, 1.0)
        
        # --- 4. Market Dynamics ---
        
        # Price Trend & Volatility
        if len(self.trade_history) > 0:
            avg_price = np.mean(self.trade_history)
            std_price = np.std(self.trade_history)
            obs[9] = avg_price / self.max_price
            obs[10] = std_price / self.max_price
        else:
            obs[9] = 0.0
            obs[10] = 0.0
            
        # Imbalance
        # We need volume at best bid/ask.
        # OrderBook doesn't easily give this without iterating.
        # Let's approximate or skip for now if too expensive.
        # Actually, we can just count how many bids are at best_bid.
        # For MVP, let's use a placeholder or simple check.
        # Let's iterate current bids/asks.
        
        bid_vol = 0
        ask_vol = 0
        
        # This iteration might be slow if many agents.
        # But num_agents is small (10-20).
        # We need to check `orderbook.bids[:, current_step]`?
        # No, bids are submitted in the current step.
        # Wait, if we are observing BEFORE action, we see the PREVIOUS step's book?
        # In AURORA, the book clears if a trade happens.
        # If no trade, orders persist.
        # So we look at `orderbook.bids[:, t_prev]`.
        
        if t_prev > 0:
             # Count bids at best_bid
             # bids is (num_buyers+1, num_times+1)
             # We can use numpy for speed
             current_bids = orderbook.bids[1:, t_prev] # Exclude 0 index
             current_asks = orderbook.asks[1:, t_prev]
             
             if best_bid > 0:
                 bid_vol = np.sum(current_bids == best_bid)
             
             if best_ask > 0:
                 ask_vol = np.sum(current_asks == best_ask)
                 
        total_vol = bid_vol + ask_vol
        if total_vol > 0:
            imbalance = (bid_vol - ask_vol) / total_vol # [-1, 1]
        else:
            imbalance = 0.0
            
        obs[11] = (imbalance + 1) / 2 # Shift to [0, 1]
        
        return obs
