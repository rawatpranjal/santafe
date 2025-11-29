import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, Optional, Tuple, List

from engine.market import Market
from engine.token_generator import TokenGenerator
from engine.agent_factory import create_agent
from traders.base import Agent
from envs.features import ObservationGenerator

class RLPuppetAgent(Agent):
    """
    A puppet agent controlled by the RL environment.
    It simply regurgitates the actions set by the environment.
    """
    def __init__(self, player_id: int, is_buyer: bool, num_tokens: int, valuations: list[int]):
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.next_bid_ask: int = -99 # Default to Pass
        self.next_buy_sell: bool = False # Default to Reject

    def bid_ask(self, time: int, nobidask: int) -> None:
        self.has_responded = False

    def bid_ask_response(self) -> int:
        self.has_responded = True
        return self.next_bid_ask

    def buy_sell(self, time: int, nobuysell: int, high_bid: int, low_ask: int, high_bidder: int, low_asker: int) -> None:
        self.has_responded = False

    def buy_sell_response(self) -> bool:
        self.has_responded = True
        return self.next_buy_sell

class DoubleAuctionEnv(gym.Env):
    """
    Gymnasium environment for the Santa Fe Double Auction.
    
    Wraps the Market engine to allow a single RL agent to trade against legacy opponents.
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Configuration
        self.num_agents = config.get("num_agents", 8) # Total agents (e.g. 4 buyers, 4 sellers)
        self.num_tokens = config.get("num_tokens", 4)
        self.max_steps = config.get("max_steps", 100)
        self.min_price = config.get("min_price", 0)
        self.max_price = config.get("max_price", 1000)
        
        # RL Agent Configuration
        # We assume RL agent is always Buyer #1 for now, or configurable
        self.rl_agent_id = config.get("rl_agent_id", 1)
        self.rl_is_buyer = config.get("rl_is_buyer", True)
        
        # Opponent Configuration
        self.opponent_type = config.get("opponent_type", "ZIC")
        
        # Action Space: Discrete(7)
        # 0: Pass, 1: Accept, 2: Match Best, 3: Improve Small, 4: Improve Large, 5: Mid, 6: Truthful
        self.action_space = spaces.Discrete(7)
        
        # Observation Space: Box(12,)
        self.obs_gen = ObservationGenerator(
            max_price=self.max_price,
            max_tokens=self.num_tokens,
            max_steps=self.max_steps
        )
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.obs_gen.feature_dim,), dtype=np.float32
        )
        
        # Internal State
        self.market: Optional[Market] = None
        self.rl_agent: Optional[RLPuppetAgent] = None
        self.token_gen = TokenGenerator(1111, self.num_tokens, None) # Random seed handled in reset
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # 1. Generate Tokens
        # We need to setup N agents.
        # Assume equal split buyers/sellers
        n_buyers = self.num_agents // 2
        n_sellers = self.num_agents - n_buyers
        
        self.token_gen.new_round()
        
        agents: List[Agent] = []
        
        # Create Buyers
        for i in range(n_buyers):
            pid = i + 1
            tokens = self.token_gen.generate_tokens(True)
            
            if self.rl_is_buyer and pid == self.rl_agent_id:
                # This is our RL agent
                self.rl_agent = RLPuppetAgent(pid, True, self.num_tokens, tokens)
                agents.append(self.rl_agent)
            else:
                # Opponent
                agent = create_agent(self.opponent_type, pid, True, self.num_tokens, tokens, 
                                   seed=self.np_random.integers(0, 100000),
                                   num_times=self.max_steps, price_min=self.min_price, price_max=self.max_price)
                agents.append(agent)
                
        # Create Sellers
        for i in range(n_sellers):
            pid = n_buyers + i + 1
            tokens = self.token_gen.generate_tokens(False)
            
            if not self.rl_is_buyer and pid == self.rl_agent_id:
                # This is our RL agent
                self.rl_agent = RLPuppetAgent(pid, False, self.num_tokens, tokens)
                agents.append(self.rl_agent)
            else:
                # Opponent
                agent = create_agent(self.opponent_type, pid, False, self.num_tokens, tokens,
                                   seed=self.np_random.integers(0, 100000),
                                   num_times=self.max_steps, price_min=self.min_price, price_max=self.max_price)
                agents.append(agent)
                
        # 2. Initialize Market
        self.market = Market(
            num_buyers=n_buyers,
            num_sellers=n_sellers,
            num_times=self.max_steps,
            price_min=self.min_price,
            price_max=self.max_price,
            buyers=[a for a in agents if a.is_buyer],
            sellers=[a for a in agents if not a.is_buyer],
            seed=self.np_random.integers(0, 100000)
        )
        
        # Start agents
        for a in agents:
            a.start_period(1)
            
        # Reset Observation Generator
        self.obs_gen.reset()
        
        # 3. Generate Initial Observation
        obs = self.obs_gen.generate(self.rl_agent, self.market.orderbook, 0)
        
        return obs, {}
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.market is None or self.rl_agent is None:
            raise RuntimeError("Call reset() before step()")
            
        # 1. Map Action to Bid/Ask
        bid_price = self._map_action_to_price(action)
        
        # 2. Set Puppet State
        self.rl_agent.next_bid_ask = bid_price
        # Auto-accept if profitable (handled in buy_sell_response logic below?)
        # Actually, for simplicity, let's assume we ALWAYS accept if profitable.
        # But we need to define "profitable".
        # The Market calls buy_sell_response().
        # We can make the Puppet smart enough to check profitability itself?
        # Or we can set it here.
        # Let's make the Puppet smart:
        # But Puppet is dumb. So we set it here.
        # We need to know current high_bid/low_ask to decide?
        # But we don't know them yet! They are determined AFTER bid_ask_stage.
        # So we should set next_buy_sell to True, and let the Puppet's logic (or Market's logic) handle it?
        # Wait, `buy_sell_response` in Market checks `decision and (low_ask > 0)`.
        # If we return True, we accept WHATEVER the price is.
        # This is risky if the price moved against us?
        # In synchronized market, we see the result of bid_ask_stage before buy_sell_stage.
        # But `step()` does both at once.
        # So we can't intervene between stages.
        # We must commit to a policy for buy/sell.
        # Policy: "Accept if surplus > 0".
        # We can implement this in `RLPuppetAgent.buy_sell_response`?
        # But `RLPuppetAgent` is supposed to be dumb.
        # Let's modify `RLPuppetAgent` to take a "strategy" for buy/sell?
        # Or just hardcode "Accept if profitable" in `RLPuppetAgent`.
        # Yes, that's a reasonable assumption for a rational trader.
        # Let's update `RLPuppetAgent` on the fly? No, let's subclass it or modify it.
        # I'll modify `RLPuppetAgent` in this file to check profitability.
        
        # 3. Run Market Step
        # Capture profit before
        profit_before = self.rl_agent.period_profit
        
        # Run step
        self.market.run_time_step()
        
        # Capture profit after
        profit_after = self.rl_agent.period_profit
        step_profit = profit_after - profit_before
        
        # 4. Compute Reward
        reward = float(step_profit)
        
        # 5. Check Termination
        terminated = (self.market.current_time >= self.market.num_times) or self.market.fail_state
        truncated = False
        
        # 6. Generate Observation
        obs = self.obs_gen.generate(self.rl_agent, self.market.orderbook, self.market.current_time)
        
        return obs, reward, terminated, truncated, {}
        
    def _map_action_to_price(self, action: int) -> int:
        """
        Map discrete action to a specific price.
        
        0: Pass (-99)
        1: Accept (Market Order)
        2: Match Best (Limit Order @ Best)
        3: Improve Small (Limit Order @ Best +/- 1)
        4: Improve Large (Limit Order @ Best +/- 5)
        5: Mid-Spread
        6: Truthful
        """
        if action == 0:
            return -99 # Pass
            
        # Get current bests (from previous step)
        # Note: OrderBook state is at `current_time`.
        # If we are at start of step T, we look at T-1.
        t_prev = max(0, self.market.current_time) # Wait, current_time is updated INSIDE run_time_step
        # But here we are BEFORE run_time_step.
        # So `market.current_time` is T-1 (or 0).
        # OrderBook arrays are 1-indexed for time? No, 0-indexed usually?
        # Let's check OrderBook.
        # `self.high_bid = np.zeros(num_times + 1)`
        # So we can access `market.current_time`.
        
        t = self.market.current_time
        best_bid = int(self.market.orderbook.high_bid[t])
        best_ask = int(self.market.orderbook.low_ask[t])
        
        # Handle "None" cases
        if best_ask == 0: best_ask = self.max_price
        if best_bid == 0: best_bid = 0
        
        val = self.rl_agent.get_current_valuation()
        if val == 0: return -99 # No tokens left
        
        price = -99
        
        if self.rl_is_buyer:
            # BUYER LOGIC
            if action == 1: # Accept (Buy at Ask)
                price = best_ask
            elif action == 2: # Match Best Bid
                price = best_bid if best_bid > 0 else self.min_price
            elif action == 3: # Improve Small (Bid + 1)
                price = best_bid + 1
            elif action == 4: # Improve Large (Bid + 5)
                price = best_bid + 5
            elif action == 5: # Mid Spread
                price = int((best_bid + best_ask) / 2)
            elif action == 6: # Truthful
                price = val
                
            # Cap at valuation (Rationality constraint)
            # Or should we let RL learn this?
            # Let's let RL learn it. But "Accept" implies taking the ask.
            # If Ask > Val, we shouldn't Accept?
            # If we submit Bid > Val, we might lose money.
            # Let's cap at max_price and min_price.
            price = min(price, self.max_price)
            price = max(price, self.min_price)
            
        else:
            # SELLER LOGIC
            if action == 1: # Accept (Sell at Bid)
                price = best_bid
            elif action == 2: # Match Best Ask
                price = best_ask if best_ask < self.max_price else self.max_price
            elif action == 3: # Improve Small (Ask - 1)
                price = best_ask - 1
            elif action == 4: # Improve Large (Ask - 5)
                price = best_ask - 5
            elif action == 5: # Mid Spread
                price = int((best_bid + best_ask) / 2)
            elif action == 6: # Truthful
                price = val
                
            price = min(price, self.max_price)
            price = max(price, self.min_price)
            
        return price

    def action_masks(self) -> np.ndarray:
        """
        Return boolean mask of valid actions.
        True = Valid, False = Invalid.
        """
        mask = np.ones(7, dtype=bool)
        
        if self.rl_agent is None or self.market is None:
            return mask
            
        # 1. Check if we can trade
        if not self.rl_agent.can_trade():
            # Can only Pass
            mask[1:] = False
            return mask
            
        # Get state
        t = self.market.current_time
        best_bid = int(self.market.orderbook.high_bid[t])
        best_ask = int(self.market.orderbook.low_ask[t])
        
        # 2. Check specific actions
        if self.rl_is_buyer:
            # Accept: Requires standing Ask
            if best_ask == 0: mask[1] = False
            
            # Match Best: Requires standing Bid
            if best_bid == 0: mask[2] = False
            
            # Improve: Always possible (starts new bid)
            # But maybe meaningless if no bid to improve?
            # If no bid, Improve Small/Large acts as opening bid?
            # Let's allow it.
            pass
        else:
            # Accept: Requires standing Bid
            if best_bid == 0: mask[1] = False
            
            # Match Best: Requires standing Ask
            if best_ask == 0: mask[2] = False
            
        return mask
