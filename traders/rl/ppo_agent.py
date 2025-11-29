from typing import Any, Optional
import numpy as np
from stable_baselines3 import PPO

from traders.base import Agent
from envs.enhanced_features import EnhancedObservationGenerator
from engine.orderbook import OrderBook

class PPOAgent(Agent):
    """
    RL Agent powered by a trained PPO model.
    
    This agent bridges the gap between the Tournament engine and the Stable-Baselines3 model.
    It generates observations, masks invalid actions, and queries the policy for decisions.
    """
    
    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        model_path: str,
        max_price: int = 1000,
        max_steps: int = 100
    ) -> None:
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        
        # Load Model
        self.model = PPO.load(model_path)
        
        # Feature Generator
        self.obs_gen = EnhancedObservationGenerator(
            max_price=max_price,
            max_tokens=num_tokens,
            max_steps=max_steps
        )
        
        # Internal State
        self.current_step = 0
        self.max_price = max_price
        self.min_price = 0
        self.next_buy_sell = False
        
        # OrderBook reference (needed for observation)
        # The Market will pass this via bid_ask_result or we need a way to access it.
        # In the Tournament, agents don't hold a reference to the OrderBook directly.
        # They receive updates via bid_ask_result / buy_sell_result.
        # BUT, ObservationGenerator needs the FULL OrderBook state.
        # This is a challenge for the "Agent" interface which mimics the Java code (limited info).
        # However, for RL, we want full state.
        # We can cheat slightly and inject the orderbook reference if we are running in python.
        # Or we can reconstruct state from messages.
        # Reconstructing is hard.
        # Let's assume we can pass the orderbook to the agent.
        # We'll add a method `set_orderbook(ob)` or similar.
        self.orderbook: Optional[OrderBook] = None

    def set_orderbook(self, orderbook: OrderBook) -> None:
        """Inject orderbook reference for observation generation."""
        self.orderbook = orderbook

    def start_period(self, period_number: int) -> None:
        super().start_period(period_number)
        self.current_step = 0
        self.obs_gen.reset()

    def bid_ask(self, time: int, nobidask: int) -> None:
        self.current_step = time
        self.has_responded = False

    def bid_ask_response(self) -> int:
        if self.orderbook is None:
            # Fallback if no orderbook (shouldn't happen in proper setup)
            self.has_responded = True
            return -99
            
        # 1. Generate Observation
        obs = self.obs_gen.generate(self, self.orderbook, self.current_step)
        
        # 2. Generate Action Mask
        mask = self._get_action_mask()
        
        # 3. Predict Action
        # Note: Standard PPO doesn't support action masking directly in predict
        # We rely on the environment to handle invalid actions (or punish them)
        # For now, we just return the action index
        action, _ = self.model.predict(obs, deterministic=True)
        action_id = int(action)
        
        # 4. Map Action to Price
        price = self._map_action_to_price(action_id)
        
        # 5. Set Buy/Sell Decision (Auto-Accept if profitable)
        # We'll set this now, to be used in buy_sell_response
        self.next_buy_sell = True # Default to accept, but we'll check profitability later
        
        self.has_responded = True
        return price

    def buy_sell(self, time: int, nobuysell: int, high_bid: int, low_ask: int, high_bidder: int, low_asker: int) -> None:
        self.has_responded = False

    def buy_sell_response(self) -> bool:
        self.has_responded = True
        # Rationality check: Only accept if profitable
        # The Env logic assumed "Auto-Accept".
        # Here we can be explicit.
        
        if self.is_buyer:
            # Buying at low_ask
            # We need to know low_ask. We can get it from orderbook or args.
            # But buy_sell_response doesn't take args.
            # We can use self.orderbook.low_ask[current_step]
            t = self.current_step
            low_ask = int(self.orderbook.low_ask[t])
            val = self.get_current_valuation()
            if low_ask > 0 and val >= low_ask:
                return True
        else:
            # Selling at high_bid
            t = self.current_step
            high_bid = int(self.orderbook.high_bid[t])
            val = self.get_current_valuation()
            if high_bid > 0 and high_bid >= val:
                return True
                
        return False

    def _get_action_mask(self) -> np.ndarray:
        """
        Get valid actions with rationality constraints for 24-action space.

        Actions:
            0: Pass (always valid)
            1: Accept (profitable only)
            2-9: Improve by 0.5%, 1%, 2%, 5%, 10%, 15%, 25%, 40% of spread
            10-17: Shade 1%, 3%, 5%, 10%, 15%, 20%, 30%, 40% of valuation
            18: Truthful (always valid if can_trade)
            19: Jump Best (must stay profitable)
            20: Snipe (spread must be < 5%)
            21-23: UnderCut by 2, 5, 10
        """
        mask = np.ones(24, dtype=bool)

        if not self.can_trade():
            mask[1:] = False
            return mask

        t = self.current_step
        t_prev = max(0, t - 1)

        best_bid = int(self.orderbook.high_bid[t_prev]) if t > 0 else 0
        best_ask = int(self.orderbook.low_ask[t_prev]) if t > 0 else 0

        # Handle empty book for calculations
        effective_ask = best_ask if best_ask > 0 else self.max_price
        effective_bid = best_bid if best_bid > 0 else self.min_price
        spread = effective_ask - effective_bid

        val = self.get_current_valuation()

        # Spread improvement percentages
        improve_pcts = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.40]
        # Shade percentages
        shade_pcts = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
        # UnderCut amounts
        undercut_amts = [2, 5, 10]

        if self.is_buyer:
            # Accept: Requires standing Ask AND profitable
            if best_ask == 0 or best_ask > val:
                mask[1] = False

            # Spread improvements (2-9): Need positive spread and result < valuation
            if spread <= 0:
                mask[2:10] = False
            else:
                for i, pct in enumerate(improve_pcts):
                    new_price = effective_bid + max(1, int(pct * spread))
                    if new_price > val:
                        mask[2 + i] = False

            # Shading actions (10-17): Always valid if price >= min_price
            for i, shade in enumerate(shade_pcts):
                if int(val * (1 - shade)) < self.min_price:
                    mask[10 + i] = False

            # Truthful (18): Always valid if can_trade

            # Jump Best (19): Need existing bid and result <= valuation
            if best_bid == 0 or best_bid + 1 > val:
                mask[19] = False

            # Snipe (20): Only valid if spread < 5% AND ask is profitable
            if spread <= 0 or spread / effective_ask >= 0.05 or best_ask > val:
                mask[20] = False

            # UnderCut (21-23): Beat best bid by fixed amount
            for i, amt in enumerate(undercut_amts):
                new_price = best_bid + amt
                if best_bid == 0 or new_price > val:
                    mask[21 + i] = False

        else:  # Seller
            # Accept: Requires standing Bid AND profitable
            if best_bid == 0 or best_bid < val:
                mask[1] = False

            # Spread improvements (2-9): Need positive spread and result > valuation
            if spread <= 0:
                mask[2:10] = False
            else:
                for i, pct in enumerate(improve_pcts):
                    new_price = effective_ask - max(1, int(pct * spread))
                    if new_price < val:
                        mask[2 + i] = False

            # Shading actions (10-17): Always valid if price <= max_price
            for i, shade in enumerate(shade_pcts):
                if int(val * (1 + shade)) > self.max_price:
                    mask[10 + i] = False

            # Truthful (18): Always valid if can_trade

            # Jump Best (19): Need existing ask and result >= valuation
            if best_ask == 0 or best_ask - 1 < val:
                mask[19] = False

            # Snipe (20): Only valid if spread < 5% AND bid is profitable
            if spread <= 0 or spread / effective_ask >= 0.05 or best_bid < val:
                mask[20] = False

            # UnderCut (21-23): Beat best ask by fixed amount (lower ask)
            for i, amt in enumerate(undercut_amts):
                new_price = best_ask - amt
                if best_ask == 0 or new_price < val:
                    mask[21 + i] = False

        return mask

    def _map_action_to_price(self, action: int) -> int:
        """
        Map 24 discrete actions to prices using spread-relative and valuation-based strategies.

        Actions:
            0: Pass
            1: Accept (buy at ask / sell at bid)
            2-9: Improve by 0.5%, 1%, 2%, 5%, 10%, 15%, 25%, 40% of spread
            10-17: Shade 1%, 3%, 5%, 10%, 15%, 20%, 30%, 40% of valuation
            18: Truthful (bid/ask at valuation)
            19: Jump Best (improve by 1)
            20: Snipe (accept only if spread < 5%)
            21-23: UnderCut by 2, 5, 10
        """
        if action == 0:
            return -99  # Pass

        t = self.current_step
        t_prev = max(0, t - 1)

        best_bid = int(self.orderbook.high_bid[t_prev]) if t > 0 else 0
        best_ask = int(self.orderbook.low_ask[t_prev]) if t > 0 else 0

        # Handle empty book
        if best_ask == 0:
            best_ask = self.max_price
        if best_bid == 0:
            best_bid = self.min_price

        spread = best_ask - best_bid
        val = self.get_current_valuation()
        if val == 0:
            return -99

        price = -99

        # Spread improvement percentages
        improve_pcts = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.40]
        # Shade percentages
        shade_pcts = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
        # UnderCut amounts
        undercut_amts = [2, 5, 10]

        if self.is_buyer:
            if action == 1:    # Accept (Buy at Ask)
                price = best_ask
            elif 2 <= action <= 9:  # Spread improvements
                pct = improve_pcts[action - 2]
                price = best_bid + max(1, int(pct * spread))
            elif 10 <= action <= 17:  # Shade valuation
                shade = shade_pcts[action - 10]
                price = int(val * (1 - shade))
            elif action == 18:  # Truthful
                price = val
            elif action == 19:  # Jump Best (improve by 1)
                price = best_bid + 1
            elif action == 20:  # Snipe (accept only if spread < 5%)
                if spread > 0 and spread / best_ask < 0.05:
                    price = best_ask
                else:
                    return -99  # Pass if spread too wide
            elif 21 <= action <= 23:  # UnderCut
                amt = undercut_amts[action - 21]
                price = best_bid + amt

            # Cap at valuation (never bid above what it's worth)
            if price > val:
                price = val

        else:  # Seller
            if action == 1:    # Accept (Sell at Bid)
                price = best_bid
            elif 2 <= action <= 9:  # Spread improvements
                pct = improve_pcts[action - 2]
                price = best_ask - max(1, int(pct * spread))
            elif 10 <= action <= 17:  # Shade valuation
                shade = shade_pcts[action - 10]
                price = int(val * (1 + shade))
            elif action == 18:  # Truthful
                price = val
            elif action == 19:  # Jump Best (improve by 1)
                price = best_ask - 1
            elif action == 20:  # Snipe (accept only if spread < 5%)
                if spread > 0 and spread / best_ask < 0.05:
                    price = best_bid
                else:
                    return -99  # Pass if spread too wide
            elif 21 <= action <= 23:  # UnderCut
                amt = undercut_amts[action - 21]
                price = best_ask - amt

            # Floor at valuation (never ask below cost)
            if price < val:
                price = val

        # Ensure price is within bounds
        price = min(price, self.max_price)
        price = max(price, self.min_price)

        return price
