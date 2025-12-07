"""
Enhanced Double Auction Environment with sophisticated rewards and features.

Key improvements:
1. Rich observation space (48 features) with market microstructure and env context
2. Sophisticated reward shaping (profit + market making + exploration)
3. Proper action masking with rationality constraints
4. Curriculum support via difficulty levels
5. Environment sampling from 10 Santa Fe configurations
6. Opponent sampling from {ZIC1, ZIC2, ZIP1, ZIP2}
"""

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from engine.agent_factory import create_agent
from engine.market import Market
from engine.metrics import calculate_equilibrium_profit
from engine.token_generator import TokenGenerator, UniformTokenGenerator
from envs.enhanced_features import EnhancedObservationGenerator
from traders.base import Agent

# Santa Fe Tournament Environments
# Reference: checklists/env.md
SANTA_FE_ENVIRONMENTS = {
    "BASE": {
        "gametype": 6453,
        "max_price": 2000,
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "max_steps": 75,
    },
    "BBBS": {
        "gametype": 6453,
        "max_price": 2000,
        "num_buyers": 6,
        "num_sellers": 2,
        "num_tokens": 4,
        "max_steps": 50,
    },
    "BSSS": {
        "gametype": 6453,
        "max_price": 2000,
        "num_buyers": 2,
        "num_sellers": 6,
        "num_tokens": 4,
        "max_steps": 50,
    },
    "EQL": {
        "gametype": 0,
        "max_price": 2000,
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "max_steps": 75,
    },
    "LAD": {
        "gametype": 0,
        "max_price": 2000,
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "max_steps": 75,
    },
    "PER": {
        "gametype": 6453,
        "max_price": 2000,
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "max_steps": 75,
    },
    "SHRT": {
        "gametype": 6453,
        "max_price": 2000,
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "max_steps": 25,
    },
    "SML": {
        "gametype": 6453,
        "max_price": 2000,
        "num_buyers": 2,
        "num_sellers": 2,
        "num_tokens": 4,
        "max_steps": 50,
    },
    "RAN": {
        "gametype": 7,
        "max_price": 3000,
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "max_steps": 50,
    },
    "TOK": {
        "gametype": 6453,
        "max_price": 2000,
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 1,
        "max_steps": 25,
    },
}

# Opponent pool for mixed training
OPPONENT_POOL = ["ZIC", "ZIC2", "ZIP", "ZIP2"]


@dataclass
class RewardComponents:
    """Track different reward components for analysis."""

    trade_profit: float = 0.0
    market_making: float = 0.0
    exploration: float = 0.0
    invalid_penalty: float = 0.0
    efficiency_bonus: float = 0.0
    bid_submission: float = 0.0  # NEW: Reward for placing competitive orders
    surplus_capture: float = 0.0  # NEW: Reward for exploiting profitable opportunities
    total: float = 0.0


class EnhancedRLAgent(Agent):
    """
    Enhanced puppet agent with better buy/sell logic and tracking.
    """

    def __init__(self, player_id: int, is_buyer: bool, num_tokens: int, valuations: list[int]):
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.next_bid_ask: int = -99
        self.next_buy_sell: bool = False
        self.last_action: int = 0
        self.trades_executed: int = 0
        self.profitable_trades: int = 0

    def bid_ask(self, time: int, nobidask: int) -> None:
        self.has_responded = False

    def bid_ask_response(self) -> int:
        self.has_responded = True
        return self.next_bid_ask

    def buy_sell(
        self,
        time: int,
        nobuysell: int,
        high_bid: int,
        low_ask: int,
        high_bidder: int,
        low_asker: int,
    ) -> None:
        self.has_responded = False

        # Smart buy/sell decision based on profitability
        if self.is_buyer and low_ask > 0:
            val = self.get_current_valuation()
            self.next_buy_sell = val >= low_ask
        elif not self.is_buyer and high_bid > 0:
            val = self.get_current_valuation()
            self.next_buy_sell = high_bid >= val
        else:
            self.next_buy_sell = False

    def buy_sell_response(self) -> bool:
        self.has_responded = True
        return self.next_buy_sell


class EnhancedDoubleAuctionEnv(gym.Env):
    """
    Enhanced Gymnasium environment for Santa Fe Double Auction.

    Features:
    - Rich 24-dimensional observation space
    - Sophisticated reward shaping
    - Proper action masking with rationality
    - Curriculum learning support
    - Detailed metrics tracking
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config

        # Universal PPO Training Mode
        # When enabled, samples environment and opponents each episode
        self.sample_env = config.get("sample_env", False)
        self.sample_opponents = config.get("sample_opponents", False)
        self.env_list = config.get("env_list", list(SANTA_FE_ENVIRONMENTS.keys()))
        self.opponent_pool = config.get("opponent_pool", OPPONENT_POOL)

        # Current environment name (for logging)
        self.current_env_name = config.get("env_name", "BASE")

        # Market Configuration (defaults, may be overridden by sampled env)
        self.num_agents = config.get("num_agents", 8)
        self.num_tokens = config.get("num_tokens", 4)
        self.max_steps = config.get("max_steps", 100)
        self.min_price = config.get("min_price", 0)
        self.max_price = config.get("max_price", 1000)

        # RL Agent Configuration
        self.rl_is_buyer = config.get("rl_is_buyer", True)
        # Auto-adjust rl_agent_id for sellers (seller PIDs start at n_buyers+1)
        default_rl_id = 1 if self.rl_is_buyer else (self.num_agents // 2) + 1
        self.rl_agent_id = config.get("rl_agent_id", default_rl_id)

        # Opponent Configuration (supports curriculum)
        self.opponent_type = config.get("opponent_type", "ZIC")
        self.opponent_mix = config.get("opponent_mix", None)  # For mixed populations

        # Token Generation Configuration
        # gametype: None = UniformTokenGenerator, int = TokenGenerator with gametype formula
        self.gametype = config.get("gametype", None)

        # Reward Configuration
        # Pure profit mode: Only use raw profit as reward (can be negative)
        # This is the simplest and most direct signal for learning
        self.pure_profit_mode = config.get("pure_profit_mode", False)

        self.reward_config = {
            "profit_weight": config.get("profit_weight", 1.0),
            "market_making_weight": config.get("market_making_weight", 0.1),
            "exploration_weight": config.get("exploration_weight", 0.01),
            "invalid_penalty": config.get("invalid_penalty", -0.1),
            "efficiency_bonus_weight": config.get("efficiency_bonus_weight", 0.05),
            "normalize_rewards": config.get("normalize_rewards", True),
        }

        # Curriculum Configuration
        self.difficulty = config.get("difficulty", "easy")  # easy, medium, hard, expert

        # Action Space: 24 discrete actions (expanded for better performance)
        # 0: Pass
        # 1: Accept (buy at ask / sell at bid)
        # 2-9: Improve by 0.5%, 1%, 2%, 5%, 10%, 15%, 25%, 40% of spread (8 actions)
        # 10-17: Shade 1%, 3%, 5%, 10%, 15%, 20%, 30%, 40% of valuation (8 actions)
        # 18: Truthful (bid/ask at valuation)
        # 19: Jump Best (improve by 1)
        # 20: Snipe (accept only if spread < 5%)
        # 21-23: UnderCut by 2, 5, 10 (beat best price by fixed amount)
        self.action_space = spaces.Discrete(50)  # Expanded for more granularity

        # Observation Space: 24 features
        self.obs_gen = EnhancedObservationGenerator(
            max_price=self.max_price,
            max_tokens=self.num_tokens,
            max_steps=self.max_steps,
            num_agents=self.num_agents,
        )
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.obs_gen.feature_dim,), dtype=np.float32
        )

        # Internal State
        self.market: Market | None = None
        self.rl_agent: EnhancedRLAgent | None = None
        # Token generator: gametype-based (tournament) or uniform (default)
        n_buyers = self.num_agents // 2
        n_sellers = self.num_agents - n_buyers
        self._n_buyers = n_buyers
        self._n_sellers = n_sellers
        if self.gametype:
            # Use gametype formula for tournament-matching token distribution
            self.token_gen: TokenGenerator | UniformTokenGenerator = TokenGenerator(
                self.gametype, self.num_tokens, seed=42
            )
        else:
            # Default: uniform random valuations
            self.token_gen = UniformTokenGenerator(
                num_tokens=self.num_tokens,
                price_min=self.min_price,
                price_max=self.max_price,
                seed=42,
                num_buyers=n_buyers,
                num_sellers=n_sellers,
            )

        # Metrics Tracking
        self.episode_metrics = {
            "total_profit": 0,
            "trades_executed": 0,
            "profitable_trades": 0,
            "invalid_actions": 0,
            "market_efficiency": 0.0,
            "reward_components": RewardComponents(),
        }

        # Time-based tracking for Skeleton-style features
        self._steps_since_last_trade = 0

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Sample environment if enabled
        if self.sample_env:
            self.current_env_name = self.np_random.choice(self.env_list)
            env_config = SANTA_FE_ENVIRONMENTS[self.current_env_name]

            # Apply environment configuration
            self.gametype = env_config["gametype"]
            self.max_price = env_config["max_price"]
            self.num_tokens = env_config["num_tokens"]
            self.max_steps = env_config["max_steps"]
            self._n_buyers = env_config["num_buyers"]
            self._n_sellers = env_config["num_sellers"]
            self.num_agents = self._n_buyers + self._n_sellers

            # Update observation space dimensions (stays at 48)
            self.obs_gen = EnhancedObservationGenerator(
                max_price=self.max_price,
                max_tokens=self.num_tokens,
                max_steps=self.max_steps,
                num_agents=self.num_agents,
                num_buyers=self._n_buyers,
                num_sellers=self._n_sellers,
                gametype=self.gametype,
            )

        # Sample opponents if enabled
        if self.sample_opponents:
            # Sample opponent types for all non-RL agents
            num_opponents = self.num_agents - 1
            self.opponent_mix = [
                self.np_random.choice(self.opponent_pool) for _ in range(num_opponents)
            ]

        # Update token generator with new seed for variety
        new_seed = self.np_random.integers(0, 2**31)
        if self.gametype:
            # TokenGenerator: create new instance with new seed and call new_round
            self.token_gen = TokenGenerator(self.gametype, self.num_tokens, seed=new_seed)
            self.token_gen.new_round()
        else:
            # UniformTokenGenerator: just update rng
            n_buyers = self._n_buyers if hasattr(self, "_n_buyers") else self.num_agents // 2
            n_sellers = (
                self._n_sellers if hasattr(self, "_n_sellers") else self.num_agents - n_buyers
            )
            self.token_gen = UniformTokenGenerator(
                num_tokens=self.num_tokens,
                price_min=self.min_price,
                price_max=self.max_price,
                seed=new_seed,
                num_buyers=n_buyers,
                num_sellers=n_sellers,
            )

        # Reset metrics
        self.episode_metrics = {
            "total_profit": 0,
            "trades_executed": 0,
            "profitable_trades": 0,
            "invalid_actions": 0,
            "market_efficiency": 0.0,
            "reward_components": RewardComponents(),
        }

        # Generate agents based on difficulty/curriculum
        agents = self._create_agents()

        # Initialize Market
        n_buyers = self._n_buyers if hasattr(self, "_n_buyers") else self.num_agents // 2
        n_sellers = self._n_sellers if hasattr(self, "_n_sellers") else self.num_agents - n_buyers

        self.market = Market(
            num_buyers=n_buyers,
            num_sellers=n_sellers,
            num_times=self.max_steps,
            price_min=self.min_price,
            price_max=self.max_price,
            buyers=[a for a in agents if a.is_buyer],
            sellers=[a for a in agents if not a.is_buyer],
            seed=self.np_random.integers(0, 100000),
        )

        # Start agents
        for a in agents:
            a.start_period(1)

        # Reset Observation Generator and set env context
        self.obs_gen.reset()
        self.obs_gen.set_env_context(
            num_buyers=n_buyers,
            num_sellers=n_sellers,
            num_tokens=self.num_tokens,
            max_steps=self.max_steps,
            gametype=self.gametype if self.gametype else 6453,
        )

        # Reset time-based tracking
        self._steps_since_last_trade = 0

        # Generate Initial Observation
        obs = self.obs_gen.generate(
            self.rl_agent, self.market.orderbook, 0, self._steps_since_last_trade
        )

        # Info includes action mask and current env
        info = {
            "action_mask": self._get_action_mask(),
            "difficulty": self.difficulty,
            "env_name": self.current_env_name,
            "opponent_mix": self.opponent_mix,
        }

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.market is None or self.rl_agent is None:
            raise RuntimeError("Call reset() before step()")

        # Track action
        self.rl_agent.last_action = action

        # Check if action is valid
        mask = self._get_action_mask()
        invalid_action = not mask[action]

        # Map Action to Price
        if invalid_action:
            bid_price = -99  # Force pass
            self.episode_metrics["invalid_actions"] += 1
        else:
            bid_price = self._map_action_to_price(action)

        # Set Agent State
        self.rl_agent.next_bid_ask = bid_price

        # Capture state before market step
        profit_before = self.rl_agent.period_profit
        trades_before = self.rl_agent.num_trades

        # Run Market Step
        self.market.run_time_step()

        # Capture state after market step
        profit_after = self.rl_agent.period_profit
        trades_after = self.rl_agent.num_trades

        # Update trade tracking and time-based features
        self._steps_since_last_trade += 1  # Increment each step

        if trades_after > trades_before:
            self.rl_agent.trades_executed += 1
            self.episode_metrics["trades_executed"] += 1
            if profit_after > profit_before:
                self.rl_agent.profitable_trades += 1
                self.episode_metrics["profitable_trades"] += 1

            # Update observation generator with trade
            current_time = self.market.orderbook.current_time
            trade_price = self.market.orderbook.trade_price[current_time]
            if trade_price > 0:
                self.obs_gen.update_trade(trade_price)

            # Reset time-based counter on trade
            self._steps_since_last_trade = 0

        # Calculate Reward
        reward_components = self._calculate_reward(
            profit_before, profit_after, trades_before, trades_after, action, invalid_action
        )
        reward = reward_components.total

        # Update metrics
        self.episode_metrics["total_profit"] = profit_after
        self.episode_metrics["reward_components"] = reward_components

        # Check Termination
        terminated = (
            (self.market.current_time >= self.market.num_times)
            or self.market.fail_state
            or (not self.rl_agent.can_trade())
        )

        # Calculate market efficiency at end of episode
        if terminated and self.market is not None:
            # Calculate equilibrium profit
            all_valuations = []
            all_costs = []

            # Collect all valuations/costs from all agents
            # We need to access the agents from the market or self._create_agents logic
            # The market object has buyers and sellers lists
            for buyer in self.market.buyers:
                all_valuations.extend(buyer.valuations)
            for seller in self.market.sellers:
                all_costs.extend(seller.valuations)  # For sellers, valuations are costs

            max_profit = calculate_equilibrium_profit(all_valuations, all_costs)

            # Calculate actual total profit of all agents
            actual_profit = 0
            for buyer in self.market.buyers:
                actual_profit += buyer.period_profit
            for seller in self.market.sellers:
                actual_profit += seller.period_profit

            if max_profit > 0:
                self.episode_metrics["market_efficiency"] = actual_profit / max_profit
            else:
                self.episode_metrics["market_efficiency"] = 0.0
        truncated = False

        # Generate Next Observation
        obs = self.obs_gen.generate(
            self.rl_agent,
            self.market.orderbook,
            self.market.current_time,
            self._steps_since_last_trade,
        )

        # Prepare Info
        info = {
            "action_mask": self._get_action_mask(),
            "metrics": self.episode_metrics.copy(),
            "reward_breakdown": {
                "trade_profit": reward_components.trade_profit,
                "market_making": reward_components.market_making,
                "exploration": reward_components.exploration,
                "invalid_penalty": reward_components.invalid_penalty,
                "efficiency_bonus": reward_components.efficiency_bonus,
            },
        }

        return obs, reward, terminated, truncated, info

    def _create_agents(self) -> list[Agent]:
        """Create agents based on difficulty/curriculum settings."""
        agents: list[Agent] = []

        n_buyers = self._n_buyers if hasattr(self, "_n_buyers") else self.num_agents // 2
        n_sellers = self._n_sellers if hasattr(self, "_n_sellers") else self.num_agents - n_buyers

        self.token_gen.new_round()

        # Determine opponent types based on difficulty or sampled mix
        opponent_types = self._get_opponent_types()
        opponent_idx = 0  # Index into opponent_types list

        # Create Buyers
        for i in range(n_buyers):
            pid = i + 1
            tokens = self.token_gen.generate_tokens(True)

            if self.rl_is_buyer and pid == self.rl_agent_id:
                # This is our RL agent
                self.rl_agent = EnhancedRLAgent(pid, True, self.num_tokens, tokens)
                agents.append(self.rl_agent)
            else:
                # Opponent - use next type from list
                opp_type = opponent_types[opponent_idx % len(opponent_types)]
                opponent_idx += 1
                agent = create_agent(
                    opp_type,
                    pid,
                    True,
                    self.num_tokens,
                    tokens,
                    seed=self.np_random.integers(0, 100000),
                    num_times=self.max_steps,
                    price_min=self.min_price,
                    price_max=self.max_price,
                )
                agents.append(agent)

        # Create Sellers
        for i in range(n_sellers):
            pid = n_buyers + i + 1
            tokens = self.token_gen.generate_tokens(False)

            if not self.rl_is_buyer and pid == self.rl_agent_id:
                # This is our RL agent
                self.rl_agent = EnhancedRLAgent(pid, False, self.num_tokens, tokens)
                agents.append(self.rl_agent)
            else:
                # Opponent - use next type from list
                opp_type = opponent_types[opponent_idx % len(opponent_types)]
                opponent_idx += 1
                agent = create_agent(
                    opp_type,
                    pid,
                    False,
                    self.num_tokens,
                    tokens,
                    seed=self.np_random.integers(0, 100000),
                    num_times=self.max_steps,
                    price_min=self.min_price,
                    price_max=self.max_price,
                )
                agents.append(agent)

        return agents

    def _get_opponent_types(self) -> list[str]:
        """Get opponent types based on difficulty/curriculum."""
        if self.opponent_mix is not None:
            # Custom mix provided
            return self.opponent_mix

        if self.difficulty == "easy":
            return ["ZIC"]  # Only random traders
        elif self.difficulty == "medium":
            return ["ZIC", "ZIC", "ZIP"]  # Mostly random, some adaptive
        elif self.difficulty == "hard":
            return ["ZIC", "ZIP", "GD"]  # Mix of strategies
        elif self.difficulty == "expert":
            return ["ZIP", "GD", "Kaplan"]  # Sophisticated opponents
        elif self.opponent_type == "Mixed":
            # ALL tournament strategies - including top performers Ringuette (#1) and Ledyard (#2)
            return ["ZIC", "ZIP", "Skeleton", "GD", "Kaplan", "Ringuette", "Ledyard", "Markup"]
        else:
            return [self.opponent_type]  # Single type specified

    def _calculate_reward(
        self,
        profit_before: float,
        profit_after: float,
        trades_before: int,
        trades_after: int,
        action: int,
        invalid: bool,
    ) -> RewardComponents:
        """Calculate multi-component reward."""
        components = RewardComponents()

        # Pure profit mode: return raw profit only (can be negative)
        # This is the simplest signal - matches Chen et al. approach
        if self.pure_profit_mode:
            step_profit = profit_after - profit_before
            components.trade_profit = step_profit
            components.total = step_profit
            return components

        # 1. Trade Profit (main component)
        step_profit = profit_after - profit_before
        components.trade_profit = step_profit * self.reward_config["profit_weight"]

        # 2. Market Making Bonus (for providing liquidity)
        if action in [2, 3, 4, 5] and not invalid:  # Limit orders
            t = max(0, self.market.current_time - 1)
            best_bid = self.market.orderbook.high_bid[t]
            best_ask = self.market.orderbook.low_ask[t]

            # Reward for tightening the spread
            if best_bid > 0 and best_ask > 0:
                old_spread = best_ask - best_bid
                # Estimate new spread after our order
                if self.rl_is_buyer and action in [3, 4]:  # Improving bid
                    new_spread = best_ask - (best_bid + (1 if action == 3 else 5))
                elif not self.rl_is_buyer and action in [3, 4]:  # Improving ask
                    new_spread = (best_ask - (1 if action == 3 else 5)) - best_bid
                else:
                    new_spread = old_spread

                if new_spread < old_spread and new_spread > 0:
                    improvement = (old_spread - new_spread) / self.max_price
                    components.market_making = (
                        improvement * self.reward_config["market_making_weight"]
                    )

        # 3. Exploration Bonus (for trying different actions)
        if action != 0 and not invalid:  # Non-pass actions
            components.exploration = self.reward_config["exploration_weight"]

        # 4. Invalid Action Penalty
        if invalid:
            components.invalid_penalty = self.reward_config["invalid_penalty"]

        # 5. Efficiency Bonus (for completing trades efficiently)
        if trades_after > trades_before:
            # Calculate efficiency of the trade
            val = self.rl_agent.valuations[trades_before]
            current_time = self.market.orderbook.current_time
            trade_price = self.market.orderbook.trade_price[current_time]

            if self.rl_is_buyer:
                if trade_price > 0:
                    efficiency = (val - trade_price) / val if val > 0 else 0
                else:
                    efficiency = 0
            else:
                if trade_price > 0:
                    efficiency = (trade_price - val) / trade_price if trade_price > 0 else 0
                else:
                    efficiency = 0

            if efficiency > 0:
                components.efficiency_bonus = (
                    efficiency * self.reward_config["efficiency_bonus_weight"]
                )

        # 6. Bid Submission Bonus (NEW: reward for placing competitive orders)
        if action in [1, 2, 3, 4, 5, 6] and not invalid:  # Any non-pass action
            if "bid_submission_bonus" in self.reward_config:
                components.bid_submission = self.reward_config["bid_submission_bonus"]

        # 7. Surplus Capture Reward (NEW: reward for exploiting profitable opportunities)
        if action in [1, 2, 3, 4, 5, 6] and not invalid:  # Any non-pass action
            if "surplus_capture_weight" in self.reward_config:
                # Get current surplus from observation (feature 13 in observation space)
                obs = self._get_observation()
                surplus = obs[13]  # Surplus feature (already normalized to [0,1])
                if surplus > 0:
                    components.surplus_capture = (
                        surplus * self.reward_config["surplus_capture_weight"]
                    )

        # Calculate total
        components.total = (
            components.trade_profit
            + components.market_making
            + components.exploration
            + components.invalid_penalty
            + components.efficiency_bonus
            + components.bid_submission
            + components.surplus_capture
        )

        # Normalize if requested
        if self.reward_config["normalize_rewards"]:
            # Normalize to roughly [-1, 1] range
            max_expected_profit = self.max_price * 0.1  # 10% of max price
            if abs(components.total) > 0:
                components.total = np.tanh(components.total / max_expected_profit)

        return components

    def _get_action_mask(self) -> np.ndarray:
        """
        Get valid actions with rationality constraints for 50-action space.

        Actions:
            0: Pass (always valid)
            1: Accept (profitable only)
            2-17: Improve by spread percentages (16 levels)
            18-33: Shade valuation (16 levels)
            34: Truthful (always valid if can_trade)
            35: Jump Best (must stay profitable)
            36: Snipe (spread must be < 5%)
            37-42: UnderCut by 1, 2, 3, 5, 10, 20
            43-46: Aggressive accept at wider spreads
            47-49: Mid-spread bids/asks
        """
        mask = np.ones(50, dtype=bool)

        if self.rl_agent is None or self.market is None:
            return mask

        # Check if we can trade
        if not self.rl_agent.can_trade():
            mask[1:] = False  # Can only Pass
            return mask

        # Get market state
        t = max(0, self.market.current_time)
        if t > 0:
            t = t - 1  # Look at previous step

        best_bid = int(self.market.orderbook.high_bid[t]) if t >= 0 else 0
        best_ask = int(self.market.orderbook.low_ask[t]) if t >= 0 else 0

        # Handle empty book for calculations
        effective_ask = best_ask if best_ask > 0 else self.max_price
        effective_bid = best_bid if best_bid > 0 else self.min_price
        spread = effective_ask - effective_bid

        val = self.rl_agent.get_current_valuation()

        # More granular spread improvement percentages (16 levels)
        improve_pcts = [
            0.0025,
            0.005,
            0.0075,
            0.01,
            0.015,
            0.02,
            0.03,
            0.05,
            0.07,
            0.10,
            0.12,
            0.15,
            0.20,
            0.30,
            0.45,
            0.60,
        ]
        # More granular shade percentages (16 levels)
        shade_pcts = [
            0.005,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.07,
            0.10,
            0.12,
            0.15,
            0.18,
            0.20,
            0.25,
            0.30,
            0.40,
            0.50,
        ]
        # More undercut amounts (6 levels)
        undercut_amts = [1, 2, 3, 5, 10, 20]
        # Aggressive thresholds
        aggressive_thresholds = [0.10, 0.15, 0.20, 0.30]

        if self.rl_is_buyer:
            # Accept: Requires standing Ask AND profitable
            if best_ask == 0 or best_ask > val:
                mask[1] = False

            # Spread improvements (2-17): Need positive spread and result < valuation
            if spread <= 0:
                mask[2:18] = False
            else:
                for i, pct in enumerate(improve_pcts):
                    new_price = effective_bid + max(1, int(pct * spread))
                    if new_price > val:
                        mask[2 + i] = False

            # Shading actions (18-33): Always valid if price >= min_price
            for i, shade in enumerate(shade_pcts):
                if int(val * (1 - shade)) < self.min_price:
                    mask[18 + i] = False

            # Truthful (34): Always valid if can_trade

            # Jump Best (35): Need existing bid and result <= valuation
            if best_bid == 0 or best_bid + 1 > val:
                mask[35] = False

            # Snipe (36): Only valid if spread < 5% AND ask is profitable
            if spread <= 0 or spread / effective_ask >= 0.05 or best_ask > val:
                mask[36] = False

            # UnderCut (37-42): Beat best bid by fixed amount
            for i, amt in enumerate(undercut_amts):
                new_price = best_bid + amt
                if best_bid == 0 or new_price > val:
                    mask[37 + i] = False

            # Aggressive accepts (43-46): Accept if spread < threshold AND profitable
            for i, thresh in enumerate(aggressive_thresholds):
                if spread <= 0 or spread / effective_ask >= thresh or best_ask > val:
                    mask[43 + i] = False

            # Mid-spread bids (47-49)
            if spread <= 0:
                mask[47:50] = False
            else:
                mid_pcts = [0.40, 0.50, 0.60]
                for i, pct in enumerate(mid_pcts):
                    new_price = effective_bid + int(pct * spread)
                    if new_price > val:
                        mask[47 + i] = False

        else:  # Seller
            # Accept: Requires standing Bid AND profitable
            if best_bid == 0 or best_bid < val:
                mask[1] = False

            # Spread improvements (2-17): Need positive spread and result > valuation
            if spread <= 0:
                mask[2:18] = False
            else:
                for i, pct in enumerate(improve_pcts):
                    new_price = effective_ask - max(1, int(pct * spread))
                    if new_price < val:
                        mask[2 + i] = False

            # Shading actions (18-33): Always valid if price <= max_price
            for i, shade in enumerate(shade_pcts):
                if int(val * (1 + shade)) > self.max_price:
                    mask[18 + i] = False

            # Truthful (34): Always valid if can_trade

            # Jump Best (35): Need existing ask and result >= valuation
            if best_ask == 0 or best_ask - 1 < val:
                mask[35] = False

            # Snipe (36): Only valid if spread < 5% AND bid is profitable
            if spread <= 0 or spread / effective_ask >= 0.05 or best_bid < val:
                mask[36] = False

            # UnderCut (37-42): Beat best ask by fixed amount (lower ask)
            for i, amt in enumerate(undercut_amts):
                new_price = best_ask - amt
                if best_ask == 0 or new_price < val:
                    mask[37 + i] = False

            # Aggressive accepts (43-46): Accept if spread < threshold AND profitable
            for i, thresh in enumerate(aggressive_thresholds):
                if spread <= 0 or spread / effective_ask >= thresh or best_bid < val:
                    mask[43 + i] = False

            # Mid-spread asks (47-49)
            if spread <= 0:
                mask[47:50] = False
            else:
                mid_pcts = [0.40, 0.50, 0.60]
                for i, pct in enumerate(mid_pcts):
                    new_price = effective_ask - int(pct * spread)
                    if new_price < val:
                        mask[47 + i] = False

        return mask

    def _map_action_to_price(self, action: int) -> int:
        """
        Map 50 discrete actions to prices using spread-relative and valuation-based strategies.

        Actions:
            0: Pass
            1: Accept (buy at ask / sell at bid)
            2-17: Improve by spread percentages (16 levels: 0.25% to 60%)
            18-33: Shade valuation (16 levels: 0.5% to 50%)
            34: Truthful (bid/ask at valuation)
            35: Jump Best (improve by 1)
            36: Snipe (accept only if spread < 5%)
            37-42: UnderCut by 1, 2, 3, 5, 10, 20
            43-49: Reserved / aggressive actions
        """
        if action == 0:
            return -99  # Pass

        t = max(0, self.market.current_time - 1)
        best_bid = int(self.market.orderbook.high_bid[t]) if t >= 0 else 0
        best_ask = int(self.market.orderbook.low_ask[t]) if t >= 0 else 0

        # Handle empty book
        if best_ask == 0:
            best_ask = self.max_price
        if best_bid == 0:
            best_bid = self.min_price

        spread = best_ask - best_bid
        val = self.rl_agent.get_current_valuation()
        if val == 0:
            return -99

        price = -99

        # More granular spread improvement percentages (16 levels)
        improve_pcts = [
            0.0025,
            0.005,
            0.0075,
            0.01,
            0.015,
            0.02,
            0.03,
            0.05,
            0.07,
            0.10,
            0.12,
            0.15,
            0.20,
            0.30,
            0.45,
            0.60,
        ]
        # More granular shade percentages (16 levels)
        shade_pcts = [
            0.005,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.07,
            0.10,
            0.12,
            0.15,
            0.18,
            0.20,
            0.25,
            0.30,
            0.40,
            0.50,
        ]
        # More undercut amounts (6 levels)
        undercut_amts = [1, 2, 3, 5, 10, 20]

        if self.rl_is_buyer:
            if action == 1:  # Accept (Buy at Ask)
                price = best_ask
            elif 2 <= action <= 17:  # Spread improvements (16 levels)
                pct = improve_pcts[action - 2]
                price = best_bid + max(1, int(pct * spread))
            elif 18 <= action <= 33:  # Shade valuation (16 levels)
                shade = shade_pcts[action - 18]
                price = int(val * (1 - shade))
            elif action == 34:  # Truthful
                price = val
            elif action == 35:  # Jump Best (improve by 1)
                price = best_bid + 1
            elif action == 36:  # Snipe (accept only if spread < 5%)
                if spread > 0 and spread / best_ask < 0.05:
                    price = best_ask
                else:
                    return -99  # Pass if spread too wide
            elif 37 <= action <= 42:  # UnderCut (6 levels)
                amt = undercut_amts[action - 37]
                price = best_bid + amt
            elif 43 <= action <= 46:  # Aggressive: accept if spread < 10/15/20/30%
                thresholds = [0.10, 0.15, 0.20, 0.30]
                thresh = thresholds[action - 43]
                if spread > 0 and spread / best_ask < thresh:
                    price = best_ask
                else:
                    return -99
            elif 47 <= action <= 49:  # Mid-spread bids
                mid_pcts = [0.40, 0.50, 0.60]  # bid at 40/50/60% of spread
                pct = mid_pcts[action - 47]
                price = best_bid + int(pct * spread)

            # Cap at valuation (never bid above what it's worth)
            if price > val:
                price = val

        else:  # Seller
            if action == 1:  # Accept (Sell at Bid)
                price = best_bid
            elif 2 <= action <= 17:  # Spread improvements (16 levels)
                pct = improve_pcts[action - 2]
                price = best_ask - max(1, int(pct * spread))
            elif 18 <= action <= 33:  # Shade valuation (16 levels)
                shade = shade_pcts[action - 18]
                price = int(val * (1 + shade))
            elif action == 34:  # Truthful
                price = val
            elif action == 35:  # Jump Best (improve by 1)
                price = best_ask - 1
            elif action == 36:  # Snipe (accept only if spread < 5%)
                if spread > 0 and spread / best_ask < 0.05:
                    price = best_bid
                else:
                    return -99  # Pass if spread too wide
            elif 37 <= action <= 42:  # UnderCut (6 levels)
                amt = undercut_amts[action - 37]
                price = best_ask - amt
            elif 43 <= action <= 46:  # Aggressive: accept if spread < 10/15/20/30%
                thresholds = [0.10, 0.15, 0.20, 0.30]
                thresh = thresholds[action - 43]
                if spread > 0 and spread / best_ask < thresh:
                    price = best_bid
                else:
                    return -99
            elif 47 <= action <= 49:  # Mid-spread asks
                mid_pcts = [0.40, 0.50, 0.60]  # ask at 40/50/60% of spread
                pct = mid_pcts[action - 47]
                price = best_ask - int(pct * spread)

            # Floor at valuation (never ask below cost)
            if price < val:
                price = val

        # Ensure price is within bounds
        price = min(price, self.max_price)
        price = max(price, self.min_price)

        return price

    def render(self) -> None:
        """Render the environment (optional)."""
        if self.market is None:
            return

        print(f"\n--- Step {self.market.current_time}/{self.max_steps} ---")
        print(f"RL Agent: {'Buyer' if self.rl_is_buyer else 'Seller'} #{self.rl_agent_id}")
        print(f"Tokens: {self.rl_agent.num_trades}/{self.num_tokens}")
        print(f"Profit: {self.rl_agent.period_profit}")
        print(f"Last Action: {self.rl_agent.last_action}")

        t = max(0, self.market.current_time - 1)
        if t >= 0:
            print(f"Best Bid: {self.market.orderbook.high_bid[t]}")
            print(f"Best Ask: {self.market.orderbook.low_ask[t]}")
            current_time = self.market.orderbook.current_time
            last_trade = self.market.orderbook.trade_price[current_time] if current_time > 0 else 0
            print(f"Last Trade Price: {last_trade}")

    def update_config(self, config: dict[str, Any]) -> None:
        """
        Update environment configuration dynamically.

        Args:
            config: New configuration dictionary
        """
        # Update opponent configuration
        if "opponent_type" in config:
            self.opponent_type = config["opponent_type"]
        if "opponent_mix" in config:
            self.opponent_mix = config["opponent_mix"]

        # Update difficulty
        if "difficulty" in config:
            self.difficulty = config["difficulty"]

        # Update pure profit mode
        if "pure_profit_mode" in config:
            self.pure_profit_mode = config["pure_profit_mode"]

        # Update reward configuration
        for key, value in config.items():
            if key in self.reward_config:
                self.reward_config[key] = value

        # Update specific reward weights if provided directly
        if "profit_weight" in config:
            self.reward_config["profit_weight"] = config["profit_weight"]
        if "market_making_weight" in config:
            self.reward_config["market_making_weight"] = config["market_making_weight"]
        if "exploration_weight" in config:
            self.reward_config["exploration_weight"] = config["exploration_weight"]
        if "invalid_penalty" in config:
            self.reward_config["invalid_penalty"] = config["invalid_penalty"]
        if "efficiency_bonus_weight" in config:
            self.reward_config["efficiency_bonus_weight"] = config["efficiency_bonus_weight"]
