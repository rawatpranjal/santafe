"""
LSTM Double Auction Environment.

Extends EnhancedDoubleAuctionEnv to provide richer observations for RecurrentPPO.
Focuses on proper separation of PUBLIC (orderbook) and PRIVATE (agent-specific) info.

Two observation modes:
- Mode A (full_history): Raw orderbook history (4 features per step)
- Mode B (summary_stats): Rolling statistics of orderbook (more robust)
"""

from typing import Any

import numpy as np
from gymnasium import spaces

from envs.enhanced_double_auction_env import (
    EnhancedDoubleAuctionEnv,
)


class LSTMDoubleAuctionEnv(EnhancedDoubleAuctionEnv):
    """
    LSTM-compatible Double Auction Environment.

    Observation space:
    - PUBLIC features (from orderbook, visible to all):
        - Current high_bid / max_price
        - Current low_ask / max_price
        - Last trade_price / max_price
        - Current spread / max_price
        - Rolling mean of spread (last 10 steps)
        - Rolling std of spread (last 10 steps)
        - Trade frequency (trades in last 10 steps / 10)
        - Spread trend (positive = widening)

    - PRIVATE features (agent-specific):
        - Current valuation / max_price
        - Tokens remaining / max_tokens
        - Profit so far / max_expected_profit
        - Time remaining / max_steps
        - Is current high bidder (binary)
        - Last action was accepted (binary)

    - ENV context (from config):
        - num_buyers / 8
        - num_sellers / 8
        - num_tokens / 8
        - max_steps / 100
        - buyer_seller_ratio
        - token_variance
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

        # History tracking for rolling statistics
        self.spread_history: list[float] = []
        self.trade_history: list[bool] = []
        self.window_size = 10

        # Observation dimensions
        self.n_public = 8  # Public orderbook features
        self.n_private = 6  # Private agent features
        self.n_env = 6  # Environment context
        self.obs_dim = self.n_public + self.n_private + self.n_env  # 20 total

        # Override observation space
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset environment and clear history."""
        obs, info = super().reset(seed=seed, options=options)

        # Clear history
        self.spread_history = []
        self.trade_history = []

        # Return new observation format
        lstm_obs = self._get_lstm_observation()
        return lstm_obs, info

    def step(self, action: int):
        """Take step and return LSTM-formatted observation."""
        obs, reward, terminated, truncated, info = super().step(action)

        # Update history
        self._update_history()

        # Return LSTM observation
        lstm_obs = self._get_lstm_observation()
        return lstm_obs, reward, terminated, truncated, info

    def _update_history(self):
        """Update rolling history buffers."""
        if self.market is None:
            return

        t = max(0, self.market.current_time - 1)

        # Get current spread
        high_bid = self.market.orderbook.high_bid[t] if t >= 0 else 0
        low_ask = self.market.orderbook.low_ask[t] if t >= 0 else self.max_price
        if low_ask == 0:
            low_ask = self.max_price
        spread = (low_ask - high_bid) / self.max_price
        self.spread_history.append(spread)

        # Track trades
        trade_occurred = self.market.orderbook.trade_price[t] > 0 if t >= 0 else False
        self.trade_history.append(trade_occurred)

        # Keep only last window_size
        if len(self.spread_history) > self.window_size:
            self.spread_history = self.spread_history[-self.window_size :]
            self.trade_history = self.trade_history[-self.window_size :]

    def _get_lstm_observation(self) -> np.ndarray:
        """
        Build observation with clear PUBLIC/PRIVATE separation.

        Returns:
            obs: (20,) float32 array, all values in [0, 1]
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        idx = 0

        # === PUBLIC FEATURES (8) - Orderbook state visible to all ===

        if self.market is not None:
            t = max(0, self.market.current_time - 1)

            # Current best prices
            high_bid = self.market.orderbook.high_bid[t] if t >= 0 else 0
            low_ask = self.market.orderbook.low_ask[t] if t >= 0 else 0
            if low_ask == 0:
                low_ask = self.max_price

            # 1. High bid (normalized)
            obs[idx] = high_bid / self.max_price
            idx += 1

            # 2. Low ask (normalized)
            obs[idx] = low_ask / self.max_price
            idx += 1

            # 3. Last trade price (normalized)
            last_trade = 0
            for check_t in range(t, -1, -1):
                if self.market.orderbook.trade_price[check_t] > 0:
                    last_trade = self.market.orderbook.trade_price[check_t]
                    break
            obs[idx] = last_trade / self.max_price
            idx += 1

            # 4. Current spread (normalized)
            spread = (low_ask - high_bid) / self.max_price
            obs[idx] = min(1.0, max(0.0, spread))
            idx += 1

            # 5. Rolling mean of spread
            if len(self.spread_history) > 0:
                obs[idx] = np.mean(self.spread_history)
            idx += 1

            # 6. Rolling std of spread
            if len(self.spread_history) > 1:
                obs[idx] = min(1.0, np.std(self.spread_history))
            idx += 1

            # 7. Trade frequency (in window)
            if len(self.trade_history) > 0:
                obs[idx] = sum(self.trade_history) / len(self.trade_history)
            idx += 1

            # 8. Spread trend (widening = positive)
            if len(self.spread_history) >= 3:
                recent = np.mean(self.spread_history[-3:])
                older = (
                    np.mean(self.spread_history[:-3]) if len(self.spread_history) > 3 else recent
                )
                trend = recent - older + 0.5  # Shift to [0, 1]
                obs[idx] = min(1.0, max(0.0, trend))
            else:
                obs[idx] = 0.5  # Neutral
            idx += 1
        else:
            idx += 8

        # === PRIVATE FEATURES (6) - Agent-specific, not visible to others ===

        if self.rl_agent is not None:
            # 1. Current valuation (normalized)
            val = self.rl_agent.get_current_valuation()
            obs[idx] = val / self.max_price if val > 0 else 0
            idx += 1

            # 2. Tokens remaining (normalized)
            tokens_left = self.num_tokens - self.rl_agent.num_trades
            obs[idx] = tokens_left / self.num_tokens
            idx += 1

            # 3. Profit so far (normalized, clipped)
            max_profit = self.max_price * self.num_tokens
            profit_norm = self.rl_agent.period_profit / max_profit
            obs[idx] = min(1.0, max(-1.0, profit_norm))
            idx += 1

            # 4. Time remaining (normalized)
            time_left = self.max_steps - self.market.current_time if self.market else self.max_steps
            obs[idx] = time_left / self.max_steps
            idx += 1

            # 5. Is current high bidder (binary)
            if self.market is not None and self.rl_is_buyer:
                t = max(0, self.market.current_time - 1)
                high_bidder = self.market.orderbook.high_bidder[t]
                # Check if this agent is high bidder (need local ID mapping)
                is_high_bidder = (
                    (high_bidder == self.rl_agent.local_id)
                    if hasattr(self.rl_agent, "local_id")
                    else False
                )
                obs[idx] = 1.0 if is_high_bidder else 0.0
            idx += 1

            # 6. Last action was profitable trade (binary)
            # Simplified: check if we traded last step
            obs[idx] = 1.0 if self.rl_agent.num_trades > 0 else 0.0
            idx += 1
        else:
            idx += 6

        # === ENVIRONMENT CONTEXT (6) - Static per episode ===

        # 1. Number of buyers (normalized)
        obs[idx] = getattr(self, "num_buyers", 4) / 8.0
        idx += 1

        # 2. Number of sellers (normalized)
        obs[idx] = getattr(self, "num_sellers", 4) / 8.0
        idx += 1

        # 3. Number of tokens (normalized)
        obs[idx] = self.num_tokens / 8.0
        idx += 1

        # 4. Max steps (normalized)
        obs[idx] = self.max_steps / 100.0
        idx += 1

        # 5. Buyer/seller ratio
        total = getattr(self, "num_buyers", 4) + getattr(self, "num_sellers", 4)
        obs[idx] = getattr(self, "num_buyers", 4) / total if total > 0 else 0.5
        idx += 1

        # 6. Token variance (from gametype)
        obs[idx] = getattr(self.obs_gen, "token_variance", 0.5) if hasattr(self, "obs_gen") else 0.5
        idx += 1

        return obs

    def _get_action_mask(self) -> np.ndarray:
        """Use parent's 50-action mask."""
        return super()._get_action_mask()
