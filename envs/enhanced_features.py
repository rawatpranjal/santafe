"""
Enhanced observation generation for RL agents with sophisticated market features.

This module extends the basic features with:
- Deep market state (order book depth, liquidity)
- Strategic indicators (agent position, competition metrics)
- Historical patterns (momentum, volatility clustering)
- Microstructure signals (order flow, price impact)
"""

from collections import deque
from typing import Any

import numpy as np

from engine.orderbook import OrderBook


class EnhancedObservationGenerator:
    """
    Generates rich, normalized observation vectors for RL agents.

    Feature Groups (Total 48 features):
    1. Private State (4): valuation, inventory, time, urgency
    2. Market State (5): bid, ask, spread, mid, depth
    3. Strategic Context (5): surplus, competition, position, momentum, pressure
    4. Market Dynamics (5): trend, volatility, volume, imbalance, liquidity
    5. Microstructure (5): bid strength, ask strength, flow toxicity, price impact, efficiency
    6. Episode Structure (7): steps remaining, tokens remaining, profit, avg profit, best/avg remaining, new period
    7. Enhanced (9): last 3 prices, eq distance, phase (3), bid/ask room
    8. Time-Based (2): steps since trade, alpha urgency
    9. Environment Context (6): num_buyers, num_sellers, num_tokens, ntimes, buyer_ratio, token_variance
    """

    def __init__(
        self,
        max_price: int = 1000,
        max_tokens: int = 4,
        max_steps: int = 100,
        history_len: int = 10,
        num_agents: int = 8,
        # New: environment context params
        num_buyers: int = 4,
        num_sellers: int = 4,
        gametype: int = 6453,
    ):
        self.max_price = max_price
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.history_len = history_len
        self.num_agents = num_agents

        # Environment context (for universal PPO)
        self.num_buyers = num_buyers
        self.num_sellers = num_sellers
        self.gametype = gametype

        # Decode gametype to token variance indicator
        # gametype=0: equal endowment (low variance)
        # gametype=7: very low variance
        # gametype=6453: standard variance
        self.token_variance = self._decode_gametype_variance(gametype)

        # Feature vector size (42 original + 6 env context = 48)
        self.feature_dim = 48

        # History buffers
        self.trade_prices: deque = deque(maxlen=history_len)
        self.trade_volumes: deque = deque(maxlen=history_len)
        self.bid_history: deque = deque(maxlen=history_len)
        self.ask_history: deque = deque(maxlen=history_len)
        self.spread_history: deque = deque(maxlen=history_len)

        # Running statistics
        self.total_volume = 0
        self.avg_trade_price = 0.0
        self.price_momentum = 0.0

    def _decode_gametype_variance(self, gametype: int) -> float:
        """
        Decode gametype to a normalized token variance indicator.

        gametype=0: Equal endowment (all traders get same tokens) -> 0.0
        gametype=7: Very low variance (small weights) -> 0.1
        gametype=6453: Standard variance -> 1.0

        The gametype is a 4-digit number where each digit d determines
        weight w = 3^d - 1. Higher weights = more variance in tokens.
        """
        if gametype == 0:
            return 0.0  # Equal endowment
        elif gametype < 100:
            return 0.1  # Low variance (small gametypes like 7)

        # Decode 4-digit gametype to weights and compute normalized variance
        total_weight = 0
        temp = gametype
        for _ in range(4):
            digit = temp % 10
            temp //= 10
            total_weight += 3**digit - 1

        # Max possible weight is 4 * (3^9 - 1) ≈ 78728
        # For 6453: w = (3^3-1) + (3^5-1) + (3^4-1) + (3^6-1) = 26+242+80+728 = 1076
        # Normalize to [0, 1] using log scale
        import math

        normalized = min(1.0, math.log1p(total_weight) / math.log1p(2000))
        return normalized

    def set_env_context(
        self, num_buyers: int, num_sellers: int, num_tokens: int, max_steps: int, gametype: int
    ) -> None:
        """Update environment context for a new episode."""
        self.num_buyers = num_buyers
        self.num_sellers = num_sellers
        self.max_tokens = num_tokens
        self.max_steps = max_steps
        self.gametype = gametype
        self.token_variance = self._decode_gametype_variance(gametype)
        self.num_agents = num_buyers + num_sellers

    def reset(self) -> None:
        """Reset all history buffers and statistics."""
        self.trade_prices.clear()
        self.trade_volumes.clear()
        self.bid_history.clear()
        self.ask_history.clear()
        self.spread_history.clear()
        self.total_volume = 0
        self.avg_trade_price = 0.0
        self.price_momentum = 0.0

    def update_trade(self, price: int, volume: int = 1) -> None:
        """Update history with new trade."""
        self.trade_prices.append(price)
        self.trade_volumes.append(volume)
        self.total_volume += volume

        # Update running average
        if len(self.trade_prices) > 0:
            self.avg_trade_price = np.mean(self.trade_prices)

        # Calculate momentum
        if len(self.trade_prices) >= 2:
            recent = (
                np.mean(list(self.trade_prices)[-3:])
                if len(self.trade_prices) >= 3
                else self.trade_prices[-1]
            )
            older = (
                np.mean(list(self.trade_prices)[:-3])
                if len(self.trade_prices) > 3
                else self.trade_prices[0]
            )
            self.price_momentum = (recent - older) / self.max_price

    def update_quotes(self, bid: int, ask: int) -> None:
        """Update quote history."""
        self.bid_history.append(bid)
        self.ask_history.append(ask)
        if bid > 0 and ask > 0:
            self.spread_history.append(ask - bid)

    def generate(
        self, agent: Any, orderbook: OrderBook, current_step: int, steps_since_last_trade: int = 0
    ) -> np.ndarray:
        """
        Generate enhanced observation vector.

        Args:
            agent: The Agent instance
            orderbook: Current OrderBook state
            current_step: Current time step (1-indexed)

        Returns:
            np.ndarray: Normalized feature vector (shape=(24,), dtype=float32)
        """
        obs = np.zeros(self.feature_dim, dtype=np.float32)
        idx = 0

        # --- 1. Private State (4 features) ---

        # Current valuation
        current_holding = agent.num_trades
        if current_holding < agent.num_tokens:
            valuation = agent.valuations[current_holding]
        else:
            valuation = 0
        obs[idx] = valuation / self.max_price
        idx += 1

        # Inventory progress
        obs[idx] = current_holding / self.max_tokens
        idx += 1

        # Time progress
        time_progress = current_step / self.max_steps
        obs[idx] = time_progress
        idx += 1

        # Urgency (inventory pressure with time)
        remaining_tokens = self.max_tokens - current_holding
        remaining_time = max(0.01, 1.0 - time_progress)
        urgency = (remaining_tokens / self.max_tokens) / remaining_time
        obs[idx] = np.clip(urgency, 0.0, 1.0)
        idx += 1

        # --- 2. Market State (5 features) ---

        # Get current best bid/ask
        t = max(0, current_step - 1)  # Look at previous step's state
        best_bid = orderbook.high_bid[t] if t >= 0 else 0
        best_ask = orderbook.low_ask[t] if t >= 0 else 0

        # Update quote history
        self.update_quotes(best_bid, best_ask)

        # Normalized bid/ask
        obs[idx] = best_bid / self.max_price
        idx += 1

        display_ask = best_ask if best_ask > 0 else self.max_price
        obs[idx] = display_ask / self.max_price
        idx += 1

        # Spread (normalized)
        spread = (display_ask - best_bid) if best_bid > 0 else self.max_price
        obs[idx] = spread / self.max_price
        idx += 1

        # Mid price
        mid_price = (best_bid + display_ask) / 2 if best_bid > 0 else display_ask / 2
        obs[idx] = mid_price / self.max_price
        idx += 1

        # Market depth (number of active orders)
        if t > 0:
            active_bids = np.sum(orderbook.bids[1:, t] > 0)
            active_asks = np.sum(orderbook.asks[1:, t] > 0)
            depth = (active_bids + active_asks) / self.num_agents
        else:
            depth = 0.0
        obs[idx] = depth
        idx += 1

        # --- 3. Strategic Context (5 features) ---

        # Immediate surplus (profit opportunity)
        surplus = 0.0
        if agent.is_buyer:
            if best_ask > 0:
                surplus = (valuation - best_ask) / self.max_price
        else:
            if best_bid > 0:
                surplus = (best_bid - valuation) / self.max_price
        obs[idx] = np.clip(surplus, -1.0, 1.0)
        idx += 1

        # Competition level (agents on same side)
        if t > 0:
            if agent.is_buyer:
                competition = np.sum(orderbook.bids[1:, t] > 0) / (self.num_agents // 2)
            else:
                competition = np.sum(orderbook.asks[1:, t] > 0) / (self.num_agents // 2)
        else:
            competition = 0.5
        obs[idx] = competition
        idx += 1

        # Position in market (how aggressive is my valuation)
        if agent.is_buyer:
            # Higher valuation = stronger position
            position = valuation / self.max_price
        else:
            # Lower cost = stronger position
            position = 1.0 - (valuation / self.max_price)
        obs[idx] = position
        idx += 1

        # Price momentum
        obs[idx] = np.clip(self.price_momentum, -1.0, 1.0)
        idx += 1

        # Trading pressure (recent activity level)
        recent_volume = sum(self.trade_volumes) if len(self.trade_volumes) > 0 else 0
        max_volume = min(5, len(self.trade_volumes))  # Normalize by window size
        pressure = recent_volume / max(1, max_volume)
        obs[idx] = np.clip(pressure, 0.0, 1.0)
        idx += 1

        # --- 4. Market Dynamics (5 features) ---

        # Price trend (moving average)
        if len(self.trade_prices) > 0:
            trend = self.avg_trade_price / self.max_price
        else:
            trend = 0.5
        obs[idx] = trend
        idx += 1

        # Volatility
        if len(self.trade_prices) >= 2:
            volatility = np.std(self.trade_prices) / self.max_price
        else:
            volatility = 0.0
        obs[idx] = np.clip(volatility, 0.0, 1.0)
        idx += 1

        # Volume rate
        volume_rate = self.total_volume / max(1, current_step)
        obs[idx] = np.clip(volume_rate, 0.0, 1.0)
        idx += 1

        # Order imbalance
        if t > 0:
            bid_vol = np.sum(orderbook.bids[1:, t] > 0)
            ask_vol = np.sum(orderbook.asks[1:, t] > 0)
            total_vol = bid_vol + ask_vol
            if total_vol > 0:
                imbalance = (bid_vol - ask_vol) / total_vol
            else:
                imbalance = 0.0
        else:
            imbalance = 0.0
        obs[idx] = (imbalance + 1.0) / 2.0  # Normalize to [0, 1]
        idx += 1

        # Liquidity (inverse of average spread)
        if len(self.spread_history) > 0:
            avg_spread = np.mean(self.spread_history)
            liquidity = 1.0 - (avg_spread / self.max_price)
        else:
            liquidity = 0.5
        obs[idx] = np.clip(liquidity, 0.0, 1.0)
        idx += 1

        # --- 5. Microstructure Signals (5 features) ---

        # Bid strength (average distance from mid)
        if len(self.bid_history) > 0 and best_bid > 0:
            avg_bid = np.mean([b for b in self.bid_history if b > 0])
            bid_strength = avg_bid / mid_price if mid_price > 0 else 0.5
        else:
            bid_strength = 0.5
        obs[idx] = np.clip(bid_strength, 0.0, 1.0)
        idx += 1

        # Ask strength
        if len(self.ask_history) > 0 and best_ask > 0:
            valid_asks = [a for a in self.ask_history if a > 0]
            if valid_asks:
                avg_ask = np.mean(valid_asks)
                ask_strength = (
                    1.0 - (avg_ask - mid_price) / self.max_price if mid_price > 0 else 0.5
                )
            else:
                ask_strength = 0.5
        else:
            ask_strength = 0.5
        obs[idx] = np.clip(ask_strength, 0.0, 1.0)
        idx += 1

        # Order flow toxicity (rapid price changes)
        if len(self.trade_prices) >= 3:
            price_changes = np.diff(self.trade_prices)
            toxicity = np.std(price_changes) / self.max_price if len(price_changes) > 0 else 0.0
        else:
            toxicity = 0.0
        obs[idx] = np.clip(toxicity, 0.0, 1.0)
        idx += 1

        # Price impact (how much prices move after trades)
        if len(self.trade_prices) >= 2:
            impacts = []
            for i in range(1, len(self.trade_prices)):
                impact = abs(self.trade_prices[i] - self.trade_prices[i - 1]) / self.max_price
                impacts.append(impact)
            avg_impact = np.mean(impacts) if impacts else 0.0
        else:
            avg_impact = 0.0
        obs[idx] = np.clip(avg_impact, 0.0, 1.0)
        idx += 1

        # Market efficiency indicator (convergence to equilibrium)
        if len(self.trade_prices) >= 5:
            # Check if prices are stabilizing
            recent_std = np.std(list(self.trade_prices)[-5:])
            older_std = (
                np.std(list(self.trade_prices)[:-5]) if len(self.trade_prices) > 5 else recent_std
            )
            if older_std > 0:
                efficiency = 1.0 - (recent_std / older_std)
            else:
                efficiency = 0.5
        else:
            efficiency = 0.5
        obs[idx] = np.clip(efficiency, 0.0, 1.0)
        idx += 1

        # --- 6. Episode Structure & Planning (7 features) ---

        # Steps remaining (explicit countdown for planning)
        steps_remaining = self.max_steps - current_step
        obs[idx] = steps_remaining / self.max_steps
        idx += 1

        # Tokens remaining (how many left to trade)
        tokens_remaining = max(0, agent.num_tokens - agent.num_trades)
        obs[idx] = tokens_remaining / self.max_tokens
        idx += 1

        # Profit this episode (running total)
        obs[idx] = np.clip(agent.period_profit / (self.max_price * self.max_tokens), 0.0, 1.0)
        idx += 1

        # Average profit per token (performance tracker)
        trades_completed = max(1, agent.num_trades)  # Avoid division by zero
        avg_profit_per_token = agent.period_profit / trades_completed
        obs[idx] = np.clip(avg_profit_per_token / self.max_price, 0.0, 1.0)
        idx += 1

        # Best remaining valuation (what's left to trade)
        if tokens_remaining > 0:
            remaining_valuations = agent.valuations[agent.num_trades :]
            if agent.is_buyer:
                best_remaining = max(remaining_valuations) if remaining_valuations else 0
            else:
                best_remaining = (
                    min(remaining_valuations) if remaining_valuations else self.max_price
                )
            obs[idx] = best_remaining / self.max_price
        else:
            obs[idx] = 0.0
        idx += 1

        # Average remaining valuation (expected value of future trades)
        if tokens_remaining > 0:
            remaining_valuations = agent.valuations[agent.num_trades :]
            avg_remaining = np.mean(remaining_valuations) if remaining_valuations else 0
            obs[idx] = avg_remaining / self.max_price
        else:
            obs[idx] = 0.0
        idx += 1

        # New period flag (1.0 at start, 0.0 otherwise)
        new_period_flag = 1.0 if current_step == 1 else 0.0
        obs[idx] = new_period_flag
        idx += 1

        # --- 7. Enhanced Features for Performance (9 features) ---

        # Last 3 trade prices (individual, not averaged) - 3 features
        trade_list = list(self.trade_prices)
        for i in range(3):
            if i < len(trade_list):
                obs[idx] = trade_list[-(i + 1)] / self.max_price  # Most recent first
            else:
                obs[idx] = 0.5  # Default to mid-range if no history
            idx += 1

        # Distance from theoretical equilibrium (estimated from mid) - 1 feature
        # Use mid-price as proxy for equilibrium
        if valuation > 0:
            if agent.is_buyer:
                eq_distance = (valuation - mid_price) / self.max_price
            else:
                eq_distance = (mid_price - valuation) / self.max_price
        else:
            eq_distance = 0.0
        obs[idx] = np.clip(eq_distance, -1.0, 1.0)
        idx += 1

        # Market phase (early/mid/late one-hot) - 3 features
        phase_early = 1.0 if time_progress < 0.33 else 0.0
        phase_mid = 1.0 if 0.33 <= time_progress < 0.67 else 0.0
        phase_late = 1.0 if time_progress >= 0.67 else 0.0
        obs[idx] = phase_early
        idx += 1
        obs[idx] = phase_mid
        idx += 1
        obs[idx] = phase_late
        idx += 1

        # Bid improvement potential (how much room to improve) - 1 feature
        if agent.is_buyer and valuation > 0:
            bid_room = (
                (valuation - best_bid) / self.max_price
                if best_bid > 0
                else valuation / self.max_price
            )
        else:
            bid_room = 0.0
        obs[idx] = np.clip(bid_room, 0.0, 1.0)
        idx += 1

        # Ask improvement potential - 1 feature
        if not agent.is_buyer and valuation > 0:
            ask_room = (
                (display_ask - valuation) / self.max_price if display_ask < self.max_price else 0.5
            )
        else:
            ask_room = 0.0
        obs[idx] = np.clip(ask_room, 0.0, 1.0)
        idx += 1

        # --- 8. Time-Based Features for Skeleton-Style Strategy (2 features) ---

        # Feature 41: Steps since last trade (normalized)
        # Enables learning time-pressure behavior like Skeleton
        obs[idx] = steps_since_last_trade / self.max_steps
        idx += 1

        # Feature 42: Skeleton's alpha formula = 1/(t - lasttime)
        # This is the key signal Skeleton uses for urgency
        alpha_urgency = 1.0 / max(1, steps_since_last_trade)
        obs[idx] = min(1.0, alpha_urgency)
        idx += 1

        # --- 9. Environment Context Features (6 features) ---
        # These enable PPO to adapt to different Santa Fe environments

        # Feature 43: Number of buyers (normalized to max 8)
        obs[idx] = self.num_buyers / 8.0
        idx += 1

        # Feature 44: Number of sellers (normalized to max 8)
        obs[idx] = self.num_sellers / 8.0
        idx += 1

        # Feature 45: Number of tokens per trader (normalized to max 8)
        obs[idx] = self.max_tokens / 8.0
        idx += 1

        # Feature 46: Period length / time steps (normalized to max 100)
        obs[idx] = self.max_steps / 100.0
        idx += 1

        # Feature 47: Buyer/seller ratio (0.5 = balanced, <0.5 = seller dominated)
        total_traders = self.num_buyers + self.num_sellers
        buyer_ratio = self.num_buyers / total_traders if total_traders > 0 else 0.5
        obs[idx] = buyer_ratio
        idx += 1

        # Feature 48: Token variance indicator (from gametype)
        # 0.0 = equal endowment, 1.0 = high variance
        obs[idx] = self.token_variance
        idx += 1

        assert idx == self.feature_dim, f"Feature dimension mismatch: {idx} != {self.feature_dim}"

        return obs
