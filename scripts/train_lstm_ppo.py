#!/usr/bin/env python3
"""
LSTM-PPO Training Script.

Trains RecurrentPPO (from sb3-contrib) using orderbook history.
The LSTM handles temporal patterns that MLP cannot capture.

Usage:
    python scripts/train_lstm_ppo.py
    python scripts/train_lstm_ppo.py --timesteps 2000000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(str(Path(__file__).parent.parent))

from envs.enhanced_double_auction_env import OPPONENT_POOL, SANTA_FE_ENVIRONMENTS
from envs.lstm_double_auction_env import LSTMDoubleAuctionEnv


def make_env_fn(seed: int, rank: int):
    """Create LSTM environment factory."""

    def _init():
        config = {
            # Universal training mode
            "sample_env": True,
            "sample_opponents": True,
            "env_list": list(SANTA_FE_ENVIRONMENTS.keys()),
            "opponent_pool": OPPONENT_POOL,
            # RL agent config
            "rl_is_buyer": True,
            "rl_agent_id": 1,
            # Reward config
            "pure_profit_mode": True,
            # Default env params
            "gametype": 6453,
            "num_agents": 8,
            "num_tokens": 4,
            "max_steps": 75,
            "min_price": 0,
            "max_price": 2000,
        }
        env = LSTMDoubleAuctionEnv(config)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


class LSTMMetricsCallback(BaseCallback):
    """Track training metrics for LSTM-PPO."""

    def __init__(self, verbose: int = 0, log_freq: int = 1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_count = 0
        self.profits = []
        self.efficiencies = []
        self.ranks = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if done and i < len(infos):
                info = infos[i]
                self.episode_count += 1

                if "metrics" in info:
                    metrics = info["metrics"]
                    self.profits.append(metrics.get("total_profit", 0))
                    self.efficiencies.append(metrics.get("market_efficiency", 0))

                    # Log to tensorboard
                    self.logger.record("trading/profit", metrics.get("total_profit", 0))
                    self.logger.record("trading/efficiency", metrics.get("market_efficiency", 0))
                    self.logger.record("trading/trades", metrics.get("trades_executed", 0))

        # Periodic logging
        if self.num_timesteps % self.log_freq == 0 and self.episode_count > 0:
            if len(self.profits) > 0:
                recent_profits = self.profits[-100:]
                recent_eff = self.efficiencies[-100:]
                self.logger.record("trading/avg_profit_100", np.mean(recent_profits))
                self.logger.record("trading/avg_efficiency_100", np.mean(recent_eff))

        return True


def main():
    parser = argparse.ArgumentParser(description="LSTM-PPO Training")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lstm-hidden", type=int, default=128, help="LSTM hidden size")
    parser.add_argument(
        "--no-normalize", action="store_true", help="Disable observation normalization"
    )
    args = parser.parse_args()

    # Paths
    checkpoint_dir = Path("checkpoints/lstm_ppo")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LSTM-PPO TRAINING")
    print("=" * 70)
    print(f"Timesteps: {args.timesteps:,}")
    print(f"LSTM hidden size: {args.lstm_hidden}")
    print(f"Environments: {list(SANTA_FE_ENVIRONMENTS.keys())}")
    print(f"Opponents: {OPPONENT_POOL}")
    print("Observation dim: 20 (8 public + 6 private + 6 env)")
    print("=" * 70)

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create environment (single env for LSTM - recurrence needs careful handling)
    print("\nCreating environment...")
    train_env = DummyVecEnv([make_env_fn(args.seed, 0)])

    # Normalize observations
    if not args.no_normalize:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Create RecurrentPPO model
    print("Creating RecurrentPPO model...")
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=args.seed,
        policy_kwargs={
            "lstm_hidden_size": args.lstm_hidden,
            "n_lstm_layers": 1,
            "net_arch": dict(pi=[256], vf=[256]),
            "activation_fn": torch.nn.ReLU,
        },
    )

    # Setup callbacks
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=100_000,
                save_path=str(checkpoint_dir),
                name_prefix="lstm_ppo",
                save_vecnormalize=True,
            ),
            LSTMMetricsCallback(verbose=1, log_freq=10_000),
        ]
    )

    # Train
    print(f"\nStarting training for {args.timesteps:,} timesteps...")
    start_time = datetime.now()

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            log_interval=100,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    elapsed = datetime.now() - start_time
    print(f"\nTraining completed in {elapsed}")

    # Save final model
    final_path = checkpoint_dir / "final_model.zip"
    model.save(final_path)
    print(f"Model saved to {final_path}")

    # Save normalization stats
    if not args.no_normalize:
        norm_path = checkpoint_dir / "vec_normalize.pkl"
        train_env.save(str(norm_path))
        print(f"Normalization saved to {norm_path}")

    # Save config
    config_path = checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "timesteps": args.timesteps,
                "seed": args.seed,
                "lstm_hidden_size": args.lstm_hidden,
                "elapsed_seconds": elapsed.total_seconds(),
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("LSTM-PPO TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model: {final_path}")

    train_env.close()


if __name__ == "__main__":
    main()
