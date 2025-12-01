#!/usr/bin/env python3
"""
Train PPO Against ZIC/ZIP Mix.

Trains a buyer-specialized PPO model against ZIC and ZIP opponents,
aiming to improve performance in the zero-intelligence hierarchy.

Current PPO (trained vs Skeleton/GD): ranks 3rd behind ZIP and ZIC
Hypothesis: training against actual opponents will improve ranking

Usage:
    python scripts/train_ppo_vs_zi_mix.py
    python scripts/train_ppo_vs_zi_mix.py --timesteps 2000000
    python scripts/train_ppo_vs_zi_mix.py --resume checkpoints/ppo_vs_zi_mix/checkpoint_500000.zip
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

# Note: MaskablePPO doesn't work with VecNormalize, use regular PPO
# from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from envs.enhanced_double_auction_env import EnhancedDoubleAuctionEnv

# Environment configuration: train against ZIC/ZIP
ENV_CONFIG = {
    "num_agents": 8,  # 4 buyers + 4 sellers
    "num_tokens": 4,
    "max_steps": 100,
    "price_min": 1,
    "price_max": 1000,
    "rl_agent_id": 1,  # PPO is buyer #1
    "rl_is_buyer": True,  # Buyer-only training
    "opponent_mix": ["ZIC", "ZIP"],  # Train against ZIC and ZIP (skip ZI - loses money)
    "gametype": 6453,  # BASE environment
    "pure_profit_mode": True,  # Focus on profit maximization
}

# PPO hyperparameters
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.15,  # Start with high entropy for exploration
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

# Training configuration
TRAINING_CONFIG = {
    "total_timesteps": 1_000_000,
    "n_envs": 8,
    "save_freq": 100_000,
    "eval_episodes": 100,
    "checkpoint_dir": "checkpoints/ppo_vs_zi_mix",
    "tensorboard_log": "logs/ppo_vs_zi_mix",
}


# Note: Action masking removed - VecNormalize doesn't preserve masks
# PPO will learn to avoid invalid actions through negative rewards


class EntropyDecayCallback(BaseCallback):
    """Callback for entropy coefficient decay during training."""

    def __init__(
        self,
        initial_ent_coef: float = 0.15,
        final_ent_coef: float = 0.005,
        decay_timesteps: int = 800_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.decay_timesteps = decay_timesteps

    def _on_step(self) -> bool:
        # Calculate current entropy coefficient
        progress = min(1.0, self.num_timesteps / self.decay_timesteps)
        current_ent_coef = self.initial_ent_coef + progress * (
            self.final_ent_coef - self.initial_ent_coef
        )

        # Update model's entropy coefficient
        self.model.ent_coef = current_ent_coef

        # Log every 50k steps
        if self.num_timesteps % 50_000 == 0:
            self.logger.record("train/ent_coef", current_ent_coef)
            if self.verbose > 0:
                print(f"  Entropy coef: {current_ent_coef:.4f}")

        return True


class MetricsCallback(BaseCallback):
    """Callback for tracking trading metrics."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_count = 0
        self.recent_profits = []
        self.recent_trades = []

    def _on_step(self) -> bool:
        # Check for episode end in any environment
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [{}])[i]

                if "metrics" in info:
                    metrics = info["metrics"]
                    self.episode_count += 1
                    self.recent_profits.append(metrics.get("total_profit", 0))
                    self.recent_trades.append(metrics.get("trades_executed", 0))

                    # Log aggregate metrics every 100 episodes
                    if self.episode_count % 100 == 0:
                        avg_profit = np.mean(self.recent_profits[-100:])
                        avg_trades = np.mean(self.recent_trades[-100:])
                        self.logger.record("trading/avg_profit_100ep", avg_profit)
                        self.logger.record("trading/avg_trades_100ep", avg_trades)

                        if self.verbose > 0:
                            print(
                                f"  Episode {self.episode_count}: "
                                f"Avg profit (100ep): {avg_profit:.1f}, "
                                f"Avg trades: {avg_trades:.1f}"
                            )

        return True


def make_env(config: dict[str, Any], seed: int = 0, rank: int = 0) -> callable:
    """Create environment factory function."""

    def _init() -> gym.Env:
        env = EnhancedDoubleAuctionEnv(config)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def make_vec_env(
    config: dict[str, Any], n_envs: int = 1, seed: int = 0
) -> SubprocVecEnv | DummyVecEnv:
    """Create vectorized environments."""
    if n_envs == 1:
        return DummyVecEnv([make_env(config, seed, 0)])
    else:
        return SubprocVecEnv([make_env(config, seed, i) for i in range(n_envs)])


def evaluate_model(model: PPO, env: gym.Env, n_episodes: int = 100) -> dict[str, float]:
    """Evaluate the trained model."""
    episode_profits = []
    episode_trades = []
    episode_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        final_info = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            if done:
                final_info = info

        episode_rewards.append(episode_reward)
        if final_info and "metrics" in final_info:
            episode_profits.append(final_info["metrics"].get("total_profit", 0))
            episode_trades.append(final_info["metrics"].get("trades_executed", 0))

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_profit": float(np.mean(episode_profits)) if episode_profits else 0.0,
        "std_profit": float(np.std(episode_profits)) if episode_profits else 0.0,
        "mean_trades": float(np.mean(episode_trades)) if episode_trades else 0.0,
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PPO vs ZIC/ZIP Mix")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TRAINING_CONFIG["total_timesteps"],
        help="Total timesteps to train",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=TRAINING_CONFIG["n_envs"],
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Setup directories
    checkpoint_dir = Path(TRAINING_CONFIG["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = Path(TRAINING_CONFIG["tensorboard_log"])
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("\n" + "=" * 70)
    print("PPO TRAINING: vs ZIC/ZIP Mix")
    print("=" * 70)
    print("Objective: Beat ZIC (currently ranks 2nd) in ZI hierarchy")
    print(f"Environment: {ENV_CONFIG['num_agents']} agents, {ENV_CONFIG['num_tokens']} tokens")
    print(f"Opponents: {ENV_CONFIG['opponent_mix']}")
    print(f"Total Timesteps: {args.timesteps:,}")
    print(f"Parallel Envs: {args.n_envs}")
    print(f"Entropy Decay: {PPO_CONFIG['ent_coef']} -> 0.005")
    print(f"Checkpoint Dir: {checkpoint_dir}")
    print("=" * 70 + "\n")

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create training environment
    print("Creating training environments...")
    train_env = make_vec_env(ENV_CONFIG, n_envs=args.n_envs, seed=args.seed)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    # Create or load model
    if args.resume:
        print(f"Loading model from {args.resume}")

        # Load VecNormalize stats if available
        norm_path = Path(args.resume).parent / "vec_normalize.pkl"
        if norm_path.exists():
            print(f"Loading VecNormalize stats from {norm_path}")
            train_env = VecNormalize.load(str(norm_path), train_env)

        model = PPO.load(args.resume, env=train_env)
    else:
        print("Creating new PPO model...")
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            **PPO_CONFIG,
            verbose=1 if args.verbose else 0,
            seed=args.seed,
            tensorboard_log=None,  # Disabled - tensorboard not installed
        )

    # Setup callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=TRAINING_CONFIG["save_freq"] // args.n_envs,
            save_path=str(checkpoint_dir),
            name_prefix="ppo_zi_mix",
            save_vecnormalize=True,
        ),
        EntropyDecayCallback(
            initial_ent_coef=PPO_CONFIG["ent_coef"],
            final_ent_coef=0.005,
            decay_timesteps=int(args.timesteps * 0.8),
            verbose=1 if args.verbose else 0,
        ),
        MetricsCallback(verbose=1 if args.verbose else 0),
    ]

    # Train
    print("\nStarting training...")
    print(f"  Total timesteps: {args.timesteps:,}")
    print(f"  Learning rate: {PPO_CONFIG['learning_rate']}")
    print(f"  Batch size: {PPO_CONFIG['batch_size']}")
    print("-" * 70 + "\n")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList(callbacks),
            log_interval=10,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Save final model
    print("\nSaving final model...")
    final_model_path = checkpoint_dir / "final_model.zip"
    model.save(final_model_path)
    print(f"  Model saved to {final_model_path}")

    # Save normalization stats
    norm_path = checkpoint_dir / "vec_normalize.pkl"
    train_env.save(str(norm_path))
    print(f"  Normalization saved to {norm_path}")

    # Final evaluation
    print("\nRunning final evaluation...")
    eval_env = EnhancedDoubleAuctionEnv(ENV_CONFIG)
    results = evaluate_model(model, eval_env, n_episodes=TRAINING_CONFIG["eval_episodes"])

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print("Final Performance:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"  Mean Profit: {results['mean_profit']:.2f} +/- {results['std_profit']:.2f}")
    print(f"  Mean Trades: {results['mean_trades']:.1f}")

    # Save results
    results_path = checkpoint_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "config": {
                    "env": ENV_CONFIG,
                    "ppo": PPO_CONFIG,
                    "training": TRAINING_CONFIG,
                    "timesteps": args.timesteps,
                    "seed": args.seed,
                },
                "results": results,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    print(f"  Results saved to {results_path}")

    # Cleanup
    train_env.close()
    eval_env.close()

    print("\n" + "=" * 70)
    print("Next step: Run evaluation tournament with:")
    print("  python scripts/run_ppo_vs_zi_experiment.py \\")
    print(f"    --model {final_model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
