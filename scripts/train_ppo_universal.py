#!/usr/bin/env python3
"""
Universal PPO Training Script.

Trains a single PPO agent that:
1. Samples environments from 10 Santa Fe configurations
2. Samples opponents from {ZIC, ZIC2, ZIP, ZIP2}
3. Receives environment context features (48-dim observation)
4. Optimizes for period profit

Usage:
    python scripts/train_ppo_universal.py
    python scripts/train_ppo_universal.py --timesteps 10000000
    python scripts/train_ppo_universal.py --resume checkpoints/ppo_universal/final_model.zip
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

sys.path.append(str(Path(__file__).parent.parent))

from envs.enhanced_double_auction_env import (
    OPPONENT_POOL,
    SANTA_FE_ENVIRONMENTS,
    EnhancedDoubleAuctionEnv,
)


def make_env_fn(seed: int, rank: int):
    """Create environment factory for vectorized envs."""

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
            # Reward config - pure profit mode
            "pure_profit_mode": True,
            # Default env (overridden by sampling)
            "gametype": 6453,
            "num_agents": 8,
            "num_tokens": 4,
            "max_steps": 75,
            "min_price": 0,
            "max_price": 2000,
        }
        env = EnhancedDoubleAuctionEnv(config)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


class UniversalMetricsCallback(BaseCallback):
    """Track training metrics including env distribution."""

    def __init__(self, verbose: int = 0, log_freq: int = 1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_count = 0
        self.env_counts = {name: 0 for name in SANTA_FE_ENVIRONMENTS}
        self.opponent_counts = {opp: 0 for opp in OPPONENT_POOL}
        self.profits = []
        self.efficiencies = []

    def _on_step(self) -> bool:
        # Check for episode end
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if done and i < len(infos):
                info = infos[i]
                self.episode_count += 1

                # Track env distribution
                env_name = info.get("env_name", "BASE")
                if env_name in self.env_counts:
                    self.env_counts[env_name] += 1

                # Track opponent distribution
                opponent_mix = info.get("opponent_mix", [])
                for opp in opponent_mix:
                    opp_str = str(opp)
                    if opp_str in self.opponent_counts:
                        self.opponent_counts[opp_str] += 1

                # Track metrics
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
            # Log env distribution
            total_eps = sum(self.env_counts.values())
            if total_eps > 0:
                for name, count in self.env_counts.items():
                    self.logger.record(f"env_dist/{name}", count / total_eps)

            # Log recent performance
            if len(self.profits) > 0:
                recent_profits = self.profits[-100:]
                recent_eff = self.efficiencies[-100:]
                self.logger.record("trading/avg_profit_100", np.mean(recent_profits))
                self.logger.record("trading/avg_efficiency_100", np.mean(recent_eff))

        return True


def evaluate_per_env(model, n_episodes: int = 50) -> dict:
    """Evaluate model on each environment separately."""
    results = {}

    for env_name in SANTA_FE_ENVIRONMENTS:
        env_config = SANTA_FE_ENVIRONMENTS[env_name]
        config = {
            "sample_env": False,
            "sample_opponents": False,
            "env_name": env_name,
            "gametype": env_config["gametype"],
            "max_price": env_config["max_price"],
            "num_buyers": env_config["num_buyers"],
            "num_sellers": env_config["num_sellers"],
            "num_tokens": env_config["num_tokens"],
            "max_steps": env_config["max_steps"],
            "num_agents": env_config["num_buyers"] + env_config["num_sellers"],
            "rl_is_buyer": True,
            "rl_agent_id": 1,
            "pure_profit_mode": True,
            "opponent_type": "ZIC",  # Default opponent for eval
            "min_price": 0,
        }

        env = EnhancedDoubleAuctionEnv(config)
        env = Monitor(env)

        profits = []
        efficiencies = []
        trades = []

        for ep in range(n_episodes):
            obs, info = env.reset(seed=42 + ep)
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, trunc, info = env.step(action)

            if "metrics" in info:
                profits.append(info["metrics"].get("total_profit", 0))
                efficiencies.append(info["metrics"].get("market_efficiency", 0))
                trades.append(info["metrics"].get("trades_executed", 0))

        results[env_name] = {
            "mean_profit": float(np.mean(profits)),
            "std_profit": float(np.std(profits)),
            "mean_efficiency": float(np.mean(efficiencies)),
            "mean_trades": float(np.mean(trades)),
        }

        print(
            f"  {env_name}: profit={np.mean(profits):.0f}, "
            f"efficiency={np.mean(efficiencies):.2f}, trades={np.mean(trades):.1f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Universal PPO Training")
    parser.add_argument("--timesteps", type=int, default=5_000_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel envs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument(
        "--no-normalize", action="store_true", help="Disable observation normalization"
    )
    args = parser.parse_args()

    # Paths
    checkpoint_dir = Path("checkpoints/ppo_universal")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("UNIVERSAL PPO TRAINING")
    print("=" * 70)
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Parallel envs: {args.n_envs}")
    print(f"Environments: {list(SANTA_FE_ENVIRONMENTS.keys())}")
    print(f"Opponents: {OPPONENT_POOL}")
    print("Observation dim: 48 (42 base + 6 env context)")
    print("=" * 70)

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create vectorized environment
    print("\nCreating environments...")
    if args.n_envs == 1:
        train_env = DummyVecEnv([make_env_fn(args.seed, 0)])
    else:
        train_env = SubprocVecEnv([make_env_fn(args.seed, i) for i in range(args.n_envs)])

    # Normalize observations (recommended for RL)
    if not args.no_normalize:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Create or load model
    if args.resume:
        print(f"Loading model from {args.resume}")
        model = PPO.load(args.resume, env=train_env)

        # Load normalization stats if they exist
        if not args.no_normalize:
            norm_path = Path(args.resume).parent / "vec_normalize.pkl"
            if norm_path.exists():
                train_env = VecNormalize.load(str(norm_path), train_env)
                print(f"Loaded normalization stats from {norm_path}")
    else:
        print("Creating new PPO model...")
        # Check if tensorboard is available
        try:
            import tensorboard  # noqa: F401

            tb_log = str(checkpoint_dir / "tensorboard")
        except ImportError:
            print("Tensorboard not installed, logging disabled")
            tb_log = None

        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=1e-4,  # Slower learning (was 3e-4)
            n_steps=2048,
            batch_size=128,  # Larger batches (was 64)
            n_epochs=5,  # Less aggressive (was 10)
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,  # More conservative (was 0.2)
            ent_coef=0.1,  # High exploration for 50-action space
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            seed=args.seed,
            tensorboard_log=tb_log,
            policy_kwargs={
                "net_arch": [256, 256],
                "activation_fn": torch.nn.ReLU,
            },
        )

    # Eval only mode
    if args.eval_only:
        print("\nEvaluating model on each environment...")
        results = evaluate_per_env(model, n_episodes=50)

        # Save results
        results_path = checkpoint_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
        return

    # Setup callbacks
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=100_000,
                save_path=str(checkpoint_dir),
                name_prefix="ppo_universal",
                save_vecnormalize=True,
            ),
            UniversalMetricsCallback(verbose=1, log_freq=10_000),
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

    # Final evaluation
    print("\nFinal Evaluation per Environment:")
    print("-" * 50)
    results = evaluate_per_env(model, n_episodes=50)

    # Save results
    results_path = checkpoint_dir / "final_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "timesteps": args.timesteps,
                "seed": args.seed,
                "elapsed_seconds": elapsed.total_seconds(),
                "per_env_results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_path}")

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    avg_profit = np.mean([r["mean_profit"] for r in results.values()])
    avg_eff = np.mean([r["mean_efficiency"] for r in results.values()])
    print(f"Average profit across envs: {avg_profit:.0f}")
    print(f"Average efficiency across envs: {avg_eff:.2f}")
    print(f"Model: {final_path}")

    train_env.close()


if __name__ == "__main__":
    main()
