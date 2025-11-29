#!/usr/bin/env python3
"""
Enhanced PPO Training Script with Curriculum Learning Support.

This script provides advanced PPO training capabilities including:
1. Enhanced environment with rich features
2. Curriculum learning progression
3. Multiple training scenarios
4. Comprehensive evaluation
5. W&B integration for tracking

Usage:
    # Train with specific scenario
    python scripts/train_ppo_enhanced.py --config ppo_vs_zic
    python scripts/train_ppo_enhanced.py --config ppo_vs_kaplan
    python scripts/train_ppo_enhanced.py --config ppo_vs_mixed
    python scripts/train_ppo_enhanced.py --config ppo_curriculum

    # Resume training
    python scripts/train_ppo_enhanced.py --config ppo_vs_zic --resume checkpoint.zip

    # Custom hyperparameters
    python scripts/train_ppo_enhanced.py --config ppo_vs_zic --learning-rate 0.0001
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from envs.enhanced_double_auction_env import EnhancedDoubleAuctionEnv
from envs.curriculum_scheduler import CurriculumScheduler, CurriculumCallback

# Optional W&B integration
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ W&B not available. Install with: pip install wandb")


def load_config(config_name: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml

    config_path = Path(__file__).parent.parent / "conf" / "rl" / "experiments" / f"{config_name}.yaml"

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Process hydra-style defaults
    if "defaults" in config:
        # Load base configs
        for default in config["defaults"]:
            if isinstance(default, str) and default != "_self_":
                # Parse default path
                if "@" in default:
                    base_name, _ = default.split("@")
                else:
                    base_name = default

                # Load base config
                base_path = Path(__file__).parent.parent / "conf" / base_name.replace("/", os.sep)
                base_path = Path(str(base_path) + ".yaml")
                if base_path.exists():
                    with open(base_path, 'r') as f:
                        base_config = yaml.safe_load(f)
                    # Merge with current config (current overrides base)
                    config = merge_configs(base_config, config)

    return config


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Recursively merge configuration dictionaries."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def make_env(config: Dict[str, Any], seed: int = 0) -> gym.Env:
    """Create a single environment instance."""
    def _init():
        env = EnhancedDoubleAuctionEnv(config["env"])
        env = Monitor(env)
        return env
    return _init()


def make_vec_env(config: Dict[str, Any], n_envs: int = 1, seed: int = 0) -> SubprocVecEnv:
    """Create vectorized environments."""
    def make_env_fn(rank: int):
        def _init():
            env_config = config["env"].copy()
            env = EnhancedDoubleAuctionEnv(env_config)
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        return _init

    # Use DummyVecEnv for debugging, SubprocVecEnv for training
    if n_envs == 1:
        return DummyVecEnv([make_env_fn(0)])
    else:
        return SubprocVecEnv([make_env_fn(i) for i in range(n_envs)])


class MetricsCallback(BaseCallback):
    """Custom callback for tracking trading metrics."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_metrics = []

    def _on_step(self) -> bool:
        # Check for episode end
        if self.locals.get("dones", [False])[0]:
            info = self.locals.get("infos", [{}])[0]

            if "metrics" in info:
                metrics = info["metrics"]
                self.episode_count += 1

                # Log to tensorboard
                self.logger.record("trading/efficiency", metrics.get("market_efficiency", 0))
                self.logger.record("trading/profit", metrics.get("total_profit", 0))
                self.logger.record("trading/trades", metrics.get("trades_executed", 0))
                self.logger.record("trading/profitable_trades", metrics.get("profitable_trades", 0))
                self.logger.record("trading/invalid_actions", metrics.get("invalid_actions", 0))

                # Track for later analysis
                self.episode_metrics.append({
                    "episode": self.episode_count,
                    "timestep": self.num_timesteps,
                    **metrics
                })

        return True


def setup_training(config: Dict[str, Any], args: argparse.Namespace) -> Tuple[PPO, Any, Any]:
    """Setup training environment and model."""

    # Create environments
    print("🏗️ Creating environments...")
    n_envs = config["training"].get("n_envs", 8)
    train_env = make_vec_env(config, n_envs=n_envs, seed=args.seed)

    # Optionally normalize observations/rewards
    if config.get("normalize", True):
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    # Create eval environment (must match train_env normalization)
    eval_env = make_vec_env(config, n_envs=4, seed=args.seed + 1000)
    if config.get("normalize", True):
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

    # Create or load model
    if args.resume:
        print(f"📂 Loading model from {args.resume}")

        # Load VecNormalize statistics if they exist
        if isinstance(train_env, VecNormalize):
            norm_path = Path(args.resume).parent / "vec_normalize.pkl"
            if norm_path.exists():
                print(f"📊 Loading VecNormalize stats from {norm_path}")
                train_env = VecNormalize.load(str(norm_path), train_env)
                # Set training mode off for evaluation
                if args.evaluate_only:
                    train_env.training = False
                    train_env.norm_reward = False

                # Also load stats for eval_env if it's normalized
                if isinstance(eval_env, VecNormalize):
                    eval_env = VecNormalize.load(str(norm_path), eval_env)
                    eval_env.training = False
                    eval_env.norm_reward = False

        model = PPO.load(args.resume, env=train_env)
    else:
        print("🤖 Creating new PPO model...")

        # Get PPO config
        ppo_config = config.get("ppo", {})

        # Override with command line args
        if args.learning_rate:
            ppo_config["learning_rate"] = args.learning_rate

        # Parse policy kwargs
        policy_kwargs = ppo_config.pop("policy_kwargs", {})
        if "activation_fn" in policy_kwargs:
            # Convert string to actual function
            activation = policy_kwargs["activation_fn"]
            if activation == "torch.nn.ReLU":
                policy_kwargs["activation_fn"] = torch.nn.ReLU
            elif activation == "torch.nn.Tanh":
                policy_kwargs["activation_fn"] = torch.nn.Tanh
            else:
                policy_kwargs["activation_fn"] = torch.nn.ReLU

        # Use regular PPO instead of MaskablePPO for initial training
        # (VecEnv doesn't preserve action_mask from reset())
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            **ppo_config,
            policy_kwargs=policy_kwargs,
            verbose=1 if args.verbose else 0,
            seed=args.seed,
            tensorboard_log=config["output"]["tensorboard_log"] if not args.no_tensorboard else None
        )

    return model, train_env, eval_env


def setup_callbacks(config: Dict[str, Any], eval_env: Any, args: argparse.Namespace) -> CallbackList:
    """Setup training callbacks."""
    callbacks = []

    # Checkpoint callback
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=config["training"].get("save_freq", 50_000),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_checkpoint",
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback
    # Note: Disabled when using VecNormalize or curriculum learning to avoid sync issues
    if not config.get("normalize", True) and not config.get("curriculum", {}).get("enabled", False):
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(checkpoint_dir / "best_model"),
            log_path=str(checkpoint_dir / "eval_logs"),
            eval_freq=config["training"].get("eval_freq", 10_000),
            n_eval_episodes=config["training"].get("eval_episodes", 100),
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)

    # Metrics callback
    metrics_callback = MetricsCallback(verbose=args.verbose)
    callbacks.append(metrics_callback)

    # W&B callback
    if WANDB_AVAILABLE and config["wandb"]["enabled"] and not args.no_wandb:
        wandb.init(
            project=config["wandb"]["project"],
            name=config["wandb"]["name"].replace("${now:%Y%m%d-%H%M%S}",
                                                 datetime.now().strftime("%Y%m%d-%H%M%S")),
            config=config,
            tags=config["wandb"]["tags"],
            notes=config["wandb"]["notes"]
        )

        wandb_callback = WandbCallback(
            gradient_save_freq=1000,
            model_save_path=str(checkpoint_dir / "wandb_models"),
            verbose=1
        )
        callbacks.append(wandb_callback)

    # Curriculum callback (if enabled)
    if config.get("curriculum", {}).get("enabled", False):
        print("📚 Setting up curriculum learning...")
        scheduler = CurriculumScheduler(config, make_env, verbose=1)
        curriculum_callback = CurriculumCallback(
            scheduler,
            eval_env,
            eval_freq=config["training"].get("curriculum_eval_freq", 25_000),
            n_eval_episodes=100,
            verbose=1
        )
        callbacks.append(curriculum_callback)

    return CallbackList(callbacks)


def evaluate_final_model(model: PPO, eval_env: Any, n_episodes: int = 100) -> Dict[str, float]:
    """Evaluate the final trained model."""
    print("\n📊 Final Evaluation...")

    episode_rewards = []
    episode_lengths = []
    episode_metrics = {
        "efficiency": [],
        "profit": [],
        "trades": [],
        "profitable_trades": [],
        "invalid_actions": []
    }

    for episode in range(n_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        final_info = None

        while not (done[0] if isinstance(done, np.ndarray) else done):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            episode_length += 1

            # Save final info dict when episode ends
            episode_done = done[0] if isinstance(done, np.ndarray) else done
            if episode_done:
                final_info = info

        # Extract metrics from final info dict (after loop exits)
        if final_info is not None:
            if isinstance(final_info, list):
                final_info = final_info[0]
            if "metrics" in final_info:
                metrics = final_info["metrics"]
                episode_metrics["efficiency"].append(metrics.get("market_efficiency", 0))
                episode_metrics["profit"].append(metrics.get("total_profit", 0))
                episode_metrics["trades"].append(metrics.get("trades_executed", 0))
                episode_metrics["profitable_trades"].append(metrics.get("profitable_trades", 0))
                episode_metrics["invalid_actions"].append(metrics.get("invalid_actions", 0))

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if (episode + 1) % 20 == 0:
            print(f"  Episodes {episode + 1}/{n_episodes} complete...")

    # Calculate statistics (convert numpy types to Python native types for JSON serialization)
    results = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "mean_efficiency": float(np.mean(episode_metrics["efficiency"])) if episode_metrics["efficiency"] else 0.0,
        "mean_profit": float(np.mean(episode_metrics["profit"])) if episode_metrics["profit"] else 0.0,
        "mean_trades": float(np.mean(episode_metrics["trades"])) if episode_metrics["trades"] else 0.0,
        "mean_profitable_trades": float(np.mean(episode_metrics["profitable_trades"])) if episode_metrics["profitable_trades"] else 0.0,
        "mean_invalid_actions": float(np.mean(episode_metrics["invalid_actions"])) if episode_metrics["invalid_actions"] else 0.0,
        "profit_ratio": float(np.mean(episode_metrics["profit"]) / 100) if episode_metrics["profit"] else 0.0  # vs baseline
    }

    return results


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Enhanced PPO Training for Double Auction")

    # Configuration
    parser.add_argument("--config", type=str, default="ppo_vs_zic",
                       help="Training configuration to use (config file name from conf/rl/experiments/*.yaml)")

    # Training settings
    parser.add_argument("--timesteps", type=int, default=None,
                       help="Total timesteps to train (overrides config)")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="Learning rate (overrides config)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # Resume/evaluation
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--evaluate-only", action="store_true",
                       help="Only evaluate, don't train")

    # Logging
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--no-tensorboard", action="store_true",
                       help="Disable tensorboard logging")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable W&B logging")

    args = parser.parse_args()

    # Load configuration
    print(f"📋 Loading configuration: {args.config}")
    config = load_config(args.config)

    # Override with command line arguments
    if args.timesteps:
        config["training"]["total_timesteps"] = args.timesteps

    # Print configuration
    print("\n" + "="*60)
    print(f"🎯 PPO TRAINING: {config['experiment']['name']}")
    print(f"📝 {config['experiment']['description']}")
    print("="*60)
    print(f"Environment: {config['env']['num_agents']} agents, {config['env']['num_tokens']} tokens")
    print(f"Opponents: {config['env'].get('opponent_mix', config['env'].get('opponent_type', 'ZIC'))}")
    print(f"Total Timesteps: {config['training']['total_timesteps']:,}")
    print(f"Parallel Envs: {config['training'].get('n_envs', 8)}")
    print("="*60 + "\n")

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup training
    model, train_env, eval_env = setup_training(config, args)

    # Evaluate only mode
    if args.evaluate_only:
        # Use train_env for evaluation since it has VecNormalize stats loaded
        results = evaluate_final_model(model, train_env, n_episodes=200)
        print("\n📊 EVALUATION RESULTS:")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Market Efficiency: {results['mean_efficiency']:.3f}")
        print(f"  Total Profit: {results['mean_profit']:.2f}")
        print(f"  Profit Ratio: {results['profit_ratio']:.3f}")
        print(f"  Trades/Episode: {results['mean_trades']:.1f}")
        print(f"  Profitable Trades: {results['mean_profitable_trades']:.1f}")
        print(f"  Invalid Actions: {results['mean_invalid_actions']:.1f}")

        # Save results
        results_path = Path(config["output"]["checkpoint_dir"]) / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {results_path}")
        return

    # Setup callbacks
    callbacks = setup_callbacks(config, eval_env, args)

    # Train model
    print("\n🚀 Starting training...")
    print(f"  Total timesteps: {config['training']['total_timesteps']:,}")
    print(f"  Learning rate: {config['ppo'].get('learning_rate', 0.0003)}")
    print(f"  Batch size: {config['ppo'].get('batch_size', 64)}")
    print(f"  Entropy coefficient: {config['ppo'].get('ent_coef', 0.01)}")
    print("-"*60 + "\n")

    try:
        model.learn(
            total_timesteps=config["training"]["total_timesteps"],
            callback=callbacks,
            log_interval=100,
            progress_bar=False  # Disabled for headless training
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")

    # Save final model
    print("\n💾 Saving final model...")
    final_model_path = Path(config["output"]["model_save_path"]) / "final_model.zip"
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(final_model_path)
    print(f"  Model saved to {final_model_path}")

    # Save normalization statistics if used
    if isinstance(train_env, VecNormalize):
        norm_path = Path(config["output"]["model_save_path"]) / "vec_normalize.pkl"
        train_env.save(str(norm_path))
        print(f"  Normalization saved to {norm_path}")

    # Final evaluation
    results = evaluate_final_model(model, eval_env, n_episodes=100)

    # Print results
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("📊 Final Performance:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Market Efficiency: {results['mean_efficiency']:.3f}")
    print(f"  Total Profit: {results['mean_profit']:.2f}")
    print(f"  Profit Ratio: {results['profit_ratio']:.3f}")
    print(f"  Trades/Episode: {results['mean_trades']:.1f}")

    # Check success metrics
    target_efficiency = config["metrics"].get("target_efficiency", 0.80)
    target_profit = config["metrics"].get("target_profit_ratio", 1.0)

    print(f"\n🎯 Target Metrics:")
    print(f"  Efficiency: {results['mean_efficiency']:.3f} / {target_efficiency:.3f} ", end="")
    print("✅" if results['mean_efficiency'] >= target_efficiency else "❌")
    print(f"  Profit Ratio: {results['profit_ratio']:.3f} / {target_profit:.3f} ", end="")
    print("✅" if results['profit_ratio'] >= target_profit else "❌")

    # Save final results
    results_path = Path(config["output"]["checkpoint_dir"]) / "final_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {results_path}")

    # Close environments
    train_env.close()
    eval_env.close()

    # Finish W&B run
    if WANDB_AVAILABLE and config["wandb"]["enabled"] and not args.no_wandb:
        wandb.finish()

    print("\n" + "="*60)
    print("🎉 All done! Happy trading! 🎉")
    print("="*60)


if __name__ == "__main__":
    main()