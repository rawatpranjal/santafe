"""
Train PPO with checkpoints for learning curve generation.

This script trains PPO against ZIC opponents with pure profit reward,
saving checkpoints at key timesteps (100K, 500K, 1M, 3M) for evaluation.

The learning curve will show profit vs training steps, with horizontal
lines for fixed strategies (ZIC, ZIP, GD, Kaplan) as baselines.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from envs.vec_env_utils import make_vec_env, get_default_env_kwargs


class LearningCurveCallback(BaseCallback):
    """Custom callback to save checkpoints at specific timesteps."""

    def __init__(self, checkpoints: list[int], save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.checkpoints = sorted(checkpoints)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.saved_checkpoints = set()

    def _on_step(self) -> bool:
        for checkpoint in self.checkpoints:
            if self.num_timesteps >= checkpoint and checkpoint not in self.saved_checkpoints:
                save_file = self.save_path / f"ppo_{checkpoint // 1000}k"
                self.model.save(str(save_file))
                self.saved_checkpoints.add(checkpoint)
                print(f"\n[Checkpoint] Saved model at {checkpoint:,} steps to {save_file}")
        return True


def train_for_learning_curve(
    total_timesteps: int = 3_000_000,
    n_envs: int = 16,
    opponent_type: str = "ZIC",
    checkpoints: list[int] = None,
    output_dir: str = "./checkpoints/learning_curve",
    seed: int = 42
):
    """
    Train PPO with pure profit reward and save checkpoints.

    Args:
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        opponent_type: Type of opponent (ZIC, ZIP, GD, Kaplan, Mixed)
        checkpoints: List of timesteps to save checkpoints (default: 100K, 500K, 1M, 3M)
        output_dir: Directory to save checkpoints
        seed: Random seed
    """
    if checkpoints is None:
        checkpoints = [100_000, 500_000, 1_000_000, 3_000_000]

    print("=" * 70)
    print("PPO LEARNING CURVE TRAINING")
    print("=" * 70)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {n_envs}")
    print(f"Opponent type: {opponent_type}")
    print(f"Pure profit mode: ENABLED")
    print(f"Checkpoints at: {[f'{c//1000}K' for c in checkpoints]}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create environment with pure profit mode
    # NOTE: price_max=100 matches TokenGenerator game_type=1111 which produces
    # valuations in ~0-10 range. Using 100 ensures reasonable bid/ask spreads.
    env_kwargs = {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens_per_agent": 4,
        "max_timesteps": 100,
        "price_min": 1,
        "price_max": 100,  # Must match TokenGenerator scale
        "rl_agent_type": "buyer",
        "opponent_type": opponent_type,
        "use_enhanced_env": True,
        "pure_profit_mode": True,  # Key setting for Chen et al. style
    }

    print("\nCreating vectorized training environment...")
    train_env = make_vec_env(
        n_envs=n_envs,
        start_method="spawn",  # Safe for multiprocessing
        env_kwargs=env_kwargs,
        seed=seed
    )

    # Create MaskablePPO model
    print("Initializing MaskablePPO model...")
    model = MaskablePPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
        tensorboard_log=None  # Disable tensorboard (not installed)
    )

    # Create callback for checkpoints
    checkpoint_callback = LearningCurveCallback(
        checkpoints=checkpoints,
        save_path=output_dir,
        verbose=1
    )

    # Train
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print("-" * 70)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=False  # Disabled (requires tqdm/rich)
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # Save final model
    final_path = Path(output_dir) / "ppo_final"
    model.save(str(final_path))
    print(f"\nFinal model saved to: {final_path}")

    # Cleanup
    train_env.close()

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"\nCheckpoints saved in: {output_dir}")
    print("Next: Run evaluation script to measure profit at each checkpoint")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO for learning curve")
    parser.add_argument("--timesteps", type=int, default=3_000_000,
                       help="Total training timesteps (default: 3M)")
    parser.add_argument("--n_envs", type=int, default=16,
                       help="Number of parallel environments")
    parser.add_argument("--opponent", type=str, default="ZIC",
                       choices=["ZIC", "ZIP", "GD", "Kaplan", "Mixed"],
                       help="Opponent type to train against")
    parser.add_argument("--output", type=str, default="./checkpoints/learning_curve",
                       help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    train_for_learning_curve(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        opponent_type=args.opponent,
        output_dir=args.output,
        seed=args.seed
    )
