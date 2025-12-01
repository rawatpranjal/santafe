"""
Train PPO for all 10 tournament environments.

Each environment requires a separate training run because PPO cannot
generalize across different market configurations.

Usage:
    python scripts/train_ppo_all_envs.py                    # Train all
    python scripts/train_ppo_all_envs.py --env BASE         # Train single env
    python scripts/train_ppo_all_envs.py --env BASE BBBS    # Train subset
    python scripts/train_ppo_all_envs.py --steps 1000000    # Override steps
"""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

# Environment configurations
# Each env has: num_buyers, num_sellers, num_tokens, max_timesteps, gametype
ENV_CONFIGS: dict[str, dict[str, Any]] = {
    "BASE": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens_per_agent": 4,
        "max_timesteps": 100,
        "gametype": 6453,
    },
    "BBBS": {  # Buyer-dominated (6B/2S)
        "num_buyers": 6,
        "num_sellers": 2,
        "num_tokens_per_agent": 4,
        "max_timesteps": 100,
        "gametype": 6453,
    },
    "BSSS": {  # Seller-dominated (2B/6S)
        "num_buyers": 2,
        "num_sellers": 6,
        "num_tokens_per_agent": 4,
        "max_timesteps": 100,
        "gametype": 6453,
    },
    "EQL": {  # Equal endowment (symmetric tokens)
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens_per_agent": 4,
        "max_timesteps": 100,
        "gametype": 5555,
    },
    "RAN": {  # Random token distribution
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens_per_agent": 4,
        "max_timesteps": 100,
        "gametype": 9999,
    },
    "SHRT": {  # Short periods (20 steps)
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens_per_agent": 4,
        "max_timesteps": 20,
        "gametype": 6453,
    },
    "TOK": {  # Single token per trader
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens_per_agent": 1,
        "max_timesteps": 100,
        "gametype": 6453,
    },
    "SML": {  # Small market (2B/2S)
        "num_buyers": 2,
        "num_sellers": 2,
        "num_tokens_per_agent": 4,
        "max_timesteps": 100,
        "gametype": 6453,
    },
}

# Note: PER (single period) and LAD (low adaptivity) are identical to BASE
# for RL training because the RL env doesn't use periods/adaptivity settings.
# We skip them and use BASE model for those evaluations.


def train_env(
    env_name: str,
    config: dict[str, Any],
    total_timesteps: int = 3_000_000,
    n_envs: int = 8,
    opponent_type: str = "Mixed",
    early_stopping_patience: int = 5,
) -> bool:
    """
    Train PPO for a specific environment configuration.

    Args:
        env_name: Environment name (e.g., "BASE", "BBBS")
        config: Environment configuration dict
        total_timesteps: Max training timesteps (with early stopping)
        n_envs: Number of parallel environments
        opponent_type: Opponent type for training
        early_stopping_patience: Stop after N evals with no improvement

    Returns:
        True if training succeeded
    """
    checkpoint_dir = f"./checkpoints/ppo_{env_name.lower()}"

    # Build Hydra override command (use venv python)
    python_path = str(Path(__file__).parent.parent / ".venv" / "bin" / "python")
    cmd = [
        python_path,
        "scripts/train_ppo.py",
        f"rl.env.num_buyers={config['num_buyers']}",
        f"rl.env.num_sellers={config['num_sellers']}",
        f"rl.env.num_tokens_per_agent={config['num_tokens_per_agent']}",
        f"rl.env.max_timesteps={config['max_timesteps']}",
        f"rl.env.gametype={config['gametype']}",
        f"rl.env.opponent_type={opponent_type}",
        f"rl.total_timesteps={total_timesteps}",
        f"rl.n_envs={n_envs}",
        f"rl.checkpoint_dir={checkpoint_dir}",
        "rl.wandb.enabled=false",  # Disable W&B for batch training
        "rl.ent_coef=0.15",  # Start with high entropy
        f"+rl.early_stopping_patience={early_stopping_patience}",  # Early stopping (+ prefix for new key)
    ]

    print(f"\n{'='*80}")
    print(f"Training PPO for {env_name} environment")
    print(f"Config: {config}")
    print(f"Output: {checkpoint_dir}")
    print(f"{'='*80}\n")

    # Create log file
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"ppo_{env_name.lower()}_training.log"

    try:
        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).parent.parent,
            )

        if result.returncode == 0:
            print(f"✓ {env_name} training completed successfully")
            return True
        else:
            print(f"✗ {env_name} training failed (exit code {result.returncode})")
            return False

    except Exception as e:
        print(f"✗ {env_name} training error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train PPO for all environments")
    parser.add_argument(
        "--env",
        nargs="+",
        default=list(ENV_CONFIGS.keys()),
        help="Environments to train (default: all)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3_000_000,
        help="Max training timesteps per environment (default: 3M with early stopping)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience - stop after N evals without improvement (default: 5)",
    )
    parser.add_argument(
        "--n_envs", type=int, default=8, help="Number of parallel environments (default: 8)"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="Mixed",
        choices=["ZIC", "ZIP", "Skeleton", "Kaplan", "Mixed"],
        help="Opponent type for training (default: Mixed)",
    )
    args = parser.parse_args()

    print(f"\n{'#'*80}")
    print("# PPO Multi-Environment Training (with Early Stopping)")
    print(f"# Environments: {args.env}")
    print(f"# Max steps per env: {args.steps:,}")
    print(f"# Early stopping patience: {args.patience} evals")
    print(f"# Parallel envs: {args.n_envs}")
    print(f"# Opponent: {args.opponent}")
    print(f"# Started: {datetime.now().isoformat()}")
    print(f"{'#'*80}\n")

    results: dict[str, bool] = {}

    for env_name in args.env:
        if env_name not in ENV_CONFIGS:
            print(f"Warning: Unknown environment '{env_name}', skipping")
            continue

        success = train_env(
            env_name=env_name,
            config=ENV_CONFIGS[env_name],
            total_timesteps=args.steps,
            n_envs=args.n_envs,
            opponent_type=args.opponent,
            early_stopping_patience=args.patience,
        )
        results[env_name] = success

    # Summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")

    success_count = sum(results.values())
    total_count = len(results)

    for env_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {env_name}: {status}")

    print(f"\nCompleted: {success_count}/{total_count}")
    print(f"Finished: {datetime.now().isoformat()}")

    # Return non-zero exit code if any failed
    if success_count < total_count:
        exit(1)


if __name__ == "__main__":
    main()
