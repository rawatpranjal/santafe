"""
Evaluate PPO checkpoints for learning curve generation.

This script evaluates saved PPO checkpoints against ZIC opponents (1v7 format)
and measures profit ratios to generate the learning curve data.

Output: JSON file with profit data at each training checkpoint, ready for plotting.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sb3_contrib import MaskablePPO

from envs.enhanced_double_auction_env import EnhancedDoubleAuctionEnv


def evaluate_ppo_checkpoint(
    model_path: str,
    num_episodes: int = 30,
    opponent_type: str = "ZIC",
    seed: int = 42
) -> dict:
    """
    Evaluate a PPO checkpoint in 1v7 format against opponents.

    Args:
        model_path: Path to saved PPO model
        num_episodes: Number of evaluation episodes
        opponent_type: Type of opponent agents
        seed: Random seed

    Returns:
        Dictionary with evaluation metrics
    """
    # Create evaluation environment
    # NOTE: price_max=100 matches TokenGenerator scale (valuations in ~0-10 range)
    config = {
        "num_agents": 8,
        "num_tokens": 4,
        "max_steps": 100,
        "min_price": 1,
        "max_price": 100,  # Must match training config
        "rl_agent_id": 1,
        "rl_is_buyer": True,
        "opponent_type": opponent_type,
        "pure_profit_mode": True,
    }

    env = EnhancedDoubleAuctionEnv(config)

    # Load model
    model = MaskablePPO.load(model_path)

    # Evaluate
    profits = []
    efficiencies = []
    trades_completed = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        episode_reward = 0
        done = False

        while not done:
            action_masks = info.get("action_mask", None)
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        # Record metrics
        profits.append(info["metrics"]["total_profit"])
        efficiencies.append(info["metrics"]["market_efficiency"])
        trades_completed.append(info["metrics"]["trades_executed"])

    env.close()

    return {
        "avg_profit": float(np.mean(profits)),
        "std_profit": float(np.std(profits)),
        "avg_efficiency": float(np.mean(efficiencies)),
        "avg_trades": float(np.mean(trades_completed)),
        "num_episodes": num_episodes
    }


def evaluate_all_checkpoints(
    checkpoint_dir: str = "./checkpoints/learning_curve",
    num_episodes: int = 30,
    opponent_type: str = "ZIC",
    seed: int = 42
) -> dict:
    """
    Evaluate all checkpoints in a directory.

    Args:
        checkpoint_dir: Directory containing PPO checkpoints
        num_episodes: Episodes per evaluation
        opponent_type: Opponent type
        seed: Random seed

    Returns:
        Dictionary mapping checkpoint name to metrics
    """
    checkpoint_path = Path(checkpoint_dir)
    results = {}

    # Find all .zip files (SB3 model format)
    model_files = list(checkpoint_path.glob("*.zip"))

    if not model_files:
        print(f"No model files found in {checkpoint_dir}")
        return results

    print(f"\nFound {len(model_files)} checkpoints to evaluate")
    print("-" * 60)

    for model_file in sorted(model_files):
        name = model_file.stem
        print(f"\nEvaluating {name}...")

        try:
            metrics = evaluate_ppo_checkpoint(
                model_path=str(model_file),
                num_episodes=num_episodes,
                opponent_type=opponent_type,
                seed=seed
            )
            results[name] = metrics
            print(f"  Avg profit: {metrics['avg_profit']:.2f} +/- {metrics['std_profit']:.2f}")
            print(f"  Avg efficiency: {metrics['avg_efficiency']:.2%}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"error": str(e)}

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoints for learning curve")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/learning_curve",
                       help="Directory containing PPO checkpoints")
    parser.add_argument("--episodes", type=int, default=30,
                       help="Number of evaluation episodes per checkpoint")
    parser.add_argument("--opponent", type=str, default="ZIC",
                       help="Opponent type for evaluation")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file (default: checkpoint_dir/eval_results.json)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print("=" * 60)
    print("PPO LEARNING CURVE EVALUATION")
    print("=" * 60)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Episodes per evaluation: {args.episodes}")
    print(f"Opponent type: {args.opponent}")
    print("=" * 60)

    # Run evaluation
    results = evaluate_all_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        num_episodes=args.episodes,
        opponent_type=args.opponent,
        seed=args.seed
    )

    # Add metadata
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_dir": args.checkpoint_dir,
        "opponent_type": args.opponent,
        "episodes_per_eval": args.episodes,
        "results": results
    }

    # Save results
    output_file = args.output or Path(args.checkpoint_dir) / "eval_results.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to: {output_file}")
    print("=" * 60)

    # Summary
    print("\n--- LEARNING CURVE DATA ---")
    print(f"{'Checkpoint':<15} {'Profit':>10} {'Efficiency':>12}")
    print("-" * 40)
    for name, metrics in sorted(results.items()):
        if "error" not in metrics:
            print(f"{name:<15} {metrics['avg_profit']:>10.2f} {metrics['avg_efficiency']:>11.2%}")


if __name__ == "__main__":
    main()
