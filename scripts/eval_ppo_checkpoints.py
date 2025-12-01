#!/usr/bin/env python3
"""
Evaluate all PPO checkpoints in a directory for learning curve analysis.

Usage:
    python scripts/eval_ppo_checkpoints.py --checkpoint_dir checkpoints/ppo_v10_10M --output results/learning_curve.json
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def extract_timesteps(filename: str) -> int:
    """Extract timesteps from checkpoint filename."""
    # Match patterns like: ppo_double_auction_500000_steps.zip or final_model.zip
    match = re.search(r"(\d+)_steps", filename)
    if match:
        return int(match.group(1))
    if "final" in filename.lower():
        return float("inf")  # Final model sorts last
    return 0


def run_tournament(model_path: str, env: str = "BASE", periods: int = 10, seeds: int = 5) -> dict:
    """Run round-robin tournament and extract PPO profit."""
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/run_ppo_buyer_tournament.py",
        "--model",
        model_path,
        "--envs",
        env,
        "--periods",
        str(periods),
        "--seeds",
        str(seeds),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    # Parse output for PPO profit
    output = result.stdout + result.stderr

    # Find PPO line in rankings table
    ppo_profit = None
    ppo_rank = None

    for line in output.split("\n"):
        if line.strip().startswith("PPO"):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    ppo_profit = float(parts[1])
                    ppo_rank = int(parts[-1]) if parts[-1].isdigit() else None
                except (ValueError, IndexError):
                    pass

    return {
        "profit": ppo_profit,
        "rank": ppo_rank,
        "raw_output": output[-2000:] if len(output) > 2000 else output,  # Last 2000 chars
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoints for learning curves")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory containing checkpoints")
    parser.add_argument("--output", default="results/learning_curve.json", help="Output JSON file")
    parser.add_argument("--env", default="BASE", help="Environment to test")
    parser.add_argument("--periods", type=int, default=10, help="Periods per tournament")
    parser.add_argument("--seeds", type=int, default=5, help="Seeds per tournament")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    # Find all .zip checkpoints
    checkpoints = sorted(checkpoint_dir.glob("*.zip"), key=lambda p: extract_timesteps(p.name))

    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoints")
    print("=" * 60)

    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_dir": str(checkpoint_dir),
        "env": args.env,
        "periods": args.periods,
        "seeds": args.seeds,
        "checkpoints": [],
    }

    # Legacy baselines for reference
    results["baselines"] = {
        "Ringuette": 1414.4,
        "Ledyard": 1284.4,
        "Skeleton": 1241.4,
        "Kaplan": 1223.0,
        "GD": 1186.1,
        "Markup": 1073.7,
        "ZIC": 925.8,
        "ZIP": 830.2,
    }

    for i, checkpoint in enumerate(checkpoints, 1):
        timesteps = extract_timesteps(checkpoint.name)
        timesteps_str = f"{timesteps/1e6:.1f}M" if timesteps != float("inf") else "final"

        print(f"[{i}/{len(checkpoints)}] Evaluating {checkpoint.name} ({timesteps_str})...")

        try:
            result = run_tournament(str(checkpoint), args.env, args.periods, args.seeds)

            checkpoint_result = {
                "filename": checkpoint.name,
                "timesteps": timesteps if timesteps != float("inf") else "final",
                "profit": result["profit"],
                "rank": result["rank"],
            }

            results["checkpoints"].append(checkpoint_result)

            status = "WINNER!" if result["rank"] == 1 else f"rank {result['rank']}"
            print(f"    Profit: {result['profit']:.1f}, {status}")

        except subprocess.TimeoutExpired:
            print("    TIMEOUT - skipping")
            results["checkpoints"].append(
                {
                    "filename": checkpoint.name,
                    "timesteps": timesteps if timesteps != float("inf") else "final",
                    "profit": None,
                    "rank": None,
                    "error": "timeout",
                }
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            results["checkpoints"].append(
                {
                    "filename": checkpoint.name,
                    "timesteps": timesteps if timesteps != float("inf") else "final",
                    "profit": None,
                    "rank": None,
                    "error": str(e),
                }
            )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print(f"Results saved to: {output_path}")

    # Print summary
    print("\nLEARNING CURVE SUMMARY")
    print("-" * 60)
    print(f"{'Timesteps':<12} {'Profit':<10} {'Rank':<6} {'vs Ringuette'}")
    print("-" * 60)

    ringuette_profit = results["baselines"]["Ringuette"]

    for cp in results["checkpoints"]:
        ts = cp["timesteps"]
        ts_str = f"{ts/1e6:.1f}M" if isinstance(ts, (int, float)) and ts != "final" else "final"
        profit = cp["profit"]
        rank = cp["rank"]

        if profit is not None:
            diff = profit - ringuette_profit
            diff_str = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
            print(f"{ts_str:<12} {profit:<10.1f} {rank:<6} {diff_str}")
        else:
            print(f"{ts_str:<12} {'N/A':<10} {'N/A':<6} N/A")

    # Find crossover point
    crossover = None
    for cp in results["checkpoints"]:
        if cp["profit"] and cp["profit"] > ringuette_profit:
            crossover = cp["timesteps"]
            break

    if crossover:
        ts_str = f"{crossover/1e6:.1f}M" if isinstance(crossover, (int, float)) else crossover
        print(f"\nCROSSOVER POINT: PPO beats Ringuette at {ts_str} steps!")
    else:
        print("\nPPO has not yet beaten Ringuette")


if __name__ == "__main__":
    main()
