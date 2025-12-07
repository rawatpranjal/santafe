#!/usr/bin/env python3
"""Run all Part 1 foundational experiments with multiple seeds.

Usage:
    python scripts/run_p1_batch_seeds.py [num_rounds] [num_seeds]

    num_rounds: Number of rounds per experiment (default: 100)
    num_seeds: Number of seeds to run (default: 10)

Example:
    python scripts/run_p1_batch_seeds.py 100 10  # Full run (400 experiments)
    python scripts/run_p1_batch_seeds.py 10 2    # Quick test
"""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def run_experiments(num_rounds: int = 100, num_seeds: int = 10):
    """Run all Part 1 experiments with multiple seeds."""
    conf_dir = Path("conf/experiment/p1_foundational")
    experiments = sorted(conf_dir.glob("*.yaml"))

    total_runs = len(experiments) * num_seeds
    print(f"\n{'='*70}")
    print(f"Running {len(experiments)} Part 1 experiments x {num_seeds} seeds = {total_runs} total")
    print(f"num_rounds={num_rounds}")
    print(f"{'='*70}\n")

    # Store all results: {exp_name: {seed: efficiency}}
    all_results = defaultdict(dict)
    run_count = 0

    for seed in range(num_seeds):
        print(f"\n--- SEED {seed} ---")
        for i, exp_file in enumerate(experiments):
            run_count += 1
            exp_name = exp_file.stem
            print(f"[{run_count}/{total_runs}] {exp_name} (seed={seed})...", end=" ", flush=True)

            # Use different seeds for auction and values RNG
            auction_seed = seed * 1000 + 42
            values_seed = seed * 1000 + 123

            cmd = [
                "uv",
                "run",
                "python",
                "scripts/run_experiment.py",
                f"experiment=p1_foundational/{exp_name}",
                f"experiment.num_rounds={num_rounds}",
                f"experiment.rng_seed_auction={auction_seed}",
                f"experiment.rng_seed_values={values_seed}",
                f"experiment.output_dir=./results/{exp_name}_seed{seed}",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Extract average efficiency from output
                lines = result.stdout.split("\n")
                avg_eff = None
                for j, line in enumerate(lines):
                    if "Average Efficiency by Period" in line:
                        try:
                            for k in range(j + 1, min(j + 10, len(lines))):
                                if "dtype" in lines[k]:
                                    break
                                if lines[k].strip() and lines[k][0].isdigit():
                                    parts = lines[k].split()
                                    if len(parts) >= 2:
                                        avg_eff = float(parts[-1])
                        except Exception:
                            pass
                        break

                if avg_eff is not None:
                    print(f"OK (eff={avg_eff:.2f})")
                    all_results[exp_name][seed] = avg_eff
                else:
                    print("OK (eff=?)")
            else:
                print("FAILED")
                print(result.stderr[-500:] if result.stderr else "No error output")

    # Save raw results
    results_path = Path("results/p1_foundational_raw.json")
    with open(results_path, "w") as f:
        json.dump(dict(all_results), f, indent=2)
    print(f"\nRaw results saved to {results_path}")

    # Compute and display statistics
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS (mean ± std)")
    print(f"{'='*70}\n")

    # Group by strategy
    strategies = defaultdict(dict)
    for exp_name, seed_results in all_results.items():
        parts = exp_name.split("_")
        if len(parts) >= 4:
            strat = parts[2].upper()
            env = parts[3].upper()
            values = list(seed_results.values())
            if values:
                import statistics

                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0.0
                strategies[strat][env] = (mean_val, std_val, len(values))

    # Print by strategy
    envs = ["BASE", "BBBS", "BSSS", "EQL", "LAD", "PER", "RAN", "SHRT", "SML", "TOK"]
    for strat in sorted(strategies.keys()):
        print(f"\n{strat}:")
        for env in envs:
            if env in strategies[strat]:
                mean_val, std_val, n = strategies[strat][env]
                print(f"  {env}: {mean_val:.2f} ± {std_val:.2f} (n={n})")
            else:
                print(f"  {env}: N/A")

    # Save aggregated results
    agg_results = {}
    for strat, env_data in strategies.items():
        agg_results[strat] = {}
        for env, (mean_val, std_val, n) in env_data.items():
            agg_results[strat][env] = {"mean": mean_val, "std": std_val, "n": n}

    agg_path = Path("results/p1_foundational_aggregated.json")
    with open(agg_path, "w") as f:
        json.dump(agg_results, f, indent=2)
    print(f"\nAggregated results saved to {agg_path}")

    return agg_results


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run_experiments(rounds, seeds)
