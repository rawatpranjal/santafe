#!/usr/bin/env python3
"""Run all Part 1 foundational experiments in batch.

Usage:
    python scripts/run_p1_batch.py [num_rounds]

    num_rounds: Number of rounds per experiment (default: 100)

Example:
    python scripts/run_p1_batch.py 100  # Full run
    python scripts/run_p1_batch.py 10   # Quick test
"""
import subprocess
import sys
from pathlib import Path


def run_experiments(num_rounds: int = 100):
    """Run all Part 1 experiments."""
    conf_dir = Path("conf/experiment/p1_foundational")
    experiments = sorted(conf_dir.glob("*.yaml"))

    print(f"\n{'='*60}")
    print(f"Running {len(experiments)} Part 1 experiments (num_rounds={num_rounds})")
    print(f"{'='*60}\n")

    results = []
    for i, exp_file in enumerate(experiments):
        exp_name = exp_file.stem
        print(f"[{i+1}/{len(experiments)}] {exp_name}...", end=" ", flush=True)

        cmd = [
            "uv",
            "run",
            "python",
            "scripts/run_experiment.py",
            f"experiment=p1_foundational/{exp_name}",
            f"experiment.num_rounds={num_rounds}",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # Extract average efficiency from output
            lines = result.stdout.split("\n")
            avg_eff = "?"
            for j, line in enumerate(lines):
                if "Average Efficiency by Period" in line:
                    # Parse the next few lines for efficiency values
                    try:
                        # Look for dtype line which has the mean
                        for k in range(j + 1, min(j + 10, len(lines))):
                            if "dtype" in lines[k]:
                                break
                            if lines[k].strip() and lines[k][0].isdigit():
                                parts = lines[k].split()
                                if len(parts) >= 2:
                                    avg_eff = parts[-1][:6]  # First 6 chars
                    except Exception:
                        pass
                    break
            print(f"OK (eff~{avg_eff})")
            results.append((exp_name, "OK", avg_eff))
        else:
            print("FAILED")
            results.append((exp_name, "FAILED", "N/A"))
            print(result.stderr[-500:] if result.stderr else "No error output")

    print(f"\n{'='*60}")
    print(f"SUMMARY: Completed {sum(1 for _, s, _ in results if s == 'OK')}/{len(results)}")
    print(f"{'='*60}\n")

    # Group by strategy
    strategies = {}
    for exp_name, status, eff in results:
        # Parse strategy from name like p1_self_zic_base
        parts = exp_name.split("_")
        if len(parts) >= 3:
            strat = parts[2].upper()
            if strat not in strategies:
                strategies[strat] = []
            strategies[strat].append((exp_name, status, eff))

    for strat in sorted(strategies.keys()):
        print(f"\n{strat}:")
        for exp_name, status, eff in strategies[strat]:
            env = exp_name.split("_")[-1].upper()
            print(f"  {env}: {eff}")


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_experiments(rounds)
