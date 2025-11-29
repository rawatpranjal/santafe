#!/usr/bin/env python3
"""Run all Part 2 experiments in batch."""
import subprocess
import sys
from pathlib import Path

def run_experiments(category: str, num_rounds: int = 50, skip_gd: bool = True):
    """Run all experiments in a category."""
    conf_dir = Path("conf/experiment/p2_tournament") / category
    experiments = sorted(conf_dir.glob("*.yaml"))

    if skip_gd:
        experiments = [e for e in experiments if "_gd_" not in e.stem]

    print(f"\n{'='*60}")
    print(f"Running {len(experiments)} {category} experiments (num_rounds={num_rounds})")
    print(f"{'='*60}\n")

    results = []
    for i, exp_file in enumerate(experiments):
        exp_name = exp_file.stem
        print(f"[{i+1}/{len(experiments)}] {exp_name}...", end=" ", flush=True)

        cmd = [
            "uv", "run", "python", "scripts/run_experiment.py",
            f"experiment=p2_tournament/{category}/{exp_name}",
            f"experiment.num_rounds={num_rounds}",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # Extract efficiency from output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Average Efficiency by Period' in line:
                    break
            print("OK")
            results.append((exp_name, "OK"))
        else:
            print("FAILED")
            results.append((exp_name, "FAILED"))
            print(result.stderr[-500:] if result.stderr else "No error output")

    print(f"\n{'='*60}")
    print(f"Completed: {sum(1 for _, s in results if s == 'OK')}/{len(results)}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    category = sys.argv[1] if len(sys.argv) > 1 else "self"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    run_experiments(category, rounds)
