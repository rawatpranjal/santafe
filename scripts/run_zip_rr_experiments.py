#!/usr/bin/env python3
"""Run all ZIP round-robin experiments for Part 2."""
import subprocess
import sys

ENVS = ["base", "bbbs", "bsss", "eql", "ran", "per", "shrt", "tok", "sml", "lad"]


def run_zip_rr_experiments(num_rounds: int = 50):
    """Run ZIP round-robin experiments (13 traders: 7 buyers × 6 sellers)."""

    print("=" * 60)
    print("ZIP Round-Robin Experiments (7 buyers × 6 sellers)")
    print("=" * 60)

    for i, env in enumerate(ENVS):
        exp_name = f"p2_rr_mixed_zip_{env}"
        print(f"[{i+1}/{len(ENVS)}] {exp_name}...", end=" ", flush=True)

        cmd = [
            "uv",
            "run",
            "python",
            "scripts/run_experiment.py",
            f"experiment=p2_tournament/rr/{exp_name}",
            f"experiment.num_rounds={num_rounds}",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("OK")
        else:
            print("FAILED")
            print(result.stderr[-300:] if result.stderr else "")

    print("\n" + "=" * 60)
    print("Done! Results in results/p2_rr_mixed_zip_*")
    print("=" * 60)


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_zip_rr_experiments(rounds)
