#!/usr/bin/env python3
"""Run all ZIP experiments for Part 2."""
import subprocess
import sys

ENVS = ["base", "bbbs", "bsss", "eql", "ran", "per", "shrt", "tok", "sml", "lad"]


def run_zip_experiments(num_rounds: int = 50):
    """Run ZIP self-play and control experiments."""

    # Self-play
    print("=" * 60)
    print("ZIP Self-Play Experiments")
    print("=" * 60)

    for env in ENVS:
        exp_name = f"p2_self_zip_{env}"
        print(f"  {exp_name}...", end=" ", flush=True)

        cmd = [
            "uv",
            "run",
            "python",
            "scripts/run_experiment.py",
            f"experiment=p2_tournament/self/{exp_name}",
            f"experiment.num_rounds={num_rounds}",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("OK")
        else:
            print("FAILED")
            print(result.stderr[-300:] if result.stderr else "")

    # Control
    print("\n" + "=" * 60)
    print("ZIP Control Experiments (1 ZIP vs 7 ZIC)")
    print("=" * 60)

    for env in ENVS:
        exp_name = f"p2_ctrl_zip_{env}"
        print(f"  {exp_name}...", end=" ", flush=True)

        cmd = [
            "uv",
            "run",
            "python",
            "scripts/run_experiment.py",
            f"experiment=p2_tournament/ctrl/{exp_name}",
            f"experiment.num_rounds={num_rounds}",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("OK")
        else:
            print("FAILED")
            print(result.stderr[-300:] if result.stderr else "")

    print("\n" + "=" * 60)
    print("Done! Results in results/p2_self_zip_* and results/p2_ctrl_zip_*")
    print("=" * 60)


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_zip_experiments(rounds)
