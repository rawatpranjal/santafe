#!/usr/bin/env python3
"""
Run all Part 1 Foundational experiments (110 configs).

Usage:
    python scripts/run_p1_all.py                    # Run all 110 experiments
    python scripts/run_p1_all.py --set easy         # Run only easy-play (50 configs)
    python scripts/run_p1_all.py --set self         # Run only self-play (50 configs)
    python scripts/run_p1_all.py --set mixed        # Run only mixed-play (10 configs)
    python scripts/run_p1_all.py --strategy zip1    # Run only ZIP1 experiments
    python scripts/run_p1_all.py --env base         # Run only BASE environment
    python scripts/run_p1_all.py --dry-run          # Print commands without executing
    python scripts/run_p1_all.py --parallel 4       # Run 4 experiments in parallel
"""

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Configuration
STRATEGIES = ["zi", "zic1", "zic2", "zip1", "zip2"]
ENVS = ["base", "bbbs", "bsss", "eql", "ran", "per", "shrt", "tok", "sml", "lad"]
NUM_ROUNDS = 100
LOG_EVENTS = True


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    set_type: str  # easy, self, or mixed
    strategy: str | None  # None for mixed
    env: str

    @property
    def config_path(self) -> str:
        return f"experiment=p1_foundational/{self.name}"


def generate_experiments() -> list[ExperimentConfig]:
    """Generate all 110 experiment configurations."""
    experiments = []

    # Easy-play: 5 strategies × 10 environments = 50
    for strategy in STRATEGIES:
        for env in ENVS:
            experiments.append(
                ExperimentConfig(
                    name=f"p1_easy_{strategy}_{env}",
                    set_type="easy",
                    strategy=strategy,
                    env=env,
                )
            )

    # Self-play: 5 strategies × 10 environments = 50
    for strategy in STRATEGIES:
        for env in ENVS:
            experiments.append(
                ExperimentConfig(
                    name=f"p1_self_{strategy}_{env}",
                    set_type="self",
                    strategy=strategy,
                    env=env,
                )
            )

    # Mixed-play: 10 environments
    for env in ENVS:
        experiments.append(
            ExperimentConfig(
                name=f"p1_mixed_{env}",
                set_type="mixed",
                strategy=None,
                env=env,
            )
        )

    return experiments


def filter_experiments(
    experiments: list[ExperimentConfig],
    set_type: str | None = None,
    strategy: str | None = None,
    env: str | None = None,
) -> list[ExperimentConfig]:
    """Filter experiments based on criteria."""
    filtered = experiments

    if set_type:
        filtered = [e for e in filtered if e.set_type == set_type]

    if strategy:
        # Mixed experiments don't have a strategy
        filtered = [e for e in filtered if e.strategy == strategy]

    if env:
        filtered = [e for e in filtered if e.env == env]

    return filtered


def run_experiment(config: ExperimentConfig, dry_run: bool = False) -> dict:
    """Run a single experiment and return results."""
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/run_experiment.py",
        config.config_path,
        f"experiment.num_rounds={NUM_ROUNDS}",
    ]

    if LOG_EVENTS:
        cmd.append("log_events=true")
        cmd.append(f"experiment_id={config.name}")

    result = {
        "name": config.name,
        "set_type": config.set_type,
        "strategy": config.strategy,
        "env": config.env,
        "command": " ".join(cmd),
        "status": "pending",
        "stdout": "",
        "stderr": "",
        "returncode": None,
    }

    if dry_run:
        result["status"] = "dry_run"
        print(f"[DRY-RUN] {result['command']}")
        return result

    print(f"[RUNNING] {config.name}...")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["returncode"] = proc.returncode
        result["status"] = "success" if proc.returncode == 0 else "failed"

        # Extract efficiency from output if present
        for line in proc.stdout.split("\n")[-10:]:
            if "efficiency" in line.lower():
                print(f"  → {line.strip()}")

        if proc.returncode != 0:
            print(f"  [ERROR] {proc.stderr[-200:]}")

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        print(f"  [TIMEOUT] {config.name}")
    except Exception as e:
        result["status"] = "error"
        result["stderr"] = str(e)
        print(f"  [ERROR] {e}")

    return result


def run_all_sequential(experiments: list[ExperimentConfig], dry_run: bool = False) -> list[dict]:
    """Run experiments sequentially."""
    results = []
    total = len(experiments)

    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{total}] {exp.name}")
        result = run_experiment(exp, dry_run=dry_run)
        results.append(result)

    return results


def run_all_parallel(
    experiments: list[ExperimentConfig], max_workers: int, dry_run: bool = False
) -> list[dict]:
    """Run experiments in parallel."""
    results = []
    total = len(experiments)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_experiment, exp, dry_run): exp for exp in experiments}

        for i, future in enumerate(as_completed(futures), 1):
            exp = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = result["status"]
                print(f"[{i}/{total}] {exp.name} - {status}")
            except Exception as e:
                print(f"[{i}/{total}] {exp.name} - ERROR: {e}")
                results.append(
                    {
                        "name": exp.name,
                        "status": "error",
                        "stderr": str(e),
                    }
                )

    return results


def save_results(results: list[dict], output_path: Path) -> None:
    """Save results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "total": len(results),
                "success": sum(1 for r in results if r["status"] == "success"),
                "failed": sum(1 for r in results if r["status"] == "failed"),
                "experiments": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Part 1 Foundational experiments (110 configs)"
    )
    parser.add_argument(
        "--set",
        choices=["easy", "self", "mixed"],
        help="Run only experiments from this set",
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        help="Run only experiments for this strategy",
    )
    parser.add_argument(
        "--env",
        choices=ENVS,
        help="Run only experiments for this environment",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/p1_batch_results.json"),
        help="Output file for results",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all experiments without running",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()

    # Generate and filter experiments
    all_experiments = generate_experiments()
    experiments = filter_experiments(
        all_experiments,
        set_type=args.set,
        strategy=args.strategy,
        env=args.env,
    )

    print("Part 1 Foundational Experiments")
    print("================================")
    print(f"Total configs: {len(all_experiments)}")
    print("  Easy-play:  50 (5 strategies × 10 envs)")
    print("  Self-play:  50 (5 strategies × 10 envs)")
    print("  Mixed-play: 10 (1 setup × 10 envs)")
    print(f"\nSelected: {len(experiments)} experiments")

    if args.list:
        print("\nExperiments to run:")
        for exp in experiments:
            print(f"  {exp.name}")
        return

    if not experiments:
        print("No experiments match the filter criteria.")
        return

    # Confirm before running
    if not args.dry_run:
        print(f"\nRounds per experiment: {NUM_ROUNDS}")
        print(f"Event logging: {LOG_EVENTS}")
        print(f"Parallelism: {args.parallel}")
        if not args.yes:
            response = input("\nProceed? [y/N] ")
            if response.lower() != "y":
                print("Aborted.")
                return

    # Run experiments
    if args.parallel > 1:
        results = run_all_parallel(experiments, args.parallel, dry_run=args.dry_run)
    else:
        results = run_all_sequential(experiments, dry_run=args.dry_run)

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    timeout = sum(1 for r in results if r["status"] == "timeout")
    print(f"Success: {success}/{len(results)}")
    print(f"Failed:  {failed}/{len(results)}")
    print(f"Timeout: {timeout}/{len(results)}")

    # Save results
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_results(results, args.output)


if __name__ == "__main__":
    main()
