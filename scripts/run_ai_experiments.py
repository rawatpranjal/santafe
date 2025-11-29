#!/usr/bin/env python3
"""
AI Extension Experiments Runner.

Runs the Chen-style (1 AI vs 7 clones) and Rust-style (heterogeneous stew)
experiments with PPO and LLM agents.

Usage:
    # List available experiments
    python scripts/run_ai_experiments.py --list

    # Run a single Chen AI experiment
    python scripts/run_ai_experiments.py --experiment chen_ai/ppo_vs_kaplan

    # Run all Chen PPO experiments
    python scripts/run_ai_experiments.py --suite chen_ppo

    # Run all Chen LLM experiments
    python scripts/run_ai_experiments.py --suite chen_llm

    # Run all Rust Stew experiments
    python scripts/run_ai_experiments.py --suite rust_stew

Prerequisites:
    - PPO experiments require a trained model at checkpoints/ppo_model.zip
    - LLM experiments require OPENAI_API_KEY environment variable
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any


# Define experiment suites
CHEN_PPO_EXPERIMENTS = [
    "chen_ai/ppo_vs_kaplan",
    "chen_ai/ppo_vs_zic",
    "chen_ai/ppo_vs_zip",
    "chen_ai/ppo_vs_gd",
    "chen_ai/ppo_vs_mixed",
]

CHEN_LLM_EXPERIMENTS = [
    "chen_ai/llm_vs_kaplan",
    "chen_ai/llm_vs_zic",
    "chen_ai/llm_vs_zip",
    "chen_ai/llm_vs_gd",
    "chen_ai/llm_vs_mixed",
]

RUST_STEW_EXPERIMENTS = [
    "rust_stew/9v9_no_ai",
    "rust_stew/9v9_with_ppo",
    "rust_stew/9v9_with_llm",
    "rust_stew/18v18_no_ai",
    "rust_stew/18v18_with_ppo",
    "rust_stew/18v18_with_llm",
]

ALL_EXPERIMENTS = {
    "chen_ppo": CHEN_PPO_EXPERIMENTS,
    "chen_llm": CHEN_LLM_EXPERIMENTS,
    "chen_all": CHEN_PPO_EXPERIMENTS + CHEN_LLM_EXPERIMENTS,
    "rust_stew": RUST_STEW_EXPERIMENTS,
    "all": CHEN_PPO_EXPERIMENTS + CHEN_LLM_EXPERIMENTS + RUST_STEW_EXPERIMENTS,
}


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    experiment: str
    success: bool
    elapsed_time: float
    output_dir: str
    error: Optional[str] = None


def check_prerequisites(experiment: str) -> List[str]:
    """
    Check prerequisites for an experiment.

    Returns:
        List of warning messages (empty if all prerequisites met)
    """
    warnings = []

    # Check PPO model for PPO experiments
    if "ppo" in experiment.lower():
        model_path = Path("checkpoints/ppo_model.zip")
        if not model_path.exists():
            warnings.append(
                f"PPO model not found at {model_path}. "
                "Train a model first using scripts/train_learning_curve.py"
            )

    # Check OpenAI API key for LLM experiments
    if "llm" in experiment.lower() or "gpt" in experiment.lower():
        if not os.environ.get("OPENAI_API_KEY"):
            warnings.append(
                "OPENAI_API_KEY environment variable not set. "
                "LLM experiments will fail."
            )

    return warnings


def run_single_experiment(
    experiment: str,
    dry_run: bool = False,
    verbose: bool = True,
) -> ExperimentResult:
    """
    Run a single experiment using Hydra.

    Args:
        experiment: Experiment name (e.g., "chen_ai/ppo_vs_kaplan")
        dry_run: If True, only print command without executing
        verbose: Print progress

    Returns:
        ExperimentResult with status
    """
    start_time = time.time()

    # Build command
    cmd = [
        "uv", "run", "python", "scripts/run_experiment.py",
        f"experiment={experiment}"
    ]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Running: {experiment}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'=' * 60}")

    if dry_run:
        return ExperimentResult(
            experiment=experiment,
            success=True,
            elapsed_time=0.0,
            output_dir=f"results/{experiment.replace('/', '_')}",
        )

    # Check prerequisites
    warnings = check_prerequisites(experiment)
    if warnings:
        for w in warnings:
            print(f"WARNING: {w}")

    # Run the experiment
    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=True,
        )

        elapsed = time.time() - start_time

        return ExperimentResult(
            experiment=experiment,
            success=True,
            elapsed_time=elapsed,
            output_dir=f"results/{experiment.replace('/', '_')}",
        )

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time

        return ExperimentResult(
            experiment=experiment,
            success=False,
            elapsed_time=elapsed,
            output_dir=f"results/{experiment.replace('/', '_')}",
            error=str(e.stderr) if e.stderr else str(e),
        )


def run_suite(
    suite_name: str,
    dry_run: bool = False,
    verbose: bool = True,
) -> List[ExperimentResult]:
    """
    Run all experiments in a suite.

    Args:
        suite_name: Suite name (chen_ppo, chen_llm, rust_stew, all)
        dry_run: If True, only print commands
        verbose: Print progress

    Returns:
        List of ExperimentResults
    """
    if suite_name not in ALL_EXPERIMENTS:
        print(f"Unknown suite: {suite_name}")
        print(f"Available suites: {list(ALL_EXPERIMENTS.keys())}")
        return []

    experiments = ALL_EXPERIMENTS[suite_name]

    if verbose:
        print(f"\nRunning suite: {suite_name}")
        print(f"Experiments: {len(experiments)}")
        for exp in experiments:
            print(f"  - {exp}")

    results = []
    for i, experiment in enumerate(experiments, 1):
        if verbose:
            print(f"\n[{i}/{len(experiments)}] ", end="")

        result = run_single_experiment(experiment, dry_run, verbose)
        results.append(result)

        if not result.success:
            print(f"FAILED: {result.error}")

    return results


def list_experiments():
    """Print all available experiments and suites."""
    print("\n=== Available Experiment Suites ===\n")

    for suite_name, experiments in ALL_EXPERIMENTS.items():
        print(f"{suite_name}:")
        for exp in experiments:
            print(f"  - {exp}")
        print()

    print("=== Chen AI Experiments (1 AI vs 7 clones) ===")
    print("Purpose: Test if AI can beat specific opponent types")
    print("Market: 4v4 (8 traders), 7000 trading days")
    print("AI Position: Buyer 1 (first buyer slot)")
    print()

    print("=== Rust Stew Experiments (Heterogeneous Market) ===")
    print("Purpose: Test AI in diverse competitive market")
    print("Market: 9v9 or 18v18 (larger scale)")
    print("AI Position: One among many diverse strategies")
    print()


def save_results(results: List[ExperimentResult], output_path: str):
    """Save experiment results to JSON."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_experiments": len(results),
        "successful": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "total_time": sum(r.elapsed_time for r in results),
        "results": [
            {
                "experiment": r.experiment,
                "success": r.success,
                "elapsed_time": r.elapsed_time,
                "output_dir": r.output_dir,
                "error": r.error,
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run AI extension experiments (Chen-style and Rust-style)"
    )

    parser.add_argument(
        "--experiment", type=str,
        help="Single experiment to run (e.g., chen_ai/ppo_vs_kaplan)"
    )
    parser.add_argument(
        "--suite", type=str,
        choices=list(ALL_EXPERIMENTS.keys()),
        help="Run all experiments in a suite"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available experiments"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON for suite results"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    if not args.experiment and not args.suite:
        parser.print_help()
        print("\n\nExamples:")
        print("  # List all experiments")
        print("  python scripts/run_ai_experiments.py --list")
        print()
        print("  # Run single experiment")
        print("  python scripts/run_ai_experiments.py --experiment chen_ai/ppo_vs_kaplan")
        print()
        print("  # Run all Chen PPO experiments")
        print("  python scripts/run_ai_experiments.py --suite chen_ppo")
        return

    verbose = not args.quiet

    if args.experiment:
        # Run single experiment
        result = run_single_experiment(args.experiment, args.dry_run, verbose)

        if result.success:
            print(f"\n✓ Experiment completed in {result.elapsed_time:.1f}s")
            print(f"  Output: {result.output_dir}")
        else:
            print(f"\n✗ Experiment failed: {result.error}")
            sys.exit(1)

    elif args.suite:
        # Run suite
        results = run_suite(args.suite, args.dry_run, verbose)

        # Print summary
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        total_time = sum(r.elapsed_time for r in results)

        print(f"\n{'=' * 60}")
        print(f"SUITE COMPLETE: {args.suite}")
        print(f"{'=' * 60}")
        print(f"Successful: {successful}/{len(results)}")
        print(f"Failed: {failed}/{len(results)}")
        print(f"Total time: {total_time:.1f}s")

        if failed > 0:
            print("\nFailed experiments:")
            for r in results:
                if not r.success:
                    print(f"  - {r.experiment}: {r.error}")

        # Save results if requested
        if args.output:
            save_results(results, args.output)
        elif not args.dry_run:
            # Default output path
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"results/ai_suite_{args.suite}_{timestamp}.json"
            save_results(results, output_path)


if __name__ == "__main__":
    main()
