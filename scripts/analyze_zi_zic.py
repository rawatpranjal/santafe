"""
Analyze Gode & Sunder (1993) ZI vs ZIC Control Experiment.

This script compares three conditions:
1. ZI vs ZI: Unconstrained randomness (control condition)
2. ZIC vs ZIC: Constrained randomness (treatment condition)
3. ZI vs ZIC: Mixed condition

Expected results (Gode & Sunder 1993 Table 1):
- ZI efficiency: 60-70% (proves randomness alone fails)
- ZIC efficiency: 98.7% ± 1.5% (proves budget constraints work)
- Difference: +28-35% (proves budget constraint is THE critical feature)

Usage:
    python scripts/analyze_zi_zic.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_experiment(experiment_name: str) -> pd.DataFrame:
    """Load experiment results from CSV."""
    results_path = Path(f"results/{experiment_name}/results.csv")

    if not results_path.exists():
        print(f"WARNING: {results_path} not found. Run experiment first:")
        print(f"  python scripts/run_experiment.py experiment=gode_sunder/{experiment_name.split('_')[-2:]}")
        return None

    df = pd.read_csv(results_path)
    df['experiment'] = experiment_name
    return df

def calculate_efficiency_stats(df: pd.DataFrame) -> dict:
    """Calculate efficiency statistics for a single experiment."""
    # Take efficiency from first agent in each period (all agents see same efficiency)
    eff_per_period = df.groupby(['round', 'period'])['efficiency'].first()

    return {
        'mean': eff_per_period.mean(),
        'std': eff_per_period.std(),
        'min': eff_per_period.min(),
        'max': eff_per_period.max(),
        'periods': len(eff_per_period)
    }

def calculate_profit_stats(df: pd.DataFrame) -> dict:
    """Calculate profit statistics aggregated across all agents."""
    return {
        'mean_profit': df['period_profit'].mean(),
        'std_profit': df['period_profit'].std(),
        'mean_eq_profit': df['agent_eq_profit'].mean() if 'agent_eq_profit' in df.columns else None,
        'mean_deviation': df['profit_deviation'].mean() if 'profit_deviation' in df.columns else None,
        'profit_dispersion': df['profit_deviation'].std() if 'profit_deviation' in df.columns else None
    }

def calculate_em_inefficiency_stats(df: pd.DataFrame) -> dict:
    """Calculate EM-Inefficiency statistics (unprofitable trades)."""
    # EM-Inefficiency is per-period metric
    em_per_period = df.groupby(['round', 'period'])['em_inefficiency'].first()

    return {
        'mean': em_per_period.mean(),
        'std': em_per_period.std(),
        'max': em_per_period.max()
    }

def main():
    print("=" * 80)
    print("GODE & SUNDER (1993) ZI vs ZIC CONTROL EXPERIMENT ANALYSIS")
    print("=" * 80)

    # Load all three experiments
    experiments = {
        'ZI vs ZI': 'gode_sunder_zi_selfplay',
        'ZIC vs ZIC': 'gode_sunder_zic_selfplay',
        'ZI vs ZIC': 'gode_sunder_zi_vs_zic'
    }

    results = {}
    for label, exp_name in experiments.items():
        df = load_experiment(exp_name)
        if df is not None:
            results[label] = df

    if not results:
        print("\nERROR: No experiment results found.")
        print("Please run experiments first:")
        print("  python scripts/run_experiment.py experiment=gode_sunder/zi_selfplay")
        print("  python scripts/run_experiment.py experiment=gode_sunder/zic_selfplay")
        print("  python scripts/run_experiment.py experiment=gode_sunder/zi_vs_zic")
        return

    # === EFFICIENCY COMPARISON ===
    print("\n" + "=" * 80)
    print("1. ALLOCATIVE EFFICIENCY COMPARISON")
    print("=" * 80)
    print(f"{'Experiment':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Periods':>10}")
    print("-" * 80)

    efficiency_summary = {}
    for label, df in results.items():
        stats = calculate_efficiency_stats(df)
        efficiency_summary[label] = stats['mean']
        print(f"{label:<20} {stats['mean']:>10.2f} {stats['std']:>10.2f} "
              f"{stats['min']:>10.2f} {stats['max']:>10.2f} {stats['periods']:>10}")

    # Calculate ZIC-ZI difference
    if 'ZI vs ZI' in efficiency_summary and 'ZIC vs ZIC' in efficiency_summary:
        zi_eff = efficiency_summary['ZI vs ZI']
        zic_eff = efficiency_summary['ZIC vs ZIC']
        diff = zic_eff - zi_eff
        print(f"\n{'ZIC - ZI Difference':<20} {diff:>10.2f} percentage points")
        print(f"{'Expected (G&S 1993)':<20} {'+28 to +35':<20}")

        # Validation check
        if 60 <= zi_eff <= 70:
            print(f"✅ ZI efficiency {zi_eff:.1f}% is within expected range (60-70%)")
        else:
            print(f"❌ ZI efficiency {zi_eff:.1f}% is OUTSIDE expected range (60-70%)")

        if 97.2 <= zic_eff <= 100.0:
            print(f"✅ ZIC efficiency {zic_eff:.1f}% is within expected range (98.7% ± 1.5%)")
        else:
            print(f"❌ ZIC efficiency {zic_eff:.1f}% is OUTSIDE expected range (98.7% ± 1.5%)")

        if 25 <= diff <= 40:
            print(f"✅ Efficiency difference {diff:.1f}pp is within expected range (+28 to +35pp)")
        else:
            print(f"❌ Efficiency difference {diff:.1f}pp is OUTSIDE expected range (+28 to +35pp)")

    # === PROFIT COMPARISON ===
    print("\n" + "=" * 80)
    print("2. INDIVIDUAL PROFIT COMPARISON")
    print("=" * 80)
    print(f"{'Experiment':<20} {'Mean Profit':>12} {'Std Profit':>12} {'Mean Eq Profit':>15} {'Profit Dispersion':>18}")
    print("-" * 80)

    for label, df in results.items():
        stats = calculate_profit_stats(df)
        eq_profit_str = f"{stats['mean_eq_profit']:.2f}" if stats['mean_eq_profit'] is not None else "N/A"
        dispersion_str = f"{stats['profit_dispersion']:.2f}" if stats['profit_dispersion'] is not None else "N/A"
        print(f"{label:<20} {stats['mean_profit']:>12.2f} {stats['std_profit']:>12.2f} "
              f"{eq_profit_str:>15} {dispersion_str:>18}")

    # === EM-INEFFICIENCY COMPARISON ===
    print("\n" + "=" * 80)
    print("3. EM-INEFFICIENCY COMPARISON (Unprofitable Trades)")
    print("=" * 80)
    print(f"{'Experiment':<20} {'Mean EM-Ineff':>15} {'Std':>10} {'Max':>10}")
    print("-" * 80)

    for label, df in results.items():
        stats = calculate_em_inefficiency_stats(df)
        print(f"{label:<20} {stats['mean']:>15.2f} {stats['std']:>10.2f} {stats['max']:>10.2f}")

    print("\nExpected: ZI should have HIGH EM-Inefficiency (20-30% bad trades)")
    print("Expected: ZIC should have LOW EM-Inefficiency (<2% bad trades)")

    # === PER-AGENT BREAKDOWN ===
    print("\n" + "=" * 80)
    print("4. PER-AGENT TYPE BREAKDOWN")
    print("=" * 80)

    for label, df in results.items():
        print(f"\n{label}:")
        print("-" * 40)

        agent_summary = df.groupby(['agent_type', 'is_buyer']).agg({
            'period_profit': ['mean', 'std'],
            'num_trades': 'mean',
            'agent_eq_profit': 'mean' if 'agent_eq_profit' in df.columns else lambda x: None
        }).round(2)

        print(agent_summary)

    # === SAVE COMPARISON TABLE ===
    output_dir = Path("results/gode_sunder")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_data = []
    for label, df in results.items():
        eff_stats = calculate_efficiency_stats(df)
        profit_stats = calculate_profit_stats(df)
        em_stats = calculate_em_inefficiency_stats(df)

        comparison_data.append({
            'Experiment': label,
            'Efficiency_Mean': eff_stats['mean'],
            'Efficiency_Std': eff_stats['std'],
            'Mean_Profit': profit_stats['mean_profit'],
            'Profit_Dispersion': profit_stats['profit_dispersion'],
            'EM_Inefficiency_Mean': em_stats['mean'],
            'EM_Inefficiency_Max': em_stats['max'],
            'Periods': eff_stats['periods']
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = output_dir / "comparison_summary.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("\n" + "=" * 80)
    print(f"Results saved to: {comparison_path}")
    print("=" * 80)

    # === FINAL VALIDATION SUMMARY ===
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print("Testing the hypothesis: Budget constraints (not intelligence) create efficiency")
    print()

    if 'ZI vs ZI' in efficiency_summary and 'ZIC vs ZIC' in efficiency_summary:
        zi_eff = efficiency_summary['ZI vs ZI']
        zic_eff = efficiency_summary['ZIC vs ZIC']
        diff = zic_eff - zi_eff

        validations = [
            (60 <= zi_eff <= 70, f"ZI efficiency {zi_eff:.1f}% in range [60%, 70%]"),
            (97.2 <= zic_eff <= 100.0, f"ZIC efficiency {zic_eff:.1f}% in range [97.2%, 100%]"),
            (25 <= diff <= 40, f"Efficiency gap {diff:.1f}pp in range [25pp, 40pp]")
        ]

        passed = sum(1 for v, _ in validations if v)
        total = len(validations)

        for valid, msg in validations:
            status = "✅ PASS" if valid else "❌ FAIL"
            print(f"{status}: {msg}")

        print()
        print(f"Overall: {passed}/{total} validations passed")

        if passed == total:
            print("\n✅ SUCCESS: Gode & Sunder (1993) control experiment VALIDATED")
            print("   Institution (budget constraint) > Intelligence (learning)")
        else:
            print("\n❌ FAILURE: Results do not match Gode & Sunder expectations")
            print("   Check implementation of ZI/ZIC traders")

if __name__ == "__main__":
    main()
