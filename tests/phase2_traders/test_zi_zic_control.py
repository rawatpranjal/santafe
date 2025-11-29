"""
Gode & Sunder (1993) Control Experiment: ZI vs ZIC

This is THE critical control experiment proving "Institution > Intelligence".
It validates that budget constraints (not learning) create market efficiency.

Multi-Token Setup (G&S standard market):
- 5 Buyers with 4 tokens each (decreasing marginal valuations)
- 5 Sellers with 4 tokens each (increasing marginal costs)
- Creates supply/demand curves with clear competitive equilibrium
- Max Surplus: ~330 (depends on exact token values)

Expected Results:
- ZI efficiency: 60-70% (random bids WITHOUT constraints = poor efficiency)
  -> ZI completes ALL trades including unprofitable buyer-seller pairs
- ZIC efficiency: 97.2-100% (random bids WITH constraints = near-optimal)
  -> ZIC only completes profitable trades (buyer_value > seller_cost)
- Difference: +28-40pp (proves budget constraint is THE critical feature)

Key Insight: ZI's inefficiency comes from completing TOO MANY trades
(including negative-surplus pairs), not from missing trades. ZIC's budget
constraint prevents those unprofitable trades.

Reference: Gode & Sunder (1993) "Allocative Efficiency of Markets with
Zero-Intelligence Traders: Market as a Partial Substitute for Individual Rationality"
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple, Any

from engine.market import Market
from engine.efficiency import (
    extract_trades_from_orderbook,
    calculate_actual_surplus,
    calculate_max_surplus,
    calculate_allocative_efficiency,
    calculate_em_inefficiency,
)
from traders.legacy.zi import ZI
from traders.legacy.zic import ZIC


# =============================================================================
# FIXTURES AND HELPERS
# =============================================================================

# Multi-token setup (4 tokens per trader, like G&S 1993)
# Standard decreasing/increasing schedules that create proper supply/demand curves
BUYER_VALUATIONS = [
    [100, 90, 80, 70],  # Buyer 1: High-value buyer
    [95, 85, 75, 65],   # Buyer 2
    [90, 80, 70, 60],   # Buyer 3
    [85, 75, 65, 55],   # Buyer 4
    [80, 70, 60, 50],   # Buyer 5: Low-value buyer
]

SELLER_COSTS = [
    [40, 50, 60, 70],   # Seller 1: Low-cost seller
    [45, 55, 65, 75],   # Seller 2
    [50, 60, 70, 80],   # Seller 3
    [55, 65, 75, 85],   # Seller 4
    [60, 70, 80, 90],   # Seller 5: High-cost seller
]

# Total tokens: 5 buyers * 4 tokens = 20 buyer tokens
#               5 sellers * 4 tokens = 20 seller tokens


def run_market_experiment(
    agent_type: str,
    num_rounds: int = 30,
    seed: int = 42,
    num_steps: int = 100,
) -> Dict[str, Any]:
    """
    Run market experiment with multi-token symmetric setup.

    Args:
        agent_type: "ZI" or "ZIC"
        num_rounds: Number of rounds to run for statistical significance
        seed: Random seed for reproducibility
        num_steps: Number of time steps per round (100 allows near-completion)

    Returns:
        Dict with efficiency metrics
    """
    max_surplus = calculate_max_surplus(BUYER_VALUATIONS, SELLER_COSTS)
    num_tokens = 4  # Tokens per trader

    # Select agent class
    AgentClass = ZI if agent_type == "ZI" else ZIC

    efficiencies: List[float] = []
    em_inefficiencies: List[float] = []
    trade_counts: List[int] = []

    for r in range(1, num_rounds + 1):
        # Create agents with multi-token setup
        buyers = []
        for i, vals in enumerate(BUYER_VALUATIONS):
            agent = AgentClass(
                player_id=i + 1,
                is_buyer=True,
                num_tokens=num_tokens,
                valuations=vals.copy(),
                price_min=0,
                price_max=150,  # Set above max buyer value, below 2x
                seed=seed + r * 10 + i,
            )
            buyers.append(agent)

        sellers = []
        for i, costs in enumerate(SELLER_COSTS):
            agent = AgentClass(
                player_id=len(buyers) + i + 1,
                is_buyer=False,
                num_tokens=num_tokens,
                valuations=costs.copy(),
                price_min=0,
                price_max=150,
                seed=seed + r * 10 + len(buyers) + i,
            )
            sellers.append(agent)

        # Create market
        market = Market(
            num_buyers=len(buyers),
            num_sellers=len(sellers),
            price_min=0,
            price_max=150,
            num_times=num_steps,
            buyers=buyers,
            sellers=sellers,
            seed=seed + r * 100,
        )

        # Run market
        for _ in range(num_steps):
            market.run_time_step()

        # Extract trades
        trades = extract_trades_from_orderbook(market.orderbook, num_steps)

        # Build valuation dicts
        buyer_valuations_dict = {
            i + 1: BUYER_VALUATIONS[i] for i in range(len(buyers))
        }
        seller_costs_dict = {i + 1: SELLER_COSTS[i] for i in range(len(sellers))}

        # Calculate metrics
        actual_surplus = calculate_actual_surplus(
            trades, buyer_valuations_dict, seller_costs_dict
        )
        efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)
        em_ineff = calculate_em_inefficiency(
            trades, buyer_valuations_dict, seller_costs_dict
        )

        efficiencies.append(efficiency)
        em_inefficiencies.append(em_ineff)
        trade_counts.append(len(trades))

    return {
        "agent_type": agent_type,
        "mean_efficiency": float(np.mean(efficiencies)),
        "std_efficiency": float(np.std(efficiencies)),
        "min_efficiency": float(np.min(efficiencies)),
        "max_efficiency": float(np.max(efficiencies)),
        "mean_em_inefficiency": float(np.mean(em_inefficiencies)),
        "mean_trades": float(np.mean(trade_counts)),
        "efficiencies": efficiencies,
        "em_inefficiencies": em_inefficiencies,
    }


# =============================================================================
# TEST 1: ZI EFFICIENCY RANGE
# =============================================================================


@pytest.mark.slow
def test_zi_efficiency_range() -> None:
    """
    Test that ZI (unconstrained) achieves 60-70% efficiency.

    ZI Rule: Random bids WITHOUT budget constraints = poor efficiency.
    Expected: ~65% (Gode & Sunder 1993 Table 1)

    Key insight: ZI completes ALL trades (including unprofitable ones),
    resulting in lower efficiency despite higher trade volume.
    """
    results = run_market_experiment("ZI", num_rounds=30, seed=42)

    zi_eff = results["mean_efficiency"]

    # Print for debugging
    print(f"\nZI Efficiency: {zi_eff:.2f}% (expected: 60-70%)")
    print(f"Std: {results['std_efficiency']:.2f}%")
    print(f"Avg trades: {results['mean_trades']:.1f}")

    # Accept slightly wider range (58-72%) due to implementation variance
    assert 58 <= zi_eff <= 72, (
        f"ZI efficiency {zi_eff:.2f}% outside expected range [58%, 72%]. "
        f"Expected ~65% per Gode & Sunder (1993)."
    )


# =============================================================================
# TEST 2: ZIC EFFICIENCY RANGE
# =============================================================================


@pytest.mark.slow
def test_zic_efficiency_range() -> None:
    """
    Test that ZIC (constrained) achieves 97.2-100% efficiency.

    ZIC Rule: Random bids WITH budget constraints = near-optimal efficiency.
    Expected: 98.7% (Gode & Sunder 1993 Table 1)

    Key insight: ZIC only completes profitable trades where buyer_value > seller_cost.
    """
    results = run_market_experiment("ZIC", num_rounds=30, seed=42)

    zic_eff = results["mean_efficiency"]

    # Print for debugging
    print(f"\nZIC Efficiency: {zic_eff:.2f}% (expected: 97.2-100%)")
    print(f"Std: {results['std_efficiency']:.2f}%")
    print(f"Avg trades: {results['mean_trades']:.1f}")

    assert 97.0 <= zic_eff <= 100.0, (
        f"ZIC efficiency {zic_eff:.2f}% outside expected range [97.0%, 100%]. "
        f"Expected 98.7% per Gode & Sunder (1993)."
    )


# =============================================================================
# TEST 3: ZI VS ZIC EFFICIENCY DIFFERENCE
# =============================================================================


@pytest.mark.slow
def test_zi_zic_efficiency_difference() -> None:
    """
    Test that ZIC - ZI efficiency difference is +25-45%.

    This is THE critical control experiment proving "Institution > Intelligence".
    The budget constraint alone (without learning) creates ~30% efficiency gain.
    """
    zi_results = run_market_experiment("ZI", num_rounds=30, seed=42)
    zic_results = run_market_experiment("ZIC", num_rounds=30, seed=42)

    zi_eff = zi_results["mean_efficiency"]
    zic_eff = zic_results["mean_efficiency"]
    diff = zic_eff - zi_eff

    # Print for debugging
    print(f"\nZI Efficiency: {zi_eff:.2f}%")
    print(f"ZIC Efficiency: {zic_eff:.2f}%")
    print(f"Difference: {diff:.2f} percentage points (expected: 25-45pp)")
    print(f"ZI trades: {zi_results['mean_trades']:.1f}")
    print(f"ZIC trades: {zic_results['mean_trades']:.1f}")

    # Use wider range (25-45) to account for variance
    assert 25 <= diff <= 45, (
        f"ZIC-ZI difference {diff:.2f}pp outside expected range [25pp, 45pp]. "
        f"Expected +28-40pp per Gode & Sunder (1993). "
        f"ZI={zi_eff:.2f}%, ZIC={zic_eff:.2f}%"
    )


# =============================================================================
# TEST 4: ZI HAS HIGHER EM-INEFFICIENCY
# =============================================================================


@pytest.mark.slow
def test_zi_em_inefficiency_higher() -> None:
    """
    Test that ZI has significantly higher EM-Inefficiency than ZIC.

    EM-Inefficiency = trades where buyer_value < seller_cost (bad trades).
    ZI should have much more unprofitable trades than ZIC.
    """
    zi_results = run_market_experiment("ZI", num_rounds=30, seed=42)
    zic_results = run_market_experiment("ZIC", num_rounds=30, seed=42)

    zi_em = zi_results["mean_em_inefficiency"]
    zic_em = zic_results["mean_em_inefficiency"]

    # Print for debugging
    print(f"\nZI EM-Inefficiency: {zi_em:.2f}")
    print(f"ZIC EM-Inefficiency: {zic_em:.2f}")

    # ZI should have significantly more unprofitable trades
    # ZIC's constraint prevents most bad trades
    # Note: ZIC should have ~0 EM-inefficiency, so we check ZI > 0
    assert zi_em > 0, (
        f"ZI should have positive EM-Inefficiency (unprofitable trades), "
        f"but got {zi_em:.2f}."
    )

    # If ZIC has any EM-inefficiency, ZI should have more
    if zic_em > 0:
        assert zi_em > zic_em, (
            f"ZI EM-Inefficiency ({zi_em:.2f}) should be higher than "
            f"ZIC ({zic_em:.2f})."
        )


# =============================================================================
# TEST 5: COMPREHENSIVE GODE & SUNDER VALIDATION
# =============================================================================


@pytest.mark.slow
def test_gode_sunder_control_experiment() -> None:
    """
    Comprehensive validation of the Gode & Sunder (1993) control experiment.

    This test validates ALL criteria at once:
    1. ZI efficiency: 58-72%
    2. ZIC efficiency: 97-100%
    3. Difference: 25-45pp
    4. ZI EM-Inefficiency > 0 (has unprofitable trades)

    If all pass, we've proven: "Institution > Intelligence"
    """
    zi_results = run_market_experiment("ZI", num_rounds=30, seed=42)
    zic_results = run_market_experiment("ZIC", num_rounds=30, seed=42)

    zi_eff = zi_results["mean_efficiency"]
    zic_eff = zic_results["mean_efficiency"]
    diff = zic_eff - zi_eff
    zi_em = zi_results["mean_em_inefficiency"]
    zic_em = zic_results["mean_em_inefficiency"]

    # Print comparison table
    print("\n" + "=" * 70)
    print("GODE & SUNDER (1993) CONTROL EXPERIMENT VALIDATION")
    print("=" * 70)
    print(f"{'Metric':<30} {'ZI':>15} {'ZIC':>15}")
    print("-" * 70)
    print(f"{'Mean Efficiency (%)':<30} {zi_eff:>15.2f} {zic_eff:>15.2f}")
    print(f"{'Std Efficiency (%)':<30} {zi_results['std_efficiency']:>15.2f} {zic_results['std_efficiency']:>15.2f}")
    print(f"{'Mean Trades':<30} {zi_results['mean_trades']:>15.1f} {zic_results['mean_trades']:>15.1f}")
    print(f"{'Mean EM-Inefficiency':<30} {zi_em:>15.2f} {zic_em:>15.2f}")
    print(f"{'Efficiency Difference':<30} {diff:>15.2f} pp")
    print("=" * 70)

    # Collect validation results
    validations: List[Tuple[str, bool, str]] = []

    # Check 1: ZI efficiency
    zi_valid = 58 <= zi_eff <= 72
    validations.append(("ZI efficiency 58-72%", zi_valid, f"{zi_eff:.2f}%"))

    # Check 2: ZIC efficiency
    zic_valid = 97.0 <= zic_eff <= 100.0
    validations.append(("ZIC efficiency 97-100%", zic_valid, f"{zic_eff:.2f}%"))

    # Check 3: Difference
    diff_valid = 25 <= diff <= 45
    validations.append(("Difference 25-45pp", diff_valid, f"{diff:.2f}pp"))

    # Check 4: EM-Inefficiency
    em_valid = zi_em > 0
    validations.append(
        ("ZI has unprofitable trades", em_valid, f"{zi_em:.2f}")
    )

    # Print results
    print("\nValidation Results:")
    for name, passed, value in validations:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {value}")

    passed_count = sum(v[1] for v in validations)
    total_count = len(validations)
    print(f"\nOverall: {passed_count}/{total_count} validations passed")

    # Assert all validations pass
    assert all(v[1] for v in validations), (
        f"Gode & Sunder (1993) control experiment failed. "
        f"{passed_count}/{total_count} validations passed. "
        f"Failed: {[v[0] for v in validations if not v[1]]}"
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
