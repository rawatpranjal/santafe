"""
Test ZI vs ZIC with FIXED symmetric tokens.

This is the proper Gode & Sunder (1993) control experiment.
Uses fixed symmetric supply/demand curves to isolate the effect of budget constraints.

Fixed Tokens (G&S symmetric market):
- Buyers: [100, 90, 80, 70, 60] (high to low)
- Sellers: [40, 50, 60, 70, 80] (low to high)
- Competitive Equilibrium: Price ~60, Quantity = 3
- Max Surplus: (100-40) + (90-50) + (80-60) = 60 + 40 + 20 = 120

Expected Results:
- ZI efficiency: 60-70% (random bids miss profitable trades)
- ZIC efficiency: 98.7% (budget constraints enable price discovery)
- Difference: +28-35% (proves budget constraint is THE critical feature)
"""

import numpy as np
from engine.market import Market
from engine.agent_factory import create_agent
from engine.efficiency import (
    extract_trades_from_orderbook,
    calculate_actual_surplus,
    calculate_max_surplus,
    calculate_allocative_efficiency,
    calculate_em_inefficiency,
)

def run_fixed_token_experiment(agent_type: str, num_rounds: int = 50, seed: int = 42) -> dict:
    """
    Run experiment with fixed symmetric tokens.

    Args:
        agent_type: "ZI" or "ZIC"
        num_rounds: Number of rounds to run
        seed: Random seed for agent behavior

    Returns:
        Dict with efficiency metrics
    """
    # Fixed symmetric tokens (G&S 1993)
    buyer_valuations = [[100], [90], [80], [70], [60]]
    seller_costs = [[40], [50], [60], [70], [80]]

    # Calculate max surplus (should be 120)
    max_surplus = calculate_max_surplus(buyer_valuations, seller_costs)

    print(f"\n{'='*80}")
    print(f"Testing {agent_type} with Fixed Symmetric Tokens")
    print(f"{'='*80}")
    print(f"Buyer valuations: {[v[0] for v in buyer_valuations]}")
    print(f"Seller costs: {[c[0] for c in seller_costs]}")
    print(f"Max surplus: {max_surplus}")
    print(f"Competitive equilibrium: Price ~60, Quantity = 3")
    print(f"{'='*80}\n")

    efficiencies = []
    em_inefficiencies = []
    actual_surpluses = []
    num_trades_list = []

    for r in range(1, num_rounds + 1):
        # Create agents with FIXED tokens
        buyers = []
        for i, vals in enumerate(buyer_valuations):
            agent = create_agent(
                agent_type,
                player_id=i+1,
                is_buyer=True,
                num_tokens=1,
                valuations=vals.copy(),
                price_min=0,
                price_max=200,
                seed=seed + r*10 + i
            )
            buyers.append(agent)

        sellers = []
        for i, costs in enumerate(seller_costs):
            agent = create_agent(
                agent_type,
                player_id=len(buyers)+i+1,
                is_buyer=False,
                num_tokens=1,
                valuations=costs.copy(),
                price_min=0,
                price_max=200,
                seed=seed + r*10 + len(buyers) + i
            )
            sellers.append(agent)

        # Create market
        market = Market(
            num_buyers=len(buyers),
            num_sellers=len(sellers),
            price_min=0,
            price_max=200,
            num_times=50,
            buyers=buyers,
            sellers=sellers,
            seed=seed + r*100
        )

        # Run market
        for _ in range(50):
            market.run_time_step()

        # Extract trades
        trades = extract_trades_from_orderbook(market.orderbook, 50)

        # Build valuation dicts for efficiency calculation
        buyer_valuations_dict = {i+1: buyer_valuations[i] for i in range(len(buyers))}
        seller_costs_dict = {i+1: seller_costs[i] for i in range(len(sellers))}

        # Calculate metrics
        actual_surplus = calculate_actual_surplus(trades, buyer_valuations_dict, seller_costs_dict)
        efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)
        em_ineff = calculate_em_inefficiency(trades, buyer_valuations_dict, seller_costs_dict)

        efficiencies.append(efficiency)
        em_inefficiencies.append(em_ineff)
        actual_surpluses.append(actual_surplus)
        num_trades_list.append(len(trades))

        if r <= 5 or r % 10 == 0:
            print(f"Round {r:2d}: Efficiency {efficiency:6.2f}%, "
                  f"Surplus {actual_surplus:3d}/{max_surplus}, "
                  f"Trades {len(trades)}, EM-Ineff {em_ineff:4d}")

    return {
        'agent_type': agent_type,
        'mean_efficiency': np.mean(efficiencies),
        'std_efficiency': np.std(efficiencies),
        'min_efficiency': np.min(efficiencies),
        'max_efficiency': np.max(efficiencies),
        'mean_em_inefficiency': np.mean(em_inefficiencies),
        'mean_trades': np.mean(num_trades_list),
        'max_surplus': max_surplus,
        'efficiencies': efficiencies,
    }

def main():
    print("\n" + "="*80)
    print("GODE & SUNDER (1993) CONTROL EXPERIMENT - FIXED SYMMETRIC TOKENS")
    print("="*80)

    # Run ZI experiment
    zi_results = run_fixed_token_experiment("ZI", num_rounds=50, seed=42)

    # Run ZIC experiment
    zic_results = run_fixed_token_experiment("ZIC", num_rounds=50, seed=42)

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON: ZI vs ZIC")
    print("="*80)
    print(f"{'Metric':<30} {'ZI':>15} {'ZIC':>15} {'Difference':>15}")
    print("-"*80)

    zi_eff = zi_results['mean_efficiency']
    zic_eff = zic_results['mean_efficiency']
    diff = zic_eff - zi_eff

    print(f"{'Mean Efficiency (%)':<30} {zi_eff:>15.2f} {zic_eff:>15.2f} {diff:>15.2f}")
    print(f"{'Std Efficiency (%)':<30} {zi_results['std_efficiency']:>15.2f} {zic_results['std_efficiency']:>15.2f} {'':>15}")
    print(f"{'Min Efficiency (%)':<30} {zi_results['min_efficiency']:>15.2f} {zic_results['min_efficiency']:>15.2f} {'':>15}")
    print(f"{'Max Efficiency (%)':<30} {zi_results['max_efficiency']:>15.2f} {zic_results['max_efficiency']:>15.2f} {'':>15}")
    print(f"{'Mean EM-Inefficiency':<30} {zi_results['mean_em_inefficiency']:>15.2f} {zic_results['mean_em_inefficiency']:>15.2f} {'':>15}")
    print(f"{'Mean Trades':<30} {zi_results['mean_trades']:>15.2f} {zic_results['mean_trades']:>15.2f} {'':>15}")

    print("\n" + "="*80)
    print("VALIDATION AGAINST GODE & SUNDER (1993)")
    print("="*80)

    validations = []

    # Check ZI efficiency
    if 60 <= zi_eff <= 70:
        print(f"✅ PASS: ZI efficiency {zi_eff:.1f}% is within expected range [60%, 70%]")
        validations.append(True)
    else:
        print(f"❌ FAIL: ZI efficiency {zi_eff:.1f}% is OUTSIDE expected range [60%, 70%]")
        validations.append(False)

    # Check ZIC efficiency
    if 97.2 <= zic_eff <= 100.0:
        print(f"✅ PASS: ZIC efficiency {zic_eff:.1f}% is within expected range [97.2%, 100%]")
        validations.append(True)
    else:
        print(f"❌ FAIL: ZIC efficiency {zic_eff:.1f}% is OUTSIDE expected range [97.2%, 100%]")
        validations.append(False)

    # Check difference
    if 25 <= diff <= 40:
        print(f"✅ PASS: Efficiency difference {diff:.1f}pp is within expected range [25pp, 40pp]")
        validations.append(True)
    else:
        print(f"❌ FAIL: Efficiency difference {diff:.1f}pp is OUTSIDE expected range [25pp, 40pp]")
        validations.append(False)

    # Check EM-Inefficiency
    zi_em = zi_results['mean_em_inefficiency']
    zic_em = zic_results['mean_em_inefficiency']
    if zi_em > zic_em * 2:  # ZI should have significantly more bad trades
        print(f"✅ PASS: ZI has higher EM-Inefficiency ({zi_em:.1f} vs {zic_em:.1f})")
        validations.append(True)
    else:
        print(f"❌ FAIL: ZI EM-Inefficiency ({zi_em:.1f}) not significantly higher than ZIC ({zic_em:.1f})")
        validations.append(False)

    passed = sum(validations)
    total = len(validations)

    print("\n" + "="*80)
    print(f"OVERALL: {passed}/{total} validations passed")
    print("="*80)

    if passed == total:
        print("\n✅ SUCCESS: Budget constraints create efficiency without intelligence")
        print("   Institution (budget constraint) > Intelligence (learning)")
        print("   The Gode & Sunder (1993) hypothesis is VALIDATED")
    else:
        print("\n⚠️  PARTIAL SUCCESS: Some validations failed")
        print("   Review ZI/ZIC implementation or market parameters")

    print()

if __name__ == "__main__":
    main()
