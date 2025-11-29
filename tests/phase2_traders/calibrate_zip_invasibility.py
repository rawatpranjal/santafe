#!/usr/bin/env python3
"""
ZIP Invasibility-Focused Hyperparameter Calibration.

Primary Goal: Optimize ZIP for "1 vs 7 ZIC" invasibility test
- Maximize per-agent profit ratio (target: >3.0x)
- Maintain reasonable self-play efficiency (floor: >70%)

Tests ~25-30 configurations focusing on profit extraction parameters.
Runtime: ~5-10 minutes
"""

import numpy as np
from engine.market import Market
from traders.legacy.zip import ZIP
from traders.legacy.zic import ZIC


def run_invasibility_test(zip_params, num_sessions=5, seed_offset=0):
    """
    Run 1 ZIP buyer vs 3 ZIC buyers + 4 ZIC sellers.

    Returns per-agent profit ratio and other metrics.
    """
    results = {
        'zip_profit': [],
        'zic_avg_profit': [],
        'profit_ratio': [],
        'efficiency': []
    }

    # Symmetric market: 4v4, 4 tokens each
    buyer_vals = [
        [307, 227, 162, 106],  # ZIP buyer (player 1)
        [297, 232, 182, 116],  # ZIC buyer
        [286, 224, 163, 99],   # ZIC buyer
        [298, 243, 196, 133]   # ZIC buyer
    ]
    seller_costs = [
        [107, 143, 196, 295],  # ZIC seller
        [109, 152, 207, 290],  # ZIC seller
        [107, 167, 227, 297],  # ZIC seller
        [105, 157, 209, 289]   # ZIC seller
    ]

    for session in range(num_sessions):
        # Create 1 ZIP buyer
        zip_buyer = ZIP(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=buyer_vals[0],
            num_times=200,
            price_min=0,
            price_max=400,
            seed=seed_offset * 100 + session,
            **zip_params
        )

        # Create 3 ZIC buyers
        zic_buyers = []
        for i in range(1, 4):
            agent = ZIC(
                player_id=i + 1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_vals[i],
                num_times=200,
                price_min=0,
                price_max=400,
                seed=seed_offset * 100 + session + i
            )
            zic_buyers.append(agent)

        # Create 4 ZIC sellers
        zic_sellers = []
        for i in range(4):
            agent = ZIC(
                player_id=i + 5,
                is_buyer=False,
                num_tokens=4,
                valuations=seller_costs[i],
                num_times=200,
                price_min=0,
                price_max=400,
                seed=seed_offset * 100 + session + i + 4
            )
            zic_sellers.append(agent)

        all_agents = [zip_buyer] + zic_buyers + zic_sellers

        # Run market
        for a in all_agents:
            a.start_period(1)

        market = Market(
            num_buyers=4,
            num_sellers=4,
            price_min=0,
            price_max=400,
            num_times=200,
            buyers=[zip_buyer] + zic_buyers,
            sellers=zic_sellers
        )

        while market.current_time < market.num_times:
            market.run_time_step()

        for a in all_agents:
            a.end_period()

        # Calculate metrics
        zip_profit = zip_buyer.period_profit
        zic_profits = [a.period_profit for a in zic_buyers + zic_sellers]
        zic_avg_profit = np.mean(zic_profits) if zic_profits else 0

        # Per-agent profit ratio
        profit_ratio = zip_profit / zic_avg_profit if zic_avg_profit > 0 else 0

        # Total efficiency
        all_vals = sorted([v for vals in buyer_vals for v in vals if v > 0], reverse=True)
        all_costs = sorted([c for costs in seller_costs for c in costs if c > 0])
        max_surplus = sum(v - c for v, c in zip(all_vals, all_costs) if v > c)
        total_profit = sum(a.period_profit for a in all_agents)
        efficiency = (total_profit / max_surplus * 100) if max_surplus > 0 else 0

        results['zip_profit'].append(zip_profit)
        results['zic_avg_profit'].append(zic_avg_profit)
        results['profit_ratio'].append(profit_ratio)
        results['efficiency'].append(efficiency)

    return {
        'mean_profit_ratio': np.mean(results['profit_ratio']),
        'std_profit_ratio': np.std(results['profit_ratio']),
        'mean_efficiency': np.mean(results['efficiency']),
        'mean_zip_profit': np.mean(results['zip_profit']),
        'mean_zic_profit': np.mean(results['zic_avg_profit'])
    }


def run_selfplay_test(zip_params, num_sessions=3):
    """
    Run ZIP vs ZIP self-play to ensure efficiency doesn't collapse.

    Returns mean self-play efficiency.
    """
    efficiencies = []

    # Symmetric market: 5v5
    buyer_vals = [[307, 227, 162, 106]] * 5
    seller_costs = [[107, 143, 196, 295]] * 5

    for session in range(num_sessions):
        # Create ZIP buyers
        buyers = []
        for i in range(5):
            agent = ZIP(
                player_id=i + 1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_vals[i],
                num_times=200,
                price_min=0,
                price_max=400,
                seed=session * 100 + i,
                **zip_params
            )
            buyers.append(agent)

        # Create ZIP sellers
        sellers = []
        for i in range(5):
            agent = ZIP(
                player_id=i + 6,
                is_buyer=False,
                num_tokens=4,
                valuations=seller_costs[i],
                num_times=200,
                price_min=0,
                price_max=400,
                seed=session * 100 + i + 5,
                **zip_params
            )
            sellers.append(agent)

        all_agents = buyers + sellers

        # Run market
        for a in all_agents:
            a.start_period(1)

        market = Market(
            num_buyers=5,
            num_sellers=5,
            price_min=0,
            price_max=400,
            num_times=200,
            buyers=buyers,
            sellers=sellers
        )

        while market.current_time < market.num_times:
            market.run_time_step()

        for a in all_agents:
            a.end_period()

        # Calculate efficiency
        all_vals = sorted([v for vals in buyer_vals for v in vals if v > 0], reverse=True)
        all_costs = sorted([c for costs in seller_costs for c in costs if c > 0])
        max_surplus = sum(v - c for v, c in zip(all_vals, all_costs) if v > c)
        total_profit = sum(a.period_profit for a in all_agents)
        efficiency = (total_profit / max_surplus * 100) if max_surplus > 0 else 0

        efficiencies.append(efficiency)

    return np.mean(efficiencies)


def main():
    print("=" * 80)
    print("ZIP INVASIBILITY-FOCUSED HYPERPARAMETER CALIBRATION")
    print("=" * 80)
    print()
    print("Test: 1 ZIP buyer vs 7 ZIC agents (3 buyers + 4 sellers)")
    print("Primary Metric: Per-agent profit ratio (target: >3.0x)")
    print("Secondary Metric: Self-play efficiency (floor: >70%)")
    print()

    # Define configurations
    # Focus on profit extraction: higher beta, higher gamma, higher margin
    configs = []

    # 1. Baseline (current default)
    configs.append({'name': '01_Baseline', 'beta': 0.20, 'gamma': 0.25, 'margin': 0.20})

    # 2-7. Systematic grid around profit extraction region
    # Higher margins for better profit extraction
    configs.append({'name': '02_HighMargin', 'beta': 0.20, 'gamma': 0.25, 'margin': 0.30})
    configs.append({'name': '03_VHighMargin', 'beta': 0.20, 'gamma': 0.25, 'margin': 0.35})

    # Higher gamma for stronger momentum in AURORA
    configs.append({'name': '04_HighMomentum', 'beta': 0.20, 'gamma': 0.30, 'margin': 0.25})
    configs.append({'name': '05_VHighMomentum', 'beta': 0.20, 'gamma': 0.35, 'margin': 0.25})

    # Higher beta for faster learning
    configs.append({'name': '06_HighBeta', 'beta': 0.30, 'gamma': 0.25, 'margin': 0.25})
    configs.append({'name': '07_VHighBeta', 'beta': 0.40, 'gamma': 0.25, 'margin': 0.25})

    # 8-13. Combined high settings
    configs.append({'name': '08_HighAll', 'beta': 0.30, 'gamma': 0.30, 'margin': 0.30})
    configs.append({'name': '09_VHighAll', 'beta': 0.35, 'gamma': 0.35, 'margin': 0.35})
    configs.append({'name': '10_FastAggressive', 'beta': 0.40, 'gamma': 0.30, 'margin': 0.30})
    configs.append({'name': '11_HighProfit', 'beta': 0.25, 'gamma': 0.30, 'margin': 0.35})
    configs.append({'name': '12_Balanced+', 'beta': 0.25, 'gamma': 0.25, 'margin': 0.25})
    configs.append({'name': '13_ModAggressive', 'beta': 0.30, 'gamma': 0.28, 'margin': 0.28})

    # 14-18. Explore lower beta with high margin (steady extraction)
    configs.append({'name': '14_SteadyHigh', 'beta': 0.15, 'gamma': 0.30, 'margin': 0.30})
    configs.append({'name': '15_SteadyVHigh', 'beta': 0.15, 'gamma': 0.30, 'margin': 0.35})
    configs.append({'name': '16_LowBetaHiMom', 'beta': 0.15, 'gamma': 0.35, 'margin': 0.28})
    configs.append({'name': '17_SlowSteady', 'beta': 0.12, 'gamma': 0.28, 'margin': 0.32})
    configs.append({'name': '18_Conservative', 'beta': 0.18, 'gamma': 0.28, 'margin': 0.32})

    # 19-23. Extreme high beta (very aggressive)
    configs.append({'name': '19_UltraFast', 'beta': 0.50, 'gamma': 0.25, 'margin': 0.25})
    configs.append({'name': '20_UltraAgg', 'beta': 0.45, 'gamma': 0.32, 'margin': 0.32})
    configs.append({'name': '21_Blitz', 'beta': 0.50, 'gamma': 0.30, 'margin': 0.30})
    configs.append({'name': '22_Dominator', 'beta': 0.38, 'gamma': 0.35, 'margin': 0.35})
    configs.append({'name': '23_MaxProfit', 'beta': 0.32, 'gamma': 0.33, 'margin': 0.38})

    # 24-28. R/A perturbation exploration (with best core config from above)
    # We'll use a moderate aggressive base for R/A tests
    base_r_a = {'beta': 0.30, 'gamma': 0.30, 'margin': 0.30}

    configs.append({
        'name': '24_RA_Wide',
        **base_r_a,
        'R_increase_min': 1.0, 'R_increase_max': 1.10,
        'R_decrease_min': 0.90, 'R_decrease_max': 1.0,
        'A_increase_min': 0.0, 'A_increase_max': 0.10,
        'A_decrease_min': -0.10, 'A_decrease_max': 0.0
    })

    configs.append({
        'name': '25_RA_Tight',
        **base_r_a,
        'R_increase_min': 1.0, 'R_increase_max': 1.02,
        'R_decrease_min': 0.98, 'R_decrease_max': 1.0,
        'A_increase_min': 0.0, 'A_increase_max': 0.02,
        'A_decrease_min': -0.02, 'A_decrease_max': 0.0
    })

    configs.append({
        'name': '26_RA_Aggressive',
        **base_r_a,
        'R_increase_min': 1.0, 'R_increase_max': 1.15,
        'R_decrease_min': 0.85, 'R_decrease_max': 1.0,
        'A_increase_min': 0.05, 'A_increase_max': 0.15,
        'A_decrease_min': -0.15, 'A_decrease_max': -0.05
    })

    configs.append({
        'name': '27_RA_Asymmetric',
        **base_r_a,
        'R_increase_min': 1.0, 'R_increase_max': 1.08,
        'R_decrease_min': 0.95, 'R_decrease_max': 1.0,
        'A_increase_min': 0.0, 'A_increase_max': 0.08,
        'A_decrease_min': -0.05, 'A_decrease_max': 0.0
    })

    configs.append({
        'name': '28_RA_Conservative',
        **base_r_a,
        'R_increase_min': 1.0, 'R_increase_max': 1.03,
        'R_decrease_min': 0.97, 'R_decrease_max': 1.0,
        'A_increase_min': 0.0, 'A_increase_max': 0.03,
        'A_decrease_min': -0.03, 'A_decrease_max': 0.0
    })

    print(f"Testing {len(configs)} configurations...")
    print()

    # Run tests
    results = []
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] Testing {config['name']}...", end=' ', flush=True)

        # Extract ZIP parameters
        zip_params = {k: v for k, v in config.items() if k != 'name'}

        # Run invasibility test (primary)
        inv_results = run_invasibility_test(zip_params, num_sessions=5)

        # Run self-play test (secondary)
        selfplay_eff = run_selfplay_test(zip_params, num_sessions=3)

        # Composite score: 70% invasibility + 30% self-play
        # Normalize profit ratio to 0-100 scale (assuming max ~5.0x)
        inv_score = min(inv_results['mean_profit_ratio'] / 5.0 * 100, 100)
        composite_score = 0.70 * inv_score + 0.30 * selfplay_eff

        results.append({
            'config': config['name'],
            'beta': config['beta'],
            'gamma': config['gamma'],
            'margin': config.get('margin', 0.20),
            'profit_ratio': inv_results['mean_profit_ratio'],
            'profit_ratio_std': inv_results['std_profit_ratio'],
            'inv_efficiency': inv_results['mean_efficiency'],
            'zip_profit': inv_results['mean_zip_profit'],
            'zic_profit': inv_results['mean_zic_profit'],
            'selfplay_eff': selfplay_eff,
            'composite_score': composite_score
        })

        print(f"Ratio={inv_results['mean_profit_ratio']:.2f}x, SelfPlay={selfplay_eff:.1f}%, Score={composite_score:.1f}")

    # Sort by composite score
    results.sort(key=lambda x: x['composite_score'], reverse=True)

    # Display results
    print()
    print("=" * 80)
    print("RESULTS (sorted by composite score)")
    print("=" * 80)
    print()
    print(f"{'Rank':<5} {'Config':<18} {'Beta':<6} {'Gamma':<7} {'Margin':<7} {'Ratio':<8} {'±Std':<7} {'Self%':<7} {'Score':<7}")
    print("-" * 80)

    for i, r in enumerate(results, 1):
        marker = "★★★" if i <= 3 else ("★★" if i <= 5 else ("★" if i <= 10 else ""))
        print(f"{i:<5} {r['config']:<18} {r['beta']:<6.2f} {r['gamma']:<7.2f} {r['margin']:<7.2f} "
              f"{r['profit_ratio']:<8.2f} {r['profit_ratio_std']:<7.2f} {r['selfplay_eff']:<7.1f} "
              f"{r['composite_score']:<7.1f} {marker}")

    print()
    print("=" * 80)
    print("TOP 3 CONFIGURATIONS")
    print("=" * 80)

    for i in range(min(3, len(results))):
        r = results[i]
        print()
        print(f"Rank {i+1}: {r['config']}")
        print(f"  Parameters: beta={r['beta']}, gamma={r['gamma']}, margin={r['margin']}")
        print(f"  Profit Ratio: {r['profit_ratio']:.2f}x (±{r['profit_ratio_std']:.2f})")
        print(f"  Self-Play Eff: {r['selfplay_eff']:.1f}%")
        print(f"  Composite Score: {r['composite_score']:.1f}/100")
        print(f"  ZIP Profit: ${r['zip_profit']:.0f}, ZIC Avg Profit: ${r['zic_profit']:.0f}")
        print(f"  Invasibility Market Eff: {r['inv_efficiency']:.1f}%")

        if r['profit_ratio'] >= 3.0 and r['selfplay_eff'] >= 70:
            print("  ✅ MEETS SUCCESS CRITERIA")
        elif r['profit_ratio'] >= 2.5:
            print("  ⚠️ Good profit ratio, needs improvement")
        else:
            print("  ❌ Below target")

    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    best = results[0]
    if best['profit_ratio'] >= 3.0 and best['selfplay_eff'] >= 70:
        print(f"✅ SUCCESS! Config '{best['config']}' meets all criteria.")
        print(f"   Recommended defaults: beta={best['beta']}, gamma={best['gamma']}, margin={best['margin']}")
    elif best['profit_ratio'] >= 2.0:
        print(f"⚠️ PARTIAL SUCCESS. Config '{best['config']}' shows promise.")
        print(f"   Consider: beta={best['beta']}, gamma={best['gamma']}, margin={best['margin']}")
        print("   May need further tuning or algorithmic improvements.")
    else:
        print("❌ NO CONFIG MEETS TARGETS. Consider:")
        print("   1. Further hyperparameter exploration")
        print("   2. Algorithmic changes to margin update logic")
        print("   3. Review _should_lower_margin() implementation")

    print()
    print("Current defaults: beta=0.20, gamma=0.25, margin=0.20")
    print(f"Best found: beta={best['beta']}, gamma={best['gamma']}, margin={best['margin']}")
    print()


if __name__ == "__main__":
    main()
