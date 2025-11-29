#!/usr/bin/env python3
"""
ZIP Hyperparameter Calibration against ZIC.

Tests various ZIP parameter configurations to find optimal settings
for competing against ZIC agents across different market scenarios.
"""

import numpy as np
from itertools import product
from engine.market import Market
from traders.legacy.zip import ZIP
from traders.legacy.zic import ZIC


def run_zip_vs_zic(
    buyer_vals,
    seller_costs,
    num_sessions=10,
    num_steps=200,
    price_max=400,
    zip_params=None,
    zip_vs_zip=False
):
    """
    Run ZIP buyers vs ZIC sellers market sessions (or ZIP vs ZIP if flag set).

    Returns:
        dict with extensive profit metrics
    """
    zip_params = zip_params or {}

    results = {
        'zip_profit': [], 'zic_profit': [], 'efficiency': [],
        'zip_trades': [], 'zic_trades': [],
        'zip_per_agent': [], 'zic_per_agent': []
    }

    for session in range(num_sessions):
        num_buyers = len(buyer_vals)
        num_sellers = len(seller_costs)

        # Create ZIP buyers
        buyers = []
        for i, vals in enumerate(buyer_vals):
            agent = ZIP(
                player_id=i + 1,
                is_buyer=True,
                num_tokens=len(vals),
                valuations=vals,
                num_times=num_steps,
                price_min=0,
                price_max=price_max,
                seed=session * 100 + i,
                **zip_params
            )
            buyers.append(agent)

        # Create sellers (ZIP or ZIC)
        sellers = []
        if zip_vs_zip:
            # ZIP vs ZIP self-play
            for i, costs in enumerate(seller_costs):
                agent = ZIP(
                    player_id=num_buyers + i + 1,
                    is_buyer=False,
                    num_tokens=len(costs),
                    valuations=costs,
                    num_times=num_steps,
                    price_min=0,
                    price_max=price_max,
                    seed=session * 100 + num_buyers + i,
                    **zip_params
                )
                sellers.append(agent)
        else:
            # ZIP vs ZIC
            for i, costs in enumerate(seller_costs):
                agent = ZIC(
                    player_id=num_buyers + i + 1,
                    is_buyer=False,
                    num_tokens=len(costs),
                    valuations=costs,
                    num_times=num_steps,
                    price_min=0,
                    price_max=price_max,
                    seed=session * 100 + num_buyers + i
                )
                sellers.append(agent)

        # Run market
        for a in buyers + sellers:
            a.start_period(1)

        market = Market(
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            price_min=0,
            price_max=price_max,
            num_times=num_steps,
            buyers=buyers,
            sellers=sellers
        )

        while market.current_time < market.num_times:
            market.run_time_step()

        for a in buyers + sellers:
            a.end_period()

        # Collect results
        zip_profit = sum(a.period_profit for a in buyers)
        zic_profit = sum(a.period_profit for a in sellers)
        total_profit = zip_profit + zic_profit

        zip_trades = sum(a.num_trades for a in buyers)
        zic_trades = sum(a.num_trades for a in sellers)

        # Per-agent profits
        zip_per_agent_profit = [a.period_profit for a in buyers]
        zic_per_agent_profit = [a.period_profit for a in sellers]

        # Calculate max surplus
        all_vals = sorted([v for vals in buyer_vals for v in vals if v > 0], reverse=True)
        all_costs = sorted([c for costs in seller_costs for c in costs if c > 0])
        max_surplus = sum(v - c for v, c in zip(all_vals, all_costs) if v > c)

        efficiency = (total_profit / max_surplus * 100) if max_surplus > 0 else 0

        results['zip_profit'].append(zip_profit)
        results['zic_profit'].append(zic_profit)
        results['efficiency'].append(efficiency)
        results['zip_trades'].append(zip_trades)
        results['zic_trades'].append(zic_trades)
        results['zip_per_agent'].extend(zip_per_agent_profit)
        results['zic_per_agent'].extend(zic_per_agent_profit)

    # Summary stats
    avg_zip = np.mean(results['zip_profit'])
    avg_zic = np.mean(results['zic_profit'])
    total_avg = avg_zip + avg_zic

    # Profit domination metrics
    profit_share = (avg_zip / total_avg * 100) if total_avg > 0 else 50.0
    profit_ratio = (avg_zip / avg_zic) if avg_zic > 0 else float('inf')

    return {
        'zip_profit_mean': avg_zip,
        'zic_profit_mean': avg_zic,
        'efficiency_mean': np.mean(results['efficiency']),
        'zip_won': avg_zip > avg_zic,
        'zip_profit_std': np.std(results['zip_profit']),
        'zic_profit_std': np.std(results['zic_profit']),
        'profit_share': profit_share,  # ZIP's share of total profit
        'profit_ratio': profit_ratio,   # ZIP/ZIC ratio
        'zip_trades_mean': np.mean(results['zip_trades']),
        'zic_trades_mean': np.mean(results['zic_trades']),
        'zip_per_agent_mean': np.mean(results['zip_per_agent']),
        'zip_per_agent_std': np.std(results['zip_per_agent']),
        'zic_per_agent_mean': np.mean(results['zic_per_agent']),
        'zic_per_agent_std': np.std(results['zic_per_agent'])
    }


def test_hyperparameter_config(config, scenarios):
    """Test a single FIXED hyperparameter configuration across all scenarios."""

    # Pass fixed values (not ranges)
    zip_params = {
        'beta': config['beta'],
        'gamma': config['gamma'],
        'margin': config['margin_sellers']  # Will use for sellers, buyers get negative
    }

    results = {}
    for scenario_name, scenario_config in scenarios.items():
        buyer_vals = scenario_config['buyers']
        seller_costs = scenario_config['sellers']
        zip_vs_zip = scenario_config.get('zip_vs_zip', False)

        result = run_zip_vs_zic(
            buyer_vals, seller_costs,
            num_sessions=5,
            zip_params=zip_params,
            zip_vs_zip=zip_vs_zip
        )
        results[scenario_name] = result

    # Aggregate metrics
    total_wins = sum(1 for r in results.values() if r['zip_won'])
    avg_efficiency = np.mean([r['efficiency_mean'] for r in results.values()])
    avg_profit_share = np.mean([r['profit_share'] for r in results.values()])
    avg_profit_ratio = np.mean([r['profit_ratio'] for r in results.values() if r['profit_ratio'] != float('inf')])

    return {
        'config': config,
        'results': results,
        'total_wins': total_wins,
        'avg_efficiency': avg_efficiency,
        'avg_profit_share': avg_profit_share,
        'avg_profit_ratio': avg_profit_ratio,
        'score': total_wins * 100 + avg_efficiency  # Combined score
    }


def main():
    """Run hyperparameter calibration with FIXED values."""

    print("ZIP Hyperparameter Calibration (Fixed Values)")
    print("=" * 70)

    # Define test scenarios (including ZIP vs ZIP self-play)
    scenarios = {
        'symmetric': {
            'buyers': [[300], [275], [250], [225]],
            'sellers': [[100], [125], [150], [175]],
            'zip_vs_zip': False
        },
        'flat_supply': {
            'buyers': [[325], [300], [275], [250], [225], [200]],
            'sellers': [[200]] * 6,
            'zip_vs_zip': False
        },
        'asymmetric': {
            'buyers': [[350], [300], [250]],
            'sellers': [[100], [150], [200], [250]],
            'zip_vs_zip': False
        },
        'high_competition': {
            'buyers': [[300], [290], [280], [270], [260]],
            'sellers': [[150], [160], [170], [180], [190]],
            'zip_vs_zip': False
        },
        'zip_vs_zip': {
            'buyers': [[300], [275], [250], [225]],
            'sellers': [[100], [125], [150], [175]],
            'zip_vs_zip': True  # Both sides use ZIP
        }
    }

    # Define fixed configurations to test
    configs = [
        {
            'name': '1. Paper Midpoint',
            'beta': 0.3,
            'gamma': 0.05,
            'margin_sellers': 0.20
        },
        {
            'name': '2. Higher Momentum',
            'beta': 0.3,
            'gamma': 0.20,
            'margin_sellers': 0.20
        },
        {
            'name': '3. High Momentum + High Beta',
            'beta': 0.5,
            'gamma': 0.20,
            'margin_sellers': 0.20
        },
        {
            'name': '4. Conservative',
            'beta': 0.2,
            'gamma': 0.10,
            'margin_sellers': 0.10
        },
        {
            'name': '5. Aggressive Margins',
            'beta': 0.3,
            'gamma': 0.15,
            'margin_sellers': 0.30
        },
        {
            'name': '6. Fast Learner',
            'beta': 0.6,
            'gamma': 0.25,
            'margin_sellers': 0.20
        },
        {
            'name': '7. Balanced',
            'beta': 0.35,
            'gamma': 0.15,
            'margin_sellers': 0.15
        },
        {
            'name': '8. High Momentum Low Beta',
            'beta': 0.2,
            'gamma': 0.25,
            'margin_sellers': 0.20
        }
    ]

    best_config = None
    best_score = -1

    for config in configs:
        print(f"\n{'='*70}")
        print(f"{config['name']}")
        print(f"  β={config['beta']:.2f}, γ={config['gamma']:.2f}, margin={config['margin_sellers']:.2f}")
        print(f"{'='*70}")

        result = test_hyperparameter_config(config, scenarios)

        print(f"Score: {result['score']:.1f}")
        print(f"Wins: {result['total_wins']}/5")
        print(f"Avg Efficiency: {result['avg_efficiency']:.1f}%")
        print(f"Avg Profit Share: {result['avg_profit_share']:.1f}% (ZIP's share of total)")
        print(f"Avg Profit Ratio: {result['avg_profit_ratio']:.2f}x (ZIP/ZIC)")

        # Print per-scenario results with profit metrics
        print("\nPer-Scenario Results:")
        for scenario_name, scenario_result in result['results'].items():
            won = "✓" if scenario_result['zip_won'] else "✗"
            print(f"  {scenario_name:20s}: {won} | "
                  f"Eff={scenario_result['efficiency_mean']:5.1f}% | "
                  f"Share={scenario_result['profit_share']:5.1f}% | "
                  f"Ratio={scenario_result['profit_ratio']:4.2f}x")

        if result['score'] > best_score:
            best_score = result['score']
            best_config = result

    print("\n" + "=" * 70)
    print("BEST CONFIGURATION:")
    print("=" * 70)
    print(f"Name: {best_config['config']['name']}")
    print(f"Score: {best_config['score']:.1f}")
    print(f"Wins: {best_config['total_wins']}/5")
    print(f"Avg Efficiency: {best_config['avg_efficiency']:.1f}%")
    print(f"Avg Profit Share: {best_config['avg_profit_share']:.1f}% (ZIP captures this much profit)")
    print(f"Avg Profit Ratio: {best_config['avg_profit_ratio']:.2f}x (ZIP/ZIC)")
    print("\nFixed Hyperparameters:")
    print(f"  beta: {best_config['config']['beta']}")
    print(f"  gamma: {best_config['config']['gamma']}")
    print(f"  margin_sellers: {best_config['config']['margin_sellers']}")
    print(f"  margin_buyers: -{best_config['config']['margin_sellers']} (symmetric)")

    # Check if ZIP crushes ZIC (>65% profit share, >2x ratio)
    if best_config['avg_profit_share'] > 65.0 and best_config['avg_profit_ratio'] > 2.0:
        print("\n✅ ZIP CRUSHES ZIC (profit share >65%, ratio >2x)")
    elif best_config['avg_profit_share'] > 55.0:
        print("\n⚠️  ZIP dominates but doesn't crush ZIC (profit share 55-65%)")
    else:
        print("\n❌ ZIP does not dominate ZIC adequately")


if __name__ == "__main__":
    main()
