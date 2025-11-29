#!/usr/bin/env python3
"""
Jacobson Hyperparameter Calibration.

Goals:
1. Maximize individual profit vs ZIC in 1v1 (target: >1.2x profit ratio)
2. Maximize self-play efficiency (target: 85%+, current: 72.9%)
3. Balance profit distribution in symmetric markets (target: 40-60%)
4. Robust performance across diverse scenarios (target: <10% std dev)

Multi-Objective Scoring:
- Profit ratio vs ZIC (40% weight)
- Self-play efficiency (30% weight)
- Profit balance (20% weight)
- Robustness (10% weight)

Tests 20 configurations across 10 market scenarios.
"""

import numpy as np
from engine.market import Market
from traders.legacy.jacobson import Jacobson
from traders.legacy.zic import ZIC
from engine.efficiency import (
    calculate_max_surplus,
    calculate_actual_surplus,
    calculate_allocative_efficiency,
    extract_trades_from_orderbook
)


def run_market_session(
    buyer_vals,
    seller_costs,
    num_steps=150,
    price_max=200,
    jacobson_params=None,
    jacobson_vs_jacobson=False,
    jacobson_vs_zic_mode='buyers',  # 'buyers', 'sellers', or 'both'
    seed_offset=0
):
    """
    Run a single market session.

    Args:
        buyer_vals: List of buyer valuation lists
        seller_costs: List of seller cost lists
        num_steps: Number of time steps
        price_max: Maximum price
        jacobson_params: Dict of hyperparameters for Jacobson agents
        jacobson_vs_jacobson: If True, both sides are Jacobson
        jacobson_vs_zic_mode: Which side has Jacobson ('buyers', 'sellers', 'both')
        seed_offset: Random seed offset

    Returns:
        dict with detailed metrics including individual profits
    """
    jacobson_params = jacobson_params or {}

    num_buyers = len(buyer_vals)
    num_sellers = len(seller_costs)

    # Create buyers
    buyers = []
    for i, vals in enumerate(buyer_vals):
        if jacobson_vs_jacobson or jacobson_vs_zic_mode in ['buyers', 'both']:
            agent = Jacobson(
                player_id=i + 1,
                is_buyer=True,
                num_tokens=len(vals),
                valuations=vals,
                num_times=num_steps,
                price_min=0,
                price_max=price_max,
                seed=seed_offset * 100 + i,
                **jacobson_params
            )
        else:
            agent = ZIC(
                player_id=i + 1,
                is_buyer=True,
                num_tokens=len(vals),
                valuations=vals,
                price_min=0,
                price_max=price_max,
                seed=seed_offset * 100 + i
            )
        buyers.append(agent)

    # Create sellers
    sellers = []
    for i, costs in enumerate(seller_costs):
        if jacobson_vs_jacobson or jacobson_vs_zic_mode in ['sellers', 'both']:
            agent = Jacobson(
                player_id=num_buyers + i + 1,
                is_buyer=False,
                num_tokens=len(costs),
                valuations=costs,
                num_times=num_steps,
                price_min=0,
                price_max=price_max,
                seed=seed_offset * 100 + num_buyers + i,
                **jacobson_params
            )
        else:
            agent = ZIC(
                player_id=num_buyers + i + 1,
                is_buyer=False,
                num_tokens=len(costs),
                valuations=costs,
                price_min=0,
                price_max=price_max,
                seed=seed_offset * 100 + num_buyers + i
            )
        sellers.append(agent)

    # Initialize period
    for a in buyers + sellers:
        a.start_period(1)

    # Create and run market
    market = Market(
        num_buyers=num_buyers,
        num_sellers=num_sellers,
        price_min=0,
        price_max=price_max,
        num_times=num_steps,
        buyers=buyers,
        sellers=sellers
    )

    for _ in range(num_steps):
        if not market.run_time_step():
            break

    # Extract trades and calculate metrics
    trades = extract_trades_from_orderbook(market.orderbook, num_steps)

    # Build valuation dicts using local indices (1-based)
    buyer_vals_dict = {i+1: buyers[i].valuations for i in range(num_buyers)}
    seller_costs_dict = {i+1: sellers[i].valuations for i in range(num_sellers)}

    actual_surplus = calculate_actual_surplus(trades, buyer_vals_dict, seller_costs_dict)
    max_surplus = calculate_max_surplus(
        [b.valuations for b in buyers],
        [s.valuations for s in sellers]
    )
    efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus) if max_surplus > 0 else 0

    # Calculate individual profits
    jacobson_profits = []
    zic_profits = []
    all_profits = []

    for buyer in buyers:
        all_profits.append(buyer.period_profit)
        if isinstance(buyer, Jacobson):
            jacobson_profits.append(buyer.period_profit)
        else:
            zic_profits.append(buyer.period_profit)

    for seller in sellers:
        all_profits.append(seller.period_profit)
        if isinstance(seller, Jacobson):
            jacobson_profits.append(seller.period_profit)
        else:
            zic_profits.append(seller.period_profit)

    return {
        'efficiency': efficiency,
        'actual_surplus': actual_surplus,
        'max_surplus': max_surplus,
        'jacobson_profits': jacobson_profits,
        'zic_profits': zic_profits,
        'all_profits': all_profits,
        'num_trades': len(trades),
        'buyers': buyers,
        'sellers': sellers
    }


def define_scenarios():
    """Define 10 diverse market scenarios for calibration."""
    scenarios = []

    # Scenario 1: Symmetric self-play (2v2)
    scenarios.append({
        'name': 'Self-Play 2v2',
        'buyer_vals': [[100, 90, 80], [95, 85, 75]],
        'seller_costs': [[40, 50, 60], [45, 55, 65]],
        'mode': 'self-play'
    })

    # Scenario 2: 1v1 vs ZIC (buyers)
    scenarios.append({
        'name': '1v1 Jacobson Buyer vs ZIC Seller',
        'buyer_vals': [[100, 90, 80]],
        'seller_costs': [[40, 50, 60]],
        'mode': 'buyers'
    })

    # Scenario 3: 1v1 vs ZIC (sellers)
    scenarios.append({
        'name': '1v1 ZIC Buyer vs Jacobson Seller',
        'buyer_vals': [[100, 90, 80]],
        'seller_costs': [[40, 50, 60]],
        'mode': 'sellers'
    })

    # Scenario 4: Asymmetric supply (2 buyers, 1 seller)
    scenarios.append({
        'name': 'Asymmetric 2v1 (excess demand)',
        'buyer_vals': [[100, 90], [95, 85]],
        'seller_costs': [[40, 50, 60, 70]],
        'mode': 'self-play'
    })

    # Scenario 5: Asymmetric demand (1 buyer, 2 sellers)
    scenarios.append({
        'name': 'Asymmetric 1v2 (excess supply)',
        'buyer_vals': [[100, 90, 80, 70]],
        'seller_costs': [[40, 50], [45, 55]],
        'mode': 'self-play'
    })

    # Scenario 6: Tight equilibrium (small surplus)
    scenarios.append({
        'name': 'Tight Equilibrium',
        'buyer_vals': [[85, 80, 75]],
        'seller_costs': [[70, 75, 80]],
        'mode': 'buyers'
    })

    # Scenario 7: Wide spread (large surplus)
    scenarios.append({
        'name': 'Wide Spread',
        'buyer_vals': [[150, 140, 130]],
        'seller_costs': [[20, 30, 40]],
        'mode': 'sellers'
    })

    # Scenario 8: Multi-agent competitive (3v3)
    scenarios.append({
        'name': 'Competitive 3v3',
        'buyer_vals': [[100, 90, 80], [98, 88, 78], [95, 85, 75]],
        'seller_costs': [[40, 50, 60], [42, 52, 62], [45, 55, 65]],
        'mode': 'self-play'
    })

    # Scenario 9: Single token (simplest case)
    scenarios.append({
        'name': 'Single Token',
        'buyer_vals': [[100]],
        'seller_costs': [[50]],
        'mode': 'buyers'
    })

    # Scenario 10: Box design (excess supply and demand)
    scenarios.append({
        'name': 'Box Design 2v2',
        'buyer_vals': [[120, 100], [110, 90]],
        'seller_costs': [[60, 80], [70, 90]],
        'mode': 'self-play'
    })

    return scenarios


def define_parameter_grid():
    """
    Define 20-configuration coarse grid for hyperparameter search.

    Parameters:
    - bid_ask_offset: [0.5, 1.0, 2.0, 3.0]
    - trade_weight_multiplier: [1.0, 2.0, 3.0, 4.0]
    - confidence_base: [0.001, 0.01, 0.05, 0.1]
    - time_pressure_multiplier: [1.0, 2.0, 3.0, 4.0]
    """
    configs = []

    # Config 0: Baseline (original defaults)
    configs.append({
        'name': 'Baseline (Original)',
        'params': {
            'bid_ask_offset': 1.0,
            'trade_weight_multiplier': 2.0,
            'confidence_base': 0.01,
            'time_pressure_multiplier': 2.0
        }
    })

    # Configs 1-4: Vary bid_ask_offset (aggressiveness)
    for offset in [0.5, 1.5, 2.5, 3.0]:
        configs.append({
            'name': f'Offset_{offset}',
            'params': {
                'bid_ask_offset': offset,
                'trade_weight_multiplier': 2.0,
                'confidence_base': 0.01,
                'time_pressure_multiplier': 2.0
            }
        })

    # Configs 5-7: Vary trade_weight_multiplier (learning speed)
    for multiplier in [1.0, 3.0, 4.0]:
        configs.append({
            'name': f'TradeWeight_{multiplier}',
            'params': {
                'bid_ask_offset': 1.0,
                'trade_weight_multiplier': multiplier,
                'confidence_base': 0.01,
                'time_pressure_multiplier': 2.0
            }
        })

    # Configs 8-10: Vary confidence_base (confidence growth)
    for base in [0.001, 0.05, 0.1]:
        configs.append({
            'name': f'ConfBase_{base}',
            'params': {
                'bid_ask_offset': 1.0,
                'trade_weight_multiplier': 2.0,
                'confidence_base': base,
                'time_pressure_multiplier': 2.0
            }
        })

    # Configs 11-13: Vary time_pressure_multiplier (urgency)
    for pressure in [1.0, 3.0, 4.0]:
        configs.append({
            'name': f'TimePressure_{pressure}',
            'params': {
                'bid_ask_offset': 1.0,
                'trade_weight_multiplier': 2.0,
                'confidence_base': 0.01,
                'time_pressure_multiplier': pressure
            }
        })

    # Configs 14-19: Random combinations (Latin Hypercube-inspired)
    import random
    random.seed(42)
    for i in range(6):
        configs.append({
            'name': f'Random_{i+1}',
            'params': {
                'bid_ask_offset': random.choice([0.75, 1.5, 2.0, 2.75]),
                'trade_weight_multiplier': random.choice([1.5, 2.5, 3.5]),
                'confidence_base': random.choice([0.005, 0.02, 0.075]),
                'time_pressure_multiplier': random.choice([1.5, 2.5, 3.5])
            }
        })

    return configs


def evaluate_config(config, scenarios, num_reps=5):
    """
    Evaluate a single configuration across all scenarios.

    Returns:
        dict with aggregated metrics
    """
    config_name = config['name']
    params = config['params']

    print(f"\n{'='*60}")
    print(f"Evaluating: {config_name}")
    print(f"Parameters: {params}")
    print(f"{'='*60}")

    # Collect metrics across scenarios
    all_efficiencies = []
    all_profit_ratios = []
    all_profit_balances = []
    scenario_results = {}

    for scenario in scenarios:
        scenario_name = scenario['name']
        mode = scenario['mode']

        # Determine Jacobson placement
        if mode == 'self-play':
            jacobson_vs_jacobson = True
            jacobson_vs_zic_mode = 'both'
        elif mode == 'buyers':
            jacobson_vs_jacobson = False
            jacobson_vs_zic_mode = 'buyers'
        elif mode == 'sellers':
            jacobson_vs_jacobson = False
            jacobson_vs_zic_mode = 'sellers'
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Run multiple replications
        rep_efficiencies = []
        rep_profit_ratios = []
        rep_profit_balances = []

        for rep in range(num_reps):
            result = run_market_session(
                buyer_vals=scenario['buyer_vals'],
                seller_costs=scenario['seller_costs'],
                num_steps=150,
                price_max=200,
                jacobson_params=params,
                jacobson_vs_jacobson=jacobson_vs_jacobson,
                jacobson_vs_zic_mode=jacobson_vs_zic_mode,
                seed_offset=rep
            )

            rep_efficiencies.append(result['efficiency'])

            # Calculate profit ratio (Jacobson vs ZIC)
            jacobson_total = sum(result['jacobson_profits'])
            zic_total = sum(result['zic_profits']) if result['zic_profits'] else 1e-6

            if jacobson_vs_jacobson:
                # Self-play: profit balance (how close to 50/50)
                total_profit = sum(result['all_profits'])
                if total_profit > 0:
                    profit_share = jacobson_total / total_profit * 100
                    profit_balance = 100 - abs(50 - profit_share)  # 100 = perfect balance
                    rep_profit_balances.append(profit_balance)
                profit_ratio = 1.0  # Not applicable
            else:
                # Competitive: profit ratio
                profit_ratio = jacobson_total / zic_total if zic_total > 0 else 1.0
                rep_profit_ratios.append(profit_ratio)

        # Aggregate per scenario
        mean_efficiency = np.mean(rep_efficiencies)
        mean_profit_ratio = np.mean(rep_profit_ratios) if rep_profit_ratios else 1.0
        mean_profit_balance = np.mean(rep_profit_balances) if rep_profit_balances else 50.0

        all_efficiencies.extend(rep_efficiencies)
        if rep_profit_ratios:
            all_profit_ratios.extend(rep_profit_ratios)
        if rep_profit_balances:
            all_profit_balances.extend(rep_profit_balances)

        scenario_results[scenario_name] = {
            'efficiency': mean_efficiency,
            'profit_ratio': mean_profit_ratio,
            'profit_balance': mean_profit_balance
        }

        print(f"  {scenario_name:40s} Eff: {mean_efficiency:5.1f}%  "
              f"Ratio: {mean_profit_ratio:.2f}  Balance: {mean_profit_balance:.1f}")

    # Calculate overall metrics
    overall_efficiency = np.mean(all_efficiencies)
    overall_profit_ratio = np.mean(all_profit_ratios) if all_profit_ratios else 1.0
    overall_profit_balance = np.mean(all_profit_balances) if all_profit_balances else 50.0
    robustness_std = np.std(all_efficiencies)

    # Multi-objective composite score
    # Normalize each metric to [0, 1] range, then weight
    efficiency_score = overall_efficiency / 95.0  # Target: 95% → score=1.0
    profit_ratio_score = min(overall_profit_ratio / 1.5, 1.0)  # Target: 1.5x → score=1.0
    profit_balance_score = overall_profit_balance / 100.0  # Target: 100 (perfect) → score=1.0
    robustness_score = 1.0 / (1.0 + robustness_std / 10.0)  # Lower std = higher score

    composite_score = (
        0.30 * efficiency_score +      # 30% weight: self-play efficiency
        0.40 * profit_ratio_score +    # 40% weight: profit vs ZIC
        0.20 * profit_balance_score +  # 20% weight: profit balance
        0.10 * robustness_score         # 10% weight: robustness
    ) * 100

    print(f"\n  OVERALL:")
    print(f"    Efficiency:      {overall_efficiency:.1f}% (score: {efficiency_score:.3f})")
    print(f"    Profit Ratio:    {overall_profit_ratio:.2f}x (score: {profit_ratio_score:.3f})")
    print(f"    Profit Balance:  {overall_profit_balance:.1f} (score: {profit_balance_score:.3f})")
    print(f"    Robustness Std:  {robustness_std:.1f}% (score: {robustness_score:.3f})")
    print(f"    COMPOSITE SCORE: {composite_score:.2f}")

    return {
        'config_name': config_name,
        'params': params,
        'overall_efficiency': overall_efficiency,
        'overall_profit_ratio': overall_profit_ratio,
        'overall_profit_balance': overall_profit_balance,
        'robustness_std': robustness_std,
        'composite_score': composite_score,
        'scenario_results': scenario_results
    }


def main():
    """Main calibration loop."""
    print("="*70)
    print("JACOBSON HYPERPARAMETER CALIBRATION")
    print("="*70)
    print("\nGoals:")
    print("  1. Maximize profit vs ZIC (1v1)")
    print("  2. Maximize self-play efficiency (target: 85%+)")
    print("  3. Balance profit distribution (symmetric markets)")
    print("  4. Robust across diverse scenarios")
    print("\nScoring Weights:")
    print("  - Profit ratio vs ZIC:    40%")
    print("  - Self-play efficiency:   30%")
    print("  - Profit balance:         20%")
    print("  - Robustness:             10%")

    scenarios = define_scenarios()
    configs = define_parameter_grid()

    print(f"\nRunning calibration:")
    print(f"  - Configurations: {len(configs)}")
    print(f"  - Scenarios: {len(scenarios)}")
    print(f"  - Replications per scenario: 5")
    print(f"  - Total simulations: {len(configs) * len(scenarios) * 5}")

    all_results = []

    for config in configs:
        result = evaluate_config(config, scenarios, num_reps=5)
        all_results.append(result)

    # Sort by composite score
    all_results.sort(key=lambda x: x['composite_score'], reverse=True)

    # Print summary
    print("\n" + "="*70)
    print("CALIBRATION RESULTS (sorted by composite score)")
    print("="*70)
    print(f"{'Rank':<6} {'Config':<25} {'Composite':<10} {'Efficiency':<12} {'Profit Ratio':<14} {'Balance':<10} {'Robustness':<12}")
    print("-"*70)

    for i, result in enumerate(all_results):
        print(f"{i+1:<6} {result['config_name']:<25} "
              f"{result['composite_score']:>8.2f}  "
              f"{result['overall_efficiency']:>10.1f}%  "
              f"{result['overall_profit_ratio']:>12.2f}x  "
              f"{result['overall_profit_balance']:>8.1f}  "
              f"{result['robustness_std']:>10.1f}%")

    # Print top 5 parameter sets
    print("\n" + "="*70)
    print("TOP 5 PARAMETER SETS")
    print("="*70)

    for i in range(min(5, len(all_results))):
        result = all_results[i]
        print(f"\n{i+1}. {result['config_name']} (Composite Score: {result['composite_score']:.2f})")
        print(f"   Parameters:")
        for param, value in result['params'].items():
            print(f"     {param:30s} = {value}")
        print(f"   Metrics:")
        print(f"     Overall Efficiency:     {result['overall_efficiency']:.1f}%")
        print(f"     Profit Ratio vs ZIC:    {result['overall_profit_ratio']:.2f}x")
        print(f"     Profit Balance:         {result['overall_profit_balance']:.1f}")
        print(f"     Robustness (std dev):   {result['robustness_std']:.1f}%")

    # Recommend best config
    best = all_results[0]
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print(f"\nBest configuration: {best['config_name']}")
    print(f"Composite score: {best['composite_score']:.2f}")
    print(f"\nRecommended parameters for traders/legacy/jacobson.py:")
    print(f"```python")
    for param, value in best['params'].items():
        print(f"self.{param} = kwargs.get('{param}', {value})")
    print(f"```")

    # Compare to baseline
    baseline = next((r for r in all_results if r['config_name'] == 'Baseline (Original)'), None)
    if baseline:
        print(f"\nImprovement over baseline:")
        print(f"  Efficiency:    {baseline['overall_efficiency']:.1f}% → {best['overall_efficiency']:.1f}% "
              f"({best['overall_efficiency'] - baseline['overall_efficiency']:+.1f}%)")
        print(f"  Profit Ratio:  {baseline['overall_profit_ratio']:.2f}x → {best['overall_profit_ratio']:.2f}x "
              f"({best['overall_profit_ratio'] - baseline['overall_profit_ratio']:+.2f}x)")
        print(f"  Composite:     {baseline['composite_score']:.2f} → {best['composite_score']:.2f} "
              f"({best['composite_score'] - baseline['composite_score']:+.2f})")


if __name__ == '__main__':
    main()
