#!/usr/bin/env python3
"""
Enhanced ZIP Hyperparameter Calibration (v2).

Goals:
1. ZIP vs ZIP self-play efficiency ≥95%
2. Profit share vs ZIC ≥70%
3. Robust performance across diverse scenarios

Tests 15 configurations across 10 market scenarios.
"""

import numpy as np
from engine.market import Market
from traders.legacy.zip import ZIP
from traders.legacy.zic import ZIC


def run_market_session(
    buyer_vals,
    seller_costs,
    num_steps=200,
    price_max=400,
    zip_params=None,
    zip_vs_zip=False,
    seed_offset=0
):
    """
    Run a single market session.

    Returns:
        dict with detailed metrics
    """
    zip_params = zip_params or {}

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
            seed=seed_offset * 100 + i,
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
                seed=seed_offset * 100 + num_buyers + i,
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
                seed=seed_offset * 100 + num_buyers + i
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

    # Track prices for convergence analysis
    trade_prices = []

    while market.current_time < market.num_times:
        market.run_time_step()
        # Get trade price from orderbook for this time step
        if market.current_time > 0:
            tp = int(market.orderbook.trade_price[market.current_time])
            if tp > 0:
                trade_prices.append(tp)

    for a in buyers + sellers:
        a.end_period()

    # Collect metrics
    zip_profit = sum(a.period_profit for a in buyers)
    counterparty_profit = sum(a.period_profit for a in sellers)
    total_profit = zip_profit + counterparty_profit

    zip_trades = sum(a.num_trades for a in buyers)
    counterparty_trades = sum(a.num_trades for a in sellers)
    total_trades = zip_trades + counterparty_trades

    # Per-agent profits
    zip_per_agent = [a.period_profit for a in buyers]
    counterparty_per_agent = [a.period_profit for a in sellers]

    # Calculate max surplus
    all_vals = sorted([v for vals in buyer_vals for v in vals if v > 0], reverse=True)
    all_costs = sorted([c for costs in seller_costs for c in costs if c > 0])
    max_surplus = sum(v - c for v, c in zip(all_vals, all_costs) if v > c)
    max_trades = sum(1 for v, c in zip(all_vals, all_costs) if v > c)

    efficiency = (total_profit / max_surplus * 100) if max_surplus > 0 else 0

    # Profit distribution
    profit_share = (zip_profit / total_profit * 100) if total_profit > 0 else 50.0
    profit_ratio = (zip_profit / counterparty_profit) if counterparty_profit > 0 else float('inf')

    # Trade volume
    trade_volume = (total_trades / (max_trades * 2) * 100) if max_trades > 0 else 0  # *2 because each trade counts for both sides

    # Price convergence (std dev of last 20% of trades)
    if len(trade_prices) > 5:
        last_n = max(5, int(len(trade_prices) * 0.2))
        convergence_std = np.std(trade_prices[-last_n:])
    else:
        convergence_std = np.std(trade_prices) if trade_prices else 0

    return {
        'efficiency': efficiency,
        'zip_profit': zip_profit,
        'counterparty_profit': counterparty_profit,
        'profit_share': profit_share,
        'profit_ratio': profit_ratio,
        'zip_trades': zip_trades,
        'counterparty_trades': counterparty_trades,
        'trade_volume': trade_volume,
        'convergence_std': convergence_std,
        'zip_per_agent': zip_per_agent,
        'counterparty_per_agent': counterparty_per_agent
    }


def test_config_on_scenario(config, scenario_config, num_sessions=5):
    """Test a single configuration on a single scenario."""

    zip_params = {
        'beta': config['beta'],
        'gamma': config['gamma'],
        'margin': config['margin_sellers']
    }

    results = {
        'efficiency': [],
        'profit_share': [],
        'profit_ratio': [],
        'trade_volume': [],
        'convergence_std': []
    }

    for session in range(num_sessions):
        result = run_market_session(
            buyer_vals=scenario_config['buyers'],
            seller_costs=scenario_config['sellers'],
            zip_params=zip_params,
            zip_vs_zip=scenario_config.get('zip_vs_zip', False),
            seed_offset=session
        )

        results['efficiency'].append(result['efficiency'])
        results['profit_share'].append(result['profit_share'])
        results['profit_ratio'].append(result['profit_ratio'] if result['profit_ratio'] != float('inf') else 10.0)
        results['trade_volume'].append(result['trade_volume'])
        results['convergence_std'].append(result['convergence_std'])

    return {
        'mean_efficiency': np.mean(results['efficiency']),
        'mean_profit_share': np.mean(results['profit_share']),
        'mean_profit_ratio': np.mean(results['profit_ratio']),
        'mean_trade_volume': np.mean(results['trade_volume']),
        'mean_convergence_std': np.mean(results['convergence_std']),
        'std_efficiency': np.std(results['efficiency']),
    }


def test_hyperparameter_config(config, scenarios):
    """Test a single configuration across all scenarios."""

    results = {}
    for scenario_name, scenario_config in scenarios.items():
        result = test_config_on_scenario(config, scenario_config, num_sessions=5)
        results[scenario_name] = result

    # Aggregate metrics
    all_efficiencies = [r['mean_efficiency'] for r in results.values()]
    all_profit_shares = [r['mean_profit_share'] for r in results.values()]
    all_profit_ratios = [r['mean_profit_ratio'] for r in results.values()]

    # Separate ZIP vs ZIP from others
    zip_vs_zip_scenarios = [k for k in scenarios.keys() if scenarios[k].get('zip_vs_zip', False)]
    other_scenarios = [k for k in scenarios.keys() if not scenarios[k].get('zip_vs_zip', False)]

    zip_vs_zip_efficiency = np.mean([results[k]['mean_efficiency'] for k in zip_vs_zip_scenarios]) if zip_vs_zip_scenarios else 0
    other_profit_share = np.mean([results[k]['mean_profit_share'] for k in other_scenarios]) if other_scenarios else 0

    mean_efficiency = np.mean(all_efficiencies)
    mean_profit_share = np.mean(all_profit_shares)
    mean_profit_ratio = np.mean(all_profit_ratios)
    robustness_std = np.std(all_efficiencies)

    # Win rate (profit share > 55%)
    wins = sum(1 for ps in all_profit_shares if ps > 55.0)
    win_rate = wins / len(all_profit_shares) * 100

    # Multi-objective composite score
    # 30% self-play efficiency, 40% profit share, 20% overall efficiency, 10% robustness
    composite_score = (
        0.30 * (zip_vs_zip_efficiency / 95.0) +
        0.40 * (other_profit_share / 70.0) +
        0.20 * (mean_efficiency / 90.0) +
        0.10 * (1.0 / (robustness_std + 1.0))
    ) * 100

    return {
        'config': config,
        'results': results,
        'mean_efficiency': mean_efficiency,
        'mean_profit_share': mean_profit_share,
        'mean_profit_ratio': mean_profit_ratio,
        'robustness_std': robustness_std,
        'win_rate': win_rate,
        'zip_vs_zip_efficiency': zip_vs_zip_efficiency,
        'other_profit_share': other_profit_share,
        'composite_score': composite_score
    }


def main():
    """Run enhanced hyperparameter calibration."""

    print("="*80)
    print("ZIP Hyperparameter Calibration v2.0")
    print("Goals: Self-play ≥95%, Profit share ≥70%, Robust across 10 scenarios")
    print("="*80)

    # Define 10 market scenarios
    scenarios = {
        '1_symmetric': {
            'buyers': [[300], [275], [250], [225]],
            'sellers': [[100], [125], [150], [175]],
            'zip_vs_zip': False
        },
        '2_flat_supply': {
            'buyers': [[325], [300], [275], [250], [225], [200]],
            'sellers': [[200]] * 6,
            'zip_vs_zip': False
        },
        '3_asymmetric': {
            'buyers': [[350], [300], [250]],
            'sellers': [[100], [150], [200], [250]],
            'zip_vs_zip': False
        },
        '4_high_competition': {
            'buyers': [[300], [290], [280], [270], [260]],
            'sellers': [[150], [160], [170], [180], [190]],
            'zip_vs_zip': False
        },
        '5_zip_vs_zip': {
            'buyers': [[300], [275], [250], [225]],
            'sellers': [[100], [125], [150], [175]],
            'zip_vs_zip': True
        },
        '6_box_excess_demand': {
            'buyers': [[300]] * 7,
            'sellers': [[200]] * 5,
            'zip_vs_zip': False
        },
        '7_box_excess_supply': {
            'buyers': [[300]] * 5,
            'sellers': [[200]] * 7,
            'zip_vs_zip': False
        },
        '8_wide_spread': {
            'buyers': [[400], [380], [360], [340]],
            'sellers': [[50], [70], [90], [110]],
            'zip_vs_zip': False
        },
        '9_multi_token': {
            'buyers': [[320, 300, 280], [310, 290, 270]],
            'sellers': [[100, 120, 140], [110, 130, 150]],
            'zip_vs_zip': False
        },
        '10_tight_equilibrium': {
            'buyers': [[250], [248], [246], [244], [242], [240]],
            'sellers': [[230], [232], [234], [236], [238], [240]],
            'zip_vs_zip': False
        }
    }

    # Define 15 configurations
    configs = [
        {'name': '1. Current Best', 'beta': 0.2, 'gamma': 0.25, 'margin_sellers': 0.20},
        {'name': '2. Paper Default', 'beta': 0.3, 'gamma': 0.05, 'margin_sellers': 0.20},
        {'name': '3. Very High Momentum', 'beta': 0.2, 'gamma': 0.35, 'margin_sellers': 0.20},
        {'name': '4. Ultra High Momentum', 'beta': 0.2, 'gamma': 0.40, 'margin_sellers': 0.20},
        {'name': '5. Moderate Momentum', 'beta': 0.2, 'gamma': 0.15, 'margin_sellers': 0.20},
        {'name': '6. Very Low Beta', 'beta': 0.1, 'gamma': 0.25, 'margin_sellers': 0.20},
        {'name': '7. High Beta', 'beta': 0.4, 'gamma': 0.25, 'margin_sellers': 0.20},
        {'name': '8. Very High Beta', 'beta': 0.5, 'gamma': 0.25, 'margin_sellers': 0.20},
        {'name': '9. Low Margin', 'beta': 0.2, 'gamma': 0.25, 'margin_sellers': 0.10},
        {'name': '10. High Margin', 'beta': 0.2, 'gamma': 0.25, 'margin_sellers': 0.30},
        {'name': '11. Very High Margin', 'beta': 0.2, 'gamma': 0.25, 'margin_sellers': 0.35},
        {'name': '12. Fast Convergence', 'beta': 0.4, 'gamma': 0.35, 'margin_sellers': 0.15},
        {'name': '13. Profit Maximizer', 'beta': 0.15, 'gamma': 0.30, 'margin_sellers': 0.30},
        {'name': '14. Balanced Hybrid', 'beta': 0.25, 'gamma': 0.30, 'margin_sellers': 0.25},
        {'name': '15. Stable Performer', 'beta': 0.15, 'gamma': 0.20, 'margin_sellers': 0.15}
    ]

    # Run calibration
    all_results = []

    for i, config in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/15] {config['name']}")
        print(f"  β={config['beta']:.2f}, γ={config['gamma']:.2f}, margin={config['margin_sellers']:.2f}")
        print(f"{'='*80}")

        result = test_hyperparameter_config(config, scenarios)
        all_results.append(result)

        print(f"Composite Score: {result['composite_score']:.1f}/100")
        print(f"  Self-Play Efficiency: {result['zip_vs_zip_efficiency']:.1f}% (target: ≥95%)")
        print(f"  Profit Share (vs ZIC): {result['other_profit_share']:.1f}% (target: ≥70%)")
        print(f"  Mean Efficiency: {result['mean_efficiency']:.1f}%")
        print(f"  Robustness (Std Dev): {result['robustness_std']:.1f}%")
        print(f"  Win Rate: {result['win_rate']:.0f}%")

    # Find best configs
    best_overall = max(all_results, key=lambda r: r['composite_score'])
    best_self_play = max(all_results, key=lambda r: r['zip_vs_zip_efficiency'])
    best_profit = max(all_results, key=lambda r: r['other_profit_share'])
    best_robust = min(all_results, key=lambda r: r['robustness_std'])

    # Print summary
    print("\n" + "="*80)
    print("CALIBRATION SUMMARY")
    print("="*80)

    print(f"\n🏆 BEST OVERALL (Composite Score)")
    print(f"  {best_overall['config']['name']}")
    print(f"  β={best_overall['config']['beta']}, γ={best_overall['config']['gamma']}, margin=±{best_overall['config']['margin_sellers']}")
    print(f"  Score: {best_overall['composite_score']:.1f}/100")
    print(f"  Self-Play: {best_overall['zip_vs_zip_efficiency']:.1f}% {'✅' if best_overall['zip_vs_zip_efficiency'] >= 95 else '⚠️'}")
    print(f"  Profit Share: {best_overall['other_profit_share']:.1f}% {'✅' if best_overall['other_profit_share'] >= 70 else '⚠️'}")
    print(f"  Robustness: {best_overall['robustness_std']:.1f}% {'✅' if best_overall['robustness_std'] < 8 else '⚠️'}")

    print(f"\n🎯 BEST FOR SELF-PLAY (Goal 1)")
    print(f"  {best_self_play['config']['name']}")
    print(f"  β={best_self_play['config']['beta']}, γ={best_self_play['config']['gamma']}, margin=±{best_self_play['config']['margin_sellers']}")
    print(f"  Self-Play: {best_self_play['zip_vs_zip_efficiency']:.1f}% {'✅' if best_self_play['zip_vs_zip_efficiency'] >= 95 else '⚠️'}")

    print(f"\n💰 BEST FOR PROFIT EXTRACTION (Goal 2)")
    print(f"  {best_profit['config']['name']}")
    print(f"  β={best_profit['config']['beta']}, γ={best_profit['config']['gamma']}, margin=±{best_profit['config']['margin_sellers']}")
    print(f"  Profit Share: {best_profit['other_profit_share']:.1f}% {'✅' if best_profit['other_profit_share'] >= 70 else '⚠️'}")
    print(f"  Profit Ratio: {best_profit['mean_profit_ratio']:.2f}x")

    print(f"\n🛡️ MOST ROBUST (Goal 3)")
    print(f"  {best_robust['config']['name']}")
    print(f"  β={best_robust['config']['beta']}, γ={best_robust['config']['gamma']}, margin=±{best_robust['config']['margin_sellers']}")
    print(f"  Robustness: {best_robust['robustness_std']:.1f}% {'✅' if best_robust['robustness_std'] < 8 else '⚠️'}")

    # Check if any config meets all goals
    all_goals_met = [r for r in all_results if
                     r['zip_vs_zip_efficiency'] >= 95 and
                     r['other_profit_share'] >= 70 and
                     r['robustness_std'] < 8]

    if all_goals_met:
        print(f"\n✅ CONFIGS MEETING ALL GOALS:")
        for r in all_goals_met:
            print(f"  - {r['config']['name']} (Score: {r['composite_score']:.1f})")
    else:
        print(f"\n⚠️ NO SINGLE CONFIG MEETS ALL GOALS")
        print(f"  Recommend using different profiles for different contexts:")
        print(f"    Tournament/Self-Play: {best_self_play['config']['name']}")
        print(f"    Competitive/Profit: {best_profit['config']['name']}")

    # Parameter insights
    print(f"\n📊 PARAMETER INSIGHTS")

    # Analyze gamma effect on self-play
    gamma_self_play = [(r['config']['gamma'], r['zip_vs_zip_efficiency']) for r in all_results]
    gamma_self_play.sort()
    print(f"\n  γ (Momentum) vs Self-Play Efficiency:")
    for gamma, eff in gamma_self_play:
        print(f"    γ={gamma:.2f}: {eff:.1f}%")

    # Analyze margin effect on profit
    margin_profit = [(r['config']['margin_sellers'], r['other_profit_share']) for r in all_results]
    margin_profit.sort()
    print(f"\n  Margin vs Profit Share:")
    for margin, profit in margin_profit:
        print(f"    margin={margin:.2f}: {profit:.1f}%")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
