#!/usr/bin/env python3
"""
ZIP Hyperparameter Calibration v3 - Self-Play Optimization.

Goal: Achieve 95%+ ZIP vs ZIP self-play efficiency by testing untested R/A parameters.

Focus: Target price perturbation parameters (R_increase/decrease, A_increase/decrease)
       and smaller initial margins for faster convergence.
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
    """Run a single market session with detailed convergence tracking."""
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
    spreads = []  # Track spread evolution

    while market.current_time < market.num_times:
        market.run_time_step()

        # Get trade price from orderbook
        if market.current_time > 0:
            tp = int(market.orderbook.trade_price[market.current_time])
            if tp > 0:
                trade_prices.append(tp)

            # Track spread
            high_bid = int(market.orderbook.high_bid[market.current_time])
            low_ask = int(market.orderbook.low_ask[market.current_time])
            if high_bid > 0 and low_ask > 0:
                spreads.append(low_ask - high_bid)

    for a in buyers + sellers:
        a.end_period()

    # Calculate metrics
    zip_profit = sum(a.period_profit for a in buyers)
    counterparty_profit = sum(a.period_profit for a in sellers)
    total_profit = zip_profit + counterparty_profit

    # Max surplus
    all_vals = sorted([v for vals in buyer_vals for v in vals if v > 0], reverse=True)
    all_costs = sorted([c for costs in seller_costs for c in costs if c > 0])
    max_surplus = sum(v - c for v, c in zip(all_vals, all_costs) if v > c)

    efficiency = (total_profit / max_surplus * 100) if max_surplus > 0 else 0
    profit_share = (zip_profit / total_profit * 100) if total_profit > 0 else 50.0

    # Convergence metrics
    if len(trade_prices) > 5:
        last_n = max(5, int(len(trade_prices) * 0.2))
        convergence_std = np.std(trade_prices[-last_n:])
    else:
        convergence_std = np.std(trade_prices) if trade_prices else 0

    mean_spread = np.mean(spreads) if spreads else 0
    final_spread = spreads[-1] if spreads else 0

    return {
        'efficiency': efficiency,
        'profit_share': profit_share,
        'convergence_std': convergence_std,
        'mean_spread': mean_spread,
        'final_spread': final_spread,
        'num_trades': len(trade_prices)
    }


def test_config(config, scenarios, num_sessions=10):
    """Test a single configuration across scenarios."""

    results = {}
    for scenario_name, scenario_config in scenarios.items():
        session_results = {
            'efficiency': [],
            'profit_share': [],
            'convergence_std': [],
            'mean_spread': [],
            'final_spread': []
        }

        for session in range(num_sessions):
            result = run_market_session(
                buyer_vals=scenario_config['buyers'],
                seller_costs=scenario_config['sellers'],
                zip_params=config,
                zip_vs_zip=scenario_config.get('zip_vs_zip', False),
                seed_offset=session
            )

            for key in session_results:
                session_results[key].append(result[key])

        # Aggregate
        results[scenario_name] = {
            'mean_efficiency': np.mean(session_results['efficiency']),
            'std_efficiency': np.std(session_results['efficiency']),
            'mean_profit_share': np.mean(session_results['profit_share']),
            'mean_convergence_std': np.mean(session_results['convergence_std']),
            'mean_spread': np.mean(session_results['mean_spread']),
            'final_spread': np.mean(session_results['final_spread'])
        }

    return results


def main():
    """Run v3 calibration focused on self-play optimization."""

    print("="*80)
    print("ZIP Hyperparameter Calibration v3.0 - Self-Play Optimization")
    print("Goal: Achieve 95%+ ZIP vs ZIP efficiency via R/A tuning + smaller margins")
    print("="*80)

    # Focused scenarios for self-play testing
    scenarios = {
        'zip_vs_zip': {
            'buyers': [[300], [275], [250], [225]],
            'sellers': [[100], [125], [150], [175]],
            'zip_vs_zip': True
        },
        'symmetric': {
            'buyers': [[300], [275], [250], [225]],
            'sellers': [[100], [125], [150], [175]],
            'zip_vs_zip': False
        },
        'high_competition': {
            'buyers': [[300], [290], [280], [270], [260]],
            'sellers': [[150], [160], [170], [180], [190]],
            'zip_vs_zip': False
        }
    }

    # 8 configurations focused on self-play
    configs = [
        {
            'name': '1. Zero Margins (Upper Bound)',
            'params': {
                'beta': 0.2, 'gamma': 0.25, 'margin': 0.00
            }
        },
        {
            'name': '2. Tiny Margins (Likely Winner)',
            'params': {
                'beta': 0.2, 'gamma': 0.25, 'margin': 0.05
            }
        },
        {
            'name': '3. Double R Perturbations',
            'params': {
                'beta': 0.2, 'gamma': 0.25, 'margin': 0.15,
                'R_increase_max': 1.10, 'R_decrease_min': 0.90
            }
        },
        {
            'name': '4. AURORA Extreme',
            'params': {
                'beta': 0.25, 'gamma': 0.30, 'margin': 0.10,
                'R_increase_max': 1.08, 'R_decrease_min': 0.92,
                'A_increase_max': 0.08, 'A_decrease_min': -0.08
            }
        },
        {
            'name': '5. Wide Perturbation Bands',
            'params': {
                'beta': 0.2, 'gamma': 0.25, 'margin': 0.15,
                'R_increase_max': 1.15, 'R_decrease_min': 0.85,
                'A_increase_max': 0.10, 'A_decrease_min': -0.10
            }
        },
        {
            'name': '6. Large Absolute Perturbations',
            'params': {
                'beta': 0.2, 'gamma': 0.25, 'margin': 0.20,
                'A_increase_max': 0.10, 'A_decrease_min': -0.10
            }
        },
        {
            'name': '7. High Beta + Small Margins',
            'params': {
                'beta': 0.4, 'gamma': 0.35, 'margin': 0.10
            }
        },
        {
            'name': '8. Slow & Steady',
            'params': {
                'beta': 0.1, 'gamma': 0.15, 'margin': 0.05,
                'R_increase_max': 1.03, 'R_decrease_min': 0.97,
                'A_increase_max': 0.03, 'A_decrease_min': -0.03
            }
        },
        {
            'name': 'Baseline (v2 Winner - margin=0.30)',
            'params': {
                'beta': 0.2, 'gamma': 0.25, 'margin': 0.30
            }
        }
    ]

    all_results = []

    for i, config in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(configs)}] {config['name']}")

        # Print params
        params = config['params']
        print(f"  β={params.get('beta', 0.2):.2f}, γ={params.get('gamma', 0.25):.2f}, " +
              f"margin={params.get('margin', 0.20):.2f}")

        if 'R_increase_max' in params:
            print(f"  R: [{params.get('R_increase_min', 1.0):.2f}, {params['R_increase_max']:.2f}] / " +
                  f"[{params.get('R_decrease_min', 0.95):.2f}, {params.get('R_decrease_max', 1.0):.2f}]")
        if 'A_increase_max' in params:
            print(f"  A: [{params.get('A_increase_min', 0.0):.2f}, {params['A_increase_max']:.2f}] / " +
                  f"[{params.get('A_decrease_min', -0.05):.2f}, {params.get('A_decrease_max', 0.0):.2f}]")

        print(f"{'='*80}")

        results = test_config(params, scenarios, num_sessions=10)

        # Extract self-play efficiency
        self_play_eff = results['zip_vs_zip']['mean_efficiency']
        mean_profit_share = np.mean([r['mean_profit_share'] for k, r in results.items() if k != 'zip_vs_zip'])
        mean_eff = np.mean([r['mean_efficiency'] for r in results.values()])

        print(f"Self-Play Efficiency: {self_play_eff:.1f}% {'✅' if self_play_eff >= 95 else '⚠️' if self_play_eff >= 90 else '❌'}")
        print(f"Mean Efficiency: {mean_eff:.1f}%")
        print(f"Profit Share (vs ZIC): {mean_profit_share:.1f}%")
        print(f"Convergence (std dev): {results['zip_vs_zip']['mean_convergence_std']:.2f}")
        print(f"Final Spread: {results['zip_vs_zip']['final_spread']:.1f}")

        all_results.append({
            'config': config,
            'results': results,
            'self_play_eff': self_play_eff,
            'mean_eff': mean_eff,
            'profit_share': mean_profit_share
        })

    # Find best configs
    best_self_play = max(all_results, key=lambda r: r['self_play_eff'])
    best_balanced = max(all_results, key=lambda r: r['self_play_eff'] * 0.6 + r['profit_share'] * 0.4)

    # Summary
    print("\n" + "="*80)
    print("CALIBRATION v3 SUMMARY")
    print("="*80)

    print(f"\n🏆 BEST SELF-PLAY EFFICIENCY")
    print(f"  {best_self_play['config']['name']}")
    print(f"  Self-Play: {best_self_play['self_play_eff']:.1f}% {'✅ TARGET ACHIEVED!' if best_self_play['self_play_eff'] >= 95 else '⚠️ Close but not 95%' if best_self_play['self_play_eff'] >= 92 else '❌ Below target'}")
    print(f"  Mean Efficiency: {best_self_play['mean_eff']:.1f}%")
    print(f"  Profit Share: {best_self_play['profit_share']:.1f}%")

    print(f"\n⚖️ BEST BALANCED (60% self-play + 40% profit)")
    print(f"  {best_balanced['config']['name']}")
    print(f"  Self-Play: {best_balanced['self_play_eff']:.1f}%")
    print(f"  Profit Share: {best_balanced['profit_share']:.1f}%")

    # Check if 95% achieved
    configs_95plus = [r for r in all_results if r['self_play_eff'] >= 95]

    if configs_95plus:
        print(f"\n✅ {len(configs_95plus)} CONFIG(S) ACHIEVED 95%+ SELF-PLAY!")
        for r in configs_95plus:
            print(f"  - {r['config']['name']}: {r['self_play_eff']:.1f}%")
    else:
        max_achieved = max(r['self_play_eff'] for r in all_results)
        print(f"\n⚠️ NO CONFIG ACHIEVED 95%+ (max: {max_achieved:.1f}%)")
        if max_achieved >= 92:
            print(f"  92-94% range suggests 95% may be achievable with further tuning")
        elif max_achieved > best_self_play['self_play_eff']:
            print(f"  Improvement over v2's 90% validates R/A parameter importance")
        else:
            print(f"  90% appears to be fundamental limit for AURORA with fixed hyperparameters")

    # Parameter insights
    print(f"\n📊 KEY INSIGHTS")

    # Margin effect
    margin_results = [(r['config']['params'].get('margin', 0.20), r['self_play_eff']) for r in all_results]
    margin_results.sort()
    print(f"\n  Margin vs Self-Play:")
    for margin, eff in margin_results:
        print(f"    margin={margin:.2f}: {eff:.1f}%")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
