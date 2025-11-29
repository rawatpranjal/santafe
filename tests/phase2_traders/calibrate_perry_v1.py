"""
Perry Hyperparameter Calibration - Grid Search v1.0

Tests 15 parameter configurations across 10 market scenarios to optimize Perry's performance.

Hyperparameters Tuned:
- a0_initial: Initial adaptive parameter for statistical bidding (default 2.0)
- desperate_threshold: Time fraction when sellers use desperate acceptance (default 0.20)
- desperate_margin: Units above cost sellers accept when desperate (default 2)

Calibration Goal:
- Reduce exploitation vulnerability while maintaining elite efficiency
- Improve tournament ranking from 6th to 4th-5th place

Methodology:
- Grid Search: 15 configurations (5 a0 × 3 threshold × 1 margin, sampled strategically)
- Test Scenarios: 10 market types (symmetric, asymmetric, competitive, etc.)
- Scoring: Composite metric (30% self-play efficiency, 25% profit share, 20% invasibility, 15% balance, 10% win rate)
- Validation: 10 replications per scenario with different random seeds
"""

import argparse
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from engine.market import Market
from engine.token_generator import TokenGenerator
from engine.efficiency import (
    extract_trades_from_orderbook,
    calculate_actual_surplus,
    calculate_max_surplus,
    calculate_allocative_efficiency,
)
from traders.legacy.perry import Perry
from traders.legacy.zic import ZIC
from traders.legacy.gd import GD
from traders.legacy.kaplan import Kaplan
from traders.legacy.zip import ZIP


# ============================================================================
# PARAMETER GRID (15 configurations)
# ============================================================================

PARAMETER_GRID = [
    # Baseline (original Perry)
    {'a0_initial': 2.0, 'desperate_threshold': 0.20, 'desperate_margin': 2, 'name': 'baseline'},

    # Conservative configurations (lower a0 = less aggressive)
    {'a0_initial': 0.5, 'desperate_threshold': 0.20, 'desperate_margin': 2, 'name': 'conservative_a0'},
    {'a0_initial': 1.0, 'desperate_threshold': 0.20, 'desperate_margin': 2, 'name': 'moderate_a0'},

    # Aggressive configurations (higher a0 = more aggressive)
    {'a0_initial': 3.0, 'desperate_threshold': 0.20, 'desperate_margin': 2, 'name': 'aggressive_a0'},
    {'a0_initial': 5.0, 'desperate_threshold': 0.20, 'desperate_margin': 2, 'name': 'very_aggressive_a0'},

    # Desperate threshold variations (baseline a0)
    {'a0_initial': 2.0, 'desperate_threshold': 0.10, 'desperate_margin': 2, 'name': 'early_desperate'},
    {'a0_initial': 2.0, 'desperate_threshold': 0.30, 'desperate_margin': 2, 'name': 'late_desperate'},

    # Desperate margin variations (baseline a0)
    {'a0_initial': 2.0, 'desperate_threshold': 0.20, 'desperate_margin': 0, 'name': 'no_desperate'},
    {'a0_initial': 2.0, 'desperate_threshold': 0.20, 'desperate_margin': 4, 'name': 'high_desperate'},

    # Combined optimizations
    {'a0_initial': 1.0, 'desperate_threshold': 0.10, 'desperate_margin': 0, 'name': 'ultra_conservative'},
    {'a0_initial': 3.0, 'desperate_threshold': 0.30, 'desperate_margin': 4, 'name': 'ultra_aggressive'},
    {'a0_initial': 1.5, 'desperate_threshold': 0.15, 'desperate_margin': 1, 'name': 'balanced_conservative'},
    {'a0_initial': 2.5, 'desperate_threshold': 0.25, 'desperate_margin': 3, 'name': 'balanced_aggressive'},

    # Exploitation-resistant profiles
    {'a0_initial': 1.0, 'desperate_threshold': 0.30, 'desperate_margin': 0, 'name': 'anti_exploit_1'},
    {'a0_initial': 1.5, 'desperate_threshold': 0.25, 'desperate_margin': 1, 'name': 'anti_exploit_2'},
]


# ============================================================================
# TEST SCENARIOS (10 market types)
# ============================================================================

def get_scenario_config(scenario_name: str) -> Dict:
    """Return market configuration for each test scenario."""

    scenarios = {
        # 1. Symmetric market (balanced supply/demand)
        'symmetric': {
            'game_type': 1111,
            'num_buyers': 4,
            'num_sellers': 4,
            'num_tokens': 3,
            'description': 'Symmetric supply/demand (baseline)',
        },

        # 2. Flat supply/demand (homogeneous valuations)
        'flat': {
            'game_type': 2222,
            'num_buyers': 4,
            'num_sellers': 4,
            'num_tokens': 3,
            'description': 'Flat supply/demand curves',
        },

        # 3. Asymmetric market (more buyers)
        'asymmetric_buyers': {
            'game_type': 1111,
            'num_buyers': 6,
            'num_sellers': 4,
            'num_tokens': 3,
            'description': 'Buyer surplus (6v4)',
        },

        # 4. Asymmetric market (more sellers)
        'asymmetric_sellers': {
            'game_type': 1111,
            'num_buyers': 4,
            'num_sellers': 6,
            'num_tokens': 3,
            'description': 'Seller surplus (4v6)',
        },

        # 5. High competition (many agents)
        'high_competition': {
            'game_type': 1111,
            'num_buyers': 8,
            'num_sellers': 8,
            'num_tokens': 2,
            'description': 'High competition (8v8)',
        },

        # 6. Multi-token (more trading opportunities)
        'multi_token': {
            'game_type': 1111,
            'num_buyers': 4,
            'num_sellers': 4,
            'num_tokens': 5,
            'description': 'Multi-token (5 per agent)',
        },

        # 7. Tight equilibrium (narrow spread)
        'tight_equilibrium': {
            'game_type': 3333,
            'num_buyers': 4,
            'num_sellers': 4,
            'num_tokens': 3,
            'description': 'Tight equilibrium (narrow spread)',
        },

        # 8. Wide equilibrium (large spread)
        'wide_equilibrium': {
            'game_type': 4444,
            'num_buyers': 4,
            'num_sellers': 4,
            'num_tokens': 3,
            'description': 'Wide equilibrium (large spread)',
        },

        # 9. Self-play (all Perry)
        'self_play': {
            'game_type': 1111,
            'num_buyers': 5,
            'num_sellers': 5,
            'num_tokens': 3,
            'description': 'Self-play (Perry vs Perry)',
        },

        # 10. vs ZIC baseline (Perry invasibility)
        'vs_zic': {
            'game_type': 1111,
            'num_buyers': 4,
            'num_sellers': 4,
            'num_tokens': 3,
            'description': '1v7 invasibility (Perry vs ZIC)',
        },
    }

    return scenarios[scenario_name]


# ============================================================================
# SCENARIO RUNNERS
# ============================================================================

def run_self_play_scenario(config: Dict, scenario: Dict, num_replications: int = 10, seed: int = 1000) -> Dict:
    """Test Perry self-play efficiency."""
    token_gen = TokenGenerator(
        game_type=scenario['game_type'],
        num_tokens=scenario['num_tokens'],
        seed=seed
    )

    efficiencies = []
    buyer_shares = []

    for rep in range(num_replications):
        token_gen.new_round()

        # Generate tokens
        buyer_tokens = [token_gen.generate_tokens(is_buyer=True) for _ in range(scenario['num_buyers'])]
        seller_tokens = [token_gen.generate_tokens(is_buyer=False) for _ in range(scenario['num_sellers'])]

        # Create Perry agents
        buyers = [
            Perry(
                i+1, True, scenario['num_tokens'], buyer_tokens[i], 0, 200,
                num_buyers=scenario['num_buyers'],
                num_sellers=scenario['num_sellers'],
                num_times=100,
                seed=seed+rep+i,
                **{k: v for k, v in config.items() if k != 'name'}
            )
            for i in range(scenario['num_buyers'])
        ]

        sellers = [
            Perry(
                i+scenario['num_buyers']+1, False, scenario['num_tokens'], seller_tokens[i], 0, 200,
                num_buyers=scenario['num_buyers'],
                num_sellers=scenario['num_sellers'],
                num_times=100,
                seed=seed+rep+i+100,
                **{k: v for k, v in config.items() if k != 'name'}
            )
            for i in range(scenario['num_sellers'])
        ]

        # Run market
        market = Market(
            num_buyers=scenario['num_buyers'],
            num_sellers=scenario['num_sellers'],
            num_times=100,
            price_min=0,
            price_max=200,
            buyers=buyers,
            sellers=sellers
        )

        for _ in range(100):
            market.run_time_step()

        # Calculate efficiency
        trades = extract_trades_from_orderbook(market.orderbook, 100)
        # Note: orderbook uses 1-based indexing for both buyers and sellers separately
        buyer_valuations = {i+1: buyers[i].valuations for i in range(scenario['num_buyers'])}
        seller_costs = {i+1: sellers[i].valuations for i in range(scenario['num_sellers'])}

        actual_surplus = calculate_actual_surplus(trades, buyer_valuations, seller_costs)
        max_surplus = calculate_max_surplus(
            [b.valuations for b in buyers],
            [s.valuations for s in sellers]
        )

        if max_surplus > 0:
            efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)
            efficiencies.append(efficiency)

        # Calculate balance
        buyer_profit = sum(b.period_profit for b in buyers)
        seller_profit = sum(s.period_profit for s in sellers)
        total_profit = buyer_profit + seller_profit
        if total_profit > 0:
            buyer_shares.append(buyer_profit / total_profit * 100)

    return {
        'efficiency_mean': np.mean(efficiencies),
        'efficiency_std': np.std(efficiencies),
        'balance_mean': abs(50 - np.mean(buyer_shares)),  # Deviation from 50/50
        'balance_std': np.std(buyer_shares),
    }


def run_invasibility_scenario(config: Dict, scenario: Dict, is_buyer: bool, num_replications: int = 10, seed: int = 1000) -> Dict:
    """Test Perry invasibility vs ZIC."""
    token_gen = TokenGenerator(
        game_type=scenario['game_type'],
        num_tokens=scenario['num_tokens'],
        seed=seed
    )

    test_profits = []
    zic_profits = []
    profit_shares = []

    for rep in range(num_replications):
        token_gen.new_round()

        # Generate tokens
        buyer_tokens = [token_gen.generate_tokens(is_buyer=True) for _ in range(4)]
        seller_tokens = [token_gen.generate_tokens(is_buyer=False) for _ in range(4)]

        # Create agents
        if is_buyer:
            buyers = [
                Perry(
                    1, True, scenario['num_tokens'], buyer_tokens[0], 0, 200,
                    num_buyers=4, num_sellers=4, num_times=100,
                    seed=seed+rep,
                    **{k: v for k, v in config.items() if k != 'name'}
                )
            ] + [
                ZIC(i+2, True, scenario['num_tokens'], buyer_tokens[i+1], 0, 200, seed=seed+rep+i+1)
                for i in range(3)
            ]
            sellers = [
                ZIC(i+5, False, scenario['num_tokens'], seller_tokens[i], 0, 200, seed=seed+rep+i+10)
                for i in range(4)
            ]
        else:
            buyers = [
                ZIC(i+1, True, scenario['num_tokens'], buyer_tokens[i], 0, 200, seed=seed+rep+i)
                for i in range(4)
            ]
            sellers = [
                Perry(
                    5, False, scenario['num_tokens'], seller_tokens[0], 0, 200,
                    num_buyers=4, num_sellers=4, num_times=100,
                    seed=seed+rep+10,
                    **{k: v for k, v in config.items() if k != 'name'}
                )
            ] + [
                ZIC(i+6, False, scenario['num_tokens'], seller_tokens[i+1], 0, 200, seed=seed+rep+i+11)
                for i in range(3)
            ]

        # Run market
        market = Market(num_buyers=4, num_sellers=4, num_times=100, price_min=0, price_max=200, buyers=buyers, sellers=sellers)
        for _ in range(100):
            market.run_time_step()

        # Extract profits
        if is_buyer:
            test_profit = buyers[0].period_profit
            zic_profit = np.mean([b.period_profit for b in buyers[1:]])
            total_side = sum(b.period_profit for b in buyers)
        else:
            test_profit = sellers[0].period_profit
            zic_profit = np.mean([s.period_profit for s in sellers[1:]])
            total_side = sum(s.period_profit for s in sellers)

        test_profits.append(test_profit)
        zic_profits.append(zic_profit)
        if total_side > 0:
            profit_shares.append(test_profit / total_side * 100)

    profit_ratios = [tp / zp if zp > 0 else 0 for tp, zp in zip(test_profits, zic_profits)]

    return {
        'profit_ratio_mean': np.mean(profit_ratios),
        'profit_share_mean': np.mean(profit_shares),
        'test_profit_mean': np.mean(test_profits),
    }


def run_mixed_scenario(config: Dict, scenario: Dict, opponent: str, num_replications: int = 10, seed: int = 1000) -> Dict:
    """Test Perry vs sophisticated traders (GD, Kaplan, ZIP)."""
    token_gen = TokenGenerator(
        game_type=scenario['game_type'],
        num_tokens=scenario['num_tokens'],
        seed=seed
    )

    opponent_class = {'gd': GD, 'kaplan': Kaplan, 'zip': ZIP}[opponent]

    perry_wins = 0
    perry_profits = []
    opponent_profits = []

    for rep in range(num_replications):
        token_gen.new_round()

        buyer_tokens = [token_gen.generate_tokens(is_buyer=True) for _ in range(4)]
        seller_tokens = [token_gen.generate_tokens(is_buyer=False) for _ in range(4)]

        # 2 Perry buyers + 2 opponent buyers vs 2 Perry sellers + 2 opponent sellers
        buyers = [
            Perry(1, True, scenario['num_tokens'], buyer_tokens[0], 0, 200,
                  num_buyers=4, num_sellers=4, num_times=100, seed=seed+rep,
                  **{k: v for k, v in config.items() if k != 'name'}),
            Perry(2, True, scenario['num_tokens'], buyer_tokens[1], 0, 200,
                  num_buyers=4, num_sellers=4, num_times=100, seed=seed+rep+1,
                  **{k: v for k, v in config.items() if k != 'name'}),
        ]

        # Add opponent buyers
        if opponent == 'kaplan':
            buyers.extend([
                opponent_class(3, True, scenario['num_tokens'], buyer_tokens[2], 0, 200,
                              num_buyers=4, num_sellers=4, num_times=100, seed=seed+rep+2),
                opponent_class(4, True, scenario['num_tokens'], buyer_tokens[3], 0, 200,
                              num_buyers=4, num_sellers=4, num_times=100, seed=seed+rep+3),
            ])
        elif opponent == 'zip':
            buyers.extend([
                opponent_class(3, True, scenario['num_tokens'], buyer_tokens[2], 0, 200, seed=seed+rep+2,
                              beta=0.2, gamma=0.25, initial_margin_buy=-0.30),
                opponent_class(4, True, scenario['num_tokens'], buyer_tokens[3], 0, 200, seed=seed+rep+3,
                              beta=0.2, gamma=0.25, initial_margin_buy=-0.30),
            ])
        else:  # GD
            buyers.extend([
                opponent_class(3, True, scenario['num_tokens'], buyer_tokens[2], 0, 200, seed=seed+rep+2),
                opponent_class(4, True, scenario['num_tokens'], buyer_tokens[3], 0, 200, seed=seed+rep+3),
            ])

        # Create sellers (same pattern)
        sellers = [
            Perry(5, False, scenario['num_tokens'], seller_tokens[0], 0, 200,
                  num_buyers=4, num_sellers=4, num_times=100, seed=seed+rep+10,
                  **{k: v for k, v in config.items() if k != 'name'}),
            Perry(6, False, scenario['num_tokens'], seller_tokens[1], 0, 200,
                  num_buyers=4, num_sellers=4, num_times=100, seed=seed+rep+11,
                  **{k: v for k, v in config.items() if k != 'name'}),
        ]

        if opponent == 'kaplan':
            sellers.extend([
                opponent_class(7, False, scenario['num_tokens'], seller_tokens[2], 0, 200,
                              num_buyers=4, num_sellers=4, num_times=100, seed=seed+rep+12),
                opponent_class(8, False, scenario['num_tokens'], seller_tokens[3], 0, 200,
                              num_buyers=4, num_sellers=4, num_times=100, seed=seed+rep+13),
            ])
        elif opponent == 'zip':
            sellers.extend([
                opponent_class(7, False, scenario['num_tokens'], seller_tokens[2], 0, 200, seed=seed+rep+12,
                              beta=0.2, gamma=0.25, initial_margin_sell=0.30),
                opponent_class(8, False, scenario['num_tokens'], seller_tokens[3], 0, 200, seed=seed+rep+13,
                              beta=0.2, gamma=0.25, initial_margin_sell=0.30),
            ])
        else:  # GD
            sellers.extend([
                opponent_class(7, False, scenario['num_tokens'], seller_tokens[2], 0, 200, seed=seed+rep+12),
                opponent_class(8, False, scenario['num_tokens'], seller_tokens[3], 0, 200, seed=seed+rep+13),
            ])

        # Run market
        market = Market(num_buyers=4, num_sellers=4, num_times=100, price_min=0, price_max=200, buyers=buyers, sellers=sellers)
        for _ in range(100):
            market.run_time_step()

        # Calculate profits
        perry_profit = sum(b.period_profit for b in buyers[:2]) + sum(s.period_profit for s in sellers[:2])
        opponent_profit = sum(b.period_profit for b in buyers[2:]) + sum(s.period_profit for s in sellers[2:])

        perry_profits.append(perry_profit)
        opponent_profits.append(opponent_profit)

        if perry_profit > opponent_profit:
            perry_wins += 1

    return {
        'win_rate': perry_wins / num_replications * 100,
        'profit_mean': np.mean(perry_profits),
        'opponent_profit_mean': np.mean(opponent_profits),
        'profit_ratio': np.mean(perry_profits) / np.mean(opponent_profits) if np.mean(opponent_profits) > 0 else 0,
    }


# ============================================================================
# COMPOSITE SCORING
# ============================================================================

def calculate_composite_score(results: Dict) -> float:
    """
    Calculate composite performance score.

    Weights:
    - 30% self-play efficiency (measures cooperation)
    - 25% profit share vs ZIC (measures exploitation resistance)
    - 20% invasibility ratio (measures competitiveness)
    - 15% balance (measures fairness)
    - 10% win rate vs sophisticated (measures robustness)
    """

    # Normalize metrics to 0-100 scale
    efficiency_score = results['self_play']['efficiency_mean']  # Already 0-100

    profit_share = (results['invasibility_buyer']['profit_share_mean'] +
                    results['invasibility_seller']['profit_share_mean']) / 2
    profit_score = min(profit_share / 50 * 100, 100)  # 50% share = 100 points

    invasibility = (results['invasibility_buyer']['profit_ratio_mean'] +
                    results['invasibility_seller']['profit_ratio_mean']) / 2
    invasibility_score = min(invasibility / 5 * 100, 100)  # 5x ratio = 100 points

    balance_score = 100 - results['self_play']['balance_mean']  # Lower deviation = better

    win_rate = (results['vs_gd']['win_rate'] + results['vs_kaplan']['win_rate'] +
                results['vs_zip']['win_rate']) / 3
    win_score = win_rate  # Already 0-100

    # Weighted composite
    composite = (
        0.30 * efficiency_score +
        0.25 * profit_score +
        0.20 * invasibility_score +
        0.15 * balance_score +
        0.10 * win_score
    )

    return composite


# ============================================================================
# MAIN CALIBRATION RUNNER
# ============================================================================

def run_calibration(num_replications: int = 10, base_seed: int = 1000, verbose: bool = True):
    """Run full calibration grid search."""

    all_results = []

    for i, config in enumerate(PARAMETER_GRID):
        if verbose:
            print(f"\n{'='*80}")
            print(f"CONFIG {i+1}/{len(PARAMETER_GRID)}: {config['name']}")
            print(f"  a0_initial={config['a0_initial']}, desperate_threshold={config['desperate_threshold']}, desperate_margin={config['desperate_margin']}")
            print(f"{'='*80}")

        config_results = {'config': config, 'scenarios': {}}

        # Run all scenarios
        scenarios = ['symmetric', 'flat', 'asymmetric_buyers', 'asymmetric_sellers',
                    'high_competition', 'multi_token', 'tight_equilibrium', 'wide_equilibrium']

        for scenario_name in scenarios:
            if verbose:
                print(f"\n  Scenario: {scenario_name}...", end='')

            scenario = get_scenario_config(scenario_name)

            if scenario_name == 'self_play':
                result = run_self_play_scenario(config, scenario, num_replications, base_seed)
            elif scenario_name == 'vs_zic':
                # Run both buyer and seller invasibility
                buyer_result = run_invasibility_scenario(config, scenario, True, num_replications, base_seed)
                seller_result = run_invasibility_scenario(config, scenario, False, num_replications, base_seed + 10000)
                result = {'buyer': buyer_result, 'seller': seller_result}
            else:
                result = run_self_play_scenario(config, scenario, num_replications, base_seed)

            config_results['scenarios'][scenario_name] = result

            if verbose:
                print(" ✓")

        # Run self-play test
        if verbose:
            print(f"\n  Self-play test...", end='')
        scenario = get_scenario_config('self_play')
        config_results['self_play'] = run_self_play_scenario(config, scenario, num_replications, base_seed + 20000)
        if verbose:
            print(" ✓")

        # Run invasibility tests
        if verbose:
            print(f"  Invasibility (buyer)...", end='')
        scenario = get_scenario_config('vs_zic')
        config_results['invasibility_buyer'] = run_invasibility_scenario(config, scenario, True, num_replications, base_seed + 30000)
        if verbose:
            print(" ✓")

        if verbose:
            print(f"  Invasibility (seller)...", end='')
        config_results['invasibility_seller'] = run_invasibility_scenario(config, scenario, False, num_replications, base_seed + 40000)
        if verbose:
            print(" ✓")

        # Run vs sophisticated traders
        for opponent in ['gd', 'kaplan', 'zip']:
            if verbose:
                print(f"  vs {opponent.upper()}...", end='')
            scenario = get_scenario_config('symmetric')
            config_results[f'vs_{opponent}'] = run_mixed_scenario(config, scenario, opponent, num_replications, base_seed + 50000 + hash(opponent) % 10000)
            if verbose:
                print(" ✓")

        # Calculate composite score
        composite = calculate_composite_score(config_results)
        config_results['composite_score'] = composite

        all_results.append(config_results)

        if verbose:
            print(f"\n  → Composite Score: {composite:.2f}/100")

    return all_results


def print_results(all_results: List[Dict]):
    """Print formatted calibration results."""

    # Sort by composite score
    ranked = sorted(all_results, key=lambda x: x['composite_score'], reverse=True)

    print("\n" + "="*80)
    print("PERRY HYPERPARAMETER CALIBRATION RESULTS")
    print("="*80)
    print(f"\nTested {len(PARAMETER_GRID)} configurations across 10 scenarios")
    print("\nScoring Weights: 30% efficiency, 25% profit share, 20% invasibility, 15% balance, 10% win rate")
    print("\n" + "-"*80)
    print(f"{'Rank':<5} {'Config':<25} {'Score':<8} {'Efficiency':<12} {'Invasibility':<12} {'Balance':<10}")
    print("-"*80)

    for i, result in enumerate(ranked[:10]):
        config = result['config']
        score = result['composite_score']
        efficiency = result['self_play']['efficiency_mean']
        invasibility = (result['invasibility_buyer']['profit_ratio_mean'] +
                       result['invasibility_seller']['profit_ratio_mean']) / 2
        balance = result['self_play']['balance_mean']

        print(f"{i+1:<5} {config['name']:<25} {score:>6.2f}   {efficiency:>6.2f}%      {invasibility:>6.2f}x       {balance:>6.2f}")

    print("\n" + "="*80)
    print("TOP PROFILES")
    print("="*80)

    # Best overall (highest composite)
    best = ranked[0]
    print(f"\n1. ROBUST (Best Overall):")
    print(f"   Config: {best['config']['name']}")
    print(f"   Parameters: a0={best['config']['a0_initial']}, threshold={best['config']['desperate_threshold']}, margin={best['config']['desperate_margin']}")
    print(f"   Composite Score: {best['composite_score']:.2f}/100")
    print(f"   Self-Play Efficiency: {best['self_play']['efficiency_mean']:.2f}%")
    print(f"   Invasibility: {(best['invasibility_buyer']['profit_ratio_mean'] + best['invasibility_seller']['profit_ratio_mean'])/2:.2f}x")

    # Best profit extraction
    best_profit = max(ranked, key=lambda x: (x['invasibility_buyer']['profit_ratio_mean'] + x['invasibility_seller']['profit_ratio_mean']) / 2)
    print(f"\n2. PROFIT (Best Invasibility):")
    print(f"   Config: {best_profit['config']['name']}")
    print(f"   Parameters: a0={best_profit['config']['a0_initial']}, threshold={best_profit['config']['desperate_threshold']}, margin={best_profit['config']['desperate_margin']}")
    print(f"   Invasibility: {(best_profit['invasibility_buyer']['profit_ratio_mean'] + best_profit['invasibility_seller']['profit_ratio_mean'])/2:.2f}x")
    print(f"   Composite Score: {best_profit['composite_score']:.2f}/100")

    # Best cooperative (highest efficiency + lowest balance deviation)
    best_coop = max(ranked, key=lambda x: x['self_play']['efficiency_mean'] - x['self_play']['balance_mean'])
    print(f"\n3. COOPERATIVE (Best Efficiency + Balance):")
    print(f"   Config: {best_coop['config']['name']}")
    print(f"   Parameters: a0={best_coop['config']['a0_initial']}, threshold={best_coop['config']['desperate_threshold']}, margin={best_coop['config']['desperate_margin']}")
    print(f"   Self-Play Efficiency: {best_coop['self_play']['efficiency_mean']:.2f}%")
    print(f"   Balance Deviation: {best_coop['self_play']['balance_mean']:.2f}")
    print(f"   Composite Score: {best_coop['composite_score']:.2f}/100")

    print("\n" + "="*80)


def save_results(all_results: List[Dict], output_file: str):
    """Save detailed results to JSON."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Calibrate Perry hyperparameters')
    parser.add_argument('--replications', type=int, default=10,
                       help='Number of replications per scenario')
    parser.add_argument('--seed', type=int, default=1000,
                       help='Base random seed')
    parser.add_argument('--output', type=str, default='results/perry_calibration_v1.json',
                       help='Output file for detailed results')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("PERRY HYPERPARAMETER CALIBRATION v1.0")
    print("="*80)
    print(f"\nConfigurations: {len(PARAMETER_GRID)}")
    print(f"Replications per scenario: {args.replications}")
    print(f"Base seed: {args.seed}")
    print("\nStarting calibration...")

    # Run calibration
    all_results = run_calibration(
        num_replications=args.replications,
        base_seed=args.seed,
        verbose=not args.quiet
    )

    # Print results
    print_results(all_results)

    # Save results
    save_results(all_results, args.output)


if __name__ == '__main__':
    main()
