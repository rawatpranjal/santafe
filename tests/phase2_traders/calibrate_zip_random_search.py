#!/usr/bin/env python
"""
Random search calibration for ZIP trader hyperparameters.

This explores a much wider continuous parameter space than grid search,
testing 200 random configurations to find optimal settings for both
efficiency and profit extraction objectives.

Key improvements over grid search:
1. Continuous parameter space (not discrete grid points)
2. Explores extreme regions (very high/low learning rates)
3. Tests new parameters (r_delta, a_delta perturbations)
4. Finds Pareto frontier of efficiency vs profit trade-off
"""

import numpy as np
import polars as pl
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from engine.market import Market
from engine.agent_factory import create_agent
from engine.token_generator import TokenGenerator
from engine.efficiency import calculate_allocative_efficiency


def generate_random_config(rng: np.random.Generator) -> Dict:
    """
    Generate a random ZIP configuration from expanded parameter space.

    Args:
        rng: Random number generator

    Returns:
        Configuration dictionary
    """
    config = {
        'beta': rng.uniform(0.01, 1.0),      # Learning rate: very slow to very fast
        'gamma': rng.uniform(0.0, 0.5),       # Momentum: zero to high
        'margin': rng.uniform(0.0, 0.5),      # Initial margin: zero to 50%
        'r_delta': rng.uniform(0.01, 0.20),   # R perturbation: 1% to 20%
        'a_delta': rng.uniform(0.05, 0.20),   # A perturbation: 5% to 20%
    }
    return config


def run_scenario(
    config: Dict,
    scenario_type: str,
    n_rounds: int = 10,
    seed: int = 42
) -> Dict:
    """
    Run a single scenario with given ZIP configuration.

    Args:
        config: ZIP hyperparameter configuration
        scenario_type: Type of scenario ('self_play', 'vs_zic', 'invasibility')
        n_rounds: Number of market rounds
        seed: Random seed

    Returns:
        Results dictionary
    """
    rng = np.random.default_rng(seed)
    token_gen = TokenGenerator(game_type=1232, num_tokens=1, seed=seed)

    # Market configuration
    if scenario_type == 'invasibility':
        n_buyers = 4
        n_sellers = 4
    else:
        n_buyers = 5
        n_sellers = 5

    n_periods = 10
    n_agents = n_buyers + n_sellers

    # Track metrics
    efficiencies = []
    total_profits = []
    zip_profits = []
    zic_profits = []

    for round_idx in range(n_rounds):
        # Generate new round parameters
        token_gen.new_round()

        # Generate tokens for each agent
        buyer_tokens = []
        seller_tokens = []

        for i in range(n_buyers):
            tokens = token_gen.generate_tokens(is_buyer=True)
            buyer_tokens.append({
                'num_tokens': len(tokens),
                'valuations': tokens
            })

        for i in range(n_sellers):
            tokens = token_gen.generate_tokens(is_buyer=False)
            seller_tokens.append({
                'num_tokens': len(tokens),
                'valuations': tokens
            })

        # Create agents based on scenario
        agents = []

        if scenario_type == 'self_play':
            # All ZIP agents
            for i in range(n_buyers):
                agent = create_agent(
                    agent_type='ZIP',
                    player_id=i + 1,
                    is_buyer=True,
                    num_tokens=buyer_tokens[i]['num_tokens'],
                    valuations=buyer_tokens[i]['valuations'],
                    price_min=0,
                    price_max=500,
                    seed=seed + round_idx * 100 + i,
                    **config  # Pass ZIP config
                )
                agents.append(agent)

            for i in range(n_sellers):
                agent = create_agent(
                    agent_type='ZIP',
                    player_id=n_buyers + i + 1,
                    is_buyer=False,
                    num_tokens=seller_tokens[i]['num_tokens'],
                    valuations=seller_tokens[i]['valuations'],
                    price_min=0,
                    price_max=500,
                    seed=seed + round_idx * 100 + n_buyers + i,
                    **config  # Pass ZIP config
                )
                agents.append(agent)

        elif scenario_type == 'vs_zic':
            # 50/50 ZIP vs ZIC
            for i in range(n_buyers):
                agent_type = 'ZIP' if i < n_buyers // 2 else 'ZIC'
                kwargs = config if agent_type == 'ZIP' else {}
                agent = create_agent(
                    agent_type=agent_type,
                    player_id=i + 1,
                    is_buyer=True,
                    num_tokens=buyer_tokens[i]['num_tokens'],
                    valuations=buyer_tokens[i]['valuations'],
                    price_min=0,
                    price_max=500,
                    seed=seed + round_idx * 100 + i,
                    **kwargs
                )
                agents.append(agent)

            for i in range(n_sellers):
                agent_type = 'ZIP' if i < n_sellers // 2 else 'ZIC'
                kwargs = config if agent_type == 'ZIP' else {}
                agent = create_agent(
                    agent_type=agent_type,
                    player_id=n_buyers + i + 1,
                    is_buyer=False,
                    num_tokens=seller_tokens[i]['num_tokens'],
                    valuations=seller_tokens[i]['valuations'],
                    price_min=0,
                    price_max=500,
                    seed=seed + round_idx * 100 + n_buyers + i,
                    **kwargs
                )
                agents.append(agent)

        elif scenario_type == 'invasibility':
            # 1 ZIP vs 7 ZIC
            for i in range(n_buyers):
                agent_type = 'ZIP' if i == 0 else 'ZIC'  # First buyer is ZIP
                kwargs = config if agent_type == 'ZIP' else {}
                agent = create_agent(
                    agent_type=agent_type,
                    player_id=i + 1,
                    is_buyer=True,
                    num_tokens=buyer_tokens[i]['num_tokens'],
                    valuations=buyer_tokens[i]['valuations'],
                    price_min=0,
                    price_max=500,
                    seed=seed + round_idx * 100 + i,
                    **kwargs
                )
                agents.append(agent)

            for i in range(n_sellers):
                # All sellers are ZIC in this scenario
                agent = create_agent(
                    agent_type='ZIC',
                    player_id=n_buyers + i + 1,
                    is_buyer=False,
                    num_tokens=seller_tokens[i]['num_tokens'],
                    valuations=seller_tokens[i]['valuations'],
                    price_min=0,
                    price_max=500,
                    seed=seed + round_idx * 100 + n_buyers + i
                )
                agents.append(agent)

        # Calculate equilibrium values
        all_buyer_vals = []
        all_seller_vals = []
        for i in range(n_buyers):
            all_buyer_vals.extend(buyer_tokens[i]['valuations'])
        for i in range(n_sellers):
            all_seller_vals.extend(seller_tokens[i]['valuations'])

        all_buyer_vals.sort(reverse=True)
        all_seller_vals.sort()
        max_surplus = sum(
            v - c for v, c in zip(all_buyer_vals, all_seller_vals)
            if v > c
        )

        # Create and run market
        market = Market(
            num_buyers=n_buyers,
            num_sellers=n_sellers,
            num_periods=n_periods,
            auction_type="AURORA"
        )

        # Initialize agents
        for agent in agents:
            agent.inform_auction_params(
                auction_type="AURORA",
                n_buyers=n_buyers,
                n_sellers=n_sellers,
                n_periods=n_periods
            )

        # Run market
        market.reset()
        trades = []

        for period in range(n_periods):
            market.open_period()

            while not market.period_ended:
                # Bid/ask stage
                bids = []
                asks = []

                for agent in agents:
                    if agent.is_buyer:
                        bid = agent.bid_ask()
                        if bid is not None:
                            bids.append((agent.player_id, bid))
                    else:
                        ask = agent.bid_ask()
                        if ask is not None:
                            asks.append((agent.player_id, ask))

                bid_results = market.bid_ask_stage(bids, asks)

                # Send results to agents
                for agent_id, result in bid_results.items():
                    agent = agents[agent_id - 1]
                    agent.bid_ask_response(**result)

                # Buy/sell stage
                buys = []
                sells = []

                for agent in agents:
                    if agent.is_buyer:
                        buy = agent.buy_sell()
                        if buy == 1:
                            buys.append(agent.player_id)
                    else:
                        sell = agent.buy_sell()
                        if sell == 1:
                            sells.append(agent.player_id)

                buy_results = market.buy_sell_stage(buys, sells)

                # Send results and record trades
                for agent_id, result in buy_results.items():
                    agent = agents[agent_id - 1]
                    agent.buy_sell_response(**result)

                    if result.get('trans', 0) == 1:
                        trades.append({
                            'buyer_id': result.get('buyer_id'),
                            'seller_id': result.get('seller_id'),
                            'price': result.get('price', 0),
                            'buyer_value': buyer_tokens[result['buyer_id'] - 1]['valuations'][0] if result.get('buyer_id') and result['buyer_id'] <= n_buyers else 0,
                            'seller_cost': seller_tokens[result['seller_id'] - n_buyers - 1]['valuations'][0] if result.get('seller_id') and result['seller_id'] > n_buyers else 0,
                        })

                market.advance_time()

        # Calculate efficiency
        if max_surplus > 0:
            efficiency = calculate_allocative_efficiency(trades, max_surplus)
            efficiencies.append(efficiency)

        # Calculate profits by agent type
        agent_profits = {i: 0.0 for i in range(1, n_agents + 1)}
        agent_types = {}

        for i, agent in enumerate(agents):
            agent_types[i + 1] = type(agent).__name__

        for trade in trades:
            if trade['buyer_id']:
                buyer_profit = trade['buyer_value'] - trade['price']
                agent_profits[trade['buyer_id']] += buyer_profit

            if trade['seller_id']:
                seller_profit = trade['price'] - trade['seller_cost']
                agent_profits[trade['seller_id']] += seller_profit

        # Aggregate by type
        zip_total = sum(p for pid, p in agent_profits.items() if 'ZIP' in agent_types.get(pid, ''))
        zic_total = sum(p for pid, p in agent_profits.items() if 'ZIC' in agent_types.get(pid, ''))

        zip_profits.append(zip_total)
        if zic_total > 0:
            zic_profits.append(zic_total)

        total_profits.append(sum(agent_profits.values()))

    # Calculate summary metrics
    results = {
        'scenario': scenario_type,
        'efficiency_mean': np.mean(efficiencies) if efficiencies else 0,
        'efficiency_std': np.std(efficiencies) if efficiencies else 0,
        'total_profit_mean': np.mean(total_profits),
        'zip_profit_mean': np.mean(zip_profits) if zip_profits else 0,
        'zic_profit_mean': np.mean(zic_profits) if zic_profits else 0,
    }

    # Calculate profit metrics based on scenario
    if scenario_type == 'vs_zic' and zic_profits:
        n_zip = n_buyers // 2 + n_sellers // 2
        n_zic = n_agents - n_zip
        zip_per_agent = np.mean(zip_profits) / n_zip if n_zip > 0 else 0
        zic_per_agent = np.mean(zic_profits) / n_zic if n_zic > 0 else 0
        results['profit_ratio'] = zip_per_agent / zic_per_agent if zic_per_agent > 0 else 0
        results['profit_share'] = np.mean(zip_profits) / np.mean(total_profits) if np.mean(total_profits) > 0 else 0
    elif scenario_type == 'invasibility' and zic_profits:
        # 1 ZIP vs 7 ZIC
        zip_per_agent = np.mean(zip_profits)  # Only 1 ZIP
        zic_per_agent = np.mean(zic_profits) / 7  # 7 ZIC agents
        results['profit_ratio'] = zip_per_agent / zic_per_agent if zic_per_agent > 0 else 0
        results['profit_share'] = np.mean(zip_profits) / np.mean(total_profits) if np.mean(total_profits) > 0 else 0
    else:
        results['profit_ratio'] = 0
        results['profit_share'] = 0

    return results


def evaluate_config(config: Dict, n_rounds: int = 10, seed: int = 42) -> Dict:
    """
    Evaluate a ZIP configuration across multiple scenarios.

    Args:
        config: ZIP hyperparameter configuration
        n_rounds: Number of rounds per scenario
        seed: Base random seed

    Returns:
        Combined results dictionary
    """
    results = {'config': config}

    # Test self-play
    self_play = run_scenario(config, 'self_play', n_rounds, seed)
    results['self_play_efficiency'] = self_play['efficiency_mean']

    # Test vs ZIC (mixed market)
    vs_zic = run_scenario(config, 'vs_zic', n_rounds, seed + 1000)
    results['vs_zic_efficiency'] = vs_zic['efficiency_mean']
    results['vs_zic_profit_ratio'] = vs_zic['profit_ratio']
    results['vs_zic_profit_share'] = vs_zic['profit_share']

    # Test invasibility (1v7)
    invasibility = run_scenario(config, 'invasibility', n_rounds, seed + 2000)
    results['invasibility_efficiency'] = invasibility['efficiency_mean']
    results['invasibility_profit_ratio'] = invasibility['profit_ratio']
    results['invasibility_profit_share'] = invasibility['profit_share']

    # Calculate composite scores for different objectives
    # Efficiency-focused score (prioritize self-play and efficiency)
    results['efficiency_score'] = (
        0.5 * results['self_play_efficiency'] +
        0.3 * results['vs_zic_efficiency'] +
        0.2 * results['invasibility_efficiency']
    )

    # Profit-focused score (prioritize profit extraction)
    results['profit_score'] = (
        0.4 * results['vs_zic_profit_share'] * 100 +
        0.4 * results['invasibility_profit_share'] * 100 +
        0.2 * min(results['invasibility_profit_ratio'] * 10, 100)  # Cap at 10x
    )

    # Balanced score (equal weight to efficiency and profit)
    results['balanced_score'] = (
        0.5 * results['efficiency_score'] +
        0.5 * results['profit_score']
    )

    return results


def main():
    """Run random search for ZIP calibration."""

    # Configuration
    n_configs = 10  # Number of random configurations to test (reduced for testing)
    n_rounds = 5    # Rounds per scenario (reduced for speed)
    seed = 42

    rng = np.random.default_rng(seed)

    print("=" * 80)
    print("ZIP RANDOM SEARCH CALIBRATION")
    print(f"Testing {n_configs} random configurations")
    print("=" * 80)
    print()

    # Generate and test configurations
    all_results = []

    for i in range(n_configs):
        config = generate_random_config(rng)

        if (i + 1) % 10 == 0:
            print(f"Testing configuration {i+1}/{n_configs}...")

        results = evaluate_config(config, n_rounds, seed + i * 10000)
        all_results.append(results)

    # Convert to DataFrame for analysis
    df = pl.DataFrame(all_results)

    # Find best configurations for each objective
    print("\n" + "=" * 80)
    print("TOP CONFIGURATIONS BY OBJECTIVE")
    print("=" * 80)

    # Best for efficiency
    print("\n### BEST FOR EFFICIENCY (Self-Play Focus) ###")
    best_efficiency = df.sort("efficiency_score", descending=True).head(5)
    for row in best_efficiency.iter_rows(named=True):
        config = row['config']
        print(f"\nScore: {row['efficiency_score']:.1f}")
        print(f"  Self-play: {row['self_play_efficiency']:.1f}%")
        print(f"  vs ZIC: {row['vs_zic_efficiency']:.1f}%")
        print(f"  Config: β={config['beta']:.3f}, γ={config['gamma']:.3f}, "
              f"margin={config['margin']:.3f}, r_δ={config['r_delta']:.3f}, a_δ={config['a_delta']:.3f}")

    # Best for profit extraction
    print("\n### BEST FOR PROFIT EXTRACTION ###")
    best_profit = df.sort("profit_score", descending=True).head(5)
    for row in best_profit.iter_rows(named=True):
        config = row['config']
        print(f"\nScore: {row['profit_score']:.1f}")
        print(f"  vs ZIC ratio: {row['vs_zic_profit_ratio']:.2f}x")
        print(f"  1v7 ratio: {row['invasibility_profit_ratio']:.2f}x")
        print(f"  1v7 share: {row['invasibility_profit_share']*100:.1f}%")
        print(f"  Config: β={config['beta']:.3f}, γ={config['gamma']:.3f}, "
              f"margin={config['margin']:.3f}, r_δ={config['r_delta']:.3f}, a_δ={config['a_delta']:.3f}")

    # Best balanced
    print("\n### BEST BALANCED (Efficiency + Profit) ###")
    best_balanced = df.sort("balanced_score", descending=True).head(5)
    for row in best_balanced.iter_rows(named=True):
        config = row['config']
        print(f"\nScore: {row['balanced_score']:.1f}")
        print(f"  Self-play: {row['self_play_efficiency']:.1f}%")
        print(f"  1v7 ratio: {row['invasibility_profit_ratio']:.2f}x")
        print(f"  Config: β={config['beta']:.3f}, γ={config['gamma']:.3f}, "
              f"margin={config['margin']:.3f}, r_δ={config['r_delta']:.3f}, a_δ={config['a_delta']:.3f}")

    # Compare to existing baselines
    print("\n" + "=" * 80)
    print("COMPARISON TO EXISTING CONFIGURATIONS")
    print("=" * 80)

    # Test current v2 (profit-optimized)
    v2_config = {
        'beta': 0.2,
        'gamma': 0.25,
        'margin': 0.30,
        'r_delta': 0.05,  # Default from paper
        'a_delta': 0.10,  # Default from paper
    }
    v2_results = evaluate_config(v2_config, n_rounds, seed + 999999)

    print(f"\n### Current v2 (margin=0.30) ###")
    print(f"  Efficiency score: {v2_results['efficiency_score']:.1f}")
    print(f"  Profit score: {v2_results['profit_score']:.1f}")
    print(f"  Balanced score: {v2_results['balanced_score']:.1f}")
    print(f"  1v7 profit ratio: {v2_results['invasibility_profit_ratio']:.2f}x")

    # Test current v3 (efficiency-optimized)
    v3_config = {
        'beta': 0.1,
        'gamma': 0.15,
        'margin': 0.05,
        'r_delta': 0.05,
        'a_delta': 0.10,
    }
    v3_results = evaluate_config(v3_config, n_rounds, seed + 999998)

    print(f"\n### Current v3 (margin=0.05) ###")
    print(f"  Efficiency score: {v3_results['efficiency_score']:.1f}")
    print(f"  Profit score: {v3_results['profit_score']:.1f}")
    print(f"  Balanced score: {v3_results['balanced_score']:.1f}")
    print(f"  Self-play efficiency: {v3_results['self_play_efficiency']:.1f}%")

    # Parameter analysis
    print("\n" + "=" * 80)
    print("PARAMETER IMPORTANCE ANALYSIS")
    print("=" * 80)

    # Correlation analysis
    for param in ['beta', 'gamma', 'margin', 'r_delta', 'a_delta']:
        param_values = [r['config'][param] for r in all_results]

        # Correlation with efficiency
        eff_corr = np.corrcoef(param_values, df['efficiency_score'].to_list())[0, 1]

        # Correlation with profit
        profit_corr = np.corrcoef(param_values, df['profit_score'].to_list())[0, 1]

        print(f"\n{param}:")
        print(f"  Correlation with efficiency: {eff_corr:+.3f}")
        print(f"  Correlation with profit: {profit_corr:+.3f}")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "results" / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results
    df.write_csv(output_dir / f"zip_random_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    # Save top configurations
    best_configs = {
        'efficiency_top5': best_efficiency.to_dicts(),
        'profit_top5': best_profit.to_dicts(),
        'balanced_top5': best_balanced.to_dicts(),
        'v2_baseline': v2_results,
        'v3_baseline': v3_results,
    }

    with open(output_dir / f"zip_best_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        json.dump(convert(best_configs), f, indent=2)

    print(f"\n\nResults saved to: {output_dir}")

    # Final recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    best_overall = df.sort("balanced_score", descending=True).row(0, named=True)
    if best_overall['balanced_score'] > max(v2_results['balanced_score'], v3_results['balanced_score']):
        print(f"\n✅ Found configuration better than existing baselines!")
        print(f"   Balanced score: {best_overall['balanced_score']:.1f} "
              f"(vs v2: {v2_results['balanced_score']:.1f}, v3: {v3_results['balanced_score']:.1f})")
        config = best_overall['config']
        print(f"\n   Recommended config:")
        print(f"     beta: {config['beta']:.3f}")
        print(f"     gamma: {config['gamma']:.3f}")
        print(f"     margin: {config['margin']:.3f}")
        print(f"     r_delta: {config['r_delta']:.3f}")
        print(f"     a_delta: {config['a_delta']:.3f}")
    else:
        print(f"\n⚠️  Existing configurations remain optimal")
        print(f"   Use v2 (margin=0.30) for profit extraction")
        print(f"   Use v3 (margin=0.05) for efficiency")


if __name__ == "__main__":
    main()