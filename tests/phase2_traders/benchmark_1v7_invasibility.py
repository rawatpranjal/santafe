"""
1v7 Invasibility Benchmark - Individual Agent Performance Test

Tests whether a single agent with a different strategy can "invade" a market
dominated by ZIC agents and extract higher profits.

Methodology:
- Market: 4 buyers vs 4 sellers (8 agents total)
- Test: 1 test agent + 3 ZIC on one side, 4 ZIC on other side
- Replications: 10 runs with different random tokens
- Metrics: profit_ratio, profit_share, efficiency

This is the STANDARD individual performance test for all traders.
"""

import argparse
from engine.market import Market
from engine.token_generator import TokenGenerator
from traders.legacy.zic import ZIC
from traders.legacy.zi2 import ZI2
from traders.legacy.zip import ZIP
from traders.legacy.gd import GD
from traders.legacy.kaplan import Kaplan
from traders.legacy.perry import Perry
from traders.legacy.lin import Lin
import numpy as np


TRADER_REGISTRY = {
    'zic': ZIC,
    'zi2': ZI2,
    'zip': ZIP,
    'gd': GD,
    'kaplan': Kaplan,
    'perry': Perry,
    'lin': Lin,
}


def run_invasibility_test(
    test_agent_class,
    test_agent_name: str,
    is_buyer: bool = True,
    num_replications: int = 10,
    seed: int = 1000,
    **agent_kwargs
):
    """
    Run 1v7 invasibility test: 1 test agent vs 7 ZIC agents.

    Args:
        test_agent_class: The trader class to test
        test_agent_name: Name for display
        is_buyer: True if test agent is buyer, False if seller
        num_replications: Number of market simulations
        seed: Base random seed
        **agent_kwargs: Additional kwargs for test agent (e.g., beta, gamma for ZIP)

    Returns:
        dict with profit_ratio, profit_share, efficiency metrics
    """

    # Token generator for consistent markets
    token_gen = TokenGenerator(
        game_type=1111,  # Symmetric market
        num_tokens=3,    # 3 tokens per agent
        seed=seed
    )

    results = {
        'test_profits': [],
        'zic_profits': [],
        'test_share': [],
        'efficiencies': [],
    }

    for rep in range(num_replications):
        # Generate new tokens for this replication
        token_gen.new_round()

        # Generate 4 buyer and 4 seller token sets
        buyer_tokens = []
        seller_tokens = []
        for i in range(4):
            buyer_tokens.append(token_gen.generate_tokens(is_buyer=True))
            seller_tokens.append(token_gen.generate_tokens(is_buyer=False))

        # Create agents
        buyers = []
        sellers = []

        # Add market context for Perry and Lin (need num_buyers, num_sellers, num_times)
        if test_agent_class in (Perry, Lin):
            agent_kwargs.update({
                'num_buyers': 4,
                'num_sellers': 4,
                'num_times': 100
            })

        if is_buyer:
            # Test agent is buyer (index 0), rest are ZIC
            buyers.append(test_agent_class(
                1, True, 3, buyer_tokens[0], 0, 200,
                seed=seed+rep, **agent_kwargs
            ))
            for i in range(1, 4):
                buyers.append(ZIC(i+1, True, 3, buyer_tokens[i], 0, 200, seed=seed+rep+i))

            # All sellers are ZIC
            for i in range(4):
                sellers.append(ZIC(i+5, False, 3, seller_tokens[i], 0, 200, seed=seed+rep+i+4))

        else:
            # All buyers are ZIC
            for i in range(4):
                buyers.append(ZIC(i+1, True, 3, buyer_tokens[i], 0, 200, seed=seed+rep+i))

            # Test agent is seller (index 0), rest are ZIC
            sellers.append(test_agent_class(
                5, False, 3, seller_tokens[0], 0, 200,
                seed=seed+rep+4, **agent_kwargs
            ))
            for i in range(1, 4):
                sellers.append(ZIC(i+6, False, 3, seller_tokens[i], 0, 200, seed=seed+rep+i+5))

        # Run market
        market = Market(
            num_buyers=4,
            num_sellers=4,
            num_times=100,
            price_min=0,
            price_max=200,
            buyers=buyers,
            sellers=sellers
        )

        # Run all time steps
        for _ in range(100):
            market.run_time_step()

        # Calculate efficiency
        all_valuations = []
        for b in buyers:
            all_valuations.extend(b.valuations)
        all_valuations.sort(reverse=True)

        all_costs = []
        for s in sellers:
            all_costs.extend(s.valuations)
        all_costs.sort()

        max_surplus = 0
        for v, c in zip(all_valuations, all_costs):
            if v > c:
                max_surplus += (v - c)
            else:
                break

        buyer_profit = sum(b.period_profit for b in buyers)
        seller_profit = sum(s.period_profit for s in sellers)
        actual_surplus = buyer_profit + seller_profit
        efficiency = (actual_surplus / max_surplus) * 100 if max_surplus > 0 else 0

        # Extract profits
        if is_buyer:
            test_agent = buyers[0]
            zic_agents = buyers[1:]
        else:
            test_agent = sellers[0]
            zic_agents = sellers[1:]

        test_profit = test_agent.period_profit
        zic_profits = [agent.period_profit for agent in zic_agents]
        mean_zic_profit = np.mean(zic_profits)
        total_side_profit = test_profit + sum(zic_profits)

        # Store results
        results['test_profits'].append(test_profit)
        results['zic_profits'].append(mean_zic_profit)
        results['test_share'].append(test_profit / total_side_profit * 100 if total_side_profit > 0 else 0)
        results['efficiencies'].append(efficiency)

    # Calculate summary statistics
    profit_ratios = [
        tp / zp if zp > 0 else 0
        for tp, zp in zip(results['test_profits'], results['zic_profits'])
    ]

    summary = {
        'agent': test_agent_name,
        'side': 'buyer' if is_buyer else 'seller',
        'profit_ratio_mean': np.mean(profit_ratios),
        'profit_ratio_std': np.std(profit_ratios),
        'profit_share_mean': np.mean(results['test_share']),
        'profit_share_std': np.std(results['test_share']),
        'efficiency_mean': np.mean(results['efficiencies']),
        'efficiency_std': np.std(results['efficiencies']),
        'test_profit_mean': np.mean(results['test_profits']),
        'zic_profit_mean': np.mean(results['zic_profits']),
    }

    return summary


def print_results(buyer_results, seller_results):
    """Print formatted results table."""
    agent_name = buyer_results['agent']

    print()
    print("=" * 80)
    print(f"1v7 INVASIBILITY BENCHMARK: {agent_name.upper()}")
    print("=" * 80)
    print()
    print("Methodology: 1 test agent + 3 ZIC vs 4 ZIC")
    print("Replications: 10 markets with different random tokens")
    print()
    print("-" * 80)
    print(f"{'Metric':<30} {'As Buyer':<20} {'As Seller':<20}")
    print("-" * 80)

    print(f"{'Profit Ratio (vs mean ZIC):':<30} "
          f"{buyer_results['profit_ratio_mean']:>6.2f}x ± {buyer_results['profit_ratio_std']:.2f}    "
          f"{seller_results['profit_ratio_mean']:>6.2f}x ± {seller_results['profit_ratio_std']:.2f}")

    print(f"{'Profit Share (% of side):':<30} "
          f"{buyer_results['profit_share_mean']:>6.2f}% ± {buyer_results['profit_share_std']:.2f}%   "
          f"{seller_results['profit_share_mean']:>6.2f}% ± {seller_results['profit_share_std']:.2f}%")

    print(f"{'Market Efficiency:':<30} "
          f"{buyer_results['efficiency_mean']:>6.2f}% ± {buyer_results['efficiency_std']:.2f}%   "
          f"{seller_results['efficiency_mean']:>6.2f}% ± {seller_results['efficiency_std']:.2f}%")

    print()
    print(f"{'Mean Test Agent Profit:':<30} "
          f"{buyer_results['test_profit_mean']:>10.1f}         "
          f"{seller_results['test_profit_mean']:>10.1f}")

    print(f"{'Mean ZIC Profit (same side):':<30} "
          f"{buyer_results['zic_profit_mean']:>10.1f}         "
          f"{seller_results['zic_profit_mean']:>10.1f}")

    print()
    print("-" * 80)
    print("INTERPRETATION:")
    print(f"  • Profit ratio > 1.0 means {agent_name} outperforms ZIC")
    print(f"  • Profit share > 25% means {agent_name} extracts more than equal share")
    print(f"  • Combined metric: {((buyer_results['profit_ratio_mean'] + seller_results['profit_ratio_mean'])/2):.2f}x average invasibility")
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(description='Run 1v7 invasibility benchmark')
    parser.add_argument('--agent', type=str, default='zi2',
                       choices=['zi2', 'zip', 'gd', 'kaplan', 'zic', 'perry', 'lin'],
                       help='Agent to test')
    parser.add_argument('--replications', type=int, default=10,
                       help='Number of replications')
    parser.add_argument('--seed', type=int, default=1000,
                       help='Base random seed')

    # ZIP-specific parameters
    parser.add_argument('--beta', type=float, default=0.2,
                       help='ZIP learning rate (default: 0.2 for v2)')
    parser.add_argument('--gamma', type=float, default=0.25,
                       help='ZIP momentum (default: 0.25 for v2)')
    parser.add_argument('--margin', type=float, default=0.30,
                       help='ZIP initial margin (default: 0.30 for v2)')

    args = parser.parse_args()

    # Get agent class
    agent_class = TRADER_REGISTRY[args.agent]

    # Prepare agent kwargs
    agent_kwargs = {}
    if args.agent == 'zip':
        agent_kwargs = {
            'beta': args.beta,
            'gamma': args.gamma,
            'initial_margin_buy': -args.margin,
            'initial_margin_sell': args.margin,
        }

    print(f"\nRunning 1v7 invasibility test for {args.agent.upper()}...")
    if agent_kwargs:
        print(f"Agent config: {agent_kwargs}")

    # Run as buyer
    print("Testing as BUYER (1 test agent + 3 ZIC buyers vs 4 ZIC sellers)...")
    buyer_results = run_invasibility_test(
        agent_class, args.agent,
        is_buyer=True,
        num_replications=args.replications,
        seed=args.seed,
        **agent_kwargs
    )

    # Run as seller
    print("Testing as SELLER (4 ZIC buyers vs 1 test agent + 3 ZIC sellers)...")
    seller_results = run_invasibility_test(
        agent_class, args.agent,
        is_buyer=False,
        num_replications=args.replications,
        seed=args.seed + 10000,
        **agent_kwargs
    )

    # Print results
    print_results(buyer_results, seller_results)

    # Save to file
    output_file = f"results/invasibility_1v7_{args.agent}.txt"
    with open(output_file, 'w') as f:
        f.write(f"1v7 Invasibility Benchmark: {args.agent.upper()}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"As Buyer: {buyer_results['profit_ratio_mean']:.2f}x profit ratio, "
                f"{buyer_results['profit_share_mean']:.1f}% share\n")
        f.write(f"As Seller: {seller_results['profit_ratio_mean']:.2f}x profit ratio, "
                f"{seller_results['profit_share_mean']:.1f}% share\n")
        f.write(f"\nAverage: {((buyer_results['profit_ratio_mean'] + seller_results['profit_ratio_mean'])/2):.2f}x invasibility\n")

    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
