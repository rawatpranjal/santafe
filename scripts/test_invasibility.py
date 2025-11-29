#!/usr/bin/env python
"""
Test invasibility: 1 advanced trader vs 7 ZIC baseline traders.

This tests individual trader performance in a hostile environment,
measuring how well a single advanced trader can extract profit from
a market dominated by ZIC traders.

Configuration: 1 advanced trader + 7 ZIC traders
- As buyer: 1 advanced buyer + 3 ZIC buyers vs 4 ZIC sellers
- As seller: 1 advanced seller + 3 ZIC sellers vs 4 ZIC buyers
"""

import numpy as np
import polars as pl
from pathlib import Path
import sys
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from engine.market import Market
from engine.agent_factory import create_agent
from engine.token_generator import TokenGenerator
from engine.efficiency import calculate_allocative_efficiency

def run_invasibility_test(
    trader_type: str,
    trader_is_buyer: bool,
    n_rounds: int = 100,
    n_periods: int = 10,
    seed: int = 42
) -> Dict:
    """
    Run 1v7 invasibility test: 1 advanced trader vs 7 ZIC.

    Args:
        trader_type: Type of advanced trader ('ZIP', 'GD', 'Kaplan', 'ZI2')
        trader_is_buyer: Whether the advanced trader is a buyer
        n_rounds: Number of market rounds
        n_periods: Trading periods per round
        seed: Random seed

    Returns:
        Results dictionary with efficiency and profit metrics
    """
    rng = np.random.default_rng(seed)
    token_gen = TokenGenerator(game_type=1232, num_tokens=1, seed=seed)

    # Market configuration
    n_buyers = 4
    n_sellers = 4
    n_agents = n_buyers + n_sellers

    # Track metrics
    efficiencies = []
    advanced_profits = []
    zic_profits = []
    advanced_profit_share = []

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

        # Create agents (1 advanced + 7 ZIC)
        agents = []
        advanced_agent_id = None

        # Create buyers
        for i in range(n_buyers):
            player_id = i + 1
            is_advanced = (i == 0 and trader_is_buyer)  # First buyer is advanced if testing buyer

            if is_advanced:
                agent_type = trader_type
                advanced_agent_id = player_id
            else:
                agent_type = 'ZIC'

            agent = create_agent(
                agent_type=agent_type,
                player_id=player_id,
                is_buyer=True,
                num_tokens=buyer_tokens[i]['num_tokens'],
                valuations=buyer_tokens[i]['valuations'],
                price_min=0,
                price_max=500,
                seed=seed + round_idx * 100 + i
            )
            agents.append(agent)

        # Create sellers
        for i in range(n_sellers):
            player_id = n_buyers + i + 1
            is_advanced = (i == 0 and not trader_is_buyer)  # First seller is advanced if testing seller

            if is_advanced:
                agent_type = trader_type
                advanced_agent_id = player_id
            else:
                agent_type = 'ZIC'

            agent = create_agent(
                agent_type=agent_type,
                player_id=player_id,
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

        # Calculate metrics
        if max_surplus > 0:
            efficiency = calculate_allocative_efficiency(trades, max_surplus)
            efficiencies.append(efficiency)

        # Calculate profits
        agent_profits = {i: 0.0 for i in range(1, n_agents + 1)}

        for trade in trades:
            if trade['buyer_id']:
                buyer_profit = trade['buyer_value'] - trade['price']
                agent_profits[trade['buyer_id']] += buyer_profit

            if trade['seller_id']:
                seller_profit = trade['price'] - trade['seller_cost']
                agent_profits[trade['seller_id']] += seller_profit

        # Separate advanced vs ZIC profits
        advanced_profit = agent_profits[advanced_agent_id]
        zic_total = sum(p for pid, p in agent_profits.items() if pid != advanced_agent_id)
        zic_avg = zic_total / 7 if zic_total > 0 else 0

        advanced_profits.append(advanced_profit)
        zic_profits.append(zic_avg)

        total_profit = sum(agent_profits.values())
        if total_profit > 0:
            advanced_share = advanced_profit / total_profit
            advanced_profit_share.append(advanced_share)

    # Calculate summary statistics
    results = {
        'trader_type': trader_type,
        'trader_role': 'buyer' if trader_is_buyer else 'seller',
        'efficiency_mean': np.mean(efficiencies) if efficiencies else 0,
        'efficiency_std': np.std(efficiencies) if efficiencies else 0,
        'advanced_profit_mean': np.mean(advanced_profits),
        'advanced_profit_std': np.std(advanced_profits),
        'zic_profit_mean': np.mean(zic_profits),
        'zic_profit_std': np.std(zic_profits),
        'profit_ratio': np.mean(advanced_profits) / np.mean(zic_profits) if np.mean(zic_profits) > 0 else 0,
        'profit_share_mean': np.mean(advanced_profit_share) if advanced_profit_share else 0,
        'profit_share_std': np.std(advanced_profit_share) if advanced_profit_share else 0,
        'n_rounds': n_rounds
    }

    return results


def main():
    """Run invasibility tests for all trader types."""

    # Test configurations
    trader_types = ['ZIP', 'GD', 'Kaplan', 'ZI2']
    n_rounds = 50  # Number of market rounds per test

    # Results storage
    all_results = []

    print("=" * 60)
    print("1v7 INVASIBILITY TESTS")
    print("1 Advanced Trader vs 7 ZIC Baseline")
    print("=" * 60)
    print()

    for trader_type in trader_types:
        print(f"\nTesting {trader_type}...")

        # Test as buyer
        print(f"  - As buyer (1 {trader_type} + 3 ZIC) vs 4 ZIC sellers...")
        buyer_results = run_invasibility_test(
            trader_type=trader_type,
            trader_is_buyer=True,
            n_rounds=n_rounds,
            seed=42
        )
        all_results.append(buyer_results)

        print(f"    Efficiency: {buyer_results['efficiency_mean']:.1f}%")
        print(f"    Profit ratio: {buyer_results['profit_ratio']:.2f}x")
        print(f"    Profit share: {buyer_results['profit_share_mean']*100:.1f}%")

        # Test as seller
        print(f"  - As seller (1 {trader_type} + 3 ZIC) vs 4 ZIC buyers...")
        seller_results = run_invasibility_test(
            trader_type=trader_type,
            trader_is_buyer=False,
            n_rounds=n_rounds,
            seed=43
        )
        all_results.append(seller_results)

        print(f"    Efficiency: {seller_results['efficiency_mean']:.1f}%")
        print(f"    Profit ratio: {seller_results['profit_ratio']:.2f}x")
        print(f"    Profit share: {seller_results['profit_share_mean']*100:.1f}%")

    # Create summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE: 1v7 Invasibility Results")
    print("=" * 60)

    df = pl.DataFrame(all_results)

    # Format for display
    summary = df.select([
        pl.col("trader_type").alias("Trader"),
        pl.col("trader_role").alias("Role"),
        (pl.col("efficiency_mean")).round(1).alias("Efficiency %"),
        pl.col("profit_ratio").round(2).alias("Profit vs ZIC"),
        (pl.col("profit_share_mean") * 100).round(1).alias("Profit Share %"),
    ])

    print(summary)

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "invasibility"
    output_dir.mkdir(parents=True, exist_ok=True)

    df.write_csv(output_dir / "invasibility_1v7_results.csv")
    print(f"\nResults saved to: {output_dir / 'invasibility_1v7_results.csv'}")

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    for trader in trader_types:
        trader_df = df.filter(pl.col("trader_type") == trader)
        buyer_ratio = trader_df.filter(pl.col("trader_role") == "buyer")["profit_ratio"][0]
        seller_ratio = trader_df.filter(pl.col("trader_role") == "seller")["profit_ratio"][0]

        print(f"\n{trader}:")
        print(f"  - As buyer: {buyer_ratio:.2f}x profit vs ZIC")
        print(f"  - As seller: {seller_ratio:.2f}x profit vs ZIC")
        print(f"  - Average: {(buyer_ratio + seller_ratio)/2:.2f}x")


if __name__ == "__main__":
    main()