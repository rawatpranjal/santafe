#!/usr/bin/env python3
"""
Minimal debug script to trace Kaplan agent behavior.

This script runs a minimal Kaplan vs Kaplan scenario with comprehensive
logging enabled to identify why agents are making unprofitable trades.
"""

import logging
import sys
sys.path.insert(0, '/Users/pranjal/Code/santafe-1')

from engine.market import Market
from engine.token_generator import TokenGenerator
from traders.legacy.kaplan import Kaplan

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)-8s [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler('/tmp/kaplan_debug.log', mode='w'),
        logging.StreamHandler()
    ]
)

# Set specific loggers to WARNING for less noise
logging.getLogger('engine').setLevel(logging.WARNING)
logging.getLogger('traders.base').setLevel(logging.WARNING)

# Keep Kaplan decision/trade logs at WARNING to see them
logging.getLogger('kaplan.decision').setLevel(logging.DEBUG)
logging.getLogger('kaplan.trade').setLevel(logging.WARNING)
logging.getLogger('kaplan.ask').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info("KAPLAN DEBUG TEST - Minimal Scenario")
    logger.info("="*80)

    # Create minimal market: 2 buyers, 2 sellers, all Kaplan
    num_buyers = 2
    num_sellers = 2
    num_tokens = 3
    num_steps = 10  # Short run to keep logs manageable
    price_min = 0
    price_max = 200

    # Create token generator
    token_gen = TokenGenerator(
        game_type=6453,  # Standard SFI game type
        num_tokens=num_tokens,
        seed=42
    )
    token_gen.new_round()  # Initialize round parameters

    # Create Kaplan agents
    buyers = []
    for i in range(1, num_buyers + 1):
        agent = Kaplan(
            player_id=i,
            is_buyer=True,
            num_tokens=num_tokens,
            valuations=[0] * num_tokens,  # Will be set by token_gen
            price_min=price_min,
            price_max=price_max,
            num_times=num_steps
        )
        tokens = token_gen.generate_tokens(is_buyer=True)
        agent.valuations = tokens
        buyers.append(agent)
        logger.info(f"Buyer {i} valuations: {tokens}")

    sellers = []
    for i in range(1, num_sellers + 1):
        agent = Kaplan(
            player_id=i,
            is_buyer=False,
            num_tokens=num_tokens,
            valuations=[0] * num_tokens,  # Will be set by token_gen
            price_min=price_min,
            price_max=price_max,
            num_times=num_steps
        )
        tokens = token_gen.generate_tokens(is_buyer=False)
        agent.valuations = tokens
        sellers.append(agent)
        logger.info(f"Seller {i} costs: {tokens}")

    # Calculate equilibrium profit
    all_buyer_vals = []
    all_seller_costs = []
    for b in buyers:
        all_buyer_vals.extend(b.valuations)
    for s in sellers:
        all_seller_costs.extend(s.valuations)

    all_buyer_vals.sort(reverse=True)
    all_seller_costs.sort()

    eq_profit = 0
    for i in range(min(len(all_buyer_vals), len(all_seller_costs))):
        if all_buyer_vals[i] >= all_seller_costs[i]:
            eq_profit += all_buyer_vals[i] - all_seller_costs[i]
        else:
            break

    logger.info(f"Equilibrium profit: {eq_profit}")

    # Create market
    market = Market(
        num_buyers=num_buyers,
        num_sellers=num_sellers,
        num_times=num_steps,
        price_min=price_min,
        price_max=price_max,
        buyers=buyers,
        sellers=sellers,
        seed=42
    )

    # Notify agents to start period
    for agent in buyers + sellers:
        agent.start_period(1)

    # Run market for num_steps timesteps
    logger.info("="*80)
    logger.info("STARTING MARKET")
    logger.info("="*80)

    for t in range(1, num_steps + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"TIMESTEP {t}")
        logger.info(f"{'='*80}")
        market.run_time_step()

        # Log period profit after each timestep
        total_profit = sum(a.period_profit for a in buyers + sellers)
        logger.info(f"Timestep {t} total profit: {total_profit}")

    # Final summary
    logger.info("="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)

    total_profit = sum(a.period_profit for a in buyers + sellers)
    efficiency = (total_profit / eq_profit * 100) if eq_profit > 0 else 0.0

    logger.info(f"Equilibrium profit: {eq_profit}")
    logger.info(f"Actual profit: {total_profit}")
    logger.info(f"Efficiency: {efficiency:.2f}%")

    for i, buyer in enumerate(buyers, 1):
        logger.info(f"Buyer {i}: period_profit={buyer.period_profit}, num_trades={buyer.num_trades}")

    for i, seller in enumerate(sellers, 1):
        logger.info(f"Seller {i}: period_profit={seller.period_profit}, num_trades={seller.num_trades}")

    logger.info(f"\nFull debug logs written to: /tmp/kaplan_debug.log")
    logger.info(f"Grep for 'ACCEPT.*sniper' to find sniper trades")
    logger.info(f"Grep for 'profit=-' to find unprofitable trades")

if __name__ == "__main__":
    main()
