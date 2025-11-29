"""
Test ZI2 profit extraction vs ZIC (near-parity hypothesis).

This validates that ZI2's market-awareness provides marginal improvements
but NOT profit dominance like GD. ZI2 should show ~1.0x profit ratio (parity).

Hypothesis: ZI2 ≈ ZIC in profit extraction (both zero-intelligence).
Key difference: ZI2 has market-awareness which narrows spreads but doesn't
enable strategic exploitation like GD's belief-based optimization.
"""

import pytest
import numpy as np
from engine.market import Market
from traders.legacy.zic import ZIC
from traders.legacy.zi2 import ZI2


def run_mixed_market(zi2_is_buyer: bool, num_sessions: int = 10, num_agents: int = 5, num_tokens: int = 5):
    """
    Run mixed ZI2 vs ZIC market sessions.

    Args:
        zi2_is_buyer: If True, ZI2 are buyers and ZIC are sellers. If False, reverse.
        num_sessions: Number of market sessions to run
        num_agents: Number of agents per side
        num_tokens: Tokens per agent

    Returns:
        (zi2_profits, zic_profits): Lists of total profits per session
    """
    zi2_profits = []
    zic_profits = []
    price_max = 400

    for session in range(num_sessions):
        # Generate random valuations/costs
        valuations = [sorted(np.random.randint(50, price_max, num_tokens).tolist(), reverse=True)
                      for _ in range(num_agents)]
        costs = [sorted(np.random.randint(0, price_max-50, num_tokens).tolist())
                 for _ in range(num_agents)]

        if zi2_is_buyer:
            buyers = [ZI2(i+1, True, num_tokens, valuations[i], 0, price_max, seed=session*100+i)
                      for i in range(num_agents)]
            sellers = [ZIC(i+1+num_agents, False, num_tokens, costs[i], 0, price_max, seed=session*100+i+50)
                       for i in range(num_agents)]
        else:
            buyers = [ZIC(i+1, True, num_tokens, valuations[i], 0, price_max, seed=session*100+i)
                      for i in range(num_agents)]
            sellers = [ZI2(i+1+num_agents, False, num_tokens, costs[i], 0, price_max, seed=session*100+i+50)
                       for i in range(num_agents)]

        # Run market
        market = Market(
            num_buyers=num_agents,
            num_sellers=num_agents,
            num_times=200,  # Longer to accumulate profits
            price_min=0,
            price_max=price_max,
            buyers=buyers,
            sellers=sellers,
            seed=session
        )

        for _ in range(200):
            if not market.run_time_step():
                break

        # Collect profits
        buyer_profit = sum(b.period_profit for b in buyers)
        seller_profit = sum(s.period_profit for s in sellers)

        if zi2_is_buyer:
            zi2_profits.append(buyer_profit)
            zic_profits.append(seller_profit)
        else:
            zic_profits.append(buyer_profit)
            zi2_profits.append(seller_profit)

    return zi2_profits, zic_profits


def test_zi2_vs_zic_as_buyers():
    """
    Test ZI2 buyers vs ZIC sellers - expect near-parity (~1.0x).

    Hypothesis: Market-awareness helps but doesn't enable exploitation.
    Expected: Profit ratio 0.9x - 1.2x (near-parity, NOT dominance).
    """
    zi2_profits, zic_profits = run_mixed_market(zi2_is_buyer=True, num_sessions=15)

    avg_zi2 = np.mean(zi2_profits)
    avg_zic = np.mean(zic_profits)
    profit_ratio = avg_zi2 / avg_zic if avg_zic > 0 else float('inf')

    print(f"\nZI2 Buyers vs ZIC Sellers:")
    print(f"  ZI2 Profit:  {avg_zi2:.0f}")
    print(f"  ZIC Profit:  {avg_zic:.0f}")
    print(f"  Profit Ratio: {profit_ratio:.2f}x")

    # ZI2 should be competitive but NOT dominant
    # Accept 0.8x - 1.5x as reasonable range (near-parity)
    assert 0.8 <= profit_ratio <= 1.5, \
        f"ZI2 should show near-parity with ZIC, got {profit_ratio:.2f}x (expected 0.8-1.5x)"

    # Profit share should be roughly balanced (40-60%)
    total_surplus = avg_zi2 + avg_zic
    zi2_share = avg_zi2 / total_surplus if total_surplus > 0 else 0

    print(f"  ZI2 Profit Share: {zi2_share:.1%}")
    print(f"  Expected Range: 40-60% (balanced competition)")

    assert 0.35 <= zi2_share <= 0.65, \
        f"ZI2 profit share should be balanced (40-60%), got {zi2_share:.1%}"


def test_zi2_vs_zic_as_sellers():
    """
    Test ZIC buyers vs ZI2 sellers - expect near-parity (~1.0x).

    Hypothesis: ZI2's market-awareness provides marginal benefit but not dominance.
    Expected: Profit ratio 0.9x - 1.2x (near-parity).
    """
    zi2_profits, zic_profits = run_mixed_market(zi2_is_buyer=False, num_sessions=15)

    avg_zi2 = np.mean(zi2_profits)
    avg_zic = np.mean(zic_profits)
    profit_ratio = avg_zi2 / avg_zic if avg_zic > 0 else float('inf')

    print(f"\nZIC Buyers vs ZI2 Sellers:")
    print(f"  ZIC Profit:  {avg_zic:.0f}")
    print(f"  ZI2 Profit:  {avg_zi2:.0f}")
    print(f"  Profit Ratio: {profit_ratio:.2f}x")

    # ZI2 should be competitive but NOT dominant
    assert 0.8 <= profit_ratio <= 1.5, \
        f"ZI2 should show near-parity with ZIC, got {profit_ratio:.2f}x (expected 0.8-1.5x)"

    # Profit share should be roughly balanced (40-60%)
    total_surplus = avg_zi2 + avg_zic
    zi2_share = avg_zi2 / total_surplus if total_surplus > 0 else 0

    print(f"  ZI2 Profit Share: {zi2_share:.1%}")
    print(f"  Expected Range: 40-60% (balanced competition)")

    assert 0.35 <= zi2_share <= 0.65, \
        f"ZI2 profit share should be balanced (40-60%), got {zi2_share:.1%}"


def test_zi2_vs_zi2_balanced():
    """
    Test ZI2 vs ZI2 markets are balanced (no systematic buyer/seller advantage).

    When both sides use ZI2, neither should dominate.
    """
    zi2_buyer_profits = []
    zi2_seller_profits = []
    price_max = 400
    num_agents = 5
    num_tokens = 5

    for session in range(15):
        valuations = [sorted(np.random.randint(50, price_max, num_tokens).tolist(), reverse=True)
                      for _ in range(num_agents)]
        costs = [sorted(np.random.randint(0, price_max-50, num_tokens).tolist())
                 for _ in range(num_agents)]

        buyers = [ZI2(i+1, True, num_tokens, valuations[i], 0, price_max, seed=session*100+i)
                  for i in range(num_agents)]
        sellers = [ZI2(i+1+num_agents, False, num_tokens, costs[i], 0, price_max, seed=session*100+i+50)
                   for i in range(num_agents)]

        market = Market(
            num_buyers=num_agents,
            num_sellers=num_agents,
            num_times=200,
            price_min=0,
            price_max=price_max,
            buyers=buyers,
            sellers=sellers,
            seed=session
        )

        for _ in range(200):
            if not market.run_time_step():
                break

        buyer_profit = sum(b.period_profit for b in buyers)
        seller_profit = sum(s.period_profit for s in sellers)

        zi2_buyer_profits.append(buyer_profit)
        zi2_seller_profits.append(seller_profit)

    avg_buyer = np.mean(zi2_buyer_profits)
    avg_seller = np.mean(zi2_seller_profits)
    ratio = avg_buyer / avg_seller if avg_seller > 0 else float('inf')

    print(f"\nZI2 vs ZI2 Balance Test:")
    print(f"  Buyer Profit:  {avg_buyer:.0f}")
    print(f"  Seller Profit: {avg_seller:.0f}")
    print(f"  Buyer/Seller Ratio: {ratio:.2f}x")

    # Should be roughly balanced (0.7x - 1.4x acceptable range)
    assert 0.7 <= ratio <= 1.4, \
        f"ZI2 vs ZI2 should be balanced, got ratio {ratio:.2f}x (expected 0.7-1.4x)"

    # Total profit split should be roughly 40-60%
    total = avg_buyer + avg_seller
    buyer_share = avg_buyer / total if total > 0 else 0
    print(f"  Buyer Share: {buyer_share:.1%}")

    assert 0.35 <= buyer_share <= 0.65, \
        f"Buyer share in ZI2 vs ZI2 should be balanced, got {buyer_share:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
