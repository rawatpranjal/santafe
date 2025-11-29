"""
Behavioral Tests for Zero Intelligence Constrained (ZIC) Traders.

These tests validate that ZIC agents exhibit the expected behavior from
Gode & Sunder (1993): random bidding/asking within budget constraints,
no learning, and achieving high allocative efficiency (~98%) despite
zero intelligence.

Reference:
- Gode, D.K. and Sunder, S. (1993). "Allocative efficiency of markets
  with zero-intelligence traders: Market as a partial substitute for
  individual rationality." Journal of Political Economy, 101(1), 119-137.
"""

import pytest
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple

from traders.legacy.zic import ZIC
from engine.market import Market
from engine.efficiency import (
    extract_trades_from_orderbook,
    calculate_actual_surplus,
    calculate_max_surplus,
    calculate_allocative_efficiency,
)


# =============================================================================
# TEST 1: UNIFORM DISTRIBUTION OF BIDS/ASKS
# =============================================================================

def test_zic_uniform_bid_distribution():
    """
    Test that ZIC buyers generate random bids in valid range.

    NOTE: Despite "uniform" in the name, this test validates Java SRobotZI1 behavior,
    which uses formula: V - floor(random() * (V - min))
    This creates a SLIGHT BIAS toward higher bids (not perfectly uniform).

    Gode & Sunder (1993) describes ZIC as "uniform", but the Java implementation
    uses floor() which creates truncation bias. We match the Java baseline.
    """
    # Create a single ZIC buyer
    buyer = ZIC(
        player_id=1,
        is_buyer=True,
        num_tokens=1,
        valuations=[100],
        price_min=0,
        price_max=200,
        seed=None  # Random for distribution test
    )

    # Collect 1000 bids
    bids = []
    for i in range(1000):
        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()
        if bid >= 0:  # Valid bid
            bids.append(bid)
        # Reset for next bid
        buyer.has_responded = False
        buyer.num_trades = 0  # Reset so agent can bid again

    # Check that bids are in valid range [0, 100]
    assert all(0 <= b <= 100 for b in bids), "All bids should be in [min, valuation]"

    # Check reasonable diversity (not all the same value)
    unique_bids = set(bids)
    assert len(unique_bids) > 30, f"Should have diverse bids, got {len(unique_bids)} unique values"

    # Statistical check: mean and std should be reasonable
    # Java formula creates bias, but not extreme
    mean_bid = np.mean(bids)
    std_bid = np.std(bids)

    # Expect mean around 45-55 (close to 50 but with slight upward bias)
    # Expect std around 25-32 (close to uniform std of 28.87)
    assert 40 <= mean_bid <= 60, f"Mean bid {mean_bid:.1f} should be roughly in center"
    assert 20 <= std_bid <= 35, f"Std bid {std_bid:.1f} should show reasonable spread"


def test_zic_uniform_ask_distribution():
    """
    Test that ZIC sellers generate uniformly distributed asks.

    ZIC Rule: Seller asks are uniform random in [cost, price_max].
    """
    # Create a single ZIC seller
    seller = ZIC(
        player_id=1,
        is_buyer=False,
        num_tokens=1,
        valuations=[50],  # Cost
        price_min=0,
        price_max=200,
        seed=None
    )

    # Collect 1000 asks
    asks = []
    for i in range(1000):
        seller.bid_ask(time=1, nobidask=0)
        ask = seller.bid_ask_response()
        if ask >= 0:  # Valid ask
            asks.append(ask)
        # Reset for next ask
        seller.has_responded = False
        seller.num_trades = 0

    # Test uniformity: asks should be Uniform[50, 200]
    # Normalize to [0, 1] for KS test
    normalized_asks = [(a - 50) / 150 for a in asks]
    _, p_value = stats.kstest(normalized_asks, 'uniform')

    assert p_value > 0.05, f"Asks not uniformly distributed (p={p_value:.4f})"

    # Check mean and std
    mean_ask = np.mean(asks)
    std_ask = np.std(asks)
    expected_mean = 125.0  # Mean of Uniform[50, 200]
    expected_std = 150 / np.sqrt(12)  # Std of Uniform[50, 200]

    assert abs(mean_ask - expected_mean) < 5, f"Mean ask {mean_ask:.1f} far from expected 125"
    assert abs(std_ask - expected_std) < 5, f"Std ask {std_ask:.1f} far from expected {expected_std:.1f}"


# =============================================================================
# TEST 2: NO LEARNING (STATIONARITY)
# =============================================================================

def test_zic_no_learning_over_time():
    """
    Test that ZIC agents do not learn or adapt over time.

    ZIC Property: Bid/ask distribution should remain stationary across periods.
    Test: Compare bid distributions from early vs late periods.
    """
    buyer = ZIC(
        player_id=1,
        is_buyer=True,
        num_tokens=10,  # Multiple tokens for multiple periods
        valuations=[100] * 10,
        price_min=0,
        price_max=200,
        seed=42
    )

    early_bids = []
    late_bids = []

    # Collect bids from "early" period (first 500 calls)
    for i in range(500):
        buyer.bid_ask(time=i+1, nobidask=0)
        bid = buyer.bid_ask_response()
        if bid >= 0:
            early_bids.append(bid)
        buyer.has_responded = False
        if i % 50 == 0:  # Simulate occasional trades
            buyer.num_trades = min(buyer.num_trades + 1, 9)

    # Reset to similar state
    buyer.num_trades = 0

    # Collect bids from "late" period (next 500 calls)
    for i in range(500):
        buyer.bid_ask(time=i+501, nobidask=0)
        bid = buyer.bid_ask_response()
        if bid >= 0:
            late_bids.append(bid)
        buyer.has_responded = False
        if i % 50 == 0:  # Simulate occasional trades
            buyer.num_trades = min(buyer.num_trades + 1, 9)

    # Test if distributions are the same using Mann-Whitney U test
    # Null hypothesis: early and late bids come from same distribution
    _, p_value = stats.mannwhitneyu(early_bids, late_bids, alternative='two-sided')

    assert p_value > 0.05, f"Bid distribution changed over time (p={p_value:.4f})"

    # Also check means are similar
    mean_early = np.mean(early_bids)
    mean_late = np.mean(late_bids)
    assert abs(mean_early - mean_late) < 10, \
        f"Mean bid changed from {mean_early:.1f} to {mean_late:.1f}"


# =============================================================================
# TEST 3: INDEPENDENCE OF DECISIONS
# =============================================================================

def test_zic_bid_independence():
    """
    Test that consecutive bids are independent (no serial correlation).

    ZIC Property: Each bid is independent random draw.
    Test: Autocorrelation of bid sequence should be near zero.
    """
    buyer = ZIC(
        player_id=1,
        is_buyer=True,
        num_tokens=1,
        valuations=[100],
        price_min=0,
        price_max=200,
        seed=None
    )

    # Collect sequence of bids
    bids = []
    for i in range(500):
        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()
        if bid >= 0:
            bids.append(bid)
        buyer.has_responded = False
        buyer.num_trades = 0  # Keep agent active

    # Calculate autocorrelation at lag 1
    bids_array = np.array(bids)
    mean = np.mean(bids_array)
    c0 = np.sum((bids_array - mean) ** 2) / len(bids_array)
    c1 = np.sum((bids_array[:-1] - mean) * (bids_array[1:] - mean)) / (len(bids_array) - 1)
    autocorr = c1 / c0 if c0 > 0 else 0

    # Autocorrelation should be near 0 (independence)
    # Under null hypothesis of independence, autocorr ~ N(0, 1/n)
    # 95% CI: [-1.96/sqrt(n), 1.96/sqrt(n)]
    n = len(bids)
    ci = 1.96 / np.sqrt(n)

    assert abs(autocorr) < ci, \
        f"Autocorrelation {autocorr:.4f} outside 95% CI [{-ci:.4f}, {ci:.4f}]"


# =============================================================================
# TEST 4: ALLOCATIVE EFFICIENCY BENCHMARK
# =============================================================================

def test_zic_high_efficiency_symmetric_market():
    """
    Test that ZIC achieves ~98% allocative efficiency in symmetric markets.

    Benchmark: Gode & Sunder (1993) reported 98.7% efficiency.
    Setup: Symmetric market with overlapping supply/demand curves.
    """
    # Create symmetric market setup (similar to Gode & Sunder)
    num_agents = 5
    num_tokens = 5

    # Symmetric valuations with good overlap
    buyer_tokens = [
        [200, 180, 160, 140, 120],
        [195, 175, 155, 135, 115],
        [190, 170, 150, 130, 110],
        [185, 165, 145, 125, 105],
        [180, 160, 140, 120, 100],
    ]

    seller_tokens = [
        [20, 40, 60, 80, 100],
        [25, 45, 65, 85, 105],
        [30, 50, 70, 90, 110],
        [35, 55, 75, 95, 115],
        [40, 60, 80, 100, 120],
    ]

    efficiencies = []

    # Run 20 replications
    for rep in range(20):
        buyers = [
            ZIC(i+1, True, num_tokens, buyer_tokens[i],
                price_min=0, price_max=250, seed=rep*100+i)
            for i in range(num_agents)
        ]
        sellers = [
            ZIC(i+6, False, num_tokens, seller_tokens[i],
                price_min=0, price_max=250, seed=rep*100+i+5)
            for i in range(num_agents)
        ]

        market = Market(
            num_buyers=num_agents,
            num_sellers=num_agents,
            num_times=200,  # Sufficient time for all trades
            price_min=0,
            price_max=250,
            buyers=buyers,
            sellers=sellers,
            seed=rep
        )

        # Run market
        for _ in range(200):
            if not market.run_time_step():
                break

        # Calculate efficiency
        trades = extract_trades_from_orderbook(market.orderbook, 200)

        # Build valuations dicts using local IDs
        buyer_vals = {i+1: buyers[i].valuations for i in range(num_agents)}
        seller_costs = {i+1: sellers[i].valuations for i in range(num_agents)}

        # Calculate efficiency
        actual_surplus = calculate_actual_surplus(trades, buyer_vals, seller_costs)
        max_surplus = calculate_max_surplus(
            [b.valuations for b in buyers],
            [s.valuations for s in sellers]
        )

        if max_surplus > 0:
            efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)
            efficiencies.append(efficiency)

    # Check average efficiency
    avg_efficiency = np.mean(efficiencies)
    std_efficiency = np.std(efficiencies)

    print(f"\nZIC Efficiency: {avg_efficiency:.2f}% ± {std_efficiency:.2f}%")
    print(f"Gode & Sunder (1993) benchmark: 98.7%")

    # Allow some deviation from literature due to protocol differences
    # AURORA protocol may differ slightly from continuous double auction
    assert avg_efficiency > 90, \
        f"ZIC efficiency {avg_efficiency:.2f}% too low (expected >90%)"

    # Check consistency (low variance)
    assert std_efficiency < 10, \
        f"ZIC efficiency variance {std_efficiency:.2f}% too high"


# =============================================================================
# TEST 5: EQUILIBRIUM CONVERGENCE
# =============================================================================

def test_zic_price_convergence_to_equilibrium():
    """
    Test that ZIC markets converge to competitive equilibrium price.

    Setup: Simple market with clear equilibrium.
    Expected: Trade prices should cluster around equilibrium.
    """
    # Simple setup with clear equilibrium at price ~75
    buyers = [
        ZIC(1, True, 3, [100, 90, 80], price_min=0, price_max=150, seed=1),
        ZIC(2, True, 3, [95, 85, 75], price_min=0, price_max=150, seed=2),
    ]
    sellers = [
        ZIC(3, False, 3, [50, 60, 70], price_min=0, price_max=150, seed=3),
        ZIC(4, False, 3, [55, 65, 75], price_min=0, price_max=150, seed=4),
    ]

    market = Market(
        num_buyers=2,
        num_sellers=2,
        num_times=100,
        price_min=0,
        price_max=150,
        buyers=buyers,
        sellers=sellers,
        seed=42
    )

    # Run market
    for _ in range(100):
        if not market.run_time_step():
            break

    # Extract trade prices
    trade_prices = []
    for t in range(1, 101):
        if market.orderbook.trade_price[t] > 0:
            trade_prices.append(market.orderbook.trade_price[t])

    # Theoretical equilibrium: where supply meets demand
    # With these valuations, equilibrium should be around 70-80
    expected_equilibrium = 75.0

    # Check that trade prices converge near equilibrium
    if len(trade_prices) > 10:
        # Look at last half of trades (after initial discovery)
        late_trades = trade_prices[len(trade_prices)//2:]
        mean_price = np.mean(late_trades)
        std_price = np.std(late_trades)

        print(f"\nEquilibrium test:")
        print(f"Expected equilibrium: {expected_equilibrium}")
        print(f"Mean trade price (late): {mean_price:.2f} ± {std_price:.2f}")

        # Prices should be within reasonable range of equilibrium
        assert abs(mean_price - expected_equilibrium) < 20, \
            f"Mean price {mean_price:.2f} far from equilibrium {expected_equilibrium}"

        # Some convergence (not too volatile)
        assert std_price < 30, f"Price volatility {std_price:.2f} too high"


# =============================================================================
# TEST 6: CONSTRAINT ENFORCEMENT
# =============================================================================

def test_zic_never_violates_budget_constraint():
    """
    Test that ZIC agents NEVER violate their budget constraints.

    Critical property: Buyers never bid above valuation.
                      Sellers never ask below cost.
    """
    # Test buyers
    buyer = ZIC(
        player_id=1,
        is_buyer=True,
        num_tokens=3,
        valuations=[100, 80, 60],
        price_min=0,
        price_max=200,  # Higher than valuations
        seed=None
    )

    # Collect many bids for each token
    for token_idx in range(3):
        buyer.num_trades = token_idx  # Simulate having traded previous tokens
        valuation = buyer.valuations[token_idx]

        for _ in range(100):
            buyer.bid_ask(time=1, nobidask=0)
            bid = buyer.bid_ask_response()
            if bid >= 0:
                assert bid <= valuation, \
                    f"Buyer bid {bid} exceeds valuation {valuation} for token {token_idx}"
            buyer.has_responded = False

    # Test sellers
    seller = ZIC(
        player_id=2,
        is_buyer=False,
        num_tokens=3,
        valuations=[40, 60, 80],  # Costs
        price_min=0,  # Lower than costs
        price_max=200,
        seed=None
    )

    # Collect many asks for each token
    for token_idx in range(3):
        seller.num_trades = token_idx  # Simulate having traded previous tokens
        cost = seller.valuations[token_idx]

        for _ in range(100):
            seller.bid_ask(time=1, nobidask=0)
            ask = seller.bid_ask_response()
            if ask >= 0:
                assert ask >= cost, \
                    f"Seller ask {ask} below cost {cost} for token {token_idx}"
            seller.has_responded = False


# =============================================================================
# TEST 7: ACCEPT/REJECT LOGIC
# =============================================================================

def test_zic_accept_reject_profitable_trades():
    """
    Test that ZIC agents accept profitable trades and reject unprofitable ones.

    Rule: Buyers accept if ask <= valuation.
          Sellers accept if bid >= cost.
    """
    # Test buyer decisions
    buyer = ZIC(
        player_id=1,
        is_buyer=True,
        num_tokens=1,
        valuations=[100],
        price_min=0,
        price_max=200,
        seed=42
    )

    # Profitable trade (ask < valuation)
    # Note: Buyer must be the high bidder to accept
    buyer.buy_sell(
        time=1,
        nobuysell=0,  # Can trade
        high_bid=90,  # Buyer's bid
        low_ask=80,  # Below valuation
        high_bidder=1,  # This buyer is the high bidder
        low_asker=2
    )
    assert buyer.buy_sell_response() == True, \
        "Buyer should accept profitable trade"

    # Reset
    buyer.has_responded = False
    buyer.num_trades = 0

    # Unprofitable trade (ask > valuation)
    buyer.buy_sell(
        time=2,
        nobuysell=0,
        high_bid=0,
        low_ask=120,  # Above valuation
        high_bidder=0,
        low_asker=2
    )
    assert buyer.buy_sell_response() == False, \
        "Buyer should reject unprofitable trade"

    # Test seller decisions
    seller = ZIC(
        player_id=2,
        is_buyer=False,
        num_tokens=1,
        valuations=[50],  # Cost
        price_min=0,
        price_max=200,
        seed=42
    )

    # Profitable trade (bid > cost)
    # Note: Seller must be the low asker to accept
    seller.buy_sell(
        time=1,
        nobuysell=0,
        high_bid=70,  # Above cost
        low_ask=60,  # Seller's ask
        high_bidder=1,
        low_asker=2  # This seller is the low asker
    )
    assert seller.buy_sell_response() == True, \
        "Seller should accept profitable trade"

    # Reset
    seller.has_responded = False
    seller.num_trades = 0

    # Unprofitable trade (bid < cost)
    seller.buy_sell(
        time=2,
        nobuysell=0,
        high_bid=40,  # Below cost
        low_ask=0,
        high_bidder=1,
        low_asker=0
    )
    assert seller.buy_sell_response() == False, \
        "Seller should reject unprofitable trade"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])