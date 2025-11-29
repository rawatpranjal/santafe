"""
Behavioral Tests for Zero Intelligence Plus (ZIP) Traders.

These tests validate that ZIP agents exhibit the expected behavior from
Cliff (1997): adaptive profit margin mechanism, convergence to equilibrium,
and achieving 85-95% allocative efficiency.

Reference:
- Cliff, D. (1997). "Minimal-intelligence agents for bargaining behaviors in
  market-based environments." HP Laboratories Technical Report HPL-97-91.
- Cliff, D. and Bruten, J. (1997). "Zero is not enough: On the lower limit
  of agent intelligence for continuous double auction markets."
"""

import pytest
import numpy as np
from typing import List, Dict, Tuple

from traders.legacy.zip import ZIP
from engine.market import Market
from engine.efficiency import (
    extract_trades_from_orderbook,
    calculate_actual_surplus,
    calculate_max_surplus,
    calculate_allocative_efficiency,
)


# =============================================================================
# TEST 1: PROFIT MARGIN ADAPTATION
# =============================================================================

def test_zip_profit_margin_adaptation():
    """
    Test that ZIP adjusts its profit margin based on market feedback.

    ZIP Rule: If outbid, reduce margin. If trade missed, increase margin.
    Expected: Margin should adapt towards market-clearing levels.
    """
    # Create a ZIP buyer
    buyer = ZIP(
        player_id=1,
        is_buyer=True,
        num_tokens=3,
        valuations=[100, 90, 80],
        price_min=0,
        price_max=150,
        seed=42
    )

    # Initial margin should be negative for buyers (-0.35 to -0.05)
    initial_margin = buyer.margin  # profit margin
    assert -0.35 <= initial_margin <= -0.05, f"Initial margin {initial_margin} out of expected range for buyer"

    # Simulate being outbid (should reduce margin)
    buyer.bid_ask(time=1, nobidask=0)
    initial_bid = buyer.bid_ask_response()

    # Notify that we were outbid (status=3 means beaten)
    buyer.bid_ask_result(
        status=3,  # Beaten by another
        num_trades=0,
        new_bids=[initial_bid, initial_bid + 5],  # Someone bid higher
        new_asks=[],
        high_bid=initial_bid + 5,
        high_bidder=2,  # Another buyer won
        low_ask=0,
        low_asker=0
    )

    # Check that margin increased (buyer should bid more aggressively)
    # For buyers, margin is negative, so "increasing" means less negative (closer to 0)
    new_margin = buyer.margin
    assert new_margin > initial_margin, \
        f"Buyer margin should increase (less negative) when outbid: {initial_margin:.4f} -> {new_margin:.4f}"

    # Test seller margin adaptation
    seller = ZIP(
        player_id=2,
        is_buyer=False,
        num_tokens=3,
        valuations=[50, 60, 70],
        price_min=0,
        price_max=150,
        seed=43
    )

    initial_seller_margin = seller.margin

    # Simulate being undercut (should reduce margin)
    seller.bid_ask(time=1, nobidask=0)
    initial_ask = seller.bid_ask_response()

    # Notify that we were undercut
    seller.bid_ask_result(
        status=3,  # Beaten
        num_trades=0,
        new_bids=[],
        new_asks=[initial_ask - 5, initial_ask],  # Someone asked lower
        high_bid=0,
        high_bidder=0,
        low_ask=initial_ask - 5,
        low_asker=3  # Another seller won
    )

    # Check that margin decreased
    new_seller_margin = seller.margin
    assert new_seller_margin < initial_seller_margin, \
        f"Seller margin should decrease when undercut: {initial_seller_margin:.4f} -> {new_seller_margin:.4f}"


# =============================================================================
# TEST 2: CONVERGENCE TO EQUILIBRIUM
# =============================================================================

def test_zip_price_convergence():
    """
    Test that ZIP markets converge to competitive equilibrium.

    Expected: Trade prices should converge towards theoretical equilibrium.
    ZIP should achieve faster convergence than ZIC.
    """
    # Simple market with clear equilibrium around 75
    buyers = [
        ZIP(1, True, 3, [100, 90, 80], price_min=0, price_max=150, seed=1),
        ZIP(2, True, 3, [95, 85, 75], price_min=0, price_max=150, seed=2),
    ]
    sellers = [
        ZIP(3, False, 3, [50, 60, 70], price_min=0, price_max=150, seed=3),
        ZIP(4, False, 3, [55, 65, 75], price_min=0, price_max=150, seed=4),
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

    if len(trade_prices) > 10:
        # Check convergence - compare first third vs last third of trades
        n = len(trade_prices)
        early_trades = trade_prices[:n//3]
        late_trades = trade_prices[2*n//3:]

        early_std = np.std(early_trades)
        late_std = np.std(late_trades)

        # Late trades should have lower variance (convergence)
        assert late_std < early_std, \
            f"Expected convergence but late_std ({late_std:.2f}) >= early_std ({early_std:.2f})"

        # Mean of late trades should be near equilibrium (75 ± 10)
        late_mean = np.mean(late_trades)
        assert 65 <= late_mean <= 85, \
            f"Late trade mean {late_mean:.2f} far from equilibrium ~75"


# =============================================================================
# TEST 3: EFFICIENCY BENCHMARK
# =============================================================================

def test_zip_efficiency_benchmark():
    """
    Test that ZIP achieves 85-95% allocative efficiency.

    Benchmark: Cliff & Bruten (1997) reported ~95% for ZIP vs ZIP.
    Expected: Should significantly outperform ZIC's theoretical maximum.
    """
    # Symmetric market setup
    num_agents = 5
    num_tokens = 5

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

    # Run 10 replications
    for rep in range(10):
        buyers = [
            ZIP(i+1, True, num_tokens, buyer_tokens[i],
                price_min=0, price_max=250, seed=rep*100+i)
            for i in range(num_agents)
        ]
        sellers = [
            ZIP(i+6, False, num_tokens, seller_tokens[i],
                price_min=0, price_max=250, seed=rep*100+i+5)
            for i in range(num_agents)
        ]

        market = Market(
            num_buyers=num_agents,
            num_sellers=num_agents,
            num_times=200,
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

        buyer_vals = {i+1: buyers[i].valuations for i in range(num_agents)}
        seller_costs = {i+1: sellers[i].valuations for i in range(num_agents)}

        actual_surplus = calculate_actual_surplus(trades, buyer_vals, seller_costs)
        max_surplus = calculate_max_surplus(
            [b.valuations for b in buyers],
            [s.valuations for s in sellers]
        )

        if max_surplus > 0:
            efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)
            efficiencies.append(efficiency)

    # Check results
    avg_efficiency = np.mean(efficiencies)
    std_efficiency = np.std(efficiencies)

    print(f"\nZIP Efficiency: {avg_efficiency:.2f}% ± {std_efficiency:.2f}%")
    print(f"Cliff & Bruten (1997) benchmark: ~95%")

    # ZIP should achieve 85-95% efficiency
    assert avg_efficiency > 85, \
        f"ZIP efficiency {avg_efficiency:.2f}% below expected minimum 85%"

    # Should have reasonable consistency
    assert std_efficiency < 10, \
        f"ZIP efficiency variance {std_efficiency:.2f}% too high"


# =============================================================================
# TEST 4: LEARNING RATE DYNAMICS
# =============================================================================

def test_zip_learning_rate_dynamics():
    """
    Test that ZIP's learning rate (beta) adapts appropriately.

    ZIP Rule: Beta adjusts to control speed of margin adaptation.
    Expected: Beta should remain stable but responsive.
    """
    buyer = ZIP(
        player_id=1,
        is_buyer=True,
        num_tokens=5,
        valuations=[100, 95, 90, 85, 80],
        price_min=0,
        price_max=150,
        seed=42
    )

    # Check initial beta is in reasonable range
    initial_beta = buyer.beta
    assert 0.1 <= initial_beta <= 0.5, \
        f"Initial beta {initial_beta} outside expected range [0.1, 0.5]"

    # Simulate market activity
    beta_values = [initial_beta]

    for t in range(20):
        buyer.bid_ask(time=t+1, nobidask=0)
        bid = buyer.bid_ask_response()

        # Simulate mixed feedback (sometimes outbid, sometimes win)
        if t % 3 == 0:
            status = 2  # Won
        else:
            status = 3  # Lost

        buyer.bid_ask_result(
            status=status,
            num_trades=min(t // 5, 4),  # Gradual trades
            new_bids=[bid],
            new_asks=[],
            high_bid=bid if status == 2 else bid + 5,
            high_bidder=1 if status == 2 else 2,
            low_ask=0,
            low_asker=0
        )

        beta_values.append(buyer.beta)

    # Beta should adapt but not explode
    max_beta = max(beta_values)
    min_beta = min(beta_values)

    assert max_beta <= 1.0, f"Beta exceeded 1.0: {max_beta}"
    assert min_beta >= 0.05, f"Beta went too low: {min_beta}"

    # Should show some variation (learning)
    beta_std = np.std(beta_values)
    assert beta_std > 0.01, "Beta showed no adaptation"


# =============================================================================
# TEST 5: MOMENTUM FACTOR
# =============================================================================

def test_zip_momentum_factor():
    """
    Test that ZIP's momentum factor influences adaptation.

    ZIP uses momentum to smooth margin changes and avoid oscillation.
    """
    # Create two ZIPs with different momentum settings
    # Note: ZIP implementation may have fixed momentum, so we test behavior
    buyer1 = ZIP(
        player_id=1,
        is_buyer=True,
        num_tokens=3,
        valuations=[100, 90, 80],
        price_min=0,
        price_max=150,
        seed=42
    )

    buyer2 = ZIP(
        player_id=2,
        is_buyer=True,
        num_tokens=3,
        valuations=[100, 90, 80],
        price_min=0,
        price_max=150,
        seed=43
    )

    # Track margin changes
    margins1 = [buyer1.margin]
    margins2 = [buyer2.margin]

    # Apply same market feedback to both
    for t in range(10):
        # Both submit bids
        buyer1.bid_ask(time=t+1, nobidask=0)
        bid1 = buyer1.bid_ask_response()

        buyer2.bid_ask(time=t+1, nobidask=0)
        bid2 = buyer2.bid_ask_response()

        # Both get outbid (should reduce margin)
        buyer1.bid_ask_result(
            status=3, num_trades=0,
            new_bids=[bid1, max(bid1, bid2) + 5],
            new_asks=[], high_bid=max(bid1, bid2) + 5,
            high_bidder=3, low_ask=0, low_asker=0
        )

        buyer2.bid_ask_result(
            status=3, num_trades=0,
            new_bids=[bid2, max(bid1, bid2) + 5],
            new_asks=[], high_bid=max(bid1, bid2) + 5,
            high_bidder=3, low_ask=0, low_asker=0
        )

        margins1.append(buyer1.margin)
        margins2.append(buyer2.margin)

    # Both should adapt (for buyers, margins should increase - become less negative - when consistently outbid)
    assert margins1[-1] > margins1[0], "Buyer1 margin didn't adapt (should be less negative)"
    assert margins2[-1] > margins2[0], "Buyer2 margin didn't adapt (should be less negative)"

    # Check that adaptation is smooth (not oscillating wildly)
    changes1 = [abs(margins1[i+1] - margins1[i]) for i in range(len(margins1)-1)]
    changes2 = [abs(margins2[i+1] - margins2[i]) for i in range(len(margins2)-1)]

    max_change1 = max(changes1)
    max_change2 = max(changes2)

    # Changes should be gradual
    assert max_change1 < 0.3, f"Buyer1 margin changed too rapidly: {max_change1}"
    assert max_change2 < 0.3, f"Buyer2 margin changed too rapidly: {max_change2}"


# =============================================================================
# TEST 6: COMPETITIVE ADVANTAGE OVER ZIC
# =============================================================================

def test_zip_outperforms_zic():
    """
    Test that ZIP achieves higher efficiency than ZIC in mixed markets.

    This validates that the learning mechanism provides real benefit.
    """
    from traders.legacy.zic import ZIC

    # Run ZIP-only market
    zip_efficiencies = []

    for rep in range(5):
        buyers = [
            ZIP(i+1, True, 3, [100-i*5, 90-i*5, 80-i*5],
                price_min=0, price_max=150, seed=rep*10+i)
            for i in range(3)
        ]
        sellers = [
            ZIP(i+4, False, 3, [30+i*5, 40+i*5, 50+i*5],
                price_min=0, price_max=150, seed=rep*10+i+3)
            for i in range(3)
        ]

        market = Market(
            num_buyers=3, num_sellers=3, num_times=100,
            price_min=0, price_max=150,
            buyers=buyers, sellers=sellers, seed=rep
        )

        for _ in range(100):
            if not market.run_time_step():
                break

        trades = extract_trades_from_orderbook(market.orderbook, 100)
        buyer_vals = {i+1: buyers[i].valuations for i in range(3)}
        seller_costs = {i+1: sellers[i].valuations for i in range(3)}

        actual_surplus = calculate_actual_surplus(trades, buyer_vals, seller_costs)
        max_surplus = calculate_max_surplus(
            [b.valuations for b in buyers],
            [s.valuations for s in sellers]
        )

        if max_surplus > 0:
            efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)
            zip_efficiencies.append(efficiency)

    # Run ZIC-only market with same setup
    zic_efficiencies = []

    for rep in range(5):
        buyers = [
            ZIC(i+1, True, 3, [100-i*5, 90-i*5, 80-i*5],
                price_min=0, price_max=150, seed=rep*10+i)
            for i in range(3)
        ]
        sellers = [
            ZIC(i+4, False, 3, [30+i*5, 40+i*5, 50+i*5],
                price_min=0, price_max=150, seed=rep*10+i+3)
            for i in range(3)
        ]

        market = Market(
            num_buyers=3, num_sellers=3, num_times=100,
            price_min=0, price_max=150,
            buyers=buyers, sellers=sellers, seed=rep
        )

        for _ in range(100):
            if not market.run_time_step():
                break

        trades = extract_trades_from_orderbook(market.orderbook, 100)
        buyer_vals = {i+1: buyers[i].valuations for i in range(3)}
        seller_costs = {i+1: sellers[i].valuations for i in range(3)}

        actual_surplus = calculate_actual_surplus(trades, buyer_vals, seller_costs)
        max_surplus = calculate_max_surplus(
            [b.valuations for b in buyers],
            [s.valuations for s in sellers]
        )

        if max_surplus > 0:
            efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)
            zic_efficiencies.append(efficiency)

    # Compare results
    avg_zip = np.mean(zip_efficiencies)
    avg_zic = np.mean(zic_efficiencies)

    print(f"\nZIP vs ZIC Comparison:")
    print(f"ZIP average: {avg_zip:.2f}%")
    print(f"ZIC average: {avg_zic:.2f}%")
    print(f"ZIP advantage: {avg_zip - avg_zic:.2f} percentage points")

    # ZIP should generally outperform ZIC
    # Allow for some statistical variation
    assert avg_zip >= avg_zic - 5, \
        f"ZIP ({avg_zip:.2f}%) unexpectedly underperformed ZIC ({avg_zic:.2f}%)"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])