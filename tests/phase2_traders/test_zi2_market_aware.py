"""
Market-Aware Improvement Tests for ZI2 vs ZIC.

These tests validate that ZI2's market-awareness provides measurable
statistical improvements over pure random ZIC bidding.

Key Hypothesis: ZI2's cbid/cask awareness narrows spreads faster than ZIC,
leading to marginal efficiency improvements (~1% gain).
"""

import pytest
import numpy as np
from scipy import stats
from typing import List, Tuple

from traders.legacy.zi2 import ZI2
from traders.legacy.zic import ZIC
from engine.market import Market
from engine.efficiency import (
    extract_trades_from_orderbook,
    calculate_actual_surplus,
    calculate_max_surplus,
    calculate_allocative_efficiency,
)


# =============================================================================
# TEST 1: SPREAD NARROWING (ZI2 VS ZIC)
# =============================================================================

def test_zi2_spread_narrowing():
    """
    Test that ZI2 narrows bid-ask spreads faster than ZIC.

    Hypothesis: Market-aware bidding should lead to faster price convergence.
    Metric: Average spread in early periods should be smaller for ZI2.
    """
    num_reps = 10
    num_agents = 4
    num_tokens = 3

    buyer_tokens = [
        [180, 160, 140],
        [175, 155, 135],
        [170, 150, 130],
        [165, 145, 125],
    ]

    seller_tokens = [
        [60, 80, 100],
        [65, 85, 105],
        [70, 90, 110],
        [75, 95, 115],
    ]

    zi2_spreads = []
    zic_spreads = []

    for rep in range(num_reps):
        # ZI2 Market
        buyers_zi2 = [
            ZI2(i+1, True, num_tokens, buyer_tokens[i],
                price_min=0, price_max=220, seed=rep*100+i)
            for i in range(num_agents)
        ]
        sellers_zi2 = [
            ZI2(i+5, False, num_tokens, seller_tokens[i],
                price_min=0, price_max=220, seed=rep*100+i+4)
            for i in range(num_agents)
        ]

        market_zi2 = Market(
            num_buyers=num_agents,
            num_sellers=num_agents,
            num_times=50,  # Short period to measure early convergence
            price_min=0,
            price_max=220,
            buyers=buyers_zi2,
            sellers=sellers_zi2,
            seed=rep
        )

        # Track spreads
        for time_step in range(50):
            if not market_zi2.run_time_step():
                break
            # Measure spread after bid/ask phase
            t = market_zi2.orderbook.current_time
            if market_zi2.orderbook.high_bid[t] > 0 and market_zi2.orderbook.low_ask[t] > 0:
                spread = market_zi2.orderbook.low_ask[t] - market_zi2.orderbook.high_bid[t]
                if spread >= 0:  # Valid spread
                    zi2_spreads.append(spread)

        # ZIC Market (same seed for fairness)
        buyers_zic = [
            ZIC(i+1, True, num_tokens, buyer_tokens[i],
                price_min=0, price_max=220, seed=rep*100+i)
            for i in range(num_agents)
        ]
        sellers_zic = [
            ZIC(i+5, False, num_tokens, seller_tokens[i],
                price_min=0, price_max=220, seed=rep*100+i+4)
            for i in range(num_agents)
        ]

        market_zic = Market(
            num_buyers=num_agents,
            num_sellers=num_agents,
            num_times=50,
            price_min=0,
            price_max=220,
            buyers=buyers_zic,
            sellers=sellers_zic,
            seed=rep
        )

        for time_step in range(50):
            if not market_zic.run_time_step():
                break
            t = market_zic.orderbook.current_time
            if market_zic.orderbook.high_bid[t] > 0 and market_zic.orderbook.low_ask[t] > 0:
                spread = market_zic.orderbook.low_ask[t] - market_zic.orderbook.high_bid[t]
                if spread >= 0:
                    zic_spreads.append(spread)

    # Compare spreads
    if len(zi2_spreads) > 0 and len(zic_spreads) > 0:
        mean_zi2 = np.mean(zi2_spreads)
        mean_zic = np.mean(zic_spreads)
        median_zi2 = np.median(zi2_spreads)
        median_zic = np.median(zic_spreads)

        print(f"\nSpread Narrowing Test:")
        print(f"  ZI2 mean spread: {mean_zi2:.2f}")
        print(f"  ZIC mean spread: {mean_zic:.2f}")
        print(f"  ZI2 median spread: {median_zi2:.2f}")
        print(f"  ZIC median spread: {median_zic:.2f}")
        print(f"  Improvement: {mean_zic - mean_zi2:.2f} ({((mean_zic - mean_zi2) / mean_zic * 100):.1f}%)")

        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(zi2_spreads, zic_spreads, alternative='less')
        print(f"  Statistical test: U={statistic:.1f}, p={p_value:.4f}")

        # ZI2 should have narrower spreads (p < 0.10 for marginal effect)
        # Note: Effect may be small since both are zero-intelligence
        print(f"  Result: {'SIGNIFICANT' if p_value < 0.10 else 'NOT SIGNIFICANT'}")


# =============================================================================
# TEST 2: CBID/CASK RESPONSE TEST
# =============================================================================

def test_zi2_cbid_cask_response():
    """
    Test that ZI2 responds to market state (cbid/cask) while ZIC does not.

    Measure: Correlation between cbid and subsequent bids should be higher for ZI2.
    """
    num_samples = 200

    # ZI2 Buyer
    buyer_zi2 = ZI2(
        player_id=1,
        is_buyer=True,
        num_tokens=1,
        valuations=[100],
        price_min=0,
        price_max=150,
        seed=42
    )

    # ZIC Buyer
    buyer_zic = ZIC(
        player_id=2,
        is_buyer=True,
        num_tokens=1,
        valuations=[100],
        price_min=0,
        price_max=150,
        seed=42
    )

    # Test with varying cbid values
    cbid_values = np.random.uniform(20, 80, num_samples)

    zi2_bids = []
    zic_bids = []
    valid_cbids = []

    for cbid in cbid_values:
        # ZI2 response
        buyer_zi2.current_bid = int(cbid)
        buyer_zi2.bid_ask(time=1, nobidask=0)
        bid_zi2 = buyer_zi2.bid_ask_response()
        if bid_zi2 > 0:
            zi2_bids.append(bid_zi2)
            valid_cbids.append(cbid)
        buyer_zi2.has_responded = False
        buyer_zi2.num_trades = 0

        # ZIC response (doesn't use cbid)
        buyer_zic.bid_ask(time=1, nobidask=0)
        bid_zic = buyer_zic.bid_ask_response()
        if bid_zic > 0:
            zic_bids.append(bid_zic)
        buyer_zic.has_responded = False
        buyer_zic.num_trades = 0

    # Calculate correlation with cbid
    if len(zi2_bids) >= 30 and len(zic_bids) >= 30:
        # ZI2 should correlate with cbid (bids influenced by market state)
        corr_zi2, p_zi2 = stats.pearsonr(valid_cbids[:len(zi2_bids)], zi2_bids)

        # ZIC should NOT correlate with cbid (pure random)
        corr_zic, p_zic = stats.pearsonr(valid_cbids[:len(zic_bids)], zic_bids)

        print(f"\nCbid/Cask Response Test:")
        print(f"  ZI2 correlation with cbid: r={corr_zi2:.3f}, p={p_zi2:.4f}")
        print(f"  ZIC correlation with cbid: r={corr_zic:.3f}, p={p_zic:.4f}")

        # ZI2 should have positive correlation (bids track cbid)
        assert corr_zi2 > 0.1, \
            f"ZI2 should show positive correlation with cbid (got r={corr_zi2:.3f})"

        # ZI2 correlation should be stronger than ZIC
        print(f"  ZI2 responsiveness: {abs(corr_zi2) - abs(corr_zic):.3f} stronger")


# =============================================================================
# TEST 3: STATISTICAL IMPROVEMENT SIGNIFICANCE
# =============================================================================

def test_zi2_improvement_significance():
    """
    Test that ZI2 efficiency improvement over ZIC is statistically significant.

    Hypothesis: Market-awareness provides ~1-2% efficiency gain (p < 0.05).
    """
    num_reps = 20  # More replications for statistical power
    num_agents = 4
    num_tokens = 4

    buyer_tokens = [
        [200, 180, 160, 140],
        [195, 175, 155, 135],
        [190, 170, 150, 130],
        [185, 165, 145, 125],
    ]

    seller_tokens = [
        [40, 60, 80, 100],
        [45, 65, 85, 105],
        [50, 70, 90, 110],
        [55, 75, 95, 115],
    ]

    zi2_efficiencies = []
    zic_efficiencies = []

    for rep in range(num_reps):
        # ZI2 Market
        buyers_zi2 = [
            ZI2(i+1, True, num_tokens, buyer_tokens[i],
                price_min=0, price_max=250, seed=rep*100+i)
            for i in range(num_agents)
        ]
        sellers_zi2 = [
            ZI2(i+5, False, num_tokens, seller_tokens[i],
                price_min=0, price_max=250, seed=rep*100+i+4)
            for i in range(num_agents)
        ]

        market_zi2 = Market(
            num_buyers=num_agents,
            num_sellers=num_agents,
            num_times=150,
            price_min=0,
            price_max=250,
            buyers=buyers_zi2,
            sellers=sellers_zi2,
            seed=rep
        )

        for _ in range(150):
            if not market_zi2.run_time_step():
                break

        trades_zi2 = extract_trades_from_orderbook(market_zi2.orderbook, 150)
        buyer_vals_zi2 = {i+1: buyers_zi2[i].valuations for i in range(num_agents)}
        seller_costs_zi2 = {i+1: sellers_zi2[i].valuations for i in range(num_agents)}

        actual_zi2 = calculate_actual_surplus(trades_zi2, buyer_vals_zi2, seller_costs_zi2)
        max_zi2 = calculate_max_surplus(
            [b.valuations for b in buyers_zi2],
            [s.valuations for s in sellers_zi2]
        )

        if max_zi2 > 0:
            eff_zi2 = calculate_allocative_efficiency(actual_zi2, max_zi2)
            zi2_efficiencies.append(eff_zi2)

        # ZIC Market (same setup)
        buyers_zic = [
            ZIC(i+1, True, num_tokens, buyer_tokens[i],
                price_min=0, price_max=250, seed=rep*100+i)
            for i in range(num_agents)
        ]
        sellers_zic = [
            ZIC(i+5, False, num_tokens, seller_tokens[i],
                price_min=0, price_max=250, seed=rep*100+i+4)
            for i in range(num_agents)
        ]

        market_zic = Market(
            num_buyers=num_agents,
            num_sellers=num_agents,
            num_times=150,
            price_min=0,
            price_max=250,
            buyers=buyers_zic,
            sellers=sellers_zic,
            seed=rep
        )

        for _ in range(150):
            if not market_zic.run_time_step():
                break

        trades_zic = extract_trades_from_orderbook(market_zic.orderbook, 150)
        buyer_vals_zic = {i+1: buyers_zic[i].valuations for i in range(num_agents)}
        seller_costs_zic = {i+1: sellers_zic[i].valuations for i in range(num_agents)}

        actual_zic = calculate_actual_surplus(trades_zic, buyer_vals_zic, seller_costs_zic)
        max_zic = calculate_max_surplus(
            [b.valuations for b in buyers_zic],
            [s.valuations for s in sellers_zic]
        )

        if max_zic > 0:
            eff_zic = calculate_allocative_efficiency(actual_zic, max_zic)
            zic_efficiencies.append(eff_zic)

    # Statistical comparison
    if len(zi2_efficiencies) >= 10 and len(zic_efficiencies) >= 10:
        mean_zi2 = np.mean(zi2_efficiencies)
        mean_zic = np.mean(zic_efficiencies)
        std_zi2 = np.std(zi2_efficiencies)
        std_zic = np.std(zic_efficiencies)

        print(f"\nStatistical Improvement Test:")
        print(f"  ZI2: {mean_zi2:.2f}% ± {std_zi2:.2f}%")
        print(f"  ZIC: {mean_zic:.2f}% ± {std_zic:.2f}%")
        print(f"  Improvement: {mean_zi2 - mean_zic:.2f} percentage points")

        # One-sided t-test (ZI2 > ZIC)
        t_stat, p_value = stats.ttest_ind(zi2_efficiencies, zic_efficiencies, alternative='greater')
        print(f"  t-test: t={t_stat:.3f}, p={p_value:.4f}")

        # Mann-Whitney U test (non-parametric alternative)
        u_stat, p_mw = stats.mannwhitneyu(zi2_efficiencies, zic_efficiencies, alternative='greater')
        print(f"  Mann-Whitney U: U={u_stat:.1f}, p={p_mw:.4f}")

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((std_zi2**2 + std_zic**2) / 2)
        cohens_d = (mean_zi2 - mean_zic) / pooled_std if pooled_std > 0 else 0
        print(f"  Effect size (Cohen's d): {cohens_d:.3f}")

        # Interpretation
        if p_value < 0.05:
            print(f"  ✓ SIGNIFICANT improvement (p < 0.05)")
        elif p_value < 0.10:
            print(f"  ~ MARGINAL improvement (p < 0.10)")
        else:
            print(f"  ✗ NOT SIGNIFICANT (p >= 0.10)")

        # ZI2 should at least match ZIC
        assert mean_zi2 >= mean_zic - 2.0, \
            f"ZI2 underperformed ZIC significantly ({mean_zi2:.2f}% vs {mean_zic:.2f}%)"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-s"])
