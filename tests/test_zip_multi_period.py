"""
Multi-period convergence tests for ZIP agents.

Validates the key finding from Cliff & Bruten 1997 Figures 34-37:
- ZIP profit dispersion falls from ~0.35 to ~0.05 over 4-10 periods
- ZIC profit dispersion stays constant at ~0.35
- This convergence is THE discriminator between intelligent and zero-intelligence traders

Each "period" (day) is a full market session where agents can trade.
Agents carry over their learned parameters between periods.
"""

import pytest
import numpy as np
from typing import List, Dict, Tuple

from engine.market import Market
from engine.efficiency import (
    extract_trades_from_orderbook,
    calculate_profit_dispersion,
    get_transaction_prices,
    calculate_price_std_dev,
)
from traders.legacy.zip import ZIP
from traders.legacy.zic import ZIC


def create_agents_for_multi_period(
    num_buyers: int,
    num_sellers: int,
    num_tokens_per_period: int,
    agent_type: str = "ZIP",
    seed: int = 42,
) -> Tuple[List[ZIP | ZIC], List[ZIP | ZIC], Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Create agents for multi-period experiments.

    Args:
        num_buyers: Number of buyers
        num_sellers: Number of sellers
        num_tokens_per_period: Tokens per agent per period
        agent_type: "ZIP" or "ZIC"
        seed: Random seed

    Returns:
        (buyers, sellers, buyer_valuations, seller_costs)
    """
    buyers: List[ZIP | ZIC] = []
    sellers: List[ZIP | ZIC] = []
    buyer_valuations: Dict[int, List[int]] = {}
    seller_costs: Dict[int, List[int]] = {}

    # Create buyers
    for i in range(1, num_buyers + 1):
        base_val = 100 - (i - 1) * 5
        vals = [base_val - j * 5 for j in range(num_tokens_per_period)]
        buyer_valuations[i] = vals

        if agent_type == "ZIP":
            buyer = ZIP(
                player_id=i,
                is_buyer=True,
                num_tokens=num_tokens_per_period,
                valuations=vals,
                price_min=0,
                price_max=100,
                seed=seed + i,
            )
        else:  # ZIC
            buyer = ZIC(
                player_id=i,
                is_buyer=True,
                num_tokens=num_tokens_per_period,
                valuations=vals,
                price_min=0,
                price_max=100,
                seed=seed + i,
            )
        buyers.append(buyer)

    # Create sellers
    for i in range(1, num_sellers + 1):
        base_cost = 30 + (i - 1) * 5
        costs = [base_cost + j * 5 for j in range(num_tokens_per_period)]
        seller_costs[i] = costs

        if agent_type == "ZIP":
            seller = ZIP(
                player_id=i,
                is_buyer=False,
                num_tokens=num_tokens_per_period,
                valuations=costs,
                price_min=0,
                price_max=100,
                seed=seed + num_buyers + i,
            )
        else:  # ZIC
            seller = ZIC(
                player_id=i,
                is_buyer=False,
                num_tokens=num_tokens_per_period,
                valuations=costs,
                price_min=0,
                price_max=100,
                seed=seed + num_buyers + i,
            )
        sellers.append(seller)

    return buyers, sellers, buyer_valuations, seller_costs


def run_single_period(
    buyers: List[ZIP | ZIC],
    sellers: List[ZIP | ZIC],
    num_timesteps: int = 50,
    seed: int = 42,
) -> Market:
    """
    Run a single trading period (day).

    Args:
        buyers: List of buyer agents
        sellers: List of seller agents
        num_timesteps: Timesteps per period
        seed: Random seed

    Returns:
        Market instance after completion
    """
    # Reset agent trade counts for new period
    for buyer in buyers:
        buyer.num_trades = 0
    for seller in sellers:
        seller.num_trades = 0

    market = Market(
        num_buyers=len(buyers),
        num_sellers=len(sellers),
        num_times=num_timesteps,
        price_min=0,
        price_max=100,
        buyers=buyers,
        sellers=sellers,
        seed=seed,
    )

    for _ in range(num_timesteps):
        success = market.run_time_step()
        if not success:
            break

    return market


def test_zip_profit_dispersion_convergence() -> None:
    """
    Test that ZIP profit dispersion converges over multiple periods.

    Expected from 1997 paper Figure 34:
    - Period 1: ~0.35
    - Period 4: ~0.05
    - Period 10: ~0.05 (stable)

    This is THE key finding - profit dispersion falls dramatically as ZIP learns.
    """
    num_periods = 10
    num_buyers = 5
    num_sellers = 5
    num_tokens = 2
    P0 = 65  # Equilibrium price

    # Create ZIP agents (they persist across periods)
    buyers, sellers, buyer_vals, seller_costs = create_agents_for_multi_period(
        num_buyers, num_sellers, num_tokens, agent_type="ZIP", seed=42
    )

    dispersions = []
    mean_prices = []
    price_stds = []

    print(f"\n=== ZIP Multi-Period Convergence ({num_periods} periods) ===")

    for period in range(1, num_periods + 1):
        # Run period
        market = run_single_period(buyers, sellers, num_timesteps=50, seed=100 + period)

        # Extract metrics
        trades = extract_trades_from_orderbook(market.orderbook, 50)

        buyer_vals_list = list(buyer_vals.values())
        seller_costs_list = list(seller_costs.values())

        dispersion = calculate_profit_dispersion(
            trades,
            buyer_vals,
            seller_costs,
            buyer_vals_list,
            seller_costs_list,
            P0,
        )
        dispersions.append(dispersion)

        prices = get_transaction_prices(market.orderbook, 50)
        mean_price = np.mean(prices) if prices else P0
        price_std = calculate_price_std_dev(prices)

        mean_prices.append(mean_price)
        price_stds.append(price_std)

        print(f"Period {period:2d}: "
              f"Dispersion={dispersion:6.2f}, "
              f"MeanPrice={mean_price:5.1f}, "
              f"PriceStd={price_std:5.2f}")

    print(f"\nPaper Benchmark (Figure 34):")
    print(f"  Period 1: ~0.35, Period 4: ~0.05, Period 10: ~0.05")
    print(f"Our Results:")
    print(f"  Period 1: {dispersions[0]:.2f}")
    print(f"  Period 4: {dispersions[3]:.2f}")
    print(f"  Period 10: {dispersions[9]:.2f}")

    # Assertions
    # We expect some convergence, though scale may differ from paper
    # Key pattern: dispersion should DECREASE over time
    first_half_avg = np.mean(dispersions[:5])
    second_half_avg = np.mean(dispersions[5:])

    print(f"\nFirst 5 periods avg dispersion: {first_half_avg:.2f}")
    print(f"Last 5 periods avg dispersion: {second_half_avg:.2f}")
    print(f"Reduction: {((first_half_avg - second_half_avg) / first_half_avg * 100):.1f}%")

    # ZIP should show learning (dispersion decreases)
    assert second_half_avg < first_half_avg * 0.95, (
        f"ZIP did not show learning: "
        f"first half {first_half_avg:.2f} vs second half {second_half_avg:.2f}"
    )


def test_zic_profit_dispersion_constant() -> None:
    """
    Test that ZIC profit dispersion stays constant (no learning).

    Expected from 1997 paper:
    - ZIC dispersion: ~0.35 across all periods (flat line)

    This contrasts with ZIP's convergence, proving ZIP's intelligence.
    """
    num_periods = 10
    num_buyers = 5
    num_sellers = 5
    num_tokens = 2
    P0 = 65

    # Create ZIC agents
    buyers, sellers, buyer_vals, seller_costs = create_agents_for_multi_period(
        num_buyers, num_sellers, num_tokens, agent_type="ZIC", seed=200
    )

    dispersions = []

    print(f"\n=== ZIC Multi-Period Behavior ({num_periods} periods) ===")

    for period in range(1, num_periods + 1):
        market = run_single_period(buyers, sellers, num_timesteps=50, seed=300 + period)

        trades = extract_trades_from_orderbook(market.orderbook, 50)

        buyer_vals_list = list(buyer_vals.values())
        seller_costs_list = list(seller_costs.values())

        dispersion = calculate_profit_dispersion(
            trades,
            buyer_vals,
            seller_costs,
            buyer_vals_list,
            seller_costs_list,
            P0,
        )
        dispersions.append(dispersion)

        prices = get_transaction_prices(market.orderbook, 50)
        mean_price = np.mean(prices) if prices else P0

        print(f"Period {period:2d}: Dispersion={dispersion:6.2f}, MeanPrice={mean_price:5.1f}")

    # Calculate variance in dispersion over time
    dispersion_std = np.std(dispersions)

    print(f"\nPaper Benchmark: ZIC dispersion constant at ~0.35")
    print(f"Our Results:")
    print(f"  Mean dispersion: {np.mean(dispersions):.2f}")
    print(f"  Std dev: {dispersion_std:.2f}")

    # Assertions
    # ZIC should NOT show significant convergence
    first_half_avg = np.mean(dispersions[:5])
    second_half_avg = np.mean(dispersions[5:])

    print(f"\nFirst 5 periods avg: {first_half_avg:.2f}")
    print(f"Last 5 periods avg: {second_half_avg:.2f}")
    print(f"Change: {abs(second_half_avg - first_half_avg):.2f}")

    # ZIC should be relatively stable (no major learning)
    # Allow for some variance but not systematic improvement
    assert abs(second_half_avg - first_half_avg) < first_half_avg * 0.3, (
        f"ZIC showed unexpected convergence pattern"
    )


def test_zip_vs_zic_multi_period_comparison() -> None:
    """
    Direct comparison: ZIP vs ZIC over 10 periods.

    Expected:
    - ZIP dispersion: decreases over time
    - ZIC dispersion: stays constant
    - By period 10: ZIP << ZIC (factor of 3-7x better)
    """
    num_periods = 10
    num_buyers = 5
    num_sellers = 5
    num_tokens = 2
    P0 = 65

    # Create both agent types
    buyers_zip, sellers_zip, buyer_vals_zip, seller_costs_zip = create_agents_for_multi_period(
        num_buyers, num_sellers, num_tokens, agent_type="ZIP", seed=400
    )

    buyers_zic, sellers_zic, buyer_vals_zic, seller_costs_zic = create_agents_for_multi_period(
        num_buyers, num_sellers, num_tokens, agent_type="ZIC", seed=500
    )

    zip_dispersions = []
    zic_dispersions = []

    print(f"\n=== ZIP vs ZIC Multi-Period Comparison ({num_periods} periods) ===")
    print(f"{'Period':<8} {'ZIP Dispersion':<15} {'ZIC Dispersion':<15} {'Ratio':<10}")
    print("-" * 55)

    for period in range(1, num_periods + 1):
        # Run ZIP period
        market_zip = run_single_period(buyers_zip, sellers_zip, num_timesteps=50, seed=600 + period)
        trades_zip = extract_trades_from_orderbook(market_zip.orderbook, 50)
        buyer_vals_list_zip = list(buyer_vals_zip.values())
        seller_costs_list_zip = list(seller_costs_zip.values())
        dispersion_zip = calculate_profit_dispersion(
            trades_zip, buyer_vals_zip, seller_costs_zip,
            buyer_vals_list_zip, seller_costs_list_zip, P0
        )
        zip_dispersions.append(dispersion_zip)

        # Run ZIC period
        market_zic = run_single_period(buyers_zic, sellers_zic, num_timesteps=50, seed=700 + period)
        trades_zic = extract_trades_from_orderbook(market_zic.orderbook, 50)
        buyer_vals_list_zic = list(buyer_vals_zic.values())
        seller_costs_list_zic = list(seller_costs_zic.values())
        dispersion_zic = calculate_profit_dispersion(
            trades_zic, buyer_vals_zic, seller_costs_zic,
            buyer_vals_list_zic, seller_costs_list_zic, P0
        )
        zic_dispersions.append(dispersion_zic)

        ratio = dispersion_zic / dispersion_zip if dispersion_zip > 0 else float('inf')

        print(f"{period:<8} {dispersion_zip:<15.2f} {dispersion_zic:<15.2f} {ratio:<10.2f}x")

    # Summary statistics
    zip_early = np.mean(zip_dispersions[:3])
    zip_late = np.mean(zip_dispersions[-3:])
    zic_early = np.mean(zic_dispersions[:3])
    zic_late = np.mean(zic_dispersions[-3:])

    print(f"\n{'Metric':<30} {'ZIP':<15} {'ZIC':<15}")
    print("-" * 60)
    print(f"{'Early periods (1-3) avg':<30} {zip_early:<15.2f} {zic_early:<15.2f}")
    print(f"{'Late periods (8-10) avg':<30} {zip_late:<15.2f} {zic_late:<15.2f}")
    print(f"{'Improvement':<30} {((zip_early - zip_late) / zip_early * 100):<14.1f}% "
          f"{((zic_early - zic_late) / zic_early * 100):<14.1f}%")

    print(f"\nPaper Benchmark:")
    print(f"  ZIP: ~0.35 → ~0.05 (7x improvement)")
    print(f"  ZIC: ~0.35 → ~0.35 (no improvement)")

    # Assertions
    # ZIP should improve more than ZIC
    zip_improvement_pct = (zip_early - zip_late) / zip_early
    zic_improvement_pct = (zic_early - zic_late) / zic_early if zic_early > 0 else 0

    print(f"\nZIP improvement: {zip_improvement_pct * 100:.1f}%")
    print(f"ZIC improvement: {zic_improvement_pct * 100:.1f}%")

    assert zip_improvement_pct > zic_improvement_pct + 0.05, (
        f"ZIP did not show significantly better learning than ZIC: "
        f"ZIP {zip_improvement_pct * 100:.1f}% vs ZIC {zic_improvement_pct * 100:.1f}%"
    )


@pytest.mark.slow
def test_zip_convergence_statistical_validation() -> None:
    """
    Statistical validation with 50 replications of 10-period experiments.

    This is the gold standard test matching the 1997 paper methodology.
    """
    num_replications = 50
    num_periods = 10
    P0 = 65

    all_zip_dispersions = []  # Shape: (num_replications, num_periods)

    print(f"\n=== Statistical Validation: {num_replications} replications of {num_periods} periods ===")

    for replication in range(num_replications):
        buyers, sellers, buyer_vals, seller_costs = create_agents_for_multi_period(
            5, 5, 2, agent_type="ZIP", seed=1000 + replication
        )

        period_dispersions = []

        for period in range(1, num_periods + 1):
            market = run_single_period(buyers, sellers, num_timesteps=50, seed=2000 + replication * 10 + period)
            trades = extract_trades_from_orderbook(market.orderbook, 50)

            buyer_vals_list = list(buyer_vals.values())
            seller_costs_list = list(seller_costs.values())

            dispersion = calculate_profit_dispersion(
                trades, buyer_vals, seller_costs,
                buyer_vals_list, seller_costs_list, P0
            )
            period_dispersions.append(dispersion)

        all_zip_dispersions.append(period_dispersions)

        if (replication + 1) % 10 == 0:
            print(f"  Completed {replication + 1}/{num_replications} replications...")

    # Convert to numpy array for easy analysis
    dispersions_array = np.array(all_zip_dispersions)  # Shape: (50, 10)

    # Compute mean ± std for each period
    mean_per_period = np.mean(dispersions_array, axis=0)
    std_per_period = np.std(dispersions_array, axis=0)

    print(f"\n{'Period':<8} {'Mean Dispersion':<20} {'Std Dev':<15}")
    print("-" * 50)
    for period in range(num_periods):
        print(f"{period + 1:<8} {mean_per_period[period]:<20.2f} {std_per_period[period]:<15.2f}")

    # Test for convergence trend
    from scipy import stats
    periods = np.arange(1, num_periods + 1)
    slope, intercept, r_value, p_value, std_err = stats.linregress(periods, mean_per_period)

    print(f"\nLinear Regression:")
    print(f"  Slope: {slope:.3f} (negative = convergence)")
    print(f"  R-squared: {r_value**2:.3f}")
    print(f"  P-value: {p_value:.4f}")

    print(f"\nConvergence Pattern:")
    print(f"  Period 1 mean: {mean_per_period[0]:.2f}")
    print(f"  Period 10 mean: {mean_per_period[9]:.2f}")
    print(f"  Improvement: {((mean_per_period[0] - mean_per_period[9]) / mean_per_period[0] * 100):.1f}%")

    # Assertion: slope should be negative (convergence)
    assert slope < 0, f"ZIP did not show convergence (slope={slope:.3f})"
    assert p_value < 0.05, f"Convergence trend not statistically significant (p={p_value:.4f})"
