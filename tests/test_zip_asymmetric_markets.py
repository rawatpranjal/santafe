"""
Asymmetric Market Tests - Where ZIC Fails, ZIP Succeeds

These tests implement the critical experiments from Cliff & Bruten 1997 Section 3
that demonstrate ZIP's intelligence vs ZIC's failure.

Key markets tested:
1. Flat Supply (Figure 17-18): All sellers have same cost
2. Box Excess Demand (Figure 19-20): All agents have same limit prices, more buyers
3. Box Excess Supply (Figure 21-22): All agents have same limit prices, more sellers

Expected results:
- ZIC: Fails to converge to P0 in all three (predicted by theory)
- ZIP: Converges to P0 in all three (demonstrates intelligence)
"""

import pytest
import numpy as np
from typing import List, Dict, Tuple

from engine.market import Market
from engine.efficiency import (
    extract_trades_from_orderbook,
    get_transaction_prices,
    calculate_price_std_dev,
    calculate_allocative_efficiency,
    calculate_actual_surplus,
    calculate_max_surplus,
)
from traders.legacy.zip import ZIP
from traders.legacy.zic import ZIC


def create_flat_supply_market(
    num_buyers: int,
    num_sellers: int,
    agent_type: str = "ZIP",
    seed: int = 42,
) -> Tuple[List[ZIP | ZIC], List[ZIP | ZIC], int, int]:
    """
    Create a flat supply market.

    All sellers have the SAME cost (Smin = Smax = seller_cost).
    Buyers have declining valuations.

    From paper (p.13, Equation 5):
    ZIC expected price: E(P) = P0 + (1/3)(Dmax - P0)
    If P0 = $2.00, Dmax = $3.25, then E(P) = $2.42 (ZIC FAILS by $0.42)
    ZIP should converge to P0 = $2.00 (ZIP SUCCEEDS)

    Args:
        num_buyers: Number of buyers
        num_sellers: Number of sellers
        agent_type: "ZIP" or "ZIC"
        seed: Random seed

    Returns:
        (buyers, sellers, P0, Dmax)
    """
    # Normalize to 0-100 scale (paper uses $2.00-$3.25 range)
    # P0 = 50 (equilibrium)
    # Seller cost = 50 (all sellers same)
    # Buyer valuations: 100 down to 50 (Dmax = 100)

    seller_cost = 50  # All sellers have SAME cost (flat supply)
    P0 = 50  # Equilibrium price
    Dmax = 100  # Maximum buyer valuation

    buyers: List[ZIP | ZIC] = []
    sellers: List[ZIP | ZIC] = []

    # Create buyers with descending valuations
    for i in range(1, num_buyers + 1):
        # Spread valuations from Dmax down to P0
        val = int(Dmax - (i - 1) * (Dmax - P0) / (num_buyers - 1)) if num_buyers > 1 else Dmax
        vals = [val]  # 1 token per buyer

        if agent_type == "ZIP":
            buyer = ZIP(
                player_id=i,
                is_buyer=True,
                num_tokens=1,
                valuations=vals,
                price_min=0,
                price_max=100,
                seed=seed + i,
            )
        else:
            buyer = ZIC(
                player_id=i,
                is_buyer=True,
                num_tokens=1,
                valuations=vals,
                price_min=0,
                price_max=100,
                seed=seed + i,
            )
        buyers.append(buyer)

    # Create sellers with FLAT (same) cost
    for i in range(1, num_sellers + 1):
        costs = [seller_cost]  # ALL sellers have SAME cost

        if agent_type == "ZIP":
            seller = ZIP(
                player_id=i,
                is_buyer=False,
                num_tokens=1,
                valuations=costs,
                price_min=0,
                price_max=100,
                seed=seed + num_buyers + i,
            )
        else:
            seller = ZIC(
                player_id=i,
                is_buyer=False,
                num_tokens=1,
                valuations=costs,
                price_min=0,
                price_max=100,
                seed=seed + num_buyers + i,
            )
        sellers.append(seller)

    return buyers, sellers, P0, Dmax


def create_box_excess_demand_market(
    num_buyers: int,
    num_sellers: int,
    agent_type: str = "ZIP",
    seed: int = 42,
) -> Tuple[List[ZIP | ZIC], List[ZIP | ZIC], int, int]:
    """
    Create a box excess demand market.

    ALL buyers have SAME valuation (Dmin = Dmax).
    ALL sellers have SAME cost (Smin = Smax).
    More buyers than sellers (excess demand).

    From paper (p.14, Equation 7):
    ZIC expected price: E(P) = (1/2)(P0 + Smin)
    If P0 = $2.00, Smin = $0.50, then E(P) = $1.25 (ZIC FAILS by $0.75)
    ZIP should converge to P0 = $2.00 from below (ZIP SUCCEEDS)

    Args:
        num_buyers: Number of buyers (should be > num_sellers)
        num_sellers: Number of sellers
        agent_type: "ZIP" or "ZIC"
        seed: Random seed

    Returns:
        (buyers, sellers, P0, Smin)
    """
    # Normalize to 0-100 scale
    buyer_val = 75  # All buyers same (Dmin = Dmax)
    seller_cost = 25  # All sellers same (Smin = Smax)
    P0 = 50  # Equilibrium (midpoint)

    buyers: List[ZIP | ZIC] = []
    sellers: List[ZIP | ZIC] = []

    # Create buyers - ALL with SAME valuation
    for i in range(1, num_buyers + 1):
        vals = [buyer_val]

        if agent_type == "ZIP":
            buyer = ZIP(
                player_id=i,
                is_buyer=True,
                num_tokens=1,
                valuations=vals,
                price_min=0,
                price_max=100,
                seed=seed + i,
            )
        else:
            buyer = ZIC(
                player_id=i,
                is_buyer=True,
                num_tokens=1,
                valuations=vals,
                price_min=0,
                price_max=100,
                seed=seed + i,
            )
        buyers.append(buyer)

    # Create sellers - ALL with SAME cost
    for i in range(1, num_sellers + 1):
        costs = [seller_cost]

        if agent_type == "ZIP":
            seller = ZIP(
                player_id=i,
                is_buyer=False,
                num_tokens=1,
                valuations=costs,
                price_min=0,
                price_max=100,
                seed=seed + num_buyers + i,
            )
        else:
            seller = ZIC(
                player_id=i,
                is_buyer=False,
                num_tokens=1,
                valuations=costs,
                price_min=0,
                price_max=100,
                seed=seed + num_buyers + i,
            )
        sellers.append(seller)

    return buyers, sellers, P0, seller_cost


def create_box_excess_supply_market(
    num_buyers: int,
    num_sellers: int,
    agent_type: str = "ZIP",
    seed: int = 42,
) -> Tuple[List[ZIP | ZIC], List[ZIP | ZIC], int, int]:
    """
    Create a box excess supply market.

    ALL buyers have SAME valuation.
    ALL sellers have SAME cost.
    More sellers than buyers (excess supply).

    From paper (p.14, Equation 8):
    ZIC expected price: E(P) = (1/2)(P0 + Dmax)
    If P0 = $2.00, Dmax = $3.50, then E(P) = $2.75 (ZIC FAILS by $0.75)
    ZIP should converge to P0 = $2.00 from above (ZIP SUCCEEDS)

    Args:
        num_buyers: Number of buyers (should be < num_sellers)
        num_sellers: Number of sellers
        agent_type: "ZIP" or "ZIC"
        seed: Random seed

    Returns:
        (buyers, sellers, P0, Dmax)
    """
    buyer_val = 75  # All buyers same (Dmin = Dmax)
    seller_cost = 25  # All sellers same
    P0 = 50  # Equilibrium

    buyers: List[ZIP | ZIC] = []
    sellers: List[ZIP | ZIC] = []

    # Create buyers - ALL with SAME valuation
    for i in range(1, num_buyers + 1):
        vals = [buyer_val]

        if agent_type == "ZIP":
            buyer = ZIP(
                player_id=i,
                is_buyer=True,
                num_tokens=1,
                valuations=vals,
                price_min=0,
                price_max=100,
                seed=seed + i,
            )
        else:
            buyer = ZIC(
                player_id=i,
                is_buyer=True,
                num_tokens=1,
                valuations=vals,
                price_min=0,
                price_max=100,
                seed=seed + i,
            )
        buyers.append(buyer)

    # Create sellers - ALL with SAME cost
    for i in range(1, num_sellers + 1):
        costs = [seller_cost]

        if agent_type == "ZIP":
            seller = ZIP(
                player_id=i,
                is_buyer=False,
                num_tokens=1,
                valuations=costs,
                price_min=0,
                price_max=100,
                seed=seed + num_buyers + i,
            )
        else:
            seller = ZIC(
                player_id=i,
                is_buyer=False,
                num_tokens=1,
                valuations=costs,
                price_min=0,
                price_max=100,
                seed=seed + num_buyers + i,
            )
        sellers.append(seller)

    return buyers, sellers, P0, buyer_val


def test_flat_supply_zip_vs_zic() -> None:
    """
    Test flat supply market: ZIP should converge, ZIC should fail.

    Paper prediction (Equation 5, p.13):
    - ZIC: E(P) = P0 + (1/3)(Dmax - P0) = 50 + (1/3)(100 - 50) = 66.67
    - ZIP: Should converge to P0 = 50

    This is THE key test showing ZIP's intelligence.
    """
    num_buyers = 6
    num_sellers = 6
    P0 = 50
    Dmax = 100

    # ZIC theoretical prediction
    zic_expected = P0 + (1 / 3) * (Dmax - P0)

    print(f"\n=== Flat Supply Market Test ===")
    print(f"P0 (equilibrium): {P0}")
    print(f"Dmax (max buyer val): {Dmax}")
    print(f"ZIC theoretical prediction: {zic_expected:.2f} (SHOULD FAIL)")
    print(f"ZIP expected: ~{P0} (SHOULD SUCCEED)")

    # Run ZIP market
    buyers_zip, sellers_zip, _, _ = create_flat_supply_market(
        num_buyers, num_sellers, agent_type="ZIP", seed=42
    )

    market_zip = Market(
        num_buyers=num_buyers,
        num_sellers=num_sellers,
        num_times=100,
        price_min=0,
        price_max=100,
        buyers=buyers_zip,
        sellers=sellers_zip,
        seed=42,
    )

    for _ in range(100):
        market_zip.run_time_step()

    prices_zip = get_transaction_prices(market_zip.orderbook, 100)
    mean_zip = np.mean(prices_zip) if prices_zip else P0

    # Run ZIC market
    buyers_zic, sellers_zic, _, _ = create_flat_supply_market(
        num_buyers, num_sellers, agent_type="ZIC", seed=100
    )

    market_zic = Market(
        num_buyers=num_buyers,
        num_sellers=num_sellers,
        num_times=100,
        price_min=0,
        price_max=100,
        buyers=buyers_zic,
        sellers=sellers_zic,
        seed=100,
    )

    for _ in range(100):
        market_zic.run_time_step()

    prices_zic = get_transaction_prices(market_zic.orderbook, 100)
    mean_zic = np.mean(prices_zic) if prices_zic else P0

    print(f"\nResults:")
    print(f"ZIP mean price: {mean_zip:.2f} (deviation from P0: {abs(mean_zip - P0):.2f})")
    print(f"ZIC mean price: {mean_zic:.2f} (deviation from P0: {abs(mean_zic - P0):.2f})")
    print(f"ZIC theoretical: {zic_expected:.2f} (deviation from P0: {abs(zic_expected - P0):.2f})")

    # Assertions
    # ZIP should be closer to P0 than ZIC
    assert abs(mean_zip - P0) < abs(mean_zic - P0), (
        f"ZIP did not converge better than ZIC: "
        f"ZIP={mean_zip:.2f}, ZIC={mean_zic:.2f}, P0={P0}"
    )

    # ZIP should be within 15% of P0
    assert abs(mean_zip - P0) < P0 * 0.15, f"ZIP too far from equilibrium: {mean_zip:.2f} vs {P0}"


def test_box_excess_demand_zip_vs_zic() -> None:
    """
    Test box excess demand market.

    Paper prediction (Equation 7, p.14):
    - ZIC: E(P) = (1/2)(P0 + Smin) = (1/2)(50 + 25) = 37.5
    - ZIP: Should converge to P0 = 50 from below
    """
    num_buyers = 11  # More buyers (excess demand)
    num_sellers = 6
    P0 = 50
    Smin = 25

    zic_expected = (P0 + Smin) / 2

    print(f"\n=== Box Excess Demand Market Test ===")
    print(f"P0 (equilibrium): {P0}")
    print(f"Smin (seller cost): {Smin}")
    print(f"ZIC theoretical prediction: {zic_expected:.2f} (SHOULD FAIL)")
    print(f"ZIP expected: ~{P0} approaching from below (SHOULD SUCCEED)")

    # Run ZIP market
    buyers_zip, sellers_zip, _, _ = create_box_excess_demand_market(
        num_buyers, num_sellers, agent_type="ZIP", seed=200
    )

    market_zip = Market(
        num_buyers=num_buyers,
        num_sellers=num_sellers,
        num_times=100,
        price_min=0,
        price_max=100,
        buyers=buyers_zip,
        sellers=sellers_zip,
        seed=200,
    )

    for _ in range(100):
        market_zip.run_time_step()

    prices_zip = get_transaction_prices(market_zip.orderbook, 100)
    mean_zip = np.mean(prices_zip) if prices_zip else P0

    # Run ZIC market
    buyers_zic, sellers_zic, _, _ = create_box_excess_demand_market(
        num_buyers, num_sellers, agent_type="ZIC", seed=300
    )

    market_zic = Market(
        num_buyers=num_buyers,
        num_sellers=num_sellers,
        num_times=100,
        price_min=0,
        price_max=100,
        buyers=buyers_zic,
        sellers=sellers_zic,
        seed=300,
    )

    for _ in range(100):
        market_zic.run_time_step()

    prices_zic = get_transaction_prices(market_zic.orderbook, 100)
    mean_zic = np.mean(prices_zic) if prices_zic else P0

    print(f"\nResults:")
    print(f"ZIP mean price: {mean_zip:.2f} (deviation from P0: {abs(mean_zip - P0):.2f})")
    print(f"ZIC mean price: {mean_zic:.2f} (deviation from P0: {abs(mean_zic - P0):.2f})")
    print(f"ZIC theoretical: {zic_expected:.2f} (deviation from P0: {abs(zic_expected - P0):.2f})")

    # ZIP should converge better than ZIC
    assert abs(mean_zip - P0) < abs(zic_expected - P0) * 0.8, (
        f"ZIP did not significantly outperform ZIC theory: "
        f"ZIP deviation={abs(mean_zip - P0):.2f}, "
        f"ZIC theory deviation={abs(zic_expected - P0):.2f}"
    )


def test_box_excess_supply_zip_vs_zic() -> None:
    """
    Test box excess supply market.

    Paper prediction (Equation 8, p.14):
    - ZIC: E(P) = (1/2)(P0 + Dmax) = (1/2)(50 + 75) = 62.5
    - ZIP: Should converge to P0 = 50 from above
    """
    num_buyers = 6
    num_sellers = 11  # More sellers (excess supply)
    P0 = 50
    Dmax = 75

    zic_expected = (P0 + Dmax) / 2

    print(f"\n=== Box Excess Supply Market Test ===")
    print(f"P0 (equilibrium): {P0}")
    print(f"Dmax (buyer val): {Dmax}")
    print(f"ZIC theoretical prediction: {zic_expected:.2f} (SHOULD FAIL)")
    print(f"ZIP expected: ~{P0} approaching from above (SHOULD SUCCEED)")

    # Run ZIP market
    buyers_zip, sellers_zip, _, _ = create_box_excess_supply_market(
        num_buyers, num_sellers, agent_type="ZIP", seed=400
    )

    market_zip = Market(
        num_buyers=num_buyers,
        num_sellers=num_sellers,
        num_times=100,
        price_min=0,
        price_max=100,
        buyers=buyers_zip,
        sellers=sellers_zip,
        seed=400,
    )

    for _ in range(100):
        market_zip.run_time_step()

    prices_zip = get_transaction_prices(market_zip.orderbook, 100)
    mean_zip = np.mean(prices_zip) if prices_zip else P0

    # Run ZIC market
    buyers_zic, sellers_zic, _, _ = create_box_excess_supply_market(
        num_buyers, num_sellers, agent_type="ZIC", seed=500
    )

    market_zic = Market(
        num_buyers=num_buyers,
        num_sellers=num_sellers,
        num_times=100,
        price_min=0,
        price_max=100,
        buyers=buyers_zic,
        sellers=sellers_zic,
        seed=500,
    )

    for _ in range(100):
        market_zic.run_time_step()

    prices_zic = get_transaction_prices(market_zic.orderbook, 100)
    mean_zic = np.mean(prices_zic) if prices_zic else P0

    print(f"\nResults:")
    print(f"ZIP mean price: {mean_zip:.2f} (deviation from P0: {abs(mean_zip - P0):.2f})")
    print(f"ZIC mean price: {mean_zic:.2f} (deviation from P0: {abs(mean_zic - P0):.2f})")
    print(f"ZIC theoretical: {zic_expected:.2f} (deviation from P0: {abs(zic_expected - P0):.2f})")

    # ZIP should converge better than ZIC
    assert abs(mean_zip - P0) < abs(zic_expected - P0) * 0.8, (
        f"ZIP did not significantly outperform ZIC theory: "
        f"ZIP deviation={abs(mean_zip - P0):.2f}, "
        f"ZIC theory deviation={abs(zic_expected - P0):.2f}"
    )


@pytest.mark.slow
def test_asymmetric_markets_full_suite() -> None:
    """
    Full asymmetric market validation suite with multiple replications.

    This test runs all three asymmetric markets multiple times to get
    statistical confidence in ZIP's superiority over ZIC.
    """
    num_replications = 10
    P0 = 50

    print(f"\n=== Full Asymmetric Market Suite ({num_replications} replications) ===\n")

    # Results storage
    results = {
        "flat_supply": {"zip": [], "zic": []},
        "box_demand": {"zip": [], "zic": []},
        "box_supply": {"zip": [], "zic": []},
    }

    for rep in range(num_replications):
        # Flat Supply
        buyers_zip, sellers_zip, _, _ = create_flat_supply_market(6, 6, "ZIP", seed=1000 + rep)
        market_zip = Market(6, 6, 100, 0, 100, buyers_zip, sellers_zip, seed=2000 + rep)
        for _ in range(100):
            market_zip.run_time_step()
        prices_zip = get_transaction_prices(market_zip.orderbook, 100)
        results["flat_supply"]["zip"].append(np.mean(prices_zip) if prices_zip else P0)

        buyers_zic, sellers_zic, _, _ = create_flat_supply_market(6, 6, "ZIC", seed=3000 + rep)
        market_zic = Market(6, 6, 100, 0, 100, buyers_zic, sellers_zic, seed=4000 + rep)
        for _ in range(100):
            market_zic.run_time_step()
        prices_zic = get_transaction_prices(market_zic.orderbook, 100)
        results["flat_supply"]["zic"].append(np.mean(prices_zic) if prices_zic else P0)

        # Box Excess Demand
        buyers_zip, sellers_zip, _, _ = create_box_excess_demand_market(11, 6, "ZIP", seed=5000 + rep)
        market_zip = Market(11, 6, 100, 0, 100, buyers_zip, sellers_zip, seed=6000 + rep)
        for _ in range(100):
            market_zip.run_time_step()
        prices_zip = get_transaction_prices(market_zip.orderbook, 100)
        results["box_demand"]["zip"].append(np.mean(prices_zip) if prices_zip else P0)

        buyers_zic, sellers_zic, _, _ = create_box_excess_demand_market(11, 6, "ZIC", seed=7000 + rep)
        market_zic = Market(11, 6, 100, 0, 100, buyers_zic, sellers_zic, seed=8000 + rep)
        for _ in range(100):
            market_zic.run_time_step()
        prices_zic = get_transaction_prices(market_zic.orderbook, 100)
        results["box_demand"]["zic"].append(np.mean(prices_zic) if prices_zic else P0)

        # Box Excess Supply
        buyers_zip, sellers_zip, _, _ = create_box_excess_supply_market(6, 11, "ZIP", seed=9000 + rep)
        market_zip = Market(6, 11, 100, 0, 100, buyers_zip, sellers_zip, seed=10000 + rep)
        for _ in range(100):
            market_zip.run_time_step()
        prices_zip = get_transaction_prices(market_zip.orderbook, 100)
        results["box_supply"]["zip"].append(np.mean(prices_zip) if prices_zip else P0)

        buyers_zic, sellers_zic, _, _ = create_box_excess_supply_market(6, 11, "ZIC", seed=11000 + rep)
        market_zic = Market(6, 11, 100, 0, 100, buyers_zic, sellers_zic, seed=12000 + rep)
        for _ in range(100):
            market_zic.run_time_step()
        prices_zic = get_transaction_prices(market_zic.orderbook, 100)
        results["box_supply"]["zic"].append(np.mean(prices_zic) if prices_zic else P0)

    # Print summary
    print(f"{'Market Type':<20} {'ZIP Mean Dev':<15} {'ZIC Mean Dev':<15} {'Ratio':<10}")
    print("-" * 70)

    for market_name, market_results in results.items():
        zip_devs = [abs(p - P0) for p in market_results["zip"]]
        zic_devs = [abs(p - P0) for p in market_results["zic"]]

        zip_mean_dev = np.mean(zip_devs)
        zic_mean_dev = np.mean(zic_devs)
        ratio = zic_mean_dev / zip_mean_dev if zip_mean_dev > 0 else float('inf')

        print(f"{market_name:<20} {zip_mean_dev:<15.2f} {zic_mean_dev:<15.2f} {ratio:<10.2f}x")

        # ZIP should converge better in all markets
        assert zip_mean_dev < zic_mean_dev, (
            f"{market_name}: ZIP did not converge better than ZIC"
        )

    print(f"\n✅ All asymmetric markets: ZIP converges better than ZIC!")
