"""
Diagnostic tests to investigate asymmetric market failures.

This module contains detailed analysis to understand why ZIP is not
outperforming ZIC in asymmetric markets as predicted by the paper.
"""

import numpy as np
from typing import List, Dict, Tuple

from engine.market import Market
from engine.efficiency import get_transaction_prices
from traders.legacy.zip import ZIP
from traders.legacy.zic import ZIC


def create_flat_supply_detailed(
    num_buyers: int,
    num_sellers: int,
    agent_type: str = "ZIP",
    seed: int = 42,
) -> Tuple[List[ZIP | ZIC], List[ZIP | ZIC], int, int, Dict]:
    """Create flat supply market with detailed tracking."""
    seller_cost = 50
    P0 = 50
    Dmax = 100

    buyers: List[ZIP | ZIC] = []
    sellers: List[ZIP | ZIC] = []

    # Track valuations/costs for analysis
    metadata = {
        "buyer_valuations": {},
        "seller_costs": {},
        "P0": P0,
        "Dmax": Dmax,
        "seller_cost": seller_cost,
    }

    # Create buyers with descending valuations
    for i in range(1, num_buyers + 1):
        val = int(Dmax - (i - 1) * (Dmax - P0) / (num_buyers - 1)) if num_buyers > 1 else Dmax
        vals = [val]
        metadata["buyer_valuations"][i] = vals

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

    # Create sellers with FLAT cost
    for i in range(1, num_sellers + 1):
        costs = [seller_cost]
        metadata["seller_costs"][i] = costs

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

    return buyers, sellers, P0, Dmax, metadata


def test_flat_supply_diagnostic():
    """
    Detailed diagnostic of flat supply market.

    This test analyzes:
    1. Market setup verification
    2. Transaction price sequences
    3. Agent margin evolution (ZIP)
    4. Bid/ask patterns
    5. Time to convergence
    """
    print("\n" + "="*80)
    print("FLAT SUPPLY MARKET DIAGNOSTIC")
    print("="*80)

    num_buyers = 6
    num_sellers = 6

    # Test setup verification
    print("\n1. MARKET SETUP VERIFICATION")
    print("-" * 80)

    buyers_zip, sellers_zip, P0, Dmax, metadata = create_flat_supply_detailed(
        num_buyers, num_sellers, agent_type="ZIP", seed=42
    )

    print(f"P0 (equilibrium): {P0}")
    print(f"Dmax (max buyer valuation): {Dmax}")
    print(f"Seller cost (FLAT): {metadata['seller_cost']}")
    print(f"\nBuyer valuations:")
    for buyer_id, vals in metadata["buyer_valuations"].items():
        print(f"  Buyer {buyer_id}: {vals[0]}")
    print(f"\nSeller costs (all same):")
    for seller_id, costs in metadata["seller_costs"].items():
        print(f"  Seller {seller_id}: {costs[0]}")

    # ZIC theoretical prediction
    zic_expected = P0 + (1/3) * (Dmax - P0)
    print(f"\nZIC Theoretical Prediction: {zic_expected:.2f}")
    print(f"  (Should fail to reach P0={P0}, stuck at ~{zic_expected:.2f})")

    # Run ZIP market with detailed tracking
    print("\n2. ZIP MARKET TRANSACTION ANALYSIS")
    print("-" * 80)

    market_zip = Market(
        num_buyers=num_buyers,
        num_sellers=num_sellers,
        num_times=200,  # Longer session for convergence
        price_min=0,
        price_max=100,
        buyers=buyers_zip,
        sellers=sellers_zip,
        seed=42,
    )

    # Track ZIP margins over time
    zip_buyer_margins = {i: [] for i in range(1, num_buyers + 1)}
    zip_seller_margins = {i: [] for i in range(1, num_sellers + 1)}

    for timestep in range(200):
        # Record margins before timestep
        for i, buyer in enumerate(buyers_zip, 1):
            if hasattr(buyer, 'margin'):
                zip_buyer_margins[i].append(buyer.margin)
        for i, seller in enumerate(sellers_zip, 1):
            if hasattr(seller, 'margin'):
                zip_seller_margins[i].append(seller.margin)

        success = market_zip.run_time_step()
        if not success:
            print(f"Market failed at timestep {timestep}")
            break

    prices_zip = get_transaction_prices(market_zip.orderbook, 200)

    if prices_zip:
        print(f"Total transactions: {len(prices_zip)}")
        print(f"Price range: [{min(prices_zip)}, {max(prices_zip)}]")
        print(f"Mean price: {np.mean(prices_zip):.2f}")
        print(f"Std dev: {np.std(prices_zip):.2f}")
        print(f"Deviation from P0: {abs(np.mean(prices_zip) - P0):.2f}")

        # Early vs late prices
        if len(prices_zip) >= 20:
            early = prices_zip[:10]
            late = prices_zip[-10:]
            print(f"\nEarly prices (first 10): mean={np.mean(early):.2f}, std={np.std(early):.2f}")
            print(f"Late prices (last 10): mean={np.mean(late):.2f}, std={np.std(late):.2f}")
            print(f"Convergence: {abs(np.mean(early) - np.mean(late)):.2f} price change")
    else:
        print("NO TRADES OCCURRED!")

    # Run ZIC market for comparison
    print("\n3. ZIC MARKET TRANSACTION ANALYSIS")
    print("-" * 80)

    buyers_zic, sellers_zic, _, _, _ = create_flat_supply_detailed(
        num_buyers, num_sellers, agent_type="ZIC", seed=100
    )

    market_zic = Market(
        num_buyers=num_buyers,
        num_sellers=num_sellers,
        num_times=200,
        price_min=0,
        price_max=100,
        buyers=buyers_zic,
        sellers=sellers_zic,
        seed=100,
    )

    for _ in range(200):
        market_zic.run_time_step()

    prices_zic = get_transaction_prices(market_zic.orderbook, 200)

    if prices_zic:
        print(f"Total transactions: {len(prices_zic)}")
        print(f"Price range: [{min(prices_zic)}, {max(prices_zic)}]")
        print(f"Mean price: {np.mean(prices_zic):.2f}")
        print(f"Std dev: {np.std(prices_zic):.2f}")
        print(f"Deviation from P0: {abs(np.mean(prices_zic) - P0):.2f}")
        print(f"Deviation from theory: {abs(np.mean(prices_zic) - zic_expected):.2f}")

        if len(prices_zic) >= 20:
            early = prices_zic[:10]
            late = prices_zic[-10:]
            print(f"\nEarly prices (first 10): mean={np.mean(early):.2f}, std={np.std(early):.2f}")
            print(f"Late prices (last 10): mean={np.mean(late):.2f}, std={np.std(late):.2f}")
    else:
        print("NO TRADES OCCURRED!")

    # Compare ZIP vs ZIC
    print("\n4. COMPARISON SUMMARY")
    print("-" * 80)

    if prices_zip and prices_zic:
        print(f"{'Metric':<30} {'ZIP':<15} {'ZIC':<15} {'Winner':<10}")
        print("-" * 70)

        zip_mean = np.mean(prices_zip)
        zic_mean = np.mean(prices_zic)
        zip_dev = abs(zip_mean - P0)
        zic_dev = abs(zic_mean - P0)

        print(f"{'Mean price':<30} {zip_mean:<15.2f} {zic_mean:<15.2f} {'-':<10}")
        print(f"{'Deviation from P0':<30} {zip_dev:<15.2f} {zic_dev:<15.2f} {'ZIP' if zip_dev < zic_dev else 'ZIC':<10}")
        print(f"{'Price std dev':<30} {np.std(prices_zip):<15.2f} {np.std(prices_zic):<15.2f} {'ZIP' if np.std(prices_zip) < np.std(prices_zic) else 'ZIC':<10}")
        print(f"{'Num transactions':<30} {len(prices_zip):<15} {len(prices_zic):<15} {'-':<10}")

        print(f"\nPaper Prediction:")
        print(f"  ZIP should converge to P0={P0}")
        print(f"  ZIC should be stuck at ~{zic_expected:.2f}")
        print(f"\nActual Results:")
        print(f"  ZIP: {zip_mean:.2f} (dev={zip_dev:.2f})")
        print(f"  ZIC: {zic_mean:.2f} (dev={zic_dev:.2f})")

        if zip_dev < zic_dev:
            print(f"\n✅ ZIP WINS: Converges better than ZIC")
        else:
            print(f"\n❌ ZIC WINS: Unexpected! ZIP should outperform")
            print(f"   Investigating possible causes...")

    # Analyze ZIP margin evolution
    print("\n5. ZIP MARGIN EVOLUTION ANALYSIS")
    print("-" * 80)

    if any(zip_buyer_margins.values()):
        print("Buyer margins (first and last 10 timesteps):")
        for buyer_id in range(1, min(4, num_buyers + 1)):  # Show first 3 buyers
            margins = zip_buyer_margins[buyer_id]
            if len(margins) >= 20:
                early_margins = margins[:10]
                late_margins = margins[-10:]
                print(f"  Buyer {buyer_id}: early={np.mean(early_margins):.3f}, late={np.mean(late_margins):.3f}, change={np.mean(late_margins) - np.mean(early_margins):.3f}")

        print("\nSeller margins (first and last 10 timesteps):")
        for seller_id in range(1, min(4, num_sellers + 1)):
            margins = zip_seller_margins[seller_id]
            if len(margins) >= 20:
                early_margins = margins[:10]
                late_margins = margins[-10:]
                print(f"  Seller {seller_id}: early={np.mean(early_margins):.3f}, late={np.mean(late_margins):.3f}, change={np.mean(late_margins) - np.mean(early_margins):.3f}")

    # Diagnose potential issues
    print("\n6. POTENTIAL ISSUES")
    print("-" * 80)

    issues = []

    if not prices_zip:
        issues.append("❌ ZIP market had NO trades - agents too conservative?")
    elif len(prices_zip) < len(prices_zic) * 0.5:
        issues.append(f"⚠️ ZIP had {len(prices_zip)} trades vs ZIC {len(prices_zic)} - low trading volume")

    if prices_zip and zip_dev > zic_dev:
        issues.append(f"❌ ZIP deviation ({zip_dev:.2f}) > ZIC deviation ({zic_dev:.2f}) - contradicts paper")

    if prices_zip and prices_zic:
        if abs(zic_mean - zic_expected) > 10:
            issues.append(f"⚠️ ZIC far from theoretical ({zic_mean:.2f} vs {zic_expected:.2f}) - check RNG")

    if any(zip_buyer_margins.values()):
        # Check if margins are changing
        total_change = 0
        for margins in list(zip_buyer_margins.values()) + list(zip_seller_margins.values()):
            if len(margins) >= 20:
                change = abs(np.mean(margins[-10:]) - np.mean(margins[:10]))
                total_change += change

        avg_change = total_change / (num_buyers + num_sellers) if (num_buyers + num_sellers) > 0 else 0
        if avg_change < 0.01:
            issues.append(f"⚠️ ZIP margins barely changing (avg change={avg_change:.4f}) - not learning?")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("No obvious issues detected.")

    print("\n" + "="*80)


def test_session_length_sensitivity():
    """
    Test if longer sessions allow ZIP to converge better.

    The paper uses "trading days" which may be much longer than our 100 timesteps.
    """
    print("\n" + "="*80)
    print("SESSION LENGTH SENSITIVITY ANALYSIS")
    print("="*80)

    P0 = 50
    session_lengths = [50, 100, 200, 500]

    print(f"\n{'Length':<10} {'ZIP Dev':<15} {'ZIC Dev':<15} {'ZIP Better?':<15}")
    print("-" * 60)

    for length in session_lengths:
        # ZIP
        buyers_zip, sellers_zip, _, _, _ = create_flat_supply_detailed(
            6, 6, agent_type="ZIP", seed=42
        )
        market_zip = Market(6, 6, length, 0, 100, buyers_zip, sellers_zip, seed=42)
        for _ in range(length):
            market_zip.run_time_step()
        prices_zip = get_transaction_prices(market_zip.orderbook, length)
        zip_dev = abs(np.mean(prices_zip) - P0) if prices_zip else 999

        # ZIC
        buyers_zic, sellers_zic, _, _, _ = create_flat_supply_detailed(
            6, 6, agent_type="ZIC", seed=100
        )
        market_zic = Market(6, 6, length, 0, 100, buyers_zic, sellers_zic, seed=100)
        for _ in range(length):
            market_zic.run_time_step()
        prices_zic = get_transaction_prices(market_zic.orderbook, length)
        zic_dev = abs(np.mean(prices_zic) - P0) if prices_zic else 999

        better = "✅ YES" if zip_dev < zic_dev else "❌ NO"
        print(f"{length:<10} {zip_dev:<15.2f} {zic_dev:<15.2f} {better:<15}")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_flat_supply_diagnostic()
    test_session_length_sensitivity()
