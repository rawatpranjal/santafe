"""
Market efficiency calculations for the Santa Fe Double Auction.

This module implements the metrics used to evaluate market performance:
- Allocative Efficiency (Total Surplus / Max Surplus)
- V-Inefficiency (missed profitable trades)
- EM-Inefficiency (negative-surplus trades)

Reference: Cason & Friedman (1996) decomposition
"""

from typing import Any

import numpy as np


def calculate_max_surplus(buyer_valuations: list[list[int]], seller_costs: list[list[int]]) -> int:
    """
    Calculate the maximum possible surplus (competitive equilibrium).

    This is the total surplus that would be achieved if all profitable
    trades were executed at prices that match buyers with sellers optimally.

    Args:
        buyer_valuations: List of valuation arrays, one per buyer
                         buyer_valuations[i][j] = buyer i's value for unit j
        seller_costs: List of cost arrays, one per seller
                     seller_costs[i][j] = seller i's cost for unit j

    Returns:
        Maximum possible surplus (integer)

    Algorithm:
        1. Sort all buyer valuations descending (highest first)
        2. Sort all seller costs ascending (lowest first)
        3. Match buyers and sellers while valuation > cost
        4. Sum up (valuation - cost) for all matched pairs
    """
    # Flatten and sort buyer valuations (descending)
    all_buyer_vals = []
    for buyer_vals in buyer_valuations:
        all_buyer_vals.extend(buyer_vals)
    all_buyer_vals.sort(reverse=True)

    # Flatten and sort seller costs (ascending)
    all_seller_costs = []
    for seller_vals in seller_costs:
        all_seller_costs.extend(seller_vals)
    all_seller_costs.sort()

    # Calculate max surplus by matching highest valuation with lowest cost
    max_surplus = 0
    num_trades = min(len(all_buyer_vals), len(all_seller_costs))

    for i in range(num_trades):
        buyer_val = all_buyer_vals[i]
        seller_cost = all_seller_costs[i]

        # Only count if trade is profitable (strict inequality to match Java baseline)
        if buyer_val > seller_cost:
            max_surplus += buyer_val - seller_cost
        else:
            # No more profitable trades possible
            break

    return max_surplus


def calculate_equilibrium_price(
    buyer_valuations: list[list[int]],
    seller_costs: list[list[int]],
) -> int:
    """
    Calculate the market-clearing (competitive equilibrium) price.

    The equilibrium price is the midpoint of the marginal buyer valuation
    and marginal seller cost at the intersection of supply and demand curves.

    Args:
        buyer_valuations: List of valuation arrays, one per buyer
        seller_costs: List of cost arrays, one per seller

    Returns:
        Equilibrium price (integer, midpoint of marginal pair)

    Algorithm:
        1. Sort all valuations descending (demand curve)
        2. Sort all costs ascending (supply curve)
        3. Find intersection (where valuation <= cost)
        4. Return midpoint of marginal pair
    """
    # Flatten and sort
    all_vals = sorted([v for vals in buyer_valuations for v in vals], reverse=True)
    all_costs = sorted([c for costs in seller_costs for c in costs])

    if not all_vals or not all_costs:
        return 0

    # Find marginal pair (intersection of supply and demand)
    for i in range(min(len(all_vals), len(all_costs))):
        if all_vals[i] <= all_costs[i]:
            # Marginal pair is (all_vals[i-1], all_costs[i-1])
            if i > 0:
                return (all_vals[i - 1] + all_costs[i - 1]) // 2
            # Edge case: first pair is already unprofitable
            return (all_vals[0] + all_costs[0]) // 2

    # All trades profitable - use last pair
    n = min(len(all_vals), len(all_costs))
    return (all_vals[n - 1] + all_costs[n - 1]) // 2


def calculate_actual_surplus(
    trades: list[tuple[int, int, int, int]],
    buyer_valuations: dict[int, list[int]],
    seller_costs: dict[int, list[int]],
) -> int:
    """
    Calculate the actual surplus achieved from executed trades.

    Args:
        trades: List of (buyer_id, seller_id, price, buyer_unit_index)
               Each tuple represents a completed trade
        buyer_valuations: Dict mapping buyer_id -> list of valuations
        seller_costs: Dict mapping seller_id -> list of costs

    Returns:
        Actual surplus achieved (integer)

    Note:
        Surplus from a trade = buyer_valuation - seller_cost
        This is independent of the trade price (price just transfers value)
    """
    actual_surplus = 0

    # Track seller positions to determine which unit was traded
    seller_positions: dict[int, int] = {}
    for seller_id in seller_costs.keys():
        seller_positions[seller_id] = 0

    for buyer_id, seller_id, price, buyer_unit in trades:
        buyer_val = buyer_valuations[buyer_id][buyer_unit]

        # Get seller's unit index (how many times this seller has traded so far)
        seller_unit = seller_positions[seller_id]
        seller_cost = seller_costs[seller_id][seller_unit]

        # Increment seller's position
        seller_positions[seller_id] += 1

        # Surplus = buyer value - seller cost
        surplus = buyer_val - seller_cost
        actual_surplus += surplus

    return actual_surplus


def calculate_allocative_efficiency(actual_surplus: int, max_surplus: int) -> float:
    """
    Calculate allocative efficiency as a percentage.

    Args:
        actual_surplus: Actual surplus achieved
        max_surplus: Maximum possible surplus

    Returns:
        Efficiency as a percentage (0.0 to 100.0)

    Special cases:
        - If max_surplus == 0: returns 100.0 (trivial market)
        - If actual_surplus > max_surplus: clamps to 100.0 (shouldn't happen)
    """
    if max_surplus == 0:
        return 100.0

    efficiency = (actual_surplus / max_surplus) * 100.0

    # Clamp to [0, 100]
    return min(max(efficiency, 0.0), 100.0)


def calculate_v_inefficiency(
    actual_trades: int,
    buyer_valuations: list[list[int]],
    seller_costs: list[list[int]],
) -> int:
    """
    Calculate V-Inefficiency (missed surplus from untraded intra-marginal units).

    V-Inefficiency measures the surplus that would have been gained from
    profitable trades that were NOT executed.

    Args:
        actual_trades: Actual number of trades executed
        buyer_valuations: List of valuation arrays, one per buyer
        seller_costs: List of cost arrays, one per seller

    Returns:
        Missed surplus from untraded intra-marginal units (integer)

    Formula (from metrics.md Section 2.2):
        IM = Σ(D(q) - S(q)) for untraded intra-marginal units

    Reference: Rust, Palmer, & Miller (1993); Cason & Friedman (1996)
    """
    # Flatten and sort buyer valuations (descending) = demand curve D(q)
    all_vals = sorted([v for vals in buyer_valuations for v in vals], reverse=True)

    # Flatten and sort seller costs (ascending) = supply curve S(q)
    all_costs = sorted([c for costs in seller_costs for c in costs])

    # Find max possible profitable trades (Q*)
    max_trades = 0
    for i in range(min(len(all_vals), len(all_costs))):
        if all_vals[i] > all_costs[i]:
            max_trades += 1
        else:
            break

    # Sum surplus from untraded intra-marginal units
    missed_surplus = 0
    for i in range(actual_trades, max_trades):
        if i < len(all_vals) and i < len(all_costs):
            # This is an intra-marginal unit that should have traded
            missed_surplus += all_vals[i] - all_costs[i]

    return missed_surplus


def calculate_em_inefficiency(
    trades: list[tuple[int, int, int, int]],
    buyer_valuations: dict[int, list[int]],
    seller_costs: dict[int, list[int]],
) -> int:
    """
    Calculate EM-Inefficiency (bad trades).

    EM-Inefficiency measures trades that resulted in negative surplus
    (buyer paid more than their valuation OR seller sold below cost).

    Args:
        trades: List of (buyer_id, seller_id, price, buyer_unit)
        buyer_valuations: Dict mapping buyer_id -> list of valuations
        seller_costs: Dict mapping buyer_id -> list of costs

    Returns:
        Negative surplus from bad trades (absolute value)

    Reference: Cason & Friedman (1996)
    """
    em_inefficiency = 0

    # Track seller positions to determine which unit was traded
    seller_positions: dict[int, int] = {}
    for seller_id in seller_costs.keys():
        seller_positions[seller_id] = 0

    for buyer_id, seller_id, price, buyer_unit in trades:
        buyer_val = buyer_valuations[buyer_id][buyer_unit]

        # Get seller's unit index
        seller_unit = seller_positions[seller_id]
        seller_cost = seller_costs[seller_id][seller_unit]

        # Increment seller's position
        seller_positions[seller_id] += 1

        # Check if this was a bad trade (negative surplus)
        surplus = buyer_val - seller_cost
        if surplus < 0:
            em_inefficiency += abs(surplus)

    return em_inefficiency


def extract_trades_from_orderbook(
    orderbook: Any,  # OrderBook instance
    num_times: int,
) -> list[tuple[int, int, int, int]]:
    """
    Extract executed trades from an OrderBook instance.

    Args:
        orderbook: OrderBook instance with trade history
        num_times: Number of timesteps to extract

    Returns:
        List of (buyer_id, seller_id, price, buyer_unit_index) tuples

    Note:
        This reconstructs trades by checking trade_price array
        and position tracking (num_buys, num_sells).
        The buyer_unit_index is which unit the buyer traded (0=first, 1=second, etc.)
    """
    trades: list[tuple[int, int, int, int]] = []

    # Track positions to determine which unit was traded
    buyer_positions: dict[int, int] = {}
    seller_positions: dict[int, int] = {}

    for buyer_id in range(1, orderbook.num_buyers + 1):
        buyer_positions[buyer_id] = 0

    for seller_id in range(1, orderbook.num_sellers + 1):
        seller_positions[seller_id] = 0

    for t in range(1, num_times + 1):
        trade_price = int(orderbook.trade_price[t])

        if trade_price > 0:
            # Trade occurred - find who traded
            # Check position changes
            for buyer_id in range(1, orderbook.num_buyers + 1):
                curr_buys = int(orderbook.num_buys[buyer_id, t])

                if curr_buys > buyer_positions[buyer_id]:
                    # This buyer traded
                    buyer_unit = buyer_positions[buyer_id]  # Which unit they bought

                    # Find seller
                    for seller_id in range(1, orderbook.num_sellers + 1):
                        curr_sells = int(orderbook.num_sells[seller_id, t])

                        if curr_sells > seller_positions[seller_id]:
                            # This seller traded
                            trades.append((buyer_id, seller_id, trade_price, buyer_unit))

                            # Update positions
                            buyer_positions[buyer_id] = curr_buys
                            seller_positions[seller_id] = curr_sells
                            break
                    break

    return trades


def calculate_individual_profits(
    trades: list[tuple[int, int, int, int]],
    buyer_valuations: dict[int, list[int]],
    seller_costs: dict[int, list[int]],
) -> tuple[dict[int, int], dict[int, int]]:
    """
    Calculate individual profits for each buyer and seller.

    Args:
        trades: List of (buyer_id, seller_id, price, buyer_unit)
        buyer_valuations: Dict mapping buyer_id -> list of valuations
        seller_costs: Dict mapping seller_id -> list of costs

    Returns:
        (buyer_profits, seller_profits) where each is dict[agent_id] -> total_profit
    """
    buyer_profits: dict[int, int] = {bid: 0 for bid in buyer_valuations.keys()}
    seller_profits: dict[int, int] = {sid: 0 for sid in seller_costs.keys()}

    # Track seller positions
    seller_positions: dict[int, int] = {sid: 0 for sid in seller_costs.keys()}

    for buyer_id, seller_id, price, buyer_unit in trades:
        # Buyer profit = valuation - price
        buyer_val = buyer_valuations[buyer_id][buyer_unit]
        buyer_profits[buyer_id] += buyer_val - price

        # Seller profit = price - cost
        seller_unit = seller_positions[seller_id]
        seller_cost = seller_costs[seller_id][seller_unit]
        seller_profits[seller_id] += price - seller_cost

        seller_positions[seller_id] += 1

    return buyer_profits, seller_profits


def calculate_equilibrium_profits(
    buyer_valuations: list[list[int]],
    seller_costs: list[list[int]],
    equilibrium_price: int,
) -> tuple[dict[int, int], dict[int, int]]:
    """
    Calculate theoretical equilibrium profits for each agent.

    At competitive equilibrium, all profitable trades are executed at price P0.
    We need to determine which agents would trade and what their profits would be.

    Args:
        buyer_valuations: List of valuation arrays, one per buyer (0-indexed)
        seller_costs: List of cost arrays, one per seller (0-indexed)
        equilibrium_price: Competitive equilibrium price P0

    Returns:
        (buyer_eq_profits, seller_eq_profits) where each is dict[agent_id] -> eq_profit
        Agent IDs are 1-indexed to match actual trading.
    """
    # Flatten valuations with tracking (agent_id, unit_idx, value)
    buyer_items: list[tuple[int, int, int]] = []
    for i, vals in enumerate(buyer_valuations):
        for j, val in enumerate(vals):
            buyer_items.append((i + 1, j, val))  # 1-indexed agent IDs

    # Flatten costs with tracking
    seller_items: list[tuple[int, int, int]] = []
    for i, costs in enumerate(seller_costs):
        for j, cost in enumerate(costs):
            seller_items.append((i + 1, j, cost))  # 1-indexed agent IDs

    # Sort buyers by valuation (descending)
    buyer_items.sort(key=lambda x: x[2], reverse=True)

    # Sort sellers by cost (ascending)
    seller_items.sort(key=lambda x: x[2])

    # Match profitable trades
    buyer_eq_profits: dict[int, int] = {i + 1: 0 for i in range(len(buyer_valuations))}
    seller_eq_profits: dict[int, int] = {i + 1: 0 for i in range(len(seller_costs))}

    num_trades = min(len(buyer_items), len(seller_items))

    for i in range(num_trades):
        buyer_id, _, buyer_val = buyer_items[i]
        seller_id, _, seller_cost = seller_items[i]

        # Only trade if profitable (strict inequality to match Java baseline)
        if buyer_val > seller_cost:
            # At equilibrium, both trade at P0
            buyer_eq_profits[buyer_id] += buyer_val - equilibrium_price
            seller_eq_profits[seller_id] += equilibrium_price - seller_cost
        else:
            break

    return buyer_eq_profits, seller_eq_profits


def calculate_profit_dispersion(
    trades: list[tuple[int, int, int, int]],
    buyer_valuations: dict[int, list[int]],
    seller_costs: dict[int, list[int]],
    buyer_vals_list: list[list[int]],
    seller_costs_list: list[list[int]],
    equilibrium_price: int,
) -> float:
    """
    Calculate profit dispersion (cross-sectional RMS difference).

    This is THE key metric from Cliff & Bruten 1997 for discriminating
    between intelligent and zero-intelligence traders.

    Formula: sqrt((1/n) * Σ(actual_profit[i] - equilibrium_profit[i])²)

    Args:
        trades: List of executed trades
        buyer_valuations: Dict mapping buyer_id -> valuations
        seller_costs: Dict mapping seller_id -> costs
        buyer_vals_list: List of valuation arrays (for equilibrium calc)
        seller_costs_list: List of cost arrays (for equilibrium calc)
        equilibrium_price: Competitive equilibrium price P0

    Returns:
        Profit dispersion (float, lower is better)

    Expected values (from 1997 paper):
        - ZIP: ~0.05 after convergence
        - ZIC: ~0.35 (constant)
        - Improvement: 7-10x better
    """
    # Calculate actual profits
    buyer_actual, seller_actual = calculate_individual_profits(
        trades, buyer_valuations, seller_costs
    )

    # Calculate equilibrium profits
    buyer_eq, seller_eq = calculate_equilibrium_profits(
        buyer_vals_list, seller_costs_list, equilibrium_price
    )

    # Compute RMS difference
    total_squared_diff = 0.0
    num_agents = len(buyer_actual) + len(seller_actual)

    for buyer_id in buyer_actual:
        diff = buyer_actual[buyer_id] - buyer_eq[buyer_id]
        total_squared_diff += diff**2

    for seller_id in seller_actual:
        diff = seller_actual[seller_id] - seller_eq[seller_id]
        total_squared_diff += diff**2

    if num_agents == 0:
        return 0.0

    return float((total_squared_diff / num_agents) ** 0.5)


def get_transaction_prices(orderbook: Any, num_times: int) -> list[int]:
    """
    Extract all transaction prices from orderbook.

    Args:
        orderbook: OrderBook instance
        num_times: Number of timesteps

    Returns:
        List of transaction prices (only non-zero trades)
    """
    prices: list[int] = []

    for t in range(1, num_times + 1):
        price = int(orderbook.trade_price[t])
        if price > 0:
            prices.append(price)

    return prices


def calculate_price_std_dev(transaction_prices: list[int]) -> float:
    """
    Calculate standard deviation of transaction prices.

    Args:
        transaction_prices: List of prices from executed trades

    Returns:
        Standard deviation (float, lower indicates convergence)

    Expected values:
        - ZIP: std_dev should decline over time
        - Early periods: high variance
        - Late periods: low variance (~<5% of price range)
    """
    if len(transaction_prices) <= 1:
        return 0.0

    return float(np.std(transaction_prices))


def calculate_smiths_alpha(transaction_prices: list[int], equilibrium_price: int) -> float:
    """
    Calculate Smith's alpha convergence coefficient.

    Formula: α = 100 × σ₀ / P*
    where σ₀ = sqrt((1/k) * Σ(p_j - P*)²)

    Args:
        transaction_prices: List of transaction prices
        equilibrium_price: Competitive equilibrium price P*

    Returns:
        Alpha coefficient (float, LOWER is better)

    Interpretation:
        - α = 0: Perfect convergence (all trades at equilibrium)
        - α < 5: Excellent convergence
        - α > 20: Poor convergence

    Reference: Smith (1962) "An Experimental Study of Competitive Market Behavior"
    """
    if not transaction_prices or equilibrium_price == 0:
        return float("inf")

    k = len(transaction_prices)
    sigma_0_squared = sum((p - equilibrium_price) ** 2 for p in transaction_prices) / k
    sigma_0 = sigma_0_squared**0.5

    # Smith's original formula: α = 100 × σ₀ / P*
    return float(100.0 * sigma_0 / equilibrium_price)


def calculate_individual_efficiency_ratio(
    actual_profit: int,
    equilibrium_profit: int,
) -> float:
    """
    Calculate individual efficiency ratio (Chen et al. 2010 metric).

    The individual efficiency ratio measures how well an agent captures
    its fair share of surplus relative to competitive equilibrium.

    Formula:
        Profit_Ratio = Actual_Profit / Equilibrium_Profit

    Args:
        actual_profit: Agent's actual profit from trading
        equilibrium_profit: Agent's expected profit at competitive equilibrium

    Returns:
        Efficiency ratio (float)
        - > 1.0: Agent captures more than equilibrium share
        - = 1.0: Agent captures exactly equilibrium share
        - < 1.0: Agent captures less than equilibrium share
        - = 0.0: Agent made no profit when they should have

    Special cases:
        - If equilibrium_profit <= 0: returns 1.0 if actual >= 0, else 0.0
          (handles marginal/extramarginal traders)

    Reference:
        Chen & Tai (2010). "The Agent-Based Double Auction Markets: 15 Years On"
    """
    if equilibrium_profit <= 0:
        # Marginal or extramarginal trader - no expected profit
        return 1.0 if actual_profit >= 0 else 0.0

    return actual_profit / equilibrium_profit


# =============================================================================
# PRICE CONVERGENCE METRICS (Section 3 of metrics.md)
# =============================================================================


def calculate_rmsd(transaction_prices: list[int], equilibrium_price: int) -> float:
    """
    Calculate Root Mean Squared Deviation from equilibrium price.

    Formula (metrics.md Section 3.1):
        RMSD = sqrt((1/T) * Σ(p_t - P*)²)

    Args:
        transaction_prices: List of transaction prices
        equilibrium_price: Competitive equilibrium price P*

    Returns:
        RMSD (float, lower is better)

    Reference: Gode & Sunder (1993)
    """
    if not transaction_prices:
        return 0.0

    squared_diffs = [(p - equilibrium_price) ** 2 for p in transaction_prices]
    return float((sum(squared_diffs) / len(transaction_prices)) ** 0.5)


def calculate_volatility_pct(transaction_prices: list[int]) -> float:
    """
    Calculate price volatility as percentage (coefficient of variation).

    Formula (metrics.md Section 3.4):
        Volatility% = (σ_p / p̄) × 100

    Args:
        transaction_prices: List of transaction prices

    Returns:
        Volatility percentage (float)
        - <5% indicates good convergence
        - >20% indicates unstable market

    Reference: Santa Fe Tournament
    """
    if len(transaction_prices) <= 1:
        return 0.0

    mean_price = sum(transaction_prices) / len(transaction_prices)
    if mean_price == 0:
        return 0.0

    std_dev = float(np.std(transaction_prices))
    return 100.0 * std_dev / mean_price


def calculate_hit_rate(
    transaction_prices: list[int],
    equilibrium_price: int,
    band_pct: float = 0.05,
) -> float:
    """
    Calculate percentage of trades within ±band_pct of equilibrium.

    Formula (metrics.md Section 3.5, 3.10):
        H_k = (|{t : |p_t - P*| ≤ k% × P*}| / T) × 100

    Args:
        transaction_prices: List of transaction prices
        equilibrium_price: Competitive equilibrium price P*
        band_pct: Band width as fraction (default 0.05 = 5%)

    Returns:
        Hit rate percentage (0-100)

    Reference: Rust, Palmer, & Miller (1994) Table 4.4
    """
    if not transaction_prices or equilibrium_price == 0:
        return 0.0

    threshold = band_pct * equilibrium_price
    hits = sum(1 for p in transaction_prices if abs(p - equilibrium_price) <= threshold)
    return 100.0 * hits / len(transaction_prices)


def calculate_mad(transaction_prices: list[int], equilibrium_price: int) -> float:
    """
    Calculate Mean Absolute Deviation from equilibrium price.

    Formula (metrics.md Section 3.6):
        MAD = (1/T) * Σ|p_t - P*|

    Args:
        transaction_prices: List of transaction prices
        equilibrium_price: Competitive equilibrium price P*

    Returns:
        MAD in price units (float)

    Reference: Gjerstad & Dickhaut (1998)
    """
    if not transaction_prices:
        return 0.0

    return sum(abs(p - equilibrium_price) for p in transaction_prices) / len(transaction_prices)


def calculate_mapd(transaction_prices: list[int], equilibrium_price: int) -> float:
    """
    Calculate Mean Absolute Percentage Deviation from equilibrium.

    Formula (metrics.md Section 3.6):
        MAPD = (1/T) * Σ(|p_t - P*| / P*) × 100

    Args:
        transaction_prices: List of transaction prices
        equilibrium_price: Competitive equilibrium price P*

    Returns:
        MAPD percentage (float)

    Note: MAPD allows comparison across markets with different price levels.

    Reference: metrics.md Section 3.6
    """
    if not transaction_prices or equilibrium_price == 0:
        return 0.0

    return (
        100.0
        * sum(abs(p - equilibrium_price) / equilibrium_price for p in transaction_prices)
        / len(transaction_prices)
    )


def calculate_dev_max(transaction_prices: list[int], equilibrium_price: int) -> float:
    """
    Calculate maximum percentage deviation from equilibrium.

    Formula (metrics.md Section 3.7):
        DEV_MAX = max_t(|p_t - P*| / P*) × 100

    Args:
        transaction_prices: List of transaction prices
        equilibrium_price: Competitive equilibrium price P*

    Returns:
        Maximum percentage deviation (float)

    Reference: Rust, Palmer, & Miller (1994) Table 4.4
    """
    if not transaction_prices or equilibrium_price == 0:
        return 0.0

    return 100.0 * max(abs(p - equilibrium_price) / equilibrium_price for p in transaction_prices)


def calculate_dev_last(transaction_prices: list[int], equilibrium_price: int) -> float:
    """
    Calculate percentage deviation of last transaction from equilibrium.

    Formula (metrics.md Section 3.8):
        DEV_LAST = (|p_T - P*| / P*) × 100

    Args:
        transaction_prices: List of transaction prices
        equilibrium_price: Competitive equilibrium price P*

    Returns:
        Last transaction deviation percentage (float)

    Reference: Rust, Palmer, & Miller (1994) Table 4.4
    """
    if not transaction_prices or equilibrium_price == 0:
        return 0.0

    return 100.0 * abs(transaction_prices[-1] - equilibrium_price) / equilibrium_price


def calculate_dev_average(transaction_prices: list[int], equilibrium_price: int) -> float:
    """
    Calculate signed average percentage deviation from equilibrium.

    Formula (metrics.md Section 3.9):
        DEV_AVERAGE = (1/T) * Σ((p_t - P*) / P*) × 100

    Args:
        transaction_prices: List of transaction prices
        equilibrium_price: Competitive equilibrium price P*

    Returns:
        Signed average deviation percentage (float)
        - Positive = prices systematically above equilibrium
        - Negative = prices systematically below equilibrium

    Reference: Rust, Palmer, & Miller (1994) Table 4.4
    """
    if not transaction_prices or equilibrium_price == 0:
        return 0.0

    return (
        100.0
        * sum((p - equilibrium_price) / equilibrium_price for p in transaction_prices)
        / len(transaction_prices)
    )


# =============================================================================
# DYNAMIC METRICS (Section 5 of metrics.md)
# =============================================================================


def calculate_pct2nd(trade_times: list[float], t_max: float) -> float:
    """
    Calculate fraction of trades in second half of period.

    Formula (metrics.md Section 5.6):
        PCT2ND = (|{t : τ_t > T_max/2}| / T) × 100

    Args:
        trade_times: List of trade timestamps
        t_max: Maximum time allowed in period

    Returns:
        Percentage of trades in second half (0-100)

    Interpretation:
        High PCT2ND indicates deadline bunching / "wait in background"
        strategies (like Kaplan).

    Reference: Rust, Palmer, & Miller (1994) Table 4.4
    """
    if not trade_times or t_max <= 0:
        return 0.0

    second_half = sum(1 for t in trade_times if t > t_max / 2)
    return 100.0 * second_half / len(trade_times)


def calculate_convergence_time(
    transaction_prices: list[int],
    equilibrium_price: int,
    band_pct: float = 0.05,
) -> int:
    """
    Calculate first trade index where price enters equilibrium band.

    Formula (metrics.md Section 5.3):
        T* = min{t : |p_t - P*| ≤ 0.05 × P*}

    Args:
        transaction_prices: List of transaction prices
        equilibrium_price: Competitive equilibrium price P*
        band_pct: Band width as fraction (default 0.05 = 5%)

    Returns:
        1-indexed trade number of first convergence, or -1 if never converged

    Expected values:
        - GD: <1 period
        - ZIP: 1-2 periods
        - ZIC: Never (no learning)

    Reference: metrics.md Section 5.3
    """
    if not transaction_prices or equilibrium_price == 0:
        return -1

    threshold = band_pct * equilibrium_price
    for i, p in enumerate(transaction_prices):
        if abs(p - equilibrium_price) <= threshold:
            return i + 1  # 1-indexed

    return -1  # Never converged


def calculate_t_last(trade_times: list[float]) -> float:
    """
    Calculate time of last transaction.

    Formula (metrics.md Section 5.4):
        T_last = max_t(τ_t)

    Args:
        trade_times: List of trade timestamps

    Returns:
        Time of last trade (float)

    Interpretation:
        If T_last ≈ T_max consistently, indicates "wait in background"
        strategies (like Kaplan) causing deadline congestion.

    Reference: Rust, Palmer, & Miller (1993)
    """
    if not trade_times:
        return 0.0

    return max(trade_times)


def calculate_autocorrelation(transaction_prices: list[int], lag: int = 1) -> float:
    """
    Calculate lag-k autocorrelation of price changes.

    Formula (metrics.md Section 5.1):
        ρ = Corr(Δp_t, Δp_{t-1})
        where Δp_t = p_t - p_{t-1}

    Args:
        transaction_prices: List of transaction prices
        lag: Lag for autocorrelation (default 1)

    Returns:
        Autocorrelation coefficient (-1 to 1)
        - ρ < 0: Mean-reversion (prices overshoot then correct)
        - ρ = 0: Random walk (no predictability)
        - ρ > 0: Momentum/trending

    Expected: ρ ≈ -0.25 (Rust et al. finding)

    Reference: Rust, Palmer, & Miller (1994)
    """
    if len(transaction_prices) < lag + 2:
        return 0.0

    # Calculate price changes
    changes = [
        transaction_prices[i] - transaction_prices[i - 1] for i in range(1, len(transaction_prices))
    ]

    if len(changes) < lag + 1:
        return 0.0

    # Calculate lagged correlation
    n = len(changes) - lag
    if n <= 0:
        return 0.0

    mean = sum(changes) / len(changes)
    variance = sum((c - mean) ** 2 for c in changes) / len(changes)

    if variance == 0:
        return 0.0

    covariance = sum((changes[i] - mean) * (changes[i + lag] - mean) for i in range(n)) / n

    return covariance / variance


def calculate_rank_correlation(
    trades: list[tuple[int, int, int, int]],
    buyer_valuations: dict[int, list[int]],
    seller_costs: dict[int, list[int]],
) -> float:
    """
    Calculate Spearman rank correlation of trade order vs efficient order.

    Formula (metrics.md Section 5.5):
        ρ_s = Spearman(R_actual, R_ideal)

    Theory suggests highest-value buyer should trade with lowest-cost seller first.

    Args:
        trades: List of (buyer_id, seller_id, price, buyer_unit) tuples
        buyer_valuations: Dict mapping buyer_id -> list of valuations
        seller_costs: Dict mapping seller_id -> list of costs

    Returns:
        Spearman correlation coefficient (-1 to 1)
        - ρ_s = 1.0: Market perfectly executed most profitable trades first
        - ρ_s = 0.0: Random order
        - ρ_s < 0: Anti-efficient order

    Reference: Rust, Palmer, & Miller (1994)
    """
    if len(trades) < 2:
        return 0.0

    # Calculate surplus for each actual trade
    seller_positions: dict[int, int] = {sid: 0 for sid in seller_costs.keys()}
    actual_surpluses = []

    for buyer_id, seller_id, price, buyer_unit in trades:
        buyer_val = buyer_valuations[buyer_id][buyer_unit]
        seller_unit = seller_positions[seller_id]
        seller_cost = seller_costs[seller_id][seller_unit]
        seller_positions[seller_id] += 1

        surplus = buyer_val - seller_cost
        actual_surpluses.append(surplus)

    # Ideal order: sorted by surplus descending
    ideal_order = sorted(
        range(len(actual_surpluses)), key=lambda i: actual_surpluses[i], reverse=True
    )

    # Actual order is just 0, 1, 2, ...
    actual_ranks = list(range(len(actual_surpluses)))
    ideal_ranks = [0] * len(actual_surpluses)
    for rank, idx in enumerate(ideal_order):
        ideal_ranks[idx] = rank

    # Spearman correlation: 1 - (6 * Σd²) / (n(n²-1))
    n = len(actual_surpluses)
    d_squared_sum = sum((actual_ranks[i] - ideal_ranks[i]) ** 2 for i in range(n))

    return 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
