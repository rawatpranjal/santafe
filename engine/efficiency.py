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


def calculate_max_surplus(
    buyer_valuations: list[list[int]], seller_costs: list[list[int]]
) -> int:
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
    all_vals = sorted(
        [v for vals in buyer_valuations for v in vals], reverse=True
    )
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
    max_trades: int, actual_trades: int
) -> int:
    """
    Calculate V-Inefficiency (missed trades).

    V-Inefficiency measures how many profitable trades were NOT executed.

    Args:
        max_trades: Maximum number of profitable trades possible
        actual_trades: Actual number of trades executed

    Returns:
        Number of missed profitable trades

    Reference: Cason & Friedman (1996)
    """
    return max(0, max_trades - actual_trades)


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
        return float('inf')

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
