# tests/regression/conftest.py
"""
Shared fixtures and configuration for regression tests.

These tests verify documented results from checklists/results.md.
All thresholds are RUTHLESS - based on exact specs from verified experiments.
"""

import pytest

from engine.market import Market
from engine.metrics import calculate_equilibrium_profit
from engine.token_generator import TokenGenerator
from traders.legacy.kaplan import Kaplan
from traders.legacy.skeleton import Skeleton
from traders.legacy.zi import ZI
from traders.legacy.zic import ZIC
from traders.legacy.zip import ZIP

# =============================================================================
# Environment Configurations (from results.md)
# =============================================================================

GAMETYPE_BASE = 6453  # Santa Fe standard
PRICE_MIN = 1
PRICE_MAX = 1000

# Environment specs from results.md Configuration Reference
ENVIRONMENTS = {
    "BASE": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_times": 100,
        "gametype": 6453,
    },
    "SHRT": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_times": 20,  # Time pressure
        "gametype": 6453,
    },
    "TOK": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 1,  # Minimal market
        "num_times": 100,
        "gametype": 6453,
    },
    "EQL": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_times": 100,
        "gametype": 2222,  # Symmetric-ish token values (used in actual experiments)
    },
    "SML": {
        "num_buyers": 2,
        "num_sellers": 2,
        "num_tokens": 4,
        "num_times": 100,
        "gametype": 6453,  # Use BASE gametype for small market
    },
    "BBBS": {
        "num_buyers": 6,
        "num_sellers": 2,
        "num_tokens": 4,
        "num_times": 100,
        "gametype": 6453,
    },
    "BSSS": {
        "num_buyers": 2,
        "num_sellers": 6,
        "num_tokens": 4,
        "num_times": 100,
        "gametype": 6453,
    },
}

# Quick-run parameters (balance speed vs statistical validity)
# For tests: fewer samples than production but still statistically meaningful
QUICK_RUN = {
    "num_seeds": 3,
    "num_rounds": 5,
    "num_periods": 3,
}

# Strategy class mapping
STRATEGIES = {
    "ZI": ZI,
    "ZIC": ZIC,
    "ZIP": ZIP,
    "Skeleton": Skeleton,
    "Kaplan": Kaplan,
}


# =============================================================================
# Helper Functions
# =============================================================================


def create_traders(
    trader_class,
    num_buyers: int,
    num_sellers: int,
    num_tokens: int,
    num_times: int,
    buyer_vals: list[list[int]],
    seller_vals: list[list[int]],
    seed: int,
):
    """Create traders of a single type with given valuations."""
    buyers = []
    for i in range(num_buyers):
        buyers.append(
            trader_class(
                player_id=i + 1,
                is_buyer=True,
                num_tokens=num_tokens,
                valuations=buyer_vals[i],
                price_min=PRICE_MIN,
                price_max=PRICE_MAX,
                num_times=num_times,
                seed=seed + i,
            )
        )

    sellers = []
    for i in range(num_sellers):
        sellers.append(
            trader_class(
                player_id=i + num_buyers + 1,
                is_buyer=False,
                num_tokens=num_tokens,
                valuations=seller_vals[i],
                price_min=PRICE_MIN,
                price_max=PRICE_MAX,
                num_times=num_times,
                seed=seed + 100 + i,
            )
        )

    return buyers, sellers


def create_mixed_traders(
    buyer_types: list[type],
    seller_types: list[type],
    num_tokens: int,
    num_times: int,
    buyer_vals: list[list[int]],
    seller_vals: list[list[int]],
    seed: int,
):
    """Create traders of mixed types (one trader per type entry)."""
    buyers = []
    for i, trader_class in enumerate(buyer_types):
        buyers.append(
            trader_class(
                player_id=i + 1,
                is_buyer=True,
                num_tokens=num_tokens,
                valuations=buyer_vals[i],
                price_min=PRICE_MIN,
                price_max=PRICE_MAX,
                num_times=num_times,
                seed=seed + i,
            )
        )

    sellers = []
    num_buyers = len(buyer_types)
    for i, trader_class in enumerate(seller_types):
        sellers.append(
            trader_class(
                player_id=i + num_buyers + 1,
                is_buyer=False,
                num_tokens=num_tokens,
                valuations=seller_vals[i],
                price_min=PRICE_MIN,
                price_max=PRICE_MAX,
                num_times=num_times,
                seed=seed + 100 + i,
            )
        )

    return buyers, sellers


def run_market(market: Market) -> Market:
    """Run market to completion."""
    for _ in range(market.num_times):
        if not market.run_time_step():
            break
    return market


def calculate_efficiency(market: Market) -> float:
    """Calculate market efficiency as actual/theoretical surplus."""
    all_valuations = []
    all_costs = []
    for buyer in market.buyers:
        all_valuations.extend(buyer.valuations)
    for seller in market.sellers:
        all_costs.extend(seller.valuations)

    max_surplus = calculate_equilibrium_profit(all_valuations, all_costs)

    actual_surplus = sum(b.period_profit for b in market.buyers)
    actual_surplus += sum(s.period_profit for s in market.sellers)

    if max_surplus <= 0:
        return 0.0
    return actual_surplus / max_surplus


def run_efficiency_trials(
    trader_class,
    env_name: str = "BASE",
    num_seeds: int = QUICK_RUN["num_seeds"],
    num_rounds: int = QUICK_RUN["num_rounds"],
    num_periods: int = QUICK_RUN["num_periods"],
) -> list[float]:
    """Run multiple trials and return efficiency values.

    Uses proper round/period structure from results.md methodology.
    """
    env = ENVIRONMENTS[env_name]
    num_buyers = env["num_buyers"]
    num_sellers = env["num_sellers"]
    num_tokens = env["num_tokens"]
    num_times = env["num_times"]
    gametype = env["gametype"]

    all_efficiencies = []

    for seed_idx in range(num_seeds):
        seed = 42 + seed_idx * 100
        token_gen = TokenGenerator(gametype, num_tokens, seed)

        for round_idx in range(num_rounds):
            token_gen.new_round()

            # Generate valuations for this round
            buyer_vals = [token_gen.generate_tokens(is_buyer=True) for _ in range(num_buyers)]
            seller_vals = [token_gen.generate_tokens(is_buyer=False) for _ in range(num_sellers)]

            for period_idx in range(num_periods):
                buyers, sellers = create_traders(
                    trader_class,
                    num_buyers,
                    num_sellers,
                    num_tokens,
                    num_times,
                    buyer_vals,
                    seller_vals,
                    seed + round_idx * 100 + period_idx * 10,
                )

                market = Market(
                    num_buyers=num_buyers,
                    num_sellers=num_sellers,
                    num_times=num_times,
                    price_min=PRICE_MIN,
                    price_max=PRICE_MAX,
                    buyers=buyers,
                    sellers=sellers,
                    seed=seed + round_idx,
                )

                run_market(market)
                eff = calculate_efficiency(market)

                # Always include the efficiency value
                # Even if no trades happened, that's valid data for some strategies
                all_efficiencies.append(eff)

    return all_efficiencies


def run_pairwise_trials(
    strategy_a,
    strategy_b,
    env_name: str = "BASE",
    num_seeds: int = QUICK_RUN["num_seeds"],
    num_rounds: int = QUICK_RUN["num_rounds"],
    num_periods: int = QUICK_RUN["num_periods"],
) -> tuple[list[float], list[float], list[float]]:
    """Run pairwise matchup trials.

    Returns (efficiencies, a_profits, b_profits).
    Each side has half buyers and half sellers of that type.
    """
    env = ENVIRONMENTS[env_name]
    num_buyers = env["num_buyers"]
    num_sellers = env["num_sellers"]
    num_tokens = env["num_tokens"]
    num_times = env["num_times"]
    gametype = env["gametype"]

    all_efficiencies = []
    all_a_profits = []
    all_b_profits = []

    for seed_idx in range(num_seeds):
        seed = 42 + seed_idx * 100
        token_gen = TokenGenerator(gametype, num_tokens, seed)

        for round_idx in range(num_rounds):
            token_gen.new_round()

            buyer_vals = [token_gen.generate_tokens(is_buyer=True) for _ in range(num_buyers)]
            seller_vals = [token_gen.generate_tokens(is_buyer=False) for _ in range(num_sellers)]

            for period_idx in range(num_periods):
                # Half of each side is strategy_a, half is strategy_b
                half_buyers = num_buyers // 2
                half_sellers = num_sellers // 2

                buyer_types = [strategy_a] * half_buyers + [strategy_b] * (num_buyers - half_buyers)
                seller_types = [strategy_a] * half_sellers + [strategy_b] * (
                    num_sellers - half_sellers
                )

                buyers, sellers = create_mixed_traders(
                    buyer_types,
                    seller_types,
                    num_tokens,
                    num_times,
                    buyer_vals,
                    seller_vals,
                    seed + round_idx * 100 + period_idx * 10,
                )

                market = Market(
                    num_buyers=num_buyers,
                    num_sellers=num_sellers,
                    num_times=num_times,
                    price_min=PRICE_MIN,
                    price_max=PRICE_MAX,
                    buyers=buyers,
                    sellers=sellers,
                    seed=seed + round_idx,
                )

                run_market(market)
                eff = calculate_efficiency(market)
                all_efficiencies.append(eff)

                # Calculate profits by strategy
                a_profit = 0
                b_profit = 0

                for i, buyer in enumerate(market.buyers):
                    if i < half_buyers:
                        a_profit += buyer.period_profit
                    else:
                        b_profit += buyer.period_profit

                for i, seller in enumerate(market.sellers):
                    if i < half_sellers:
                        a_profit += seller.period_profit
                    else:
                        b_profit += seller.period_profit

                all_a_profits.append(a_profit)
                all_b_profits.append(b_profit)

    return all_efficiencies, all_a_profits, all_b_profits


def run_roundrobin_trials(
    strategies: list[type],
    env_name: str = "BASE",
    num_seeds: int = QUICK_RUN["num_seeds"],
    num_rounds: int = QUICK_RUN["num_rounds"],
    num_periods: int = QUICK_RUN["num_periods"],
) -> dict[str, list[float]]:
    """Run round-robin tournament with multiple strategies.

    Returns dict mapping strategy name to list of profits across trials.
    Assumes 4 buyers and 4 sellers (one of each strategy per side).
    """
    env = ENVIRONMENTS[env_name]
    num_tokens = env["num_tokens"]
    num_times = env["num_times"]
    gametype = env["gametype"]

    # For round-robin, use one of each strategy per side
    num_strategies = len(strategies)
    num_buyers = num_strategies
    num_sellers = num_strategies

    # Initialize profit tracking
    profits = {s.__name__: [] for s in strategies}

    for seed_idx in range(num_seeds):
        seed = 42 + seed_idx * 100
        token_gen = TokenGenerator(gametype, num_tokens, seed)

        for round_idx in range(num_rounds):
            token_gen.new_round()

            buyer_vals = [token_gen.generate_tokens(is_buyer=True) for _ in range(num_buyers)]
            seller_vals = [token_gen.generate_tokens(is_buyer=False) for _ in range(num_sellers)]

            for period_idx in range(num_periods):
                buyers, sellers = create_mixed_traders(
                    strategies,
                    strategies,
                    num_tokens,
                    num_times,
                    buyer_vals,
                    seller_vals,
                    seed + round_idx * 100 + period_idx * 10,
                )

                market = Market(
                    num_buyers=num_buyers,
                    num_sellers=num_sellers,
                    num_times=num_times,
                    price_min=PRICE_MIN,
                    price_max=PRICE_MAX,
                    buyers=buyers,
                    sellers=sellers,
                    seed=seed + round_idx,
                )

                run_market(market)

                # Record profits by strategy
                for i, strategy in enumerate(strategies):
                    strategy_profit = (
                        market.buyers[i].period_profit + market.sellers[i].period_profit
                    )
                    profits[strategy.__name__].append(strategy_profit)

    return profits


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_token_generator():
    """Create TokenGenerator with gametype=6453 (BASE)."""
    return TokenGenerator(6453, 4, seed=42)


@pytest.fixture
def quick_run_params():
    """Return quick-run parameters for tests."""
    return QUICK_RUN.copy()
