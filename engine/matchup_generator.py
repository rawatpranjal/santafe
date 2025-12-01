"""
Match-up generator for Chen et al. (2010) style experiments.

Generates random 4v4 match-ups from a strategy pool WITHOUT replacement,
following the experimental design from:

Chen & Tai (2010). "The Agent-Based Double Auction Markets: 15 Years On"

Key features:
- Random sampling from strategy pool
- Each match-up has 4 buyers and 4 sellers
- Strategies are sampled WITHOUT replacement within a match-up
- Reproducible with fixed seeds
"""

import numpy as np


class MatchupGenerator:
    """
    Generate random 4v4 match-ups from a strategy pool.

    Following Chen et al. (2010):
    - 300 random match-ups without replacement
    - Each match-up samples 8 strategies (4 buyers + 4 sellers) from the pool
    - A strategy never faces its own type within a match-up
    """

    def __init__(
        self,
        strategy_pool: list[str],
        num_matchups: int = 300,
        num_buyers: int = 4,
        num_sellers: int = 4,
        seed: int = 42,
    ) -> None:
        """
        Initialize the match-up generator.

        Args:
            strategy_pool: List of strategy names to sample from
                          (e.g., ['ZIC', 'Kaplan', 'GD', 'ZIP', 'Ledyard', 'Lin', 'Jacobson', 'Perry'])
            num_matchups: Number of match-ups to generate (default: 300)
            num_buyers: Number of buyers per match-up (default: 4)
            num_sellers: Number of sellers per match-up (default: 4)
            seed: Random seed for reproducibility (default: 42)

        Raises:
            ValueError: If pool is too small to sample required agents
        """
        total_agents = num_buyers + num_sellers
        if len(strategy_pool) < total_agents:
            raise ValueError(
                f"Strategy pool ({len(strategy_pool)}) must have at least "
                f"{total_agents} strategies for {num_buyers}v{num_sellers} match-ups"
            )

        self.strategy_pool = list(strategy_pool)
        self.num_matchups = num_matchups
        self.num_buyers = num_buyers
        self.num_sellers = num_sellers
        self.rng = np.random.default_rng(seed)

    def generate_matchups(self) -> list[tuple[list[str], list[str]]]:
        """
        Generate all match-ups.

        Returns:
            List of (buyer_types, seller_types) tuples.
            Each tuple contains two lists of strategy names.

        Example:
            [
                (['ZIC', 'Kaplan', 'GD', 'ZIP'], ['Ledyard', 'Lin', 'Jacobson', 'Perry']),
                (['Kaplan', 'GD', 'Ledyard', 'Lin'], ['ZIC', 'ZIP', 'Jacobson', 'Perry']),
                ...
            ]
        """
        matchups = []
        total_agents = self.num_buyers + self.num_sellers

        for _ in range(self.num_matchups):
            # Sample strategies WITHOUT replacement
            selected = self.rng.choice(self.strategy_pool, size=total_agents, replace=False)

            # First half are buyers, second half are sellers
            buyers = list(selected[: self.num_buyers])
            sellers = list(selected[self.num_buyers :])

            matchups.append((buyers, sellers))

        return matchups

    def generate_single_matchup(self) -> tuple[list[str], list[str]]:
        """
        Generate a single random match-up.

        Useful for testing or iterative experiments.

        Returns:
            (buyer_types, seller_types) tuple
        """
        total_agents = self.num_buyers + self.num_sellers
        selected = self.rng.choice(self.strategy_pool, size=total_agents, replace=False)
        buyers = list(selected[: self.num_buyers])
        sellers = list(selected[self.num_buyers :])
        return buyers, sellers


def get_default_strategy_pool() -> list[str]:
    """
    Get the default strategy pool matching Chen et al. (2010).

    Returns:
        List of strategy names available in the Santa Fe implementation.

    Note:
        Chen's paper incorrectly describes "Ringuette" as a background trader.
        The actual SRobotZI2.java is market-aware random, NOT a sniper.
        Use ZI2 for accurate replication.
    """
    return [
        "ZIC",  # Zero-Intelligence Constrained (Gode & Sunder 1993)
        "Kaplan",  # Sniper strategy (1993 winner)
        "GD",  # Gjerstad-Dickhaut belief-based
        "ZIP",  # Zero-Intelligence Plus (Cliff 1997)
        "Ledyard",  # Ledyard reservation price
        "TruthTeller",  # Truth Teller - bids/asks at reservation price
        "Skeleton",  # Reference strategy from SFDA
        "ZI2",  # Ringuette's actual submission (market-aware random, NOT sniper!)
        "Markup",  # Fixed markup strategy
        "ReservationPrice",  # Simplified BGAN
        "HistogramLearner",  # Simplified Empirical Bayesian
    ]


def get_chen_2010_strategy_pool() -> list[str]:
    """
    Get the exact 11-strategy pool from Chen & Tai (2010) Section 3.2.

    Returns:
        List of strategy names matching Chen's experimental setup.

    Strategy mapping from Chen paper to our implementations:
        Truth Teller -> TruthTeller (simple reservation price bidding)
        Skeleton -> Skeleton
        Kaplan -> Kaplan
        Ringuette -> ZI2 (Java SRobotZI2.java - market-aware random)
        ZIC -> ZIC
        ZIP -> ZIP
        Markup -> Markup
        GD -> GD
        BGAN -> ReservationPrice
        Ledyard -> Ledyard
        Empirical -> HistogramLearner
    """
    return [
        "TruthTeller",  # Truth Teller - bids/asks at reservation price
        "Skeleton",  # SFDA reference
        "Kaplan",  # SFDA winner
        "ZI2",  # Ringuette (Java SRobotZI2.java)
        "ZIC",  # Zero-Intelligence Constrained
        "ZIP",  # Zero-Intelligence Plus
        "Markup",  # Fixed markup
        "GD",  # Gjerstad-Dickhaut
        "ReservationPrice",  # BGAN
        "Ledyard",  # Ledyard
        "HistogramLearner",  # Empirical Bayesian
    ]
