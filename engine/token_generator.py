"""
Token Generator for Santa Fe Tournament.

Ports logic from STokenGeneratorOriginal.java.
Generates valuations/costs for agents based on a game type seed.

Also supports "uniform" mode for Gode & Sunder (1993) replication,
which uses simple uniform random tokens in [price_min, price_max].
"""

import numpy as np


class UniformTokenGenerator:
    """
    Gode & Sunder (1993) style token generator.

    Creates supply/demand curves that SPAN the full price range, with
    equilibrium roughly in the middle. This allows for both:
    - Intramarginal units (should trade)
    - Extramarginal units (should NOT trade)

    This setup allows ZI traders to make inefficient trades by:
    - Trading extramarginal units that shouldn't trade
    - Trading at prices that transfer surplus inefficiently

    Reference: Gode & Sunder (1993), Smith (1962) induced value theory.
    """

    def __init__(
        self,
        num_tokens: int,
        price_min: int,
        price_max: int,
        seed: int,
        num_buyers: int = 5,
        num_sellers: int = 5,
    ):
        self.num_tokens = num_tokens
        self.price_min = price_min
        self.price_max = price_max
        self.num_buyers = num_buyers
        self.num_sellers = num_sellers
        self.rng = np.random.default_rng(seed)

    def new_round(self) -> None:
        """No-op - each call generates fresh random tokens."""
        pass

    def generate_tokens(self, is_buyer: bool) -> list[int]:
        """
        Generate tokens with overlapping supply/demand like Gode & Sunder.

        Creates equilibrium roughly at midpoint, with:
        - Buyers: values uniformly in [midpoint-margin, max_price]
        - Sellers: costs uniformly in [min_price, midpoint+margin]

        This creates ~60% overlap (extramarginal units exist) while
        maintaining positive expected surplus for intramarginal trades.

        ZIC only trades profitably, achieving ~99% efficiency.
        ZI trades randomly, achieving ~60-70% efficiency due to:
        - Trading extramarginal units (negative surplus)
        - Random price selection transferring value inefficiently
        """
        midpoint = (self.price_min + self.price_max) // 2
        range_size = self.price_max - self.price_min
        margin = range_size // 5  # 20% overlap zone

        if is_buyer:
            # Buyers: values in [midpoint - margin, max_price]
            # This means some buyer units (with low values) are extramarginal
            low = midpoint - margin
            high = self.price_max
        else:
            # Sellers: costs in [min_price, midpoint + margin]
            # This means some seller units (with high costs) are extramarginal
            low = self.price_min
            high = midpoint + margin

        tokens = [int(self.rng.integers(low, high + 1)) for _ in range(self.num_tokens)]

        # Sort tokens
        # Buyers: High to Low (descending) - best values first
        # Sellers: Low to High (ascending) - best costs first
        tokens.sort()

        if is_buyer:
            tokens.reverse()

        return tokens


class TokenGenerator:
    """
    Generates tokens (valuations/costs) for a round.

    Logic:
    - Based on a 4-digit 'game_type' seed.
    - Digits determine weights w[1]..w[4].
    - Shared parameters A, B1, B2, C are generated per round.
    - Individual tokens add a random noise component.

    Special case gametype=0 (EQL/LAD environments):
    - All players receive the same token values shifted by a common random constant.
    - This creates "equal endowment" where relative positions are identical.
    - Reference: Santa Fe 1992 paper Table 3.1
    """

    def __init__(self, game_type: int, num_tokens: int, seed: int):
        self.game_type = game_type
        self.num_tokens = num_tokens
        self.rng = np.random.default_rng(seed)

        # For gametype 0, use 6453 weights for base token generation
        effective_game_type = 6453 if game_type == 0 else game_type
        self.w = self._determine_weights(effective_game_type)

        # Round-specific state
        self.A = 0
        self.B1 = 0
        self.B2 = 0
        self.C: list[int] = []

        # For gametype 0: shared base tokens and shift
        self._equal_buyer_tokens: list[int] = []
        self._equal_seller_tokens: list[int] = []
        self._equal_shift: int = 0

    def _determine_weights(self, i: int) -> list[int]:
        w = [0] * 5
        temp_i = i
        for x in range(1, 5):
            digit = temp_i // (10 ** (4 - x))
            temp_i -= digit * (10 ** (4 - x))
            w[x] = int(3**digit - 1)
        return w

    def new_round(self) -> None:
        """Generate shared parameters for a new round."""
        # rng.integers(low, high) -> [low, high)
        # Java: nextInt(n) -> [0, n)
        # Java: nextInt(w[1]+1) -> [0, w[1]]

        self.A = int(self.rng.integers(0, self.w[1] + 1))
        self.B1 = int(self.rng.integers(0, self.w[2] + 1))
        self.B2 = int(self.rng.integers(0, self.w[2] + 1))

        self.C = [0] * (self.num_tokens * 2 + 1)  # 1-indexed for convenience matching Java
        for x in range(1, self.num_tokens * 2 + 1):
            self.C[x] = int(self.rng.integers(0, self.w[3] + 1))

        # For gametype 0: generate shared base tokens once per round
        if self.game_type == 0:
            self._generate_equal_base_tokens()

    def _generate_equal_base_tokens(self) -> None:
        """
        Generate base tokens for EQL/LAD (gametype 0).

        All agents receive identical relative token values. The shift is
        a common random constant that moves all values together.

        Per Santa Fe paper: "all players receive the same token values
        shifted by a common random constant"
        """
        # Generate a random shift (using w[1] range like A parameter)
        self._equal_shift = int(self.rng.integers(0, self.w[1] + 1))

        # Generate base buyer tokens (shared across all buyers)
        buyer_tokens = []
        for x in range(1, self.num_tokens + 1):
            # Use B1, C[x], and noise for buyer base values
            noise = int(self.rng.integers(0, self.w[4] + 1))
            val = self._equal_shift + self.B1 + self.C[x] + noise
            buyer_tokens.append(val)
        buyer_tokens.sort(reverse=True)  # Descending for buyers
        self._equal_buyer_tokens = buyer_tokens

        # Generate base seller tokens (shared across all sellers)
        seller_tokens = []
        for x in range(1, self.num_tokens + 1):
            # Use B2, C[numTokens+x], and noise for seller base values
            noise = int(self.rng.integers(0, self.w[4] + 1))
            val = self._equal_shift + self.B2 + self.C[self.num_tokens + x] + noise
            seller_tokens.append(val)
        seller_tokens.sort()  # Ascending for sellers
        self._equal_seller_tokens = seller_tokens

    def generate_tokens(self, is_buyer: bool) -> list[int]:
        """Generate sorted tokens for a player."""
        # Special handling for gametype 0 (EQL/LAD)
        if self.game_type == 0:
            if is_buyer:
                return self._equal_buyer_tokens.copy()
            else:
                return self._equal_seller_tokens.copy()

        # Standard gametype handling
        tokens = []
        for x in range(1, self.num_tokens + 1):
            noise = int(self.rng.integers(0, self.w[4] + 1))
            val = 0
            if is_buyer:
                # A + B1 + C[x] + noise
                val = self.A + self.B1 + self.C[x] + noise
            else:
                # A + B2 + C[numTokens+x] + noise
                val = self.A + self.B2 + self.C[self.num_tokens + x] + noise
            tokens.append(val)

        # Sort tokens
        # Buyers: High to Low (Descending)
        # Sellers: Low to High (Ascending)
        tokens.sort()

        if is_buyer:
            tokens.reverse()

        return tokens


def generate_tokens(
    num_buyers: int,
    num_sellers: int,
    num_tokens: int,
    price_min: int = 0,
    price_max: int = 100,
    game_type: int = 1111,
    seed: int | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Generate tokens for all buyers and sellers.

    Simple wrapper for environments that need random token generation
    without using the full tournament token generator logic.

    Args:
        num_buyers: Number of buyer agents
        num_sellers: Number of seller agents
        num_tokens: Tokens per agent
        price_min: Minimum price
        price_max: Maximum price
        game_type: Game type seed (4 digits)
        seed: Random seed

    Returns:
        (buyer_valuations, seller_costs) - Lists of token lists
    """
    if seed is None:
        seed = np.random.randint(0, 2**31)

    generator = TokenGenerator(game_type, num_tokens, seed)
    generator.new_round()

    buyer_valuations = []
    for _ in range(num_buyers):
        tokens = generator.generate_tokens(is_buyer=True)
        buyer_valuations.append(tokens)

    seller_costs = []
    for _ in range(num_sellers):
        tokens = generator.generate_tokens(is_buyer=False)
        seller_costs.append(tokens)

    return buyer_valuations, seller_costs
