# tests/unit/engine/test_token_generator.py
"""
Adversarial tests for TokenGenerator (Santa Fe) and UniformTokenGenerator (Gode-Sunder).

These tests verify:
1. Token sorting (buyers descending, sellers ascending)
2. Reproducibility with seeds
3. Weight computation from game_type
4. Token range constraints
"""


from engine.token_generator import (
    TokenGenerator,
    UniformTokenGenerator,
    generate_tokens,
)

# =============================================================================
# Test: TokenGenerator Weight Computation
# =============================================================================


class TestTokenGeneratorWeights:
    """Tests for the 4-digit game_type to weight mapping."""

    def test_weight_computation_game_type_0000(self):
        """Game type 0000 uses 6453 weights internally for EQL/LAD environments.

        Per Santa Fe 1992 paper Table 3.1, gametype=0 is for EQL/LAD environments
        where all players receive the same token values. To generate meaningful
        tokens, we use gametype 6453 weights internally.

        Weights for 6453: w[i] = 3^digit[i] - 1
        - w[1] = 3^6 - 1 = 728
        - w[2] = 3^4 - 1 = 80
        - w[3] = 3^5 - 1 = 242
        - w[4] = 3^3 - 1 = 26
        """
        gen = TokenGenerator(game_type=0, num_tokens=4, seed=42)
        # Internally uses 6453 weights
        assert gen.w[1] == 728, f"w[1] should be 728, got {gen.w[1]}"
        assert gen.w[2] == 80, f"w[2] should be 80, got {gen.w[2]}"
        assert gen.w[3] == 242, f"w[3] should be 242, got {gen.w[3]}"
        assert gen.w[4] == 26, f"w[4] should be 26, got {gen.w[4]}"

    def test_weight_computation_game_type_1111(self):
        """Game type 1111 should give all weights = 2 (3^1 - 1 = 2)."""
        gen = TokenGenerator(game_type=1111, num_tokens=4, seed=42)
        for i in range(1, 5):
            assert gen.w[i] == 2, f"w[{i}] should be 2, got {gen.w[i]}"

    def test_weight_computation_game_type_2222(self):
        """Game type 2222 should give all weights = 8 (3^2 - 1 = 8)."""
        gen = TokenGenerator(game_type=2222, num_tokens=4, seed=42)
        for i in range(1, 5):
            assert gen.w[i] == 8, f"w[{i}] should be 8, got {gen.w[i]}"

    def test_weight_computation_game_type_3333(self):
        """Game type 3333 should give all weights = 26 (3^3 - 1 = 26)."""
        gen = TokenGenerator(game_type=3333, num_tokens=4, seed=42)
        for i in range(1, 5):
            assert gen.w[i] == 26, f"w[{i}] should be 26, got {gen.w[i]}"

    def test_weight_computation_mixed_digits(self):
        """Game type 0123 should give weights [0, 2, 8, 26]."""
        gen = TokenGenerator(game_type=123, num_tokens=4, seed=42)
        # Note: 123 is interpreted as 0123
        expected = {1: 0, 2: 2, 3: 8, 4: 26}
        for i, expected_w in expected.items():
            assert gen.w[i] == expected_w, f"w[{i}] should be {expected_w}, got {gen.w[i]}"


# =============================================================================
# Test: Token Sorting
# =============================================================================


class TestTokenSorting:
    """Tests for correct token sorting order."""

    def test_buyer_tokens_sorted_descending(self):
        """Buyer tokens must be sorted high to low (descending)."""
        gen = TokenGenerator(game_type=2222, num_tokens=4, seed=42)
        gen.new_round()
        tokens = gen.generate_tokens(is_buyer=True)

        for i in range(len(tokens) - 1):
            assert (
                tokens[i] >= tokens[i + 1]
            ), f"Buyer tokens not descending: {tokens[i]} < {tokens[i + 1]} at position {i}"

    def test_seller_tokens_sorted_ascending(self):
        """Seller tokens must be sorted low to high (ascending)."""
        gen = TokenGenerator(game_type=2222, num_tokens=4, seed=42)
        gen.new_round()
        tokens = gen.generate_tokens(is_buyer=False)

        for i in range(len(tokens) - 1):
            assert (
                tokens[i] <= tokens[i + 1]
            ), f"Seller tokens not ascending: {tokens[i]} > {tokens[i + 1]} at position {i}"

    def test_uniform_buyer_tokens_sorted_descending(self):
        """UniformTokenGenerator buyer tokens must be descending."""
        gen = UniformTokenGenerator(
            num_tokens=4,
            price_min=1,
            price_max=200,
            seed=42,
        )
        tokens = gen.generate_tokens(is_buyer=True)

        for i in range(len(tokens) - 1):
            assert tokens[i] >= tokens[i + 1], f"Uniform buyer tokens not descending: {tokens}"

    def test_uniform_seller_tokens_sorted_ascending(self):
        """UniformTokenGenerator seller tokens must be ascending."""
        gen = UniformTokenGenerator(
            num_tokens=4,
            price_min=1,
            price_max=200,
            seed=42,
        )
        tokens = gen.generate_tokens(is_buyer=False)

        for i in range(len(tokens) - 1):
            assert tokens[i] <= tokens[i + 1], f"Uniform seller tokens not ascending: {tokens}"


# =============================================================================
# Test: Reproducibility
# =============================================================================


class TestReproducibility:
    """Tests for deterministic token generation with same seed."""

    def test_same_seed_same_tokens(self):
        """Same seed should produce identical tokens."""
        gen1 = TokenGenerator(game_type=1111, num_tokens=4, seed=42)
        gen1.new_round()
        tokens1_buyer = gen1.generate_tokens(is_buyer=True)
        tokens1_seller = gen1.generate_tokens(is_buyer=False)

        gen2 = TokenGenerator(game_type=1111, num_tokens=4, seed=42)
        gen2.new_round()
        tokens2_buyer = gen2.generate_tokens(is_buyer=True)
        tokens2_seller = gen2.generate_tokens(is_buyer=False)

        assert tokens1_buyer == tokens2_buyer, "Same seed should give same buyer tokens"
        assert tokens1_seller == tokens2_seller, "Same seed should give same seller tokens"

    def test_different_seed_different_tokens(self):
        """Different seeds should (almost always) produce different tokens."""
        gen1 = TokenGenerator(game_type=2222, num_tokens=4, seed=42)
        gen1.new_round()
        tokens1 = gen1.generate_tokens(is_buyer=True)

        gen2 = TokenGenerator(game_type=2222, num_tokens=4, seed=123)
        gen2.new_round()
        tokens2 = gen2.generate_tokens(is_buyer=True)

        # With game_type 2222 (w=8), different seeds should give different results
        assert tokens1 != tokens2, "Different seeds should give different tokens"

    def test_new_round_changes_tokens(self):
        """Calling new_round() should generate different A, B, C values."""
        gen = TokenGenerator(game_type=2222, num_tokens=4, seed=42)

        gen.new_round()
        tokens1 = gen.generate_tokens(is_buyer=True)

        gen.new_round()
        tokens2 = gen.generate_tokens(is_buyer=True)

        # With high enough weights, consecutive rounds should differ
        assert tokens1 != tokens2, "new_round() should change tokens"

    def test_uniform_generator_reproducibility(self):
        """UniformTokenGenerator should be reproducible with same seed."""
        gen1 = UniformTokenGenerator(num_tokens=4, price_min=1, price_max=200, seed=42)
        tokens1 = gen1.generate_tokens(is_buyer=True)

        gen2 = UniformTokenGenerator(num_tokens=4, price_min=1, price_max=200, seed=42)
        tokens2 = gen2.generate_tokens(is_buyer=True)

        assert tokens1 == tokens2, "Same seed should give same uniform tokens"


# =============================================================================
# Test: Token Range Constraints
# =============================================================================


class TestTokenRangeConstraints:
    """Tests for token values staying within expected ranges."""

    def test_tokens_are_non_negative(self):
        """All tokens should be non-negative."""
        gen = TokenGenerator(game_type=1111, num_tokens=4, seed=42)
        gen.new_round()

        buyer_tokens = gen.generate_tokens(is_buyer=True)
        seller_tokens = gen.generate_tokens(is_buyer=False)

        for t in buyer_tokens:
            assert t >= 0, f"Buyer token negative: {t}"
        for t in seller_tokens:
            assert t >= 0, f"Seller token negative: {t}"

    def test_game_type_0000_eql_lad_equal_tokens(self):
        """Game type 0000 (EQL/LAD) gives identical tokens to all agents.

        Per Santa Fe 1992 paper Table 3.1, gametype=0 is for EQL/LAD environments
        where "all players receive the same token values shifted by a common
        random constant". All buyers get identical tokens, all sellers get
        identical tokens.
        """
        gen = TokenGenerator(game_type=0, num_tokens=4, seed=42)
        gen.new_round()

        # Generate multiple buyer and seller token sets
        buyer_tokens_1 = gen.generate_tokens(is_buyer=True)
        buyer_tokens_2 = gen.generate_tokens(is_buyer=True)
        buyer_tokens_3 = gen.generate_tokens(is_buyer=True)

        seller_tokens_1 = gen.generate_tokens(is_buyer=False)
        seller_tokens_2 = gen.generate_tokens(is_buyer=False)
        seller_tokens_3 = gen.generate_tokens(is_buyer=False)

        # All buyers should get identical tokens
        assert buyer_tokens_1 == buyer_tokens_2, "Buyers should get same tokens"
        assert buyer_tokens_2 == buyer_tokens_3, "Buyers should get same tokens"

        # All sellers should get identical tokens
        assert seller_tokens_1 == seller_tokens_2, "Sellers should get same tokens"
        assert seller_tokens_2 == seller_tokens_3, "Sellers should get same tokens"

        # Tokens should be non-zero (using 6453 weights internally)
        assert any(t > 0 for t in buyer_tokens_1), "Tokens should be non-zero"
        assert any(t > 0 for t in seller_tokens_1), "Tokens should be non-zero"

    def test_uniform_tokens_within_range(self):
        """UniformTokenGenerator tokens should be within price bounds."""
        gen = UniformTokenGenerator(
            num_tokens=10,
            price_min=50,
            price_max=150,
            seed=42,
        )

        for _ in range(5):  # Multiple rounds
            buyer_tokens = gen.generate_tokens(is_buyer=True)
            seller_tokens = gen.generate_tokens(is_buyer=False)

            for t in buyer_tokens:
                assert 50 <= t <= 150, f"Buyer token {t} out of range [50, 150]"
            for t in seller_tokens:
                assert 50 <= t <= 150, f"Seller token {t} out of range [50, 150]"


# =============================================================================
# Test: Token Count
# =============================================================================


class TestTokenCount:
    """Tests for correct number of tokens generated."""

    def test_correct_number_of_tokens(self):
        """Generator should produce exactly num_tokens tokens."""
        for num_tokens in [1, 4, 10, 20]:
            gen = TokenGenerator(game_type=1111, num_tokens=num_tokens, seed=42)
            gen.new_round()

            buyer_tokens = gen.generate_tokens(is_buyer=True)
            seller_tokens = gen.generate_tokens(is_buyer=False)

            assert (
                len(buyer_tokens) == num_tokens
            ), f"Expected {num_tokens} buyer tokens, got {len(buyer_tokens)}"
            assert (
                len(seller_tokens) == num_tokens
            ), f"Expected {num_tokens} seller tokens, got {len(seller_tokens)}"

    def test_uniform_correct_number_of_tokens(self):
        """UniformTokenGenerator should produce exactly num_tokens tokens."""
        for num_tokens in [1, 4, 10]:
            gen = UniformTokenGenerator(
                num_tokens=num_tokens,
                price_min=1,
                price_max=200,
                seed=42,
            )

            buyer_tokens = gen.generate_tokens(is_buyer=True)
            seller_tokens = gen.generate_tokens(is_buyer=False)

            assert len(buyer_tokens) == num_tokens
            assert len(seller_tokens) == num_tokens


# =============================================================================
# Test: generate_tokens() Function
# =============================================================================


class TestGenerateTokensFunction:
    """Tests for the convenience generate_tokens() wrapper function."""

    def test_returns_correct_structure(self):
        """Should return (buyer_valuations, seller_costs) lists."""
        buyer_vals, seller_costs = generate_tokens(
            num_buyers=3,
            num_sellers=2,
            num_tokens=4,
            seed=42,
        )

        assert len(buyer_vals) == 3, "Should have 3 buyer token lists"
        assert len(seller_costs) == 2, "Should have 2 seller token lists"

        for bv in buyer_vals:
            assert len(bv) == 4, "Each buyer should have 4 tokens"
        for sc in seller_costs:
            assert len(sc) == 4, "Each seller should have 4 tokens"

    def test_buyers_sorted_descending(self):
        """All buyer token lists should be sorted descending."""
        buyer_vals, _ = generate_tokens(
            num_buyers=5,
            num_sellers=5,
            num_tokens=4,
            seed=42,
        )

        for i, tokens in enumerate(buyer_vals):
            for j in range(len(tokens) - 1):
                assert tokens[j] >= tokens[j + 1], f"Buyer {i} tokens not descending: {tokens}"

    def test_sellers_sorted_ascending(self):
        """All seller token lists should be sorted ascending."""
        _, seller_costs = generate_tokens(
            num_buyers=5,
            num_sellers=5,
            num_tokens=4,
            seed=42,
        )

        for i, tokens in enumerate(seller_costs):
            for j in range(len(tokens) - 1):
                assert tokens[j] <= tokens[j + 1], f"Seller {i} tokens not ascending: {tokens}"

    def test_reproducibility_with_seed(self):
        """Same seed should produce same results."""
        result1 = generate_tokens(num_buyers=4, num_sellers=4, num_tokens=4, seed=42)
        result2 = generate_tokens(num_buyers=4, num_sellers=4, num_tokens=4, seed=42)

        assert result1 == result2, "Same seed should give same results"


# =============================================================================
# Test: Market Creation with Generated Tokens
# =============================================================================


class TestSupplyDemandProperties:
    """Tests for economic properties of generated supply/demand."""

    def test_uniform_creates_overlapping_supply_demand(self):
        """UniformTokenGenerator should create overlapping supply/demand.

        This is necessary for market efficiency experiments - we need
        some trades to be possible (overlap) and some to not be.
        """
        gen = UniformTokenGenerator(
            num_tokens=10,
            price_min=1,
            price_max=200,
            seed=42,
            num_buyers=4,
            num_sellers=4,
        )

        # Generate many sets to verify overlap
        overlaps = 0
        for _ in range(20):
            buyer_tokens = gen.generate_tokens(is_buyer=True)
            seller_tokens = gen.generate_tokens(is_buyer=False)

            # Check if max buyer value >= min seller cost (overlap exists)
            if max(buyer_tokens) >= min(seller_tokens):
                overlaps += 1

        # Should have overlap in most cases
        assert overlaps > 15, f"Expected supply/demand overlap in most cases, got {overlaps}/20"

    def test_buyer_first_token_highest_value(self):
        """Buyer's first token should be their highest valuation."""
        gen = TokenGenerator(game_type=2222, num_tokens=4, seed=42)
        gen.new_round()
        tokens = gen.generate_tokens(is_buyer=True)

        assert tokens[0] == max(tokens), "First buyer token should be highest"

    def test_seller_first_token_lowest_cost(self):
        """Seller's first token should be their lowest cost."""
        gen = TokenGenerator(game_type=2222, num_tokens=4, seed=42)
        gen.new_round()
        tokens = gen.generate_tokens(is_buyer=False)

        assert tokens[0] == min(tokens), "First seller token should be lowest"


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_token(self):
        """Should work with just 1 token per agent."""
        gen = TokenGenerator(game_type=1111, num_tokens=1, seed=42)
        gen.new_round()

        buyer_tokens = gen.generate_tokens(is_buyer=True)
        seller_tokens = gen.generate_tokens(is_buyer=False)

        assert len(buyer_tokens) == 1
        assert len(seller_tokens) == 1

    def test_large_number_of_tokens(self):
        """Should work with many tokens."""
        gen = TokenGenerator(game_type=1111, num_tokens=100, seed=42)
        gen.new_round()

        buyer_tokens = gen.generate_tokens(is_buyer=True)
        seller_tokens = gen.generate_tokens(is_buyer=False)

        assert len(buyer_tokens) == 100
        assert len(seller_tokens) == 100

        # Verify still sorted
        for i in range(99):
            assert buyer_tokens[i] >= buyer_tokens[i + 1]
            assert seller_tokens[i] <= seller_tokens[i + 1]

    def test_narrow_price_range(self):
        """UniformTokenGenerator should work with narrow price range."""
        gen = UniformTokenGenerator(
            num_tokens=4,
            price_min=100,
            price_max=105,
            seed=42,
        )

        tokens = gen.generate_tokens(is_buyer=True)
        for t in tokens:
            assert 100 <= t <= 105, f"Token {t} outside narrow range [100, 105]"

    def test_multiple_agents_different_tokens(self):
        """Different agents should get different tokens (not identical)."""
        buyer_vals, seller_costs = generate_tokens(
            num_buyers=4,
            num_sellers=4,
            num_tokens=4,
            game_type=2222,  # High variation
            seed=42,
        )

        # Check buyers aren't all identical
        unique_buyer_lists = {tuple(bv) for bv in buyer_vals}
        assert (
            len(unique_buyer_lists) > 1
        ), "Different buyers should have different tokens (with high game_type)"

        # Check sellers aren't all identical
        unique_seller_lists = {tuple(sc) for sc in seller_costs}
        assert (
            len(unique_seller_lists) > 1
        ), "Different sellers should have different tokens (with high game_type)"
