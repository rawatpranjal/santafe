# tests/unit/traders/test_zic.py
"""
Adversarial tests for ZIC (Zero Intelligence Constrained) agent.

ZIC from Gode & Sunder (1993) is the foundation - if ZIC is broken,
everything else is suspect. These tests verify:
1. Bids are within [min_price, valuation]
2. Asks are within [cost, max_price]
3. Only accepts profitable trades
4. Never trades at a loss
"""

import numpy as np

from traders.legacy.zic import ZIC

# =============================================================================
# Test: Bid Range Constraints (Buyers)
# =============================================================================


class TestBidRangeConstraints:
    """Tests that buyer bids stay within [min_price, valuation]."""

    def test_bid_never_exceeds_valuation(self):
        """Buyer bid must NEVER exceed current token valuation."""
        buyer = ZIC(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            seed=42,
        )

        for token_idx in range(4):
            valuation = buyer.valuations[token_idx]
            buyer.num_trades = token_idx

            for _ in range(100):  # Many samples
                buyer.bid_ask(time=1, nobidask=0)
                bid = buyer.bid_ask_response()

                assert (
                    bid <= valuation
                ), f"Bid {bid} exceeds valuation {valuation} for token {token_idx}"

    def test_bid_never_below_min_price(self):
        """Buyer bid must never be below min_price."""
        buyer = ZIC(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=50,  # High min price
            price_max=200,
            seed=42,
        )

        for _ in range(100):
            buyer.bid_ask(time=1, nobidask=0)
            bid = buyer.bid_ask_response()

            assert bid >= 50, f"Bid {bid} below min_price 50"

    def test_bid_range_matches_java_formula(self):
        """Bid should follow Java formula: V - floor(random * (V - min))."""
        # With seed, we should get reproducible results
        buyer = ZIC(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[100],
            price_min=0,
            price_max=200,
            seed=42,
        )

        # Generate many bids and verify they're in range
        bids = []
        for _ in range(1000):
            buyer2 = ZIC(
                player_id=1,
                is_buyer=True,
                num_tokens=1,
                valuations=[100],
                price_min=0,
                price_max=200,
                seed=None,  # Random each time
            )
            buyer2.bid_ask(time=1, nobidask=0)
            bids.append(buyer2.bid_ask_response())

        # Should see bids across the range [0, 100]
        assert min(bids) >= 0, f"Min bid {min(bids)} below 0"
        assert max(bids) <= 100, f"Max bid {max(bids)} above valuation 100"
        # Should have some variance
        assert len(set(bids)) > 10, "Not enough bid variance - suspicious"


# =============================================================================
# Test: Ask Range Constraints (Sellers)
# =============================================================================


class TestAskRangeConstraints:
    """Tests that seller asks stay within [cost, max_price]."""

    def test_ask_never_below_cost(self):
        """Seller ask must NEVER be below current token cost."""
        seller = ZIC(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],  # Costs for seller
            price_min=1,
            price_max=200,
            seed=42,
        )

        for token_idx in range(4):
            cost = seller.valuations[token_idx]
            seller.num_trades = token_idx

            for _ in range(100):
                seller.bid_ask(time=1, nobidask=0)
                ask = seller.bid_ask_response()

                assert ask >= cost, f"Ask {ask} below cost {cost} for token {token_idx}"

    def test_ask_never_exceeds_max_price(self):
        """Seller ask must never exceed max_price."""
        seller = ZIC(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=100,  # Low max price
            seed=42,
        )

        for _ in range(100):
            seller.bid_ask(time=1, nobidask=0)
            ask = seller.bid_ask_response()

            assert ask <= 100, f"Ask {ask} exceeds max_price 100"


# =============================================================================
# Test: Profitable Trade Acceptance
# =============================================================================


class TestProfitableTradeAcceptance:
    """Tests that ZIC only accepts profitable trades."""

    def test_buyer_rejects_unprofitable_ask(self):
        """Buyer should reject if ask >= valuation (no profit)."""
        buyer = ZIC(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            seed=42,
        )

        # Setup: we are high bidder, but ask is AT valuation (no profit)
        buyer.buy_sell(
            time=1,
            nobuysell=0,  # We can trade
            high_bid=95,
            low_ask=100,  # Equal to valuation - NOT profitable
            high_bidder=1,  # We are high bidder
            low_asker=2,
        )

        assert buyer.buy_sell_response() is False, "Buyer should reject trade when ask >= valuation"

    def test_buyer_accepts_profitable_ask(self):
        """Buyer should accept if ask < valuation (profitable)."""
        buyer = ZIC(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            seed=42,
        )

        # Setup: we are high bidder, ask is below valuation
        buyer.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=95,
            low_ask=90,  # Below valuation of 100 - profitable
            high_bidder=1,  # We are high bidder
            low_asker=2,
        )

        assert (
            buyer.buy_sell_response() is True
        ), "Buyer should accept trade when ask < valuation and spread crossed"

    def test_seller_rejects_unprofitable_bid(self):
        """Seller should reject if bid <= cost (no profit)."""
        seller = ZIC(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],  # Costs
            price_min=1,
            price_max=200,
            seed=42,
        )

        # Setup: we are low asker, but bid is AT cost (no profit)
        seller.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=50,  # Equal to cost - NOT profitable
            low_ask=55,
            high_bidder=2,
            low_asker=1,  # We are low asker
        )

        assert seller.buy_sell_response() is False, "Seller should reject trade when bid <= cost"

    def test_seller_accepts_profitable_bid(self):
        """Seller should accept if bid > cost (profitable)."""
        seller = ZIC(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            seed=42,
        )

        # Setup: we are low asker, bid is above cost
        seller.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=60,  # Above cost of 50 - profitable
            low_ask=55,  # We're offering 55
            high_bidder=2,
            low_asker=1,  # We are low asker
        )

        assert (
            seller.buy_sell_response() is True
        ), "Seller should accept trade when bid > cost and spread crossed"


# =============================================================================
# Test: nobuysell Flag Handling
# =============================================================================


class TestNobuysellHandling:
    """Tests that ZIC respects the nobuysell restrictions."""

    def test_respects_nobuysell_flag(self):
        """ZIC should reject trade if nobuysell > 0."""
        buyer = ZIC(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            seed=42,
        )

        # Good trade conditions but nobuysell blocks it
        buyer.buy_sell(
            time=1,
            nobuysell=4,  # Not the high bidder
            high_bid=95,
            low_ask=80,
            high_bidder=1,
            low_asker=2,
        )

        assert buyer.buy_sell_response() is False, "Should reject when nobuysell > 0"


# =============================================================================
# Test: Position Tracking
# =============================================================================


class TestPositionTracking:
    """Tests that ZIC correctly tracks its position and uses right valuation."""

    def test_uses_correct_valuation_per_token(self):
        """Each trade should use the next token's valuation."""
        buyer = ZIC(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 80, 60, 40],  # Descending
            price_min=1,
            price_max=200,
            seed=42,
        )

        # For each token, verify bid doesn't exceed that token's valuation
        for expected_val in [100, 80, 60, 40]:
            bids = []
            for _ in range(50):
                # Reset seed for reproducibility
                buyer.rng = np.random.default_rng(None)
                buyer.bid_ask(time=1, nobidask=0)
                bids.append(buyer.bid_ask_response())

            max_bid = max(bids)
            assert (
                max_bid <= expected_val
            ), f"At token {buyer.num_trades}, max bid {max_bid} > valuation {expected_val}"

            # Move to next token
            buyer.num_trades += 1

    def test_returns_zero_when_no_tokens_left(self):
        """Should return 0 (pass) when all tokens traded."""
        buyer = ZIC(
            player_id=1,
            is_buyer=True,
            num_tokens=2,
            valuations=[100, 90],
            price_min=1,
            price_max=200,
            seed=42,
        )

        # Trade all tokens
        buyer.num_trades = 2

        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()

        assert bid == 0, f"Should return 0 when no tokens left, got {bid}"


# =============================================================================
# Test: Reproducibility
# =============================================================================


class TestReproducibility:
    """Tests for deterministic behavior with same seed."""

    def test_same_seed_same_bids(self):
        """Same seed should produce same bid sequence."""
        bids1 = []
        buyer1 = ZIC(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            seed=42,
        )
        for _ in range(10):
            buyer1.bid_ask(time=1, nobidask=0)
            bids1.append(buyer1.bid_ask_response())

        bids2 = []
        buyer2 = ZIC(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            seed=42,
        )
        for _ in range(10):
            buyer2.bid_ask(time=1, nobidask=0)
            bids2.append(buyer2.bid_ask_response())

        assert bids1 == bids2, "Same seed should give same bids"


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_valuation_equals_min_price(self):
        """When valuation == min_price, should bid min_price."""
        buyer = ZIC(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[50],  # Equals min_price
            price_min=50,
            price_max=200,
            seed=42,
        )

        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()

        assert bid == 50, f"Should bid min_price when valuation == min_price, got {bid}"

    def test_cost_equals_max_price(self):
        """When cost == max_price, should ask max_price."""
        seller = ZIC(
            player_id=1,
            is_buyer=False,
            num_tokens=1,
            valuations=[200],  # Equals max_price
            price_min=1,
            price_max=200,
            seed=42,
        )

        seller.bid_ask(time=1, nobidask=0)
        ask = seller.bid_ask_response()

        assert ask == 200, f"Should ask max_price when cost == max_price, got {ask}"

    def test_valuation_below_min_price(self):
        """When valuation < min_price (extramarginal), should bid min_price."""
        buyer = ZIC(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[30],  # Below min_price of 50
            price_min=50,
            price_max=200,
            seed=42,
        )

        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()

        assert bid == 50, f"Extramarginal buyer should bid min_price, got {bid}"

    def test_cost_above_max_price(self):
        """When cost > max_price (extramarginal), should ask max_price."""
        seller = ZIC(
            player_id=1,
            is_buyer=False,
            num_tokens=1,
            valuations=[250],  # Above max_price of 200
            price_min=1,
            price_max=200,
            seed=42,
        )

        seller.bid_ask(time=1, nobidask=0)
        ask = seller.bid_ask_response()

        assert ask == 200, f"Extramarginal seller should ask max_price, got {ask}"

    def test_spread_not_crossed_no_trade(self):
        """No trade should happen if bid < ask (spread not crossed)."""
        buyer = ZIC(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            seed=42,
        )

        # Spread not crossed: bid=80, ask=90
        buyer.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=80,
            low_ask=90,  # ask > bid, spread not crossed
            high_bidder=1,
            low_asker=2,
        )

        assert buyer.buy_sell_response() is False, "Should not trade when spread not crossed"


# =============================================================================
# Test: Multiple Agents Independence
# =============================================================================


class TestMultipleAgentsIndependence:
    """Tests that multiple ZIC agents are independent."""

    def test_agents_independent_with_different_seeds(self):
        """Different seeds should produce different bid sequences."""
        buyer1 = ZIC(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            seed=42,
        )
        buyer2 = ZIC(
            player_id=2,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            seed=123,
        )

        buyer1.bid_ask(time=1, nobidask=0)
        bid1 = buyer1.bid_ask_response()

        buyer2.bid_ask(time=1, nobidask=0)
        bid2 = buyer2.bid_ask_response()

        # Unlikely to be equal with different seeds
        # Run multiple times to increase confidence
        different = False
        for _ in range(10):
            buyer1.bid_ask(time=1, nobidask=0)
            buyer2.bid_ask(time=1, nobidask=0)
            if buyer1.bid_ask_response() != buyer2.bid_ask_response():
                different = True
                break

        assert different, "Agents with different seeds should produce different bids"


# =============================================================================
# Test: Defensive Checks
# =============================================================================


class TestDefensiveChecks:
    """Tests for defensive validation in buy_sell_response."""

    def test_rejects_negative_prices(self):
        """Should reject trade with negative prices (corrupted state)."""
        buyer = ZIC(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            seed=42,
        )

        # Simulate corrupted state with negative ask
        buyer.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=80,
            low_ask=-10,  # Invalid negative price
            high_bidder=1,
            low_asker=2,
        )

        assert buyer.buy_sell_response() is False, "Should reject trade with negative prices"

    def test_rejects_when_not_winner(self):
        """Should not accept if not the high bidder / low asker."""
        buyer = ZIC(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            seed=42,
        )

        # Good trade but we're not the high bidder
        buyer.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=95,
            low_ask=80,
            high_bidder=2,  # Someone else is high bidder
            low_asker=3,
        )

        assert buyer.buy_sell_response() is False, "Should not accept if not high bidder"


# =============================================================================
# STRESS TESTS - 1000 Trial Verification
# =============================================================================


class TestZICStress1000Trials:
    """Stress tests with 1000 trials to verify budget constraints NEVER violated.

    From the ruthless testing plan:
    - ZIC buyer bids in U[MinPrice, TokenRedemptionValue] - NEVER above valuation
    - ZIC seller asks in U[TokenCost, MaxPrice] - NEVER below cost
    - Test 1000 random bids, ALL must be within budget constraint

    If even ONE violation occurs in 1000 trials, the implementation is BROKEN.
    """

    def test_buyer_never_bids_above_valuation_1000_trials(self):
        """ZIC buyer MUST bid in [MinPrice, Valuation] across 1000 trials.

        This is the DEFINING property of ZIC from Gode & Sunder (1993).
        A single violation means the budget constraint is broken.
        """
        violations = []
        valuation = 100

        for seed in range(1000):
            buyer = ZIC(
                player_id=1,
                is_buyer=True,
                num_tokens=1,
                valuations=[valuation],
                price_min=1,
                price_max=500,  # High max to make violations obvious
                seed=seed,
            )
            buyer.bid_ask(time=1, nobidask=0)
            bid = buyer.bid_ask_response()

            if bid > valuation:
                violations.append((seed, bid, valuation))

        assert len(violations) == 0, (
            f"{len(violations)}/1000 buyer bids exceeded valuation! "
            f"First 5 violations: {violations[:5]}"
        )

    def test_seller_never_asks_below_cost_1000_trials(self):
        """ZIC seller MUST ask in [Cost, MaxPrice] across 1000 trials.

        This is the DEFINING property of ZIC from Gode & Sunder (1993).
        A single violation means the budget constraint is broken.
        """
        violations = []
        cost = 80

        for seed in range(1000):
            seller = ZIC(
                player_id=1,
                is_buyer=False,
                num_tokens=1,
                valuations=[cost],
                price_min=1,  # Low min to make violations obvious
                price_max=200,
                seed=seed,
            )
            seller.bid_ask(time=1, nobidask=0)
            ask = seller.bid_ask_response()

            if ask < cost:
                violations.append((seed, ask, cost))

        assert len(violations) == 0, (
            f"{len(violations)}/1000 seller asks below cost! "
            f"First 5 violations: {violations[:5]}"
        )

    def test_buyer_all_tokens_never_exceed_valuation_1000_trials(self):
        """Test ALL 4 tokens across 1000 traders - no bid should exceed valuation."""
        valuations = [100, 85, 70, 55]
        violations = []

        for seed in range(1000):
            buyer = ZIC(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=valuations.copy(),
                price_min=1,
                price_max=500,
                seed=seed,
            )

            for token_idx in range(4):
                buyer.num_trades = token_idx
                buyer.bid_ask(time=1, nobidask=0)
                bid = buyer.bid_ask_response()
                expected_max = valuations[token_idx]

                if bid > expected_max:
                    violations.append((seed, token_idx, bid, expected_max))

        assert len(violations) == 0, (
            f"{len(violations)}/4000 bids exceeded valuation! " f"First 5: {violations[:5]}"
        )

    def test_seller_all_tokens_never_below_cost_1000_trials(self):
        """Test ALL 4 tokens across 1000 traders - no ask should be below cost."""
        costs = [40, 55, 70, 85]
        violations = []

        for seed in range(1000):
            seller = ZIC(
                player_id=1,
                is_buyer=False,
                num_tokens=4,
                valuations=costs.copy(),
                price_min=1,
                price_max=200,
                seed=seed,
            )

            for token_idx in range(4):
                seller.num_trades = token_idx
                seller.bid_ask(time=1, nobidask=0)
                ask = seller.bid_ask_response()
                expected_min = costs[token_idx]

                if ask < expected_min:
                    violations.append((seed, token_idx, ask, expected_min))

        assert len(violations) == 0, (
            f"{len(violations)}/4000 asks below cost! " f"First 5: {violations[:5]}"
        )

    def test_bid_distribution_spans_range(self):
        """Verify bids span the full valid range [min_price, valuation].

        ZIC should produce uniformly distributed bids within constraints.
        If bids cluster or don't span the range, the random generation is broken.
        """
        valuation = 100
        min_price = 20
        bids = []

        for seed in range(1000):
            buyer = ZIC(
                player_id=1,
                is_buyer=True,
                num_tokens=1,
                valuations=[valuation],
                price_min=min_price,
                price_max=500,
                seed=seed,
            )
            buyer.bid_ask(time=1, nobidask=0)
            bids.append(buyer.bid_ask_response())

        # Should cover most of the valid range
        bid_range = max(bids) - min(bids)
        expected_range = valuation - min_price  # 80

        # Should cover at least 80% of the valid range
        assert (
            bid_range >= 0.8 * expected_range
        ), f"Bid range {bid_range} doesn't span enough of valid range {expected_range}"

        # All bids should be in valid range
        assert min(bids) >= min_price, f"Min bid {min(bids)} below min_price {min_price}"
        assert max(bids) <= valuation, f"Max bid {max(bids)} above valuation {valuation}"

    def test_ask_distribution_spans_range(self):
        """Verify asks span the full valid range [cost, max_price].

        ZIC should produce uniformly distributed asks within constraints.
        """
        cost = 50
        max_price = 150
        asks = []

        for seed in range(1000):
            seller = ZIC(
                player_id=1,
                is_buyer=False,
                num_tokens=1,
                valuations=[cost],
                price_min=1,
                price_max=max_price,
                seed=seed,
            )
            seller.bid_ask(time=1, nobidask=0)
            asks.append(seller.bid_ask_response())

        # Should cover most of the valid range
        ask_range = max(asks) - min(asks)
        expected_range = max_price - cost  # 100

        # Should cover at least 80% of the valid range
        assert (
            ask_range >= 0.8 * expected_range
        ), f"Ask range {ask_range} doesn't span enough of valid range {expected_range}"

        # All asks should be in valid range
        assert min(asks) >= cost, f"Min ask {min(asks)} below cost {cost}"
        assert max(asks) <= max_price, f"Max ask {max(asks)} above max_price {max_price}"
