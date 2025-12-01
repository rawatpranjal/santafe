# tests/unit/traders/test_skeleton.py
"""
Adversarial tests for Skeleton agent.
Skeleton is the baseline strategy from the Santa Fe tournament.

Per Rust et al. (1994) Figure 4 and DATManual Section 3.4:
- Uses alpha = 0.25 + 0.1 * U[0,1] for randomization
- Bids/asks based on CBID, CASK, and own token values
- CRITICAL: Only bids/asks when there's a reference price to react to

Key invariants tested:
1. No bid when no CBID AND no CASK
2. No bid when CBID exists but no CASK
3. No ask when no CASK AND no CBID
4. No ask when CASK exists but no CBID
5. Bid formula: MOST - alpha*SPAN (Case 1) or weighted avg (Case 2)
6. Never bids above valuation (buyer rationality)
7. Never asks below cost (seller rationality)
"""

from traders.legacy.skeleton import Skeleton


class TestSkeletonBuyerNoBid:
    """Test cases where buyer should NOT bid (return 0)."""

    def test_buyer_no_bid_when_no_cbid_no_cask(self):
        """BUG 1: Buyer should NOT bid when there's no CBID and no CASK.

        Per spec: "If there is no current bid AND no current ask → don't bid"
        The buyer has nothing to react to.
        """
        buyer = Skeleton(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            seed=42,
        )

        # Set up: no current bid, no current ask
        buyer.current_bid = 0
        buyer.current_ask = 0
        buyer.nobidask = 0

        bid = buyer._request_bid()
        assert bid == 0, f"Buyer should NOT bid when no CBID and no CASK, got {bid}"

    def test_buyer_no_bid_when_cbid_exists_but_no_cask(self):
        """BUG 3: Buyer should NOT bid when CBID exists but no CASK.

        Per spec: "If there is a current bid but NO current ask → don't bid"
        Without an ask to reference, no good target price exists.
        """
        buyer = Skeleton(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            seed=42,
        )

        # Set up: current bid exists, no current ask
        buyer.current_bid = 50
        buyer.current_ask = 0
        buyer.nobidask = 0

        bid = buyer._request_bid()
        assert bid == 0, f"Buyer should NOT bid when CBID exists but no CASK, got {bid}"


class TestSkeletonSellerNoAsk:
    """Test cases where seller should NOT ask (return 0)."""

    def test_seller_no_ask_when_no_cask_no_cbid(self):
        """BUG 5: Seller should NOT ask when there's no CASK and no CBID.

        Per spec: "If there is no current ask AND no current bid → don't ask"
        The seller has nothing to react to.
        """
        seller = Skeleton(
            player_id=2,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],  # Ascending costs
            seed=42,
        )

        # Set up: no current ask, no current bid
        seller.current_ask = 0
        seller.current_bid = 0
        seller.nobidask = 0

        ask = seller._request_ask()
        assert ask == 0, f"Seller should NOT ask when no CASK and no CBID, got {ask}"

    def test_seller_no_ask_when_cask_exists_but_no_cbid(self):
        """BUG 6: Seller should NOT ask when CASK exists but no CBID.

        Per spec: "If there is a current ask but NO current bid → don't ask"
        Without a bid to reference, no good target price exists.
        """
        seller = Skeleton(
            player_id=2,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            seed=42,
        )

        # Set up: current ask exists, no current bid
        seller.current_ask = 90
        seller.current_bid = 0
        seller.nobidask = 0

        ask = seller._request_ask()
        assert ask == 0, f"Seller should NOT ask when CASK exists but no CBID, got {ask}"


class TestSkeletonBuyerCase1:
    """Test Case 1: No current bid, but current ask exists."""

    def test_buyer_case1_bids_when_cask_exists(self):
        """Case 1: Buyer SHOULD bid when no CBID but CASK exists."""
        buyer = Skeleton(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            seed=42,
        )

        # Set up: no current bid, but current ask exists
        buyer.current_bid = 0
        buyer.current_ask = 85
        buyer.nobidask = 0

        bid = buyer._request_bid()
        assert bid > 0, f"Buyer should bid when CASK exists, got {bid}"

    def test_buyer_case1_formula_most_minus_alpha_span(self):
        """Case 1 formula: Bid = MOST - alpha * SPAN.

        Where:
        - MOST = min(CASK, last_token - 1)
        - SPAN = first_token - last_token
        - alpha = 0.25 + 0.1 * U[0,1] ∈ [0.25, 0.35]
        """
        buyer = Skeleton(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            seed=42,
        )

        # Set up Case 1
        buyer.current_bid = 0
        buyer.current_ask = 85
        buyer.nobidask = 0

        # Compute expected range
        first_token = 100
        last_token = 70
        span = first_token - last_token  # 30

        # MOST = min(85, 70-1) = min(85, 69) = 69
        most = min(85, 69)

        # Bid = MOST - alpha * SPAN
        # alpha ∈ [0.25, 0.35], so bid ∈ [MOST - 0.35*30, MOST - 0.25*30]
        # bid ∈ [69 - 10.5, 69 - 7.5] = [58.5, 61.5] → [58, 61] after int()

        bid = buyer._request_bid()
        assert 58 <= bid <= 62, f"Bid should be in [58, 62], got {bid}"


class TestSkeletonBuyerCase2:
    """Test Case 2: Current bid exists, current ask exists."""

    def test_buyer_case2_weighted_average_formula(self):
        """Case 2 formula: Bid = (1-alpha)*(CBID+1) + alpha*MOST.

        Where:
        - MOST = min(CASK, token_val - 1)
        - alpha ∈ [0.25, 0.35]
        """
        buyer = Skeleton(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            seed=42,
        )

        # Set up Case 2
        buyer.current_bid = 60
        buyer.current_ask = 85
        buyer.nobidask = 0

        # token_val = 100 (first token)
        # MOST = min(85, 100-1) = min(85, 99) = 85
        most = 85
        cbid_plus_1 = 61

        # Bid = (1-alpha)*61 + alpha*85
        # alpha=0.25: (0.75)*61 + 0.25*85 = 45.75 + 21.25 = 67
        # alpha=0.35: (0.65)*61 + 0.35*85 = 39.65 + 29.75 = 69.4

        bid = buyer._request_bid()
        assert 67 <= bid <= 70, f"Bid should be in [67, 70], got {bid}"

    def test_buyer_case2_no_bid_when_most_lte_cbid(self):
        """Case 2: Buyer should NOT bid if MOST <= CBID (can't improve)."""
        buyer = Skeleton(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            seed=42,
        )

        # Set up: CBID is already at MOST or higher
        buyer.current_bid = 85  # CBID = 85
        buyer.current_ask = 90  # CASK = 90
        buyer.nobidask = 0

        # token_val = 100
        # MOST = min(90, 100-1) = min(90, 99) = 90
        # But wait, MOST = 90 > CBID = 85, so we'd bid
        # Let's use a case where MOST <= CBID
        buyer.current_bid = 95  # CBID = 95
        buyer.current_ask = 90  # CASK = 90

        # MOST = min(90, 99) = 90
        # MOST (90) <= CBID (95), so no bid

        bid = buyer._request_bid()
        assert bid == 0, f"Should not bid when MOST <= CBID, got {bid}"


class TestSkeletonSellerCase1:
    """Test Case 1 for seller: No current ask, but current bid exists."""

    def test_seller_case1_asks_when_cbid_exists(self):
        """Case 1: Seller SHOULD ask when no CASK but CBID exists."""
        seller = Skeleton(
            player_id=2,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            seed=42,
        )

        # Set up: no current ask, but current bid exists
        seller.current_ask = 0
        seller.current_bid = 55
        seller.nobidask = 0

        ask = seller._request_ask()
        assert ask > 0, f"Seller should ask when CBID exists, got {ask}"

    def test_seller_case1_formula_least_plus_alpha_span(self):
        """Case 1 formula for seller: Ask = LEAST + alpha * SPAN.

        Where:
        - LEAST = max(CBID, last_token + 1)
        - SPAN = last_token - first_token (for seller)
        - alpha ∈ [0.25, 0.35]
        """
        seller = Skeleton(
            player_id=2,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],  # Ascending costs
            seed=42,
        )

        # Set up Case 1
        seller.current_ask = 0
        seller.current_bid = 55
        seller.nobidask = 0

        # first_token = 50 (lowest cost = best)
        # last_token = 80 (highest cost = worst)
        # SPAN = 80 - 50 = 30

        # LEAST = max(55, 80+1) = max(55, 81) = 81
        least = max(55, 81)  # 81
        span = 30

        # Ask = LEAST + alpha * SPAN
        # alpha=0.25: 81 + 0.25*30 = 81 + 7.5 = 88.5
        # alpha=0.35: 81 + 0.35*30 = 81 + 10.5 = 91.5

        ask = seller._request_ask()
        assert 88 <= ask <= 92, f"Ask should be in [88, 92], got {ask}"


class TestSkeletonRationality:
    """Test that Skeleton never makes irrational trades."""

    def test_buyer_never_bids_above_valuation(self):
        """Buyer bid should never exceed current token valuation."""
        buyer = Skeleton(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            seed=42,
        )

        # Test with various market conditions
        for cask in [50, 75, 100, 150]:
            for cbid in [0, 30, 60]:
                buyer.current_ask = cask
                buyer.current_bid = cbid
                buyer.nobidask = 0

                bid = buyer._request_bid()
                if bid > 0:
                    assert (
                        bid <= buyer.valuations[buyer.num_trades]
                    ), f"Bid {bid} exceeds valuation {buyer.valuations[buyer.num_trades]}"

    def test_seller_never_asks_below_cost(self):
        """Seller ask should never be below current token cost."""
        seller = Skeleton(
            player_id=2,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            seed=42,
        )

        # Test with various market conditions
        for cbid in [40, 60, 90, 120]:
            for cask in [0, 70, 100]:
                seller.current_bid = cbid
                seller.current_ask = cask
                seller.nobidask = 0

                ask = seller._request_ask()
                if ask > 0:
                    assert (
                        ask >= seller.valuations[seller.num_trades]
                    ), f"Ask {ask} below cost {seller.valuations[seller.num_trades]}"


class TestSkeletonAlphaDistribution:
    """Test that alpha is correctly distributed in [0.25, 0.35]."""

    def test_alpha_range_affects_bid_variance(self):
        """Multiple bids should show variance due to random alpha."""
        bids = []
        for seed in range(100):
            buyer = Skeleton(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=[100, 90, 80, 70],
                seed=seed,
            )
            buyer.current_bid = 50
            buyer.current_ask = 85
            buyer.nobidask = 0

            bid = buyer._request_bid()
            if bid > 0:
                bids.append(bid)

        # Should have variance from alpha randomization
        assert len(set(bids)) > 1, "Bids should vary due to random alpha"
        # But should be bounded
        assert min(bids) >= 50, f"Min bid {min(bids)} too low"
        assert max(bids) <= 85, f"Max bid {max(bids)} too high"


class TestSkeletonNobidaskFlag:
    """Test nobidask flag handling."""

    def test_buyer_no_bid_when_nobidask_set(self):
        """Buyer should not bid when nobidask > 0."""
        buyer = Skeleton(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            seed=42,
        )

        buyer.current_bid = 50
        buyer.current_ask = 85
        buyer.nobidask = 1  # Set: cannot bid

        bid = buyer._request_bid()
        assert bid == 0, "Should not bid when nobidask > 0"

    def test_seller_no_ask_when_nobidask_set(self):
        """Seller should not ask when nobidask > 0."""
        seller = Skeleton(
            player_id=2,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            seed=42,
        )

        seller.current_bid = 55
        seller.current_ask = 90
        seller.nobidask = 1  # Set: cannot ask

        ask = seller._request_ask()
        assert ask == 0, "Should not ask when nobidask > 0"


class TestSkeletonBuySellTrivialRule:
    """Test that buy_sell_response uses the TRIVIAL rule, not complex logic.

    Per spec Section 4:
    "If it holds the current bid and can make a positive profit at the current ask,
    accept; otherwise reject."

    This is a SIMPLE rule: accept if profitable, reject otherwise.
    Skeleton should NOT use time-based thresholds or target calculations.
    """

    def test_buyer_accepts_profitable_trade(self):
        """BUG 7: Buyer should accept any profitable trade (val > ask)."""
        buyer = Skeleton(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            seed=42,
        )

        # Set up: buyer is high bidder, ask is profitable
        buyer.current_bid = 75
        buyer.current_ask = 80  # val=100, ask=80, profit=20
        buyer.current_bidder = 1  # We are high bidder
        buyer.current_asker = 2
        buyer.current_time = 50
        buyer.last_time = 0  # Long time since last trade

        result = buyer.buy_sell_response()
        assert result is True, "Buyer should accept profitable trade (val=100 > ask=80)"

    def test_buyer_rejects_unprofitable_trade(self):
        """Buyer should reject trade where ask > valuation."""
        buyer = Skeleton(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            seed=42,
        )

        # Set up: ask is higher than valuation
        buyer.current_bid = 105
        buyer.current_ask = 110  # val=100, ask=110, loss
        buyer.current_bidder = 1
        buyer.current_asker = 2
        buyer.current_time = 50
        buyer.last_time = 0

        result = buyer.buy_sell_response()
        assert result is False, "Buyer should reject unprofitable trade (val=100 < ask=110)"

    def test_buyer_accepts_marginal_profit(self):
        """BUG 7: Buyer should accept even with small profit (trivial rule)."""
        buyer = Skeleton(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            seed=42,
        )

        # Set up: marginal profit of 1
        buyer.current_bid = 98
        buyer.current_ask = 99  # val=100, ask=99, profit=1
        buyer.current_bidder = 1
        buyer.current_asker = 2
        buyer.current_time = 5  # Early in period
        buyer.last_time = 4  # Just traded

        # With trivial rule: should accept (profit > 0)
        # With complex rule: might reject due to threshold
        result = buyer.buy_sell_response()
        assert result is True, "Buyer should accept any positive profit with trivial rule"

    def test_seller_accepts_profitable_trade(self):
        """BUG 7: Seller should accept any profitable trade (bid > cost)."""
        seller = Skeleton(
            player_id=2,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            seed=42,
        )

        # Set up: seller is low asker, bid is profitable
        seller.current_bid = 70  # cost=50, bid=70, profit=20
        seller.current_ask = 65
        seller.current_bidder = 1
        seller.current_asker = 2  # We are low asker
        seller.current_time = 50
        seller.last_time = 0

        result = seller.buy_sell_response()
        assert result is True, "Seller should accept profitable trade (bid=70 > cost=50)"

    def test_seller_rejects_unprofitable_trade(self):
        """Seller should reject trade where bid < cost."""
        seller = Skeleton(
            player_id=2,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            seed=42,
        )

        # Set up: bid is lower than cost
        seller.current_bid = 40  # cost=50, bid=40, loss
        seller.current_ask = 45
        seller.current_bidder = 1
        seller.current_asker = 2
        seller.current_time = 50
        seller.last_time = 0

        result = seller.buy_sell_response()
        assert result is False, "Seller should reject unprofitable trade (bid=40 < cost=50)"

    def test_seller_accepts_marginal_profit(self):
        """BUG 7: Seller should accept even with small profit (trivial rule)."""
        seller = Skeleton(
            player_id=2,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            seed=42,
        )

        # Set up: marginal profit of 1
        seller.current_bid = 51  # cost=50, bid=51, profit=1
        seller.current_ask = 52
        seller.current_bidder = 1
        seller.current_asker = 2
        seller.current_time = 5  # Early in period
        seller.last_time = 4  # Just traded

        # With trivial rule: should accept (profit > 0)
        # With complex rule: might reject due to threshold
        result = seller.buy_sell_response()
        assert result is True, "Seller should accept any positive profit with trivial rule"

    def test_buyer_time_independence(self):
        """Skeleton's accept decision should NOT depend on time since last trade."""
        buyer = Skeleton(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            seed=42,
        )

        # Profitable trade
        buyer.current_bid = 75
        buyer.current_ask = 85  # profit = 15
        buyer.current_bidder = 1
        buyer.current_asker = 2

        # Test with different time gaps
        results = []
        for last_time, current_time in [(0, 1), (0, 50), (49, 50)]:
            buyer.last_time = last_time
            buyer.current_time = current_time
            results.append(buyer.buy_sell_response())

        # With trivial rule: ALL should be True (same profit)
        # With complex rule: might vary based on alpha = 1/(t - lasttime)
        assert all(results), f"Accept decision should be time-independent, got {results}"


class TestSkeletonNoTokens:
    """Test behavior when all tokens are traded."""

    def test_buyer_no_bid_when_no_tokens_left(self):
        """Buyer should not bid when all tokens traded."""
        buyer = Skeleton(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            seed=42,
        )

        buyer.num_trades = 4  # All traded
        buyer.current_bid = 50
        buyer.current_ask = 85
        buyer.nobidask = 0

        bid = buyer._request_bid()
        assert bid == 0, "Should not bid when no tokens left"

    def test_seller_no_ask_when_no_tokens_left(self):
        """Seller should not ask when all tokens traded."""
        seller = Skeleton(
            player_id=2,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            seed=42,
        )

        seller.num_trades = 4  # All traded
        seller.current_bid = 55
        seller.current_ask = 90
        seller.nobidask = 0

        ask = seller._request_ask()
        assert ask == 0, "Should not ask when no tokens left"
