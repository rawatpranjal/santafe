# tests/unit/traders/test_ppo_agent.py
"""
Unit tests for the PPO (Proximal Policy Optimization) Agent.

Tests cover:
- Model loading (MaskablePPO and regular PPO)
- Action masking with 24-action discrete space
- Action-to-price mapping for buyers and sellers
- Orderbook dependency handling
- Rationality constraints (never trade at a loss)
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from engine.orderbook import OrderBook
from traders.rl.ppo_agent import PPOAgent

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_model():
    """Create a mock PPO model."""
    model = Mock()
    # Default to pass action (0)
    model.predict.return_value = (np.array([0]), None)
    return model


@pytest.fixture
def orderbook():
    """Create an orderbook for testing."""
    ob = OrderBook(
        num_buyers=4, num_sellers=4, num_times=100, min_price=1, max_price=200, rng_seed=42
    )
    # Set up some initial state
    # bids array is (num_buyers+1, num_times+1), buyers are 1..4
    # asks array is (num_sellers+1, num_times+1), sellers are 1..4
    ob.high_bid[5] = 50
    ob.low_ask[5] = 60
    ob.bids[1, 5] = 50  # buyer 1 bid at time 5
    ob.asks[1, 5] = 60  # seller 1 ask at time 5
    return ob


@pytest.fixture
def buyer_valuations():
    return [100, 90, 80, 70]


@pytest.fixture
def seller_valuations():
    return [30, 40, 50, 60]


# =============================================================================
# Test: Model Loading
# =============================================================================


class TestModelLoading:
    """Tests for model loading and fallback behavior."""

    def test_loads_maskable_ppo_first(self, buyer_valuations):
        """Should try MaskablePPO first."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_model = Mock()
            mock_maskable.load.return_value = mock_model

            agent = PPOAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_valuations,
                model_path="fake_path.zip",
            )

            mock_maskable.load.assert_called_once_with("fake_path.zip")
            assert agent.use_action_masking == True
            assert agent.model is mock_model

    def test_falls_back_to_regular_ppo(self, buyer_valuations):
        """Should fall back to regular PPO if MaskablePPO fails."""
        with (
            patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable,
            patch("traders.rl.ppo_agent.PPO") as mock_ppo,
        ):
            mock_maskable.load.side_effect = ValueError("Not a MaskablePPO model")
            mock_model = Mock()
            mock_ppo.load.return_value = mock_model

            agent = PPOAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_valuations,
                model_path="regular_ppo.zip",
            )

            mock_ppo.load.assert_called_once_with("regular_ppo.zip")
            assert agent.use_action_masking == False
            assert agent.model is mock_model


# =============================================================================
# Test: Orderbook Dependency
# =============================================================================


class TestOrderbookDependency:
    """Tests for orderbook handling."""

    def test_set_orderbook(self, buyer_valuations, orderbook):
        """Should store orderbook reference."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_valuations,
                model_path="fake.zip",
            )

            assert agent.orderbook is None
            agent.set_orderbook(orderbook)
            assert agent.orderbook is orderbook

    def test_bid_ask_response_returns_pass_without_orderbook(self, buyer_valuations):
        """Without orderbook, should return pass (-99)."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_valuations,
                model_path="fake.zip",
            )

            agent.bid_ask(time=1, nobidask=0)
            response = agent.bid_ask_response()
            assert response == -99


# =============================================================================
# Test: Action Space (24 Actions)
# =============================================================================


class TestActionSpace:
    """Tests for the 24-action discrete space."""

    def test_pass_action_always_valid(self, buyer_valuations, orderbook):
        """Action 0 (pass) should always be valid."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_valuations,
                model_path="fake.zip",
            )
            agent.set_orderbook(orderbook)
            agent.current_step = 5

            mask = agent._get_action_mask()
            assert mask[0] == True  # Pass always valid

    def test_cannot_trade_masks_all_but_pass(self, buyer_valuations, orderbook):
        """When cannot trade, only pass should be valid."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_valuations,
                model_path="fake.zip",
            )
            agent.set_orderbook(orderbook)
            agent.current_step = 5

            # Trade all tokens
            agent.num_trades = 4

            mask = agent._get_action_mask()
            assert mask[0] == True  # Pass valid
            assert not any(mask[1:])  # All others invalid


# =============================================================================
# Test: Action Masking - Buyer
# =============================================================================


class TestBuyerActionMasking:
    """Tests for buyer-specific action masking."""

    def test_accept_masked_when_no_ask(self, buyer_valuations, orderbook):
        """Accept (action 1) should be masked when no ask exists."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_valuations,
                model_path="fake.zip",
            )
            # Clear ask
            orderbook.low_ask[5] = 0
            agent.set_orderbook(orderbook)
            agent.current_step = 6  # Will look at t=5

            mask = agent._get_action_mask()
            assert mask[1] == False  # Accept invalid

    def test_accept_masked_when_ask_above_valuation(self, buyer_valuations, orderbook):
        """Accept should be masked when ask > valuation."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=[50, 40, 30, 20],  # Low valuations
                model_path="fake.zip",
            )
            orderbook.low_ask[5] = 100  # Ask above all valuations
            agent.set_orderbook(orderbook)
            agent.current_step = 6

            mask = agent._get_action_mask()
            assert mask[1] == False  # Accept invalid

    def test_accept_valid_when_profitable(self, buyer_valuations, orderbook):
        """Accept should be valid when ask <= valuation."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_valuations,  # First valuation = 100
                model_path="fake.zip",
            )
            orderbook.low_ask[5] = 60  # Ask below valuation
            agent.set_orderbook(orderbook)
            agent.current_step = 6

            mask = agent._get_action_mask()
            assert mask[1] == True  # Accept valid

    def test_spread_improvements_masked_when_no_spread(self, buyer_valuations, orderbook):
        """Spread improvement actions (2-9) should be masked with no spread."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_valuations,
                model_path="fake.zip",
            )
            # Set bid >= ask (no positive spread)
            orderbook.high_bid[5] = 60
            orderbook.low_ask[5] = 50
            agent.set_orderbook(orderbook)
            agent.current_step = 6

            mask = agent._get_action_mask()
            # Actions 2-9 should be masked
            assert not any(mask[2:10])

    def test_truthful_always_valid_for_buyer(self, buyer_valuations, orderbook):
        """Truthful (action 18) should be valid when can trade."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_valuations,
                model_path="fake.zip",
            )
            agent.set_orderbook(orderbook)
            agent.current_step = 6

            mask = agent._get_action_mask()
            assert mask[18] == True  # Truthful valid

    def test_snipe_masked_when_spread_too_wide(self, buyer_valuations, orderbook):
        """Snipe (action 20) should be masked when spread >= 5%."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_valuations,
                model_path="fake.zip",
            )
            # Wide spread: 50 to 100 = 50% spread
            orderbook.high_bid[5] = 50
            orderbook.low_ask[5] = 100
            agent.set_orderbook(orderbook)
            agent.current_step = 6

            mask = agent._get_action_mask()
            assert mask[20] == False  # Snipe invalid


# =============================================================================
# Test: Action Masking - Seller
# =============================================================================


class TestSellerActionMasking:
    """Tests for seller-specific action masking."""

    def test_accept_masked_when_no_bid(self, seller_valuations, orderbook):
        """Accept (action 1) should be masked when no bid exists."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=5,
                is_buyer=False,
                num_tokens=4,
                valuations=seller_valuations,
                model_path="fake.zip",
            )
            orderbook.high_bid[5] = 0  # No bid
            agent.set_orderbook(orderbook)
            agent.current_step = 6

            mask = agent._get_action_mask()
            assert mask[1] == False  # Accept invalid

    def test_accept_masked_when_bid_below_cost(self, seller_valuations, orderbook):
        """Accept should be masked when bid < cost."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=5,
                is_buyer=False,
                num_tokens=4,
                valuations=[50, 60, 70, 80],  # Costs
                model_path="fake.zip",
            )
            orderbook.high_bid[5] = 40  # Below all costs
            agent.set_orderbook(orderbook)
            agent.current_step = 6

            mask = agent._get_action_mask()
            assert mask[1] == False  # Accept invalid

    def test_accept_valid_when_profitable_for_seller(self, seller_valuations, orderbook):
        """Accept should be valid when bid >= cost."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=5,
                is_buyer=False,
                num_tokens=4,
                valuations=seller_valuations,  # First cost = 30
                model_path="fake.zip",
            )
            orderbook.high_bid[5] = 50  # Above first cost
            agent.set_orderbook(orderbook)
            agent.current_step = 6

            mask = agent._get_action_mask()
            assert mask[1] == True  # Accept valid


# =============================================================================
# Test: Action to Price Mapping - Buyer
# =============================================================================


class TestBuyerPriceMapping:
    """Tests for buyer action-to-price mapping."""

    def test_pass_returns_negative_99(self, buyer_valuations, orderbook):
        """Pass action should return -99."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_valuations,
                model_path="fake.zip",
            )
            agent.set_orderbook(orderbook)
            agent.current_step = 6

            price = agent._map_action_to_price(0)
            assert price == -99

    def test_accept_returns_best_ask(self, buyer_valuations, orderbook):
        """Accept action should return best ask price."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_valuations,
                model_path="fake.zip",
            )
            orderbook.low_ask[5] = 75
            agent.set_orderbook(orderbook)
            agent.current_step = 6

            price = agent._map_action_to_price(1)
            assert price == 75

    def test_truthful_returns_valuation(self, buyer_valuations, orderbook):
        """Truthful action should return current valuation."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_valuations,
                model_path="fake.zip",
            )
            agent.set_orderbook(orderbook)
            agent.current_step = 6

            price = agent._map_action_to_price(18)
            assert price == 100  # First valuation

    def test_buyer_price_capped_at_valuation(self, buyer_valuations, orderbook):
        """Buyer prices should never exceed valuation."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=[50, 40, 30, 20],  # Low valuations
                model_path="fake.zip",
            )
            # Set high ask that would be accepted
            orderbook.low_ask[5] = 100
            agent.set_orderbook(orderbook)
            agent.current_step = 6

            # Accept would try to bid at 100, but should be capped at 50
            price = agent._map_action_to_price(1)
            # Accept returns best_ask directly, but the mask should prevent this
            # The cap happens for other actions
            # Let's test jump best
            orderbook.high_bid[5] = 100
            price = agent._map_action_to_price(19)  # Jump best = bid + 1 = 101, capped to 50
            assert price <= 50


# =============================================================================
# Test: Action to Price Mapping - Seller
# =============================================================================


class TestSellerPriceMapping:
    """Tests for seller action-to-price mapping."""

    def test_accept_returns_best_bid(self, seller_valuations, orderbook):
        """Accept action should return best bid price."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=5,
                is_buyer=False,
                num_tokens=4,
                valuations=seller_valuations,
                model_path="fake.zip",
            )
            orderbook.high_bid[5] = 55
            agent.set_orderbook(orderbook)
            agent.current_step = 6

            price = agent._map_action_to_price(1)
            assert price == 55

    def test_truthful_returns_cost_for_seller(self, seller_valuations, orderbook):
        """Truthful action should return current cost for seller."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=5,
                is_buyer=False,
                num_tokens=4,
                valuations=seller_valuations,
                model_path="fake.zip",
            )
            agent.set_orderbook(orderbook)
            agent.current_step = 6

            price = agent._map_action_to_price(18)
            assert price == 30  # First cost

    def test_seller_price_floored_at_cost(self, seller_valuations, orderbook):
        """Seller prices should never go below cost."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=5,
                is_buyer=False,
                num_tokens=4,
                valuations=[80, 90, 100, 110],  # High costs
                model_path="fake.zip",
            )
            # Set low bid
            orderbook.high_bid[5] = 50
            agent.set_orderbook(orderbook)
            agent.current_step = 6

            # Jump best for seller = ask - 1
            orderbook.low_ask[5] = 60
            price = agent._map_action_to_price(19)  # Jump best = 60 - 1 = 59, floored to 80
            assert price >= 80


# =============================================================================
# Test: Buy/Sell Response (Rationality)
# =============================================================================


class TestBuySellResponse:
    """Tests for buy/sell response rationality."""

    def test_buyer_accepts_profitable_trade(self, buyer_valuations, orderbook):
        """Buyer should accept when ask <= valuation."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_valuations,
                model_path="fake.zip",
            )
            orderbook.low_ask[5] = 80  # Below first valuation of 100
            agent.set_orderbook(orderbook)
            agent.current_step = 5

            agent.buy_sell(time=5, nobuysell=0, high_bid=50, low_ask=80, high_bidder=1, low_asker=5)
            result = agent.buy_sell_response()
            assert result == True

    def test_buyer_rejects_unprofitable_trade(self, buyer_valuations, orderbook):
        """Buyer should reject when ask > valuation."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_valuations,
                model_path="fake.zip",
            )
            orderbook.low_ask[5] = 150  # Above first valuation of 100
            agent.set_orderbook(orderbook)
            agent.current_step = 5

            agent.buy_sell(
                time=5, nobuysell=0, high_bid=50, low_ask=150, high_bidder=1, low_asker=5
            )
            result = agent.buy_sell_response()
            assert result == False

    def test_seller_accepts_profitable_trade(self, seller_valuations, orderbook):
        """Seller should accept when bid >= cost."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=5,
                is_buyer=False,
                num_tokens=4,
                valuations=seller_valuations,
                model_path="fake.zip",
            )
            orderbook.high_bid[5] = 50  # Above first cost of 30
            agent.set_orderbook(orderbook)
            agent.current_step = 5

            agent.buy_sell(time=5, nobuysell=0, high_bid=50, low_ask=80, high_bidder=1, low_asker=5)
            result = agent.buy_sell_response()
            assert result == True

    def test_seller_rejects_unprofitable_trade(self, seller_valuations, orderbook):
        """Seller should reject when bid < cost."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=5,
                is_buyer=False,
                num_tokens=4,
                valuations=seller_valuations,
                model_path="fake.zip",
            )
            orderbook.high_bid[5] = 20  # Below first cost of 30
            agent.set_orderbook(orderbook)
            agent.current_step = 5

            agent.buy_sell(time=5, nobuysell=0, high_bid=20, low_ask=80, high_bidder=1, low_asker=5)
            result = agent.buy_sell_response()
            assert result == False


# =============================================================================
# Test: Period Reset
# =============================================================================


class TestPeriodReset:
    """Tests for period start/reset behavior."""

    def test_start_period_resets_step_counter(self, buyer_valuations, orderbook):
        """start_period should reset current_step."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_valuations,
                model_path="fake.zip",
            )
            agent.current_step = 50
            agent._steps_since_last_trade = 10

            agent.start_period(period_number=2)

            assert agent.current_step == 0
            assert agent._steps_since_last_trade == 0

    def test_start_period_resets_observation_generator(self, buyer_valuations):
        """start_period should reset observation generator."""
        with patch("traders.rl.ppo_agent.MaskablePPO") as mock_maskable:
            mock_maskable.load.return_value = Mock()
            agent = PPOAgent(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=buyer_valuations,
                model_path="fake.zip",
            )
            # Add some history
            agent.obs_gen.trade_prices.append(100)
            agent.obs_gen.total_volume = 5

            agent.start_period(period_number=2)

            assert len(agent.obs_gen.trade_prices) == 0
            assert agent.obs_gen.total_volume == 0


# =============================================================================
# Test: Integration with Real Checkpoint (Optional)
# =============================================================================


class TestRealCheckpoint:
    """Integration tests with real checkpoints (skipped if unavailable or obs space mismatch)."""

    @pytest.fixture
    def checkpoint_path(self):
        """Get a real checkpoint path if available."""
        import os

        path = "/Users/pranjal/Code/santafe-1/checkpoints/ppo_vs_zic/best_model/best_model.zip"
        if os.path.exists(path):
            return path
        pytest.skip("No checkpoint available for integration test")

    @pytest.mark.skip(
        reason="Model was trained with 24-dim obs, current EnhancedObservationGenerator produces 42-dim"
    )
    def test_load_real_checkpoint(self, checkpoint_path, buyer_valuations, orderbook):
        """Should load and use a real checkpoint."""
        agent = PPOAgent(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=buyer_valuations,
            model_path=checkpoint_path,
            max_price=200,
            max_steps=100,
        )
        agent.set_orderbook(orderbook)

        # Should be able to generate a response
        agent.bid_ask(time=6, nobidask=0)
        price = agent.bid_ask_response()

        # Should return a valid price or pass
        assert price == -99 or (0 <= price <= 200)

    @pytest.mark.skip(
        reason="Model was trained with 24-dim obs, current EnhancedObservationGenerator produces 42-dim"
    )
    def test_real_checkpoint_action_is_rational(self, checkpoint_path, buyer_valuations, orderbook):
        """Real checkpoint should produce rational actions."""
        agent = PPOAgent(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=buyer_valuations,
            model_path=checkpoint_path,
            max_price=200,
            max_steps=100,
        )
        agent.set_orderbook(orderbook)

        # Set up market state
        orderbook.high_bid[5] = 50
        orderbook.low_ask[5] = 80

        agent.bid_ask(time=6, nobidask=0)
        price = agent.bid_ask_response()

        # If bidding, should not exceed valuation
        if price > 0:
            assert (
                price <= buyer_valuations[0]
            ), f"Buyer bid {price} exceeds valuation {buyer_valuations[0]}"
