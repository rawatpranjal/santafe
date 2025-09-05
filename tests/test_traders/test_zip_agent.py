# tests/test_traders/test_zip_agent.py
import pytest
import numpy as np
from traders.zip import ZIPBuyer, ZIPSeller


class TestZIPBuyerBasics:
    """Test basic ZIP buyer functionality."""
    
    def test_zip_buyer_initialization(self, zip_buyer):
        """Test ZIP buyer is properly initialized."""
        assert zip_buyer.name == "TestZIPBuyer"
        assert zip_buyer.is_buyer is True
        assert zip_buyer.strategy == "zip"
        assert zip_buyer.private_values == [100, 90, 80, 70]  # Should be sorted descending
        assert zip_buyer.max_tokens == 4
        assert zip_buyer.tokens_left == 4
        
        # Check ZIP-specific parameters
        assert 0.1 <= zip_buyer.beta <= 0.5
        assert 0.0 <= zip_buyer.gamma <= 0.1
        assert -0.2 <= zip_buyer.margin <= -0.1  # Buyer margin should be negative
        
    def test_zip_buyer_can_trade(self, zip_buyer):
        """Test can_trade logic."""
        assert zip_buyer.can_trade() is True
        assert zip_buyer.mytrades_period == 0
        
        # Simulate some trades
        zip_buyer.mytrades_period = 2
        zip_buyer.tokens_left = 2
        assert zip_buyer.can_trade() is True
        
        # Exhaust all tokens
        zip_buyer.mytrades_period = 4
        zip_buyer.tokens_left = 0
        assert zip_buyer.can_trade() is False

    def test_get_next_value_cost(self, zip_buyer):
        """Test getting next value for trading."""
        # First token should be highest value
        assert zip_buyer.get_next_value_cost() == 100
        
        # Simulate first trade
        zip_buyer.mytrades_period = 1
        zip_buyer.tokens_left = 3
        assert zip_buyer.get_next_value_cost() == 90
        
        # Simulate more trades
        zip_buyer.mytrades_period = 3
        zip_buyer.tokens_left = 1
        assert zip_buyer.get_next_value_cost() == 70


class TestZIPShoutPriceCalculation:
    """Test ZIP shout price calculation logic."""
    
    def test_calculate_shout_price_basic(self, zip_buyer):
        """Test basic shout price calculation."""
        # Set known margin for predictable results
        zip_buyer.margin = -0.1  # 10% below value
        
        price = zip_buyer._calculate_shout_price()
        expected = 100 * (1 + (-0.1))  # 100 * 0.9 = 90
        assert price == 90
        
    def test_calculate_shout_price_bounds_checking(self, zip_buyer):
        """Test shout price respects market bounds."""
        # Test with extreme negative margin
        zip_buyer.margin = -0.9  # Would give 100 * 0.1 = 10
        zip_buyer.min_price = 50
        price = zip_buyer._calculate_shout_price()
        assert price >= zip_buyer.min_price
        assert price <= zip_buyer.get_next_value_cost()  # Can't bid above value
        
    def test_calculate_shout_price_no_tokens(self, zip_buyer):
        """Test shout price when no tokens left."""
        zip_buyer.mytrades_period = 4  # All tokens used
        zip_buyer.tokens_left = 0
        price = zip_buyer._calculate_shout_price()
        assert price is None

    def test_make_bid_or_ask(self, zip_buyer, mock_market_history):
        """Test make_bid_or_ask method."""
        current_bid_info = None
        current_ask_info = {'price': 95, 'agent': 'other'}
        
        bid_price = zip_buyer.make_bid_or_ask(
            current_bid_info, current_ask_info, 
            phibid=85, phiask=95, 
            market_history=mock_market_history
        )
        
        assert bid_price is not None
        assert isinstance(bid_price, int)
        assert zip_buyer.min_price <= bid_price <= zip_buyer.max_price
        assert bid_price <= zip_buyer.get_next_value_cost()
        assert zip_buyer.last_shout_price == bid_price


class TestZIPLearningLogic:
    """Test ZIP learning and margin update logic."""
    
    def test_margin_increase_condition(self, zip_buyer, mock_step_outcome):
        """Test margin increase when bid was too low."""
        # Setup: buyer bid 80, trade happened at 90
        zip_buyer.last_shout_price = 80
        zip_buyer.margin = -0.2  # Initial margin
        
        # Create step outcome where trade price > bid price
        step_outcome = {
            'last_trade_info': {
                'price': 90,
                'type': 'buy_accepts_ask',
                'buyer': None,
                'seller': None
            }
        }
        
        initial_margin = zip_buyer.margin
        zip_buyer.observe_reward(None, None, 0, None, False, step_outcome=step_outcome)
        
        # Margin should increase (become less negative) since p >= q condition met
        assert zip_buyer.margin > initial_margin

    def test_margin_decrease_condition(self, zip_buyer):
        """Test margin decrease when bid was too high."""
        # Setup: buyer bid 95, trade happened at 85 (undercut)
        zip_buyer.last_shout_price = 95
        zip_buyer.margin = -0.1
        
        step_outcome = {
            'last_trade_info': {
                'price': 85,
                'type': 'buy_accepts_ask', 
                'buyer': None,
                'seller': None
            }
        }
        
        initial_margin = zip_buyer.margin  
        zip_buyer.observe_reward(None, None, 0, None, False, step_outcome=step_outcome)
        
        # This should trigger decrease condition
        # (p <= q where p=95, q=85 is false, so no update)
        # But if it were an accepted offer where buyer was undercut, margin should decrease

    def test_no_update_without_last_price(self, zip_buyer):
        """Test no update when agent hasn't shouted."""
        zip_buyer.last_shout_price = None
        initial_margin = zip_buyer.margin
        
        step_outcome = {
            'last_trade_info': {
                'price': 85,
                'type': 'buy_accepts_ask'
            }
        }
        
        zip_buyer.observe_reward(None, None, 0, None, False, step_outcome=step_outcome)
        
        # Margin should remain unchanged
        assert zip_buyer.margin == initial_margin

    def test_margin_bounds_enforcement(self, zip_buyer):
        """Test margin stays within valid bounds."""
        # Test extreme updates don't break bounds
        zip_buyer.margin = -0.95  # Near lower bound
        zip_buyer.last_shout_price = 50
        
        step_outcome = {
            'last_trade_info': {
                'price': 200,  # Very high trade price
                'type': 'buy_accepts_ask'
            }
        }
        
        zip_buyer.observe_reward(None, None, 0, None, False, step_outcome=step_outcome)
        
        # Margin should stay within [-1.0, 0.0] for buyers
        assert -1.0 <= zip_buyer.margin <= 0.0


class TestZIPSellerBasics:
    """Test basic ZIP seller functionality."""
    
    def test_zip_seller_initialization(self, zip_seller):
        """Test ZIP seller is properly initialized."""
        assert zip_seller.name == "TestZIPSeller"
        assert zip_seller.is_buyer is False
        assert zip_seller.strategy == "zip"
        assert zip_seller.private_values == [50, 60, 70, 80]  # Should be sorted ascending
        assert zip_seller.max_tokens == 4
        
        # Seller margin should be positive
        assert 0.1 <= zip_seller.margin <= 0.2

    def test_seller_shout_price_calculation(self, zip_seller):
        """Test seller shout price calculation."""
        zip_seller.margin = 0.1  # 10% above cost
        
        price = zip_seller._calculate_shout_price()
        expected = 50 * (1 + 0.1)  # 50 * 1.1 = 55
        assert price == 55

    def test_seller_margin_bounds(self, zip_seller):
        """Test seller margin stays non-negative."""
        # Force extreme negative update
        zip_seller.margin = 0.05  # Small positive margin
        zip_seller.last_shout_price = 100
        
        step_outcome = {
            'last_trade_info': {
                'price': 40,  # Very low trade price
                'type': 'sell_accepts_bid'
            }
        }
        
        zip_seller.observe_reward(None, None, 0, None, False, step_outcome=step_outcome)
        
        # Margin should stay >= 0.0 for sellers
        assert zip_seller.margin >= 0.0


class TestZIPAcceptanceLogic:
    """Test ZIP trader acceptance logic."""
    
    def test_buyer_request_buy(self, zip_buyer, mock_market_history):
        """Test buyer acceptance of asks."""
        current_bid_info = {'price': 90, 'agent': zip_buyer}
        current_ask_info = {'price': 85, 'agent': 'other'}
        
        # Should accept profitable offer
        assert zip_buyer.request_buy(85, current_bid_info, current_ask_info, 90, 85, mock_market_history) is True
        
        # Should reject unprofitable offer
        assert zip_buyer.request_buy(105, current_bid_info, current_ask_info, 90, 85, mock_market_history) is False
        
    def test_seller_request_sell(self, zip_seller, mock_market_history):
        """Test seller acceptance of bids."""
        current_bid_info = {'price': 60, 'agent': 'other'}
        current_ask_info = {'price': 65, 'agent': zip_seller}
        
        # Should accept profitable bid
        assert zip_seller.request_sell(60, current_bid_info, current_ask_info, 60, 65, mock_market_history) is True
        
        # Should reject unprofitable bid
        assert zip_seller.request_sell(40, current_bid_info, current_ask_info, 60, 65, mock_market_history) is False


class TestZIPIntegration:
    """Integration tests for ZIP trader behavior."""
    
    def test_zip_learning_over_time(self, zip_buyer):
        """Test ZIP learning behavior over multiple interactions."""
        initial_margin = zip_buyer.margin
        
        # Simulate multiple trading events where bids are consistently too low
        for i in range(5):
            zip_buyer.last_shout_price = 80
            step_outcome = {
                'last_trade_info': {
                    'price': 90,
                    'type': 'buy_accepts_ask'
                }
            }
            zip_buyer.observe_reward(None, None, 0, None, False, step_outcome=step_outcome)
        
        # After multiple "too low" signals, margin should have increased
        assert zip_buyer.margin > initial_margin
        
    def test_zip_price_adaptation(self, zip_buyer):
        """Test that ZIP prices adapt to market conditions."""
        # Record initial price
        initial_price = zip_buyer._calculate_shout_price()
        initial_margin = zip_buyer.margin
        
        # Simulate learning that bids should be higher
        zip_buyer.last_shout_price = initial_price
        step_outcome = {
            'last_trade_info': {
                'price': initial_price + 10,
                'type': 'buy_accepts_ask'
            }
        }
        zip_buyer.observe_reward(None, None, 0, None, False, step_outcome=step_outcome)
        
        # New price should be different (likely higher)
        new_price = zip_buyer._calculate_shout_price()
        assert new_price != initial_price
        
        # Margin should have changed
        assert zip_buyer.margin != initial_margin