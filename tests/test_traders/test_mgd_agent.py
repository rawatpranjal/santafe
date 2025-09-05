# tests/test_traders/test_mgd_agent.py
import pytest
import numpy as np
from traders.mgd import MGDBuyer, MGDSeller


class TestMGDBuyerBasics:
    """Test basic MGD buyer functionality."""
    
    @pytest.fixture
    def mgd_buyer(self, sample_market_params, reset_random):
        """Create an MGD buyer for testing."""
        buyer = MGDBuyer(
            name="TestMGDBuyer",
            is_buyer=True,
            private_values=[100, 90, 80, 70],
            history_len=20,
            use_multi_unit=True
        )
        buyer.update_game_params(sample_market_params)
        return buyer
    
    def test_mgd_buyer_initialization(self, mgd_buyer):
        """Test MGD buyer is properly initialized."""
        assert mgd_buyer.name == "TestMGDBuyer"
        assert mgd_buyer.is_buyer is True
        assert mgd_buyer.strategy == "mgd"
        assert mgd_buyer.private_values == [100, 90, 80, 70]
        assert mgd_buyer.max_tokens == 4
        assert mgd_buyer.history_len == 20
        assert mgd_buyer.use_multi_unit is True
        assert len(mgd_buyer.market_ask_history) == 0
        assert mgd_buyer.prev_period_highest_trade_price is None
        assert mgd_buyer.prev_period_lowest_trade_price is None

    def test_mgd_buyer_can_trade(self, mgd_buyer):
        """Test can_trade logic."""
        assert mgd_buyer.can_trade() is True
        
        # Exhaust all tokens
        mgd_buyer.mytrades_period = 4
        mgd_buyer.tokens_left = 0
        assert mgd_buyer.can_trade() is False

    def test_update_history(self, mgd_buyer):
        """Test market ask history updates."""
        # Valid ask info
        ask_info = {'price': 85, 'agent': 'other'}
        mgd_buyer._update_history(ask_info)
        assert len(mgd_buyer.market_ask_history) == 1
        assert mgd_buyer.market_ask_history[0] == 85
        
        # Invalid ask info should be ignored
        mgd_buyer._update_history({'price': 'invalid'})
        assert len(mgd_buyer.market_ask_history) == 1  # No change
        
        mgd_buyer._update_history(None)
        assert len(mgd_buyer.market_ask_history) == 1  # No change


class TestMGDBelief:
    """Test MGD belief function and probability calculations."""
    
    @pytest.fixture
    def mgd_buyer_with_history(self, sample_market_params):
        """Create MGD buyer with some ask history."""
        buyer = MGDBuyer(
            name="TestMGD",
            is_buyer=True,
            private_values=[100, 90, 80, 70],
            history_len=10
        )
        buyer.update_game_params(sample_market_params)
        # Add some ask prices to history
        for price in [85, 90, 95, 80, 92]:
            buyer.market_ask_history.append(price)
        return buyer
    
    def test_basic_gd_belief(self, mgd_buyer_with_history):
        """Test basic GD belief calculation without MGD modifications."""
        buyer = mgd_buyer_with_history
        
        # Ask history: [85, 90, 95, 80, 92]
        # For bid=87: asks <= 87 are [85, 80] = 2/5 = 0.4
        prob = buyer._estimate_prob_accept_mgd(87)
        assert abs(prob - 0.4) < 0.01
        
        # For bid=100: all asks <= 100, so 5/5 = 1.0
        prob = buyer._estimate_prob_accept_mgd(100)
        assert abs(prob - 1.0) < 0.01

    def test_mgd_modification_high_bound(self, mgd_buyer_with_history):
        """Test MGD modification for bids above previous high."""
        buyer = mgd_buyer_with_history
        buyer.prev_period_highest_trade_price = 85
        
        # Bid above previous high should have prob=0
        prob = buyer._estimate_prob_accept_mgd(90)
        assert prob == 0.0
        
        # Bid at previous high should use GD calculation
        prob = buyer._estimate_prob_accept_mgd(85)
        assert prob > 0.0

    def test_mgd_modification_low_bound(self, mgd_buyer_with_history):
        """Test MGD modification for bids below previous low."""
        buyer = mgd_buyer_with_history
        buyer.prev_period_lowest_trade_price = 85
        
        # Bid below previous low should have prob=1
        prob = buyer._estimate_prob_accept_mgd(80)
        assert prob == 1.0
        
        # Bid at previous low should use GD calculation
        prob = buyer._estimate_prob_accept_mgd(85)
        assert prob > 0.0 and prob < 1.0

    def test_empty_history(self, sample_market_params):
        """Test behavior with no market history."""
        buyer = MGDBuyer("Test", True, [100], history_len=5)
        buyer.update_game_params(sample_market_params)
        
        prob = buyer._estimate_prob_accept_mgd(50)
        assert prob == 0.0


class TestMGDOptimization:
    """Test MGD bidding optimization logic."""
    
    @pytest.fixture
    def mgd_buyer_optimized(self, sample_market_params):
        """Create MGD buyer set up for optimization testing."""
        buyer = MGDBuyer(
            name="OptTest",
            is_buyer=True,
            private_values=[100, 90, 80, 70],
            history_len=10,
            use_multi_unit=True
        )
        buyer.update_game_params(sample_market_params)
        
        # Add consistent ask history for predictable optimization
        for price in [85, 87, 90, 83, 88]:
            buyer.market_ask_history.append(price)
        
        return buyer
    
    def test_single_unit_optimization(self, mgd_buyer_optimized):
        """Test single unit bid optimization."""
        buyer = mgd_buyer_optimized
        buyer.use_multi_unit = False
        
        bid = buyer._calculate_optimal_bid_single_unit()
        assert bid is not None
        assert isinstance(bid, int)
        assert buyer.min_price <= bid <= buyer.get_next_value_cost()

    def test_multi_unit_optimization(self, mgd_buyer_optimized):
        """Test multi-unit bid optimization.""" 
        buyer = mgd_buyer_optimized
        buyer.use_multi_unit = True
        
        bid = buyer._calculate_optimal_bid_multi_unit()
        assert bid is not None
        assert isinstance(bid, int)
        assert buyer.min_price <= bid <= buyer.max_price

    def test_no_profitable_bid(self, sample_market_params):
        """Test behavior when no profitable bid exists."""
        buyer = MGDBuyer("Test", True, [50], history_len=5)
        buyer.update_game_params(sample_market_params)
        
        # All asks are below buyer value - should find no profitable bid
        for price in [60, 65, 70]:
            buyer.market_ask_history.append(price)
        
        bid = buyer._calculate_optimal_bid_single_unit()
        assert bid is None

    def test_make_bid_or_ask(self, mgd_buyer_optimized, mock_market_history):
        """Test the main make_bid_or_ask method."""
        buyer = mgd_buyer_optimized
        
        current_bid_info = None
        current_ask_info = {'price': 88, 'agent': 'other'}
        
        bid = buyer.make_bid_or_ask(
            current_bid_info, current_ask_info,
            phibid=85, phiask=88,
            market_history=mock_market_history
        )
        
        # Should return a reasonable bid
        if bid is not None:
            assert isinstance(bid, int)
            assert buyer.min_price <= bid <= buyer.max_price
            assert bid <= buyer.get_next_value_cost()


class TestMGDSeller:
    """Test MGD seller functionality."""
    
    @pytest.fixture
    def mgd_seller(self, sample_market_params, reset_random):
        """Create an MGD seller for testing."""
        seller = MGDSeller(
            name="TestMGDSeller",
            is_buyer=False,
            private_values=[50, 60, 70, 80],
            history_len=15,
            use_multi_unit=True
        )
        seller.update_game_params(sample_market_params)
        return seller
    
    def test_mgd_seller_initialization(self, mgd_seller):
        """Test MGD seller is properly initialized."""
        assert mgd_seller.name == "TestMGDSeller"
        assert mgd_seller.is_buyer is False
        assert mgd_seller.strategy == "mgd"
        assert mgd_seller.private_values == [50, 60, 70, 80]
        assert mgd_seller.history_len == 15
        assert len(mgd_seller.market_bid_history) == 0

    def test_seller_belief_function(self, mgd_seller):
        """Test seller belief calculation."""
        # Add bid history: [60, 65, 70, 55, 68]
        for price in [60, 65, 70, 55, 68]:
            mgd_seller.market_bid_history.append(price)
        
        # For ask=62: bids >= 62 are [65, 70, 68] = 3/5 = 0.6
        prob = mgd_seller._estimate_prob_accept_mgd(62)
        assert abs(prob - 0.6) < 0.01

    def test_seller_mgd_modifications(self, mgd_seller):
        """Test seller-specific MGD modifications."""
        # Add some bid history
        for price in [60, 65, 70]:
            mgd_seller.market_bid_history.append(price)
        
        # Set previous period bounds
        mgd_seller.prev_period_lowest_trade_price = 62
        mgd_seller.prev_period_highest_trade_price = 68
        
        # Ask below prev low should have prob=0
        prob = mgd_seller._estimate_prob_accept_mgd(60)
        assert prob == 0.0
        
        # Ask above prev high should have prob=1
        prob = mgd_seller._estimate_prob_accept_mgd(70)
        assert prob == 1.0


class TestMGDPeriodUpdates:
    """Test MGD period-end updates and state management."""
    
    def test_update_end_of_period(self, sample_market_params):
        """Test period-end updates with trade prices."""
        buyer = MGDBuyer("Test", True, [100], history_len=5)
        buyer.update_game_params(sample_market_params)
        
        # Update with trade prices
        trade_prices = [75, 80, 85, 78]
        buyer.update_end_of_period(trade_prices)
        
        assert buyer.prev_period_lowest_trade_price == 75
        assert buyer.prev_period_highest_trade_price == 85

    def test_update_end_of_period_no_trades(self, sample_market_params):
        """Test period-end updates with no trades."""
        buyer = MGDBuyer("Test", True, [100], history_len=5)
        buyer.update_game_params(sample_market_params)
        
        # Set initial bounds
        buyer.prev_period_lowest_trade_price = 80
        buyer.prev_period_highest_trade_price = 90
        
        # Update with empty trade list
        buyer.update_end_of_period([])
        
        # Should keep existing bounds
        assert buyer.prev_period_lowest_trade_price == 80
        assert buyer.prev_period_highest_trade_price == 90

    def test_reset_learning_state(self, sample_market_params):
        """Test learning state reset for new rounds."""
        buyer = MGDBuyer("Test", True, [100], history_len=5)
        buyer.update_game_params(sample_market_params)
        
        # Set up some state
        buyer.market_ask_history.append(85)
        buyer.prev_period_lowest_trade_price = 80
        buyer.prev_period_highest_trade_price = 90
        
        # Reset for new round
        buyer._reset_learning_state()
        
        assert len(buyer.market_ask_history) == 0
        assert buyer.prev_period_lowest_trade_price is None
        assert buyer.prev_period_highest_trade_price is None


class TestMGDAcceptance:
    """Test MGD trader acceptance logic."""
    
    def test_buyer_request_buy(self, sample_market_params, mock_market_history):
        """Test MGD buyer acceptance logic."""
        buyer = MGDBuyer("Test", True, [100, 90], history_len=5)
        buyer.update_game_params(sample_market_params)
        
        current_bid_info = {'price': 85, 'agent': buyer}
        current_ask_info = {'price': 80, 'agent': 'other'}
        
        # Should accept profitable offer
        accept = buyer.request_buy(80, current_bid_info, current_ask_info, 85, 80, mock_market_history)
        assert accept is True
        
        # Should reject unprofitable offer
        accept = buyer.request_buy(105, current_bid_info, current_ask_info, 85, 80, mock_market_history)
        assert accept is False

    def test_seller_request_sell(self, sample_market_params, mock_market_history):
        """Test MGD seller acceptance logic."""
        seller = MGDSeller("Test", False, [50, 60], history_len=5)
        seller.update_game_params(sample_market_params)
        
        current_bid_info = {'price': 65, 'agent': 'other'}
        current_ask_info = {'price': 70, 'agent': seller}
        
        # Should accept profitable bid
        accept = seller.request_sell(65, current_bid_info, current_ask_info, 65, 70, mock_market_history)
        assert accept is True
        
        # Should reject unprofitable bid
        accept = seller.request_sell(45, current_bid_info, current_ask_info, 65, 70, mock_market_history)
        assert accept is False