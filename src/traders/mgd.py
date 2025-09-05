# traders/mgd.py
import logging
import numpy as np
from collections import deque
from .base import BaseTrader


class MGDBuyer(BaseTrader):
    """
    Modified Gjerstad-Dickhaut (MGD) Buyer Strategy:
    Based on Tesauro & Das (2001) improvements to the original GD strategy.
    
    Key modifications:
    1. Uses persistent memory of previous period's trade prices
    2. Modified belief function with price bounds from previous period
    3. Multi-unit bidding optimization (optional)
    
    References:
    - Tesauro, G. and Das, R. (2001). High-performance bidding agents for the continuous double auction. 
    """
    def __init__(self, name, is_buyer, private_values, history_len=50, use_multi_unit=True, **kwargs):
        super().__init__(name, True, private_values, strategy="mgd")
        self.logger = logging.getLogger(f'trader.{self.name}')
        
        if history_len <= 0:
            history_len = 1
        self.history_len = history_len
        self.use_multi_unit = use_multi_unit
        
        # Store recent market observations (ask prices relevant for buyer)
        self.market_ask_history = deque(maxlen=self.history_len)
        
        # Previous period price bounds for MGD modification
        self.prev_period_highest_trade_price = None
        self.prev_period_lowest_trade_price = None
        
        self.logger.debug(f"Initialized MGD Buyer: history_len={self.history_len}, multi_unit={self.use_multi_unit}")

    def _reset_learning_state(self):
        """Clear history at the start of a new ROUND."""
        self.market_ask_history.clear()
        self.prev_period_highest_trade_price = None
        self.prev_period_lowest_trade_price = None
        self.logger.debug("MGD history cleared for new round.")

    def update_end_of_period(self, period_trade_prices):
        """Update previous period price bounds from completed period."""
        super().update_end_of_period(period_trade_prices)
        
        if period_trade_prices:
            self.prev_period_highest_trade_price = max(period_trade_prices)
            self.prev_period_lowest_trade_price = min(period_trade_prices)
            self.logger.debug(f"MGD updated prev period bounds: [{self.prev_period_lowest_trade_price:.1f}, {self.prev_period_highest_trade_price:.1f}]")
        else:
            # No trades in previous period - keep existing bounds or use None
            self.logger.debug("MGD: No trades in previous period, keeping existing bounds")

    def _update_history(self, current_ask_info):
        """Add current best ask to history if it exists."""
        if current_ask_info and isinstance(current_ask_info.get('price'), (int, float)):
            price = current_ask_info['price']
            if self.min_price <= price <= self.max_price:
                self.market_ask_history.append(price)

    def _estimate_prob_accept_mgd(self, bid_price):
        """
        Modified belief function from Tesauro & Das (2001).
        
        First calculates basic GD belief, then applies MGD modifications:
        - Reset f(p) = 0 for prices above previous period's highest trade price
        - Reset f(p) = 1 for prices below previous period's lowest trade price
        """
        if not self.market_ask_history or bid_price < self.min_price:
            return 0.0

        # Step 1: Calculate basic GD belief function
        relevant_asks = np.array(list(self.market_ask_history))
        if len(relevant_asks) == 0:
            return 0.0

        # Basic GD: probability = fraction of historical asks <= current potential bid
        accepting_asks_count = np.sum(relevant_asks <= bid_price)
        base_probability = accepting_asks_count / len(relevant_asks)
        
        # Step 2: Apply MGD modifications using previous period price bounds
        modified_probability = base_probability
        
        if self.prev_period_highest_trade_price is not None:
            # Reset f(p) = 1 for bids above previous period's highest trade price
            # High bids above what traded before will definitely be accepted by sellers
            if bid_price > self.prev_period_highest_trade_price:
                modified_probability = 1.0
                self.logger.debug(f"MGD: Bid {bid_price} > prev_high {self.prev_period_highest_trade_price:.1f}, setting prob=1")
                
        if self.prev_period_lowest_trade_price is not None:
            # Reset f(p) = 0 for bids below previous period's lowest trade price  
            # Low bids below what traded before will not be accepted by sellers
            if bid_price < self.prev_period_lowest_trade_price:
                modified_probability = 0.0
                self.logger.debug(f"MGD: Bid {bid_price} < prev_low {self.prev_period_lowest_trade_price:.1f}, setting prob=0")
        
        return modified_probability

    def _calculate_expected_profit_single(self, bid_price, value):
        """Calculate expected profit for a single unit at given bid price."""
        prob_accept = self._estimate_prob_accept_mgd(bid_price)
        expected_profit = prob_accept * (value - bid_price)
        return expected_profit

    def _calculate_optimal_bid_multi_unit(self):
        """
        Multi-unit bidding optimization.
        Considers expected profit for quotes based on all remaining tokens.
        Returns the bid that maximizes expected total profit.
        """
        if not self.can_trade():
            return None
            
        # Get all remaining values
        remaining_values = []
        for i in range(self.mytrades_period, self.max_tokens):
            if i < len(self.private_values):
                remaining_values.append(self.private_values[i])
        
        if not remaining_values:
            return None
            
        best_expected_profit = -1e-9
        best_bid = None
        
        # Consider different bid prices and their impact on all remaining units
        for potential_bid in range(self.min_price, max(remaining_values) + 1):
            total_expected_profit = 0.0
            
            # Calculate expected profit considering all remaining units
            for value in remaining_values:
                if potential_bid <= value:  # Only profitable trades
                    unit_expected_profit = self._calculate_expected_profit_single(potential_bid, value)
                    total_expected_profit += unit_expected_profit
                else:
                    break  # Since values are sorted, no more profitable trades
            
            if total_expected_profit > best_expected_profit:
                best_expected_profit = total_expected_profit
                best_bid = potential_bid
        
        return best_bid if best_expected_profit > 0 else None

    def _calculate_optimal_bid_single_unit(self):
        """Standard GD-style bidding for single unit (next token only)."""
        value = self.get_next_value_cost()
        if value is None:
            return None

        best_expected_profit = -1e-9
        best_bid = None

        # Iterate through potential profitable bids
        for potential_bid in range(self.min_price, value + 1):
            expected_profit = self._calculate_expected_profit_single(potential_bid, value)
            
            if expected_profit > best_expected_profit:
                best_expected_profit = expected_profit
                best_bid = potential_bid

        return best_bid if best_expected_profit > 0 else None

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Choose bid to maximize expected profit using MGD logic."""
        self._update_history(current_ask_info)

        if not self.can_trade():
            return None

        # Choose between single-unit and multi-unit optimization
        if self.use_multi_unit and self.max_tokens - self.mytrades_period > 1:
            optimal_bid = self._calculate_optimal_bid_multi_unit()
        else:
            optimal_bid = self._calculate_optimal_bid_single_unit()

        if optimal_bid is not None:
            self.logger.debug(f"MGD proposing bid {optimal_bid} ({'multi' if self.use_multi_unit else 'single'}-unit)")
            
        return optimal_bid

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Accept if profitable (like ZIC)."""
        if not self.can_trade() or current_offer_price is None:
            return False
        
        value = self.get_next_value_cost()
        is_profitable = (value is not None and current_offer_price <= value)
        
        if is_profitable:
            self._clear_rl_step_state()
            
        return is_profitable

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False


class MGDSeller(BaseTrader):
    """
    Modified Gjerstad-Dickhaut (MGD) Seller Strategy.
    
    Mirrors the buyer logic with appropriate modifications for sellers:
    - Uses bid history instead of ask history
    - Modifies belief function with previous period bounds
    - Multi-unit optimization for asks
    """
    def __init__(self, name, is_buyer, private_values, history_len=50, use_multi_unit=True, **kwargs):
        super().__init__(name, False, private_values, strategy="mgd")
        self.logger = logging.getLogger(f'trader.{self.name}')
        
        if history_len <= 0:
            history_len = 1
        self.history_len = history_len  
        self.use_multi_unit = use_multi_unit
        
        # Store recent market bid observations
        self.market_bid_history = deque(maxlen=self.history_len)
        
        # Previous period price bounds
        self.prev_period_highest_trade_price = None
        self.prev_period_lowest_trade_price = None
        
        self.logger.debug(f"Initialized MGD Seller: history_len={self.history_len}, multi_unit={self.use_multi_unit}")

    def _reset_learning_state(self):
        """Clear history at the start of a new ROUND."""
        self.market_bid_history.clear()
        self.prev_period_highest_trade_price = None
        self.prev_period_lowest_trade_price = None
        self.logger.debug("MGD seller history cleared for new round.")

    def update_end_of_period(self, period_trade_prices):
        """Update previous period price bounds from completed period."""
        super().update_end_of_period(period_trade_prices)
        
        if period_trade_prices:
            self.prev_period_highest_trade_price = max(period_trade_prices)
            self.prev_period_lowest_trade_price = min(period_trade_prices)
            self.logger.debug(f"MGD seller updated prev period bounds: [{self.prev_period_lowest_trade_price:.1f}, {self.prev_period_highest_trade_price:.1f}]")

    def _update_history(self, current_bid_info):
        """Add current best bid to history if it exists and is valid."""
        if current_bid_info and isinstance(current_bid_info.get('price'), (int, float)):
            price = current_bid_info['price']
            if self.min_price <= price <= self.max_price:
                self.market_bid_history.append(price)

    def _estimate_prob_accept_mgd(self, ask_price):
        """
        Modified belief function for sellers.
        
        Calculates probability that an ask at ask_price will be accepted,
        with MGD modifications based on previous period bounds.
        """
        if not self.market_bid_history or ask_price > self.max_price:
            return 0.0

        # Step 1: Basic GD belief - fraction of historical bids >= ask price
        relevant_bids = np.array(list(self.market_bid_history))
        if len(relevant_bids) == 0:
            return 0.0

        accepting_bids_count = np.sum(relevant_bids >= ask_price)
        base_probability = accepting_bids_count / len(relevant_bids)
        
        # Step 2: MGD modifications
        modified_probability = base_probability
        
        if self.prev_period_lowest_trade_price is not None:
            # Reset f(p) = 1 for asks below previous period's lowest trade price
            # Low asks below what traded before will definitely be accepted by buyers
            if ask_price < self.prev_period_lowest_trade_price:
                modified_probability = 1.0
                self.logger.debug(f"MGD: Ask {ask_price} < prev_low {self.prev_period_lowest_trade_price:.1f}, setting prob=1")
                
        if self.prev_period_highest_trade_price is not None:
            # Reset f(p) = 0 for asks above previous period's highest trade price
            # High asks above what traded before will not be accepted by buyers
            if ask_price > self.prev_period_highest_trade_price:
                modified_probability = 0.0
                self.logger.debug(f"MGD: Ask {ask_price} > prev_high {self.prev_period_highest_trade_price:.1f}, setting prob=0")
        
        return modified_probability

    def _calculate_expected_profit_single(self, ask_price, cost):
        """Calculate expected profit for a single unit at given ask price."""
        prob_accept = self._estimate_prob_accept_mgd(ask_price)
        expected_profit = prob_accept * (ask_price - cost)
        return expected_profit

    def _calculate_optimal_ask_multi_unit(self):
        """Multi-unit asking optimization for sellers."""
        if not self.can_trade():
            return None
            
        # Get all remaining costs
        remaining_costs = []
        for i in range(self.mytrades_period, self.max_tokens):
            if i < len(self.private_values):
                remaining_costs.append(self.private_values[i])
        
        if not remaining_costs:
            return None
            
        best_expected_profit = -1e-9
        best_ask = None
        
        # Consider different ask prices
        for potential_ask in range(min(remaining_costs), self.max_price + 1):
            total_expected_profit = 0.0
            
            # Calculate expected profit considering all remaining units
            for cost in remaining_costs:
                if potential_ask >= cost:  # Only profitable trades
                    unit_expected_profit = self._calculate_expected_profit_single(potential_ask, cost)
                    total_expected_profit += unit_expected_profit
                else:
                    break  # Since costs are sorted, no more profitable trades
            
            if total_expected_profit > best_expected_profit:
                best_expected_profit = total_expected_profit
                best_ask = potential_ask
        
        return best_ask if best_expected_profit > 0 else None

    def _calculate_optimal_ask_single_unit(self):
        """Standard GD-style asking for single unit."""
        cost = self.get_next_value_cost()
        if cost is None:
            return None

        best_expected_profit = -1e-9
        best_ask = None

        # Iterate through potential profitable asks
        for potential_ask in range(cost, self.max_price + 1):
            expected_profit = self._calculate_expected_profit_single(potential_ask, cost)
            
            if expected_profit > best_expected_profit:
                best_expected_profit = expected_profit
                best_ask = potential_ask

        return best_ask if best_expected_profit > 0 else None

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Choose ask to maximize expected profit using MGD logic."""
        self._update_history(current_bid_info)

        if not self.can_trade():
            return None

        # Choose between single-unit and multi-unit optimization
        if self.use_multi_unit and self.max_tokens - self.mytrades_period > 1:
            optimal_ask = self._calculate_optimal_ask_multi_unit()
        else:
            optimal_ask = self._calculate_optimal_ask_single_unit()

        if optimal_ask is not None:
            self.logger.debug(f"MGD proposing ask {optimal_ask} ({'multi' if self.use_multi_unit else 'single'}-unit)")
            
        return optimal_ask

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Accept if profitable."""
        if not self.can_trade() or current_bid_price is None:
            return False
        
        cost = self.get_next_value_cost()
        is_profitable = (cost is not None and current_bid_price >= cost)
        
        if is_profitable:
            self._clear_rl_step_state()
            
        return is_profitable