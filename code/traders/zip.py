# traders/zip.py
import random
import logging
import numpy as np
from .base import BaseTrader

class ZipBuyer(BaseTrader):
    """
    ZIP Buyer Strategy (Simplified):
    - Maintains a profit margin `self.margin` in [0, 1].
    - Bids based on `value * (1 - self.margin)`.
    - Adapts margin slightly when a trade occurs using a Widrow-Hoff inspired rule.
    - Accepts profitable offers like ZIC.
    """
    def __init__(self, name, is_buyer, private_values, margin_init=0.05, beta=0.1, gamma=0.1, shout_probability=0.8, **kwargs):
        # Ensure is_buyer is True for ZipBuyer
        super().__init__(name, True, private_values, strategy="zip")
        self.logger = logging.getLogger(f'trader.{self.name}')
        # Clamp initial margin to valid range [0, 1] for buyers
        if not (0 <= margin_init <= 1):
            self.logger.warning(f"Initial margin {margin_init} out of [0,1]. Clamping.")
            margin_init = np.clip(margin_init, 0.0, 1.0)
        self.margin = margin_init
        self.beta = beta           # Learning rate for target adjustment
        self.gamma = gamma         # Momentum factor
        self.momentum = 0.0        # Smoothed margin adjustment
        self.shout_probability = shout_probability # Likelihood of submitting a quote
        self.last_bid_price = None # Store last shouted bid price
        self.logger.debug(f"Initialized ZIP Buyer: margin={self.margin:.3f}, beta={self.beta}, gamma={self.gamma}, shout_prob={self.shout_probability}")

    def get_next_value_cost(self):
        """ Gets the value for the next token. """
        return super().get_next_value_cost()

    def _update_margin(self, trade_price, traded_value):
        """ Update margin based on the price and value of the completed trade. """
        if traded_value is None or trade_price is None or traded_value <= 0:
            self.logger.debug("Cannot update margin: missing value/price or value<=0.")
            return

        # Heuristic: Target a margin based on the realized profit.
        # If profit was high (price << value), decrease margin (bid higher next time).
        # If profit was low (price ≈ value), increase margin (bid lower next time).
        # Let's target a price halfway between trade_price and value as a reference.
        target_price_heuristic = (trade_price + traded_value) / 2.0
        # Convert target price to target margin: margin = 1 - (price / value)
        target_margin = 1.0 - (target_price_heuristic / traded_value)
        target_margin = np.clip(target_margin, 0.0, 1.0) # Ensure target margin is valid [0, 1]

        # Calculate raw delta towards the target margin
        raw_delta = self.beta * (target_margin - self.margin)
        # Apply momentum smoothing
        delta = self.gamma * self.momentum + (1.0 - self.gamma) * raw_delta
        self.momentum = delta # Store momentum for next update

        # Update the actual margin
        self.margin += delta
        self.margin = np.clip(self.margin, 0.0, 1.0) # Clamp margin to [0, 1]
        self.logger.debug(f"Updated margin after trade. Price={trade_price}, V_Traded={traded_value}, TargetM={target_margin:.3f}, NewM={self.margin:.3f}")

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Calculate and potentially submit a bid based on current margin. """
        if not self.can_trade(): return None

        # Decide whether to shout based on probability
        if random.random() >= self.shout_probability:
            # self.logger.debug("Decided not to shout bid.")
            return None

        value = self.get_next_value_cost()
        if value is None: return None # Should not happen if can_trade is true

        # Calculate bid based on current margin
        bid_price = value * (1.0 - self.margin)
        # Clamp to market bounds and ensure integer price
        bid_price = max(self.min_price, min(self.max_price, int(round(bid_price))))

        # Final check: ensure bid is profitable (<= value)
        if bid_price > value:
             # This might happen due to rounding or if margin calculation dips below 0 temporarily before clamping
             # self.logger.warning(f"Calculated bid {bid_price} > value {value}. Clamping to value.")
             bid_price = value

        # Ensure bid doesn't fall below min_price after clamping to value
        bid_price = max(self.min_price, bid_price)

        self.last_bid_price = bid_price
        # self.logger.debug(f"Proposing bid {bid_price} (V={value}, M={self.margin:.3f})")
        return bid_price

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept offer if profitable (like ZIC). """
        if not self.can_trade() or current_offer_price is None: return False
        value = self.get_next_value_cost()
        is_profitable = (value is not None and current_offer_price <= value)
        if is_profitable:
            # self.logger.debug(f"Requesting BUY at {current_offer_price} (Value={value})")
            # Clear potential RL state inherited from base class
            self._current_step_state = None; self._current_step_action = None
            self._current_step_log_prob = None; self._current_step_value = None
        # NOTE: A more complex ZIP might update margin here if a highly profitable offer was missed.
        return is_profitable

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Buyers do not accept bids. """
        return False

    def record_trade(self, period, step, price):
        """ Record the trade and trigger margin update. """
        # Get the value of the token *being* traded before updating state
        value_traded = self.get_next_value_cost()
        # Call base class to update profit, tokens_left, etc.
        profit_increment = super().record_trade(period, step, price)
        # If trade was successful (profit_increment is not None), update margin
        if profit_increment is not None and value_traded is not None:
             self._update_margin(price, value_traded) # Pass price and value involved
        return profit_increment


class ZipSeller(BaseTrader):
    """
    ZIP Seller Strategy (Simplified):
    - Maintains a profit margin `self.margin` >= 0.
    - Asks based on `cost * (1 + self.margin)`.
    - Adapts margin slightly when a trade occurs.
    - Accepts profitable bids like ZIC.
    """
    def __init__(self, name, is_buyer, private_values, margin_init=0.05, beta=0.1, gamma=0.1, shout_probability=0.8, **kwargs):
        # Ensure is_buyer is False for ZipSeller
        super().__init__(name, False, private_values, strategy="zip")
        self.logger = logging.getLogger(f'trader.{self.name}')
        # Clamp initial margin to be non-negative for sellers
        if margin_init < 0:
            self.logger.warning(f"Initial margin {margin_init} is negative. Clamping to 0.")
            margin_init = max(0.0, margin_init)
        self.margin = margin_init
        self.beta = beta
        self.gamma = gamma
        self.momentum = 0.0
        self.shout_probability = shout_probability
        self.last_ask_price = None
        self.logger.debug(f"Initialized ZIP Seller: margin={self.margin:.3f}, beta={self.beta}, gamma={self.gamma}, shout_prob={self.shout_probability}")

    def get_next_value_cost(self):
        """ Gets the cost for the next token. """
        return super().get_next_value_cost()

    def _update_margin(self, trade_price, traded_cost):
        """ Update margin based on the price and cost of the completed trade. """
        if traded_cost is None or trade_price is None or traded_cost <= 0:
            self.logger.debug("Cannot update margin: missing cost/price or cost<=0.")
            return

        # Heuristic: Target a margin based on realized profit.
        # If profit was high (price >> cost), decrease margin (ask lower next time).
        # If profit was low (price ≈ cost), increase margin (ask higher next time).
        # Target price halfway between trade_price and cost.
        target_price_heuristic = (trade_price + traded_cost) / 2.0
        # Convert target price to target margin: margin = (price / cost) - 1
        target_margin = (target_price_heuristic / traded_cost) - 1.0
        target_margin = max(0.0, target_margin) # Ensure target margin is non-negative

        # Calculate delta and apply momentum
        raw_delta = self.beta * (target_margin - self.margin)
        delta = self.gamma * self.momentum + (1.0 - self.gamma) * raw_delta
        self.momentum = delta

        # Update margin
        self.margin += delta
        self.margin = max(0.0, self.margin) # Ensure margin remains non-negative
        self.logger.debug(f"Updated margin after trade. Price={trade_price}, C_Traded={traded_cost}, TargetM={target_margin:.3f}, NewM={self.margin:.3f}")

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Calculate and potentially submit an ask based on current margin. """
        if not self.can_trade(): return None

        if random.random() >= self.shout_probability:
            # self.logger.debug("Decided not to shout ask.")
            return None

        cost = self.get_next_value_cost()
        if cost is None: return None

        # Calculate ask price
        ask_price = cost * (1.0 + self.margin)
        ask_price = max(self.min_price, min(self.max_price, int(round(ask_price))))

        # Ensure ask is profitable (>= cost)
        if ask_price < cost:
             # self.logger.warning(f"Calculated ask {ask_price} < cost {cost}. Clamping to cost.")
             ask_price = cost

        # Ensure ask doesn't exceed max_price after clamping to cost
        ask_price = min(self.max_price, ask_price)

        self.last_ask_price = ask_price
        # self.logger.debug(f"Proposing ask {ask_price} (C={cost}, M={self.margin:.3f})")
        return ask_price

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
         """ Sellers do not accept asks. """
         return False

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept bid if profitable (like ZIC). """
        if not self.can_trade() or current_bid_price is None: return False
        cost = self.get_next_value_cost()
        is_profitable = (cost is not None and current_bid_price >= cost)
        if is_profitable:
            # self.logger.debug(f"Requesting SELL at {current_bid_price} (Cost={cost})")
            # Clear potential RL state inherited from base class
            self._current_step_state = None; self._current_step_action = None
            self._current_step_log_prob = None; self._current_step_value = None
        # NOTE: A more complex ZIP might update margin here if a highly profitable bid was missed.
        return is_profitable

    def record_trade(self, period, step, price):
        """ Record the trade and trigger margin update. """
        cost_traded = self.get_next_value_cost()
        profit_increment = super().record_trade(period, step, price)
        if profit_increment is not None and cost_traded is not None:
             self._update_margin(price, cost_traded)
        return profit_increment