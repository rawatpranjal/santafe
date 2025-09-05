# traders/zip.py
import random
import logging
import numpy as np
# Import the *updated* BaseTrader
from .base import BaseTrader

class ZIPTrader(BaseTrader):
    """
    Base class for ZIP agents, encapsulating the core learning mechanism.
    Based on Cliff & Bruten (1997).
    """
    def __init__(self, name, is_buyer, private_values, strategy="zip", **kwargs):
        super().__init__(name, is_buyer, private_values, strategy=strategy)
        self.logger = logging.getLogger(f'trader.{self.name}.ZIPLogic')

        # --- Learning Parameters (Randomly initialized per agent) ---
        # Learning rate for price updates
        self.beta = kwargs.get('zip_beta', random.uniform(0.1, 0.5))
        # Momentum coefficient (0 = no momentum)
        self.gamma = kwargs.get('zip_gamma', random.uniform(0.0, 0.1))
        # Target price relative noise range (R factor)
        self.r_noise_range = kwargs.get('zip_r_noise', 0.05) # R will be 1 +/- this
        # Target price absolute noise range (A factor)
        self.a_noise_range = kwargs.get('zip_a_noise', 0.05) # A will be +/- this

        # Initial margin range (different for buyers/sellers)
        if self.is_buyer:
            # Buyer margin mu is in [-1, 0]
            margin_init_low = kwargs.get('zip_buyer_margin_low', -0.35)
            margin_init_high = kwargs.get('zip_buyer_margin_high', -0.05)
            self.margin = random.uniform(margin_init_low, margin_init_high)
            # Ensure it's within the valid range [-1, 0]
            self.margin = np.clip(self.margin, -1.0, 0.0)
        else:
            # Seller margin mu is in [0, inf)
            margin_init_low = kwargs.get('zip_seller_margin_low', 0.05)
            margin_init_high = kwargs.get('zip_seller_margin_high', 0.35)
            self.margin = random.uniform(margin_init_low, margin_init_high)
            # Ensure it's non-negative
            self.margin = max(0.0, self.margin)

        # --- Internal State ---
        self.price_momentum = 0.0    # Stores the momentum term 'i(t)' for price updates
        self.last_shout_price = None # Store the last price *this agent* calculated/shouted

        # self.logger.debug(f"Initialized ZIP {('Buyer' if is_buyer else 'Seller')}: "
        #                   f"μ={self.margin:.3f}, β={self.beta:.3f}, γ={self.gamma:.3f}, "
        #                   f"R_noise=±{self.r_noise_range}, A_noise=±{self.a_noise_range}")

    def _calculate_shout_price(self):
        """ Calculates the shout price based on current margin and value/cost. """
        limit_price = self.get_next_value_cost()
        if limit_price is None:
            return None

        # Equation 9: pi(t) = λi,j * (1 + μi(t))
        # Ensure limit_price is float for calculation
        try:
            limit_price_f = float(limit_price)
            shout_price = limit_price_f * (1.0 + self.margin)
        except ValueError:
             self.logger.warning(f"Invalid limit price {limit_price}. Cannot calculate shout.")
             return None

        # Clamp to market bounds
        shout_price = np.clip(shout_price, self.min_price, self.max_price)

        # Apply budget constraint (re-clamp based on role)
        if self.is_buyer:
            shout_price = min(shout_price, limit_price_f) # Cannot bid more than value
        else:
            shout_price = max(shout_price, limit_price_f) # Cannot ask less than cost

        # Final clamp to market bounds after budget constraint
        shout_price = np.clip(shout_price, self.min_price, self.max_price)

        return int(round(shout_price))


    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Calculate and return the agent's shout price. """
        if not self.can_trade():
            self.last_shout_price = None # Clear last shout if cannot trade
            return None

        shout_price = self._calculate_shout_price()
        self.last_shout_price = shout_price # Store the price we *would* shout

        # ZIP doesn't use probabilistic shouting from paper
        # self.logger.debug(f"Proposing {'bid' if self.is_buyer else 'ask'} {shout_price} "
        #                   f"(Limit={self.get_next_value_cost()}, μ={self.margin:.3f})")
        return shout_price

    def _calculate_and_apply_update(self, event_price_q, increase_margin):
        """ Calculates and applies the margin update using Widrow-Hoff delta rule. """
        limit_price = self.get_next_value_cost()
        current_shout_price = self.last_shout_price # Price based on current margin

        # Need limit price and current shout price to proceed
        if limit_price is None or current_shout_price is None:
            # self.logger.debug("Skipping update: missing limit price or last shout price.")
            return
        try:
            limit_price_f = float(limit_price)
            current_shout_price_f = float(current_shout_price)
            event_price_q_f = float(event_price_q)
        except (ValueError, TypeError):
            self.logger.warning("Skipping update: non-numeric price involved.")
            return

        # --- Calculate Target Price τ (Equation 14) ---
        # Determine R and A noise based on whether to increase/decrease margin
        # Note: Increase margin means *lower* target price for buyer, *higher* for seller
        if self.is_buyer:
             # Increase margin -> Lower target price (R<1, A<0)
             # Decrease margin -> Higher target price (R>1, A>0)
             increase_target = not increase_margin
        else: # Seller
             # Increase margin -> Higher target price (R>1, A>0)
             # Decrease margin -> Lower target price (R<1, A<0)
             increase_target = increase_margin

        # Generate random R and A components
        if increase_target:
            R = 1.0 + random.uniform(0, self.r_noise_range) # R > 1
            A = random.uniform(0, self.a_noise_range)       # A > 0
        else:
            R = 1.0 - random.uniform(0, self.r_noise_range) # R < 1
            A = random.uniform(-self.a_noise_range, 0)      # A < 0

        target_price_tau = R * event_price_q_f + A
        # Clamp target price to market bounds? Paper doesn't specify, but seems reasonable.
        target_price_tau = np.clip(target_price_tau, self.min_price, self.max_price)

        # --- Calculate Price Update using Delta Rule and Momentum ---
        # Delta based on *price* difference (Equation 13)
        price_delta = self.beta * (target_price_tau - current_shout_price_f)

        # Update momentum term 'i' (Equation 15) - this is momentum on the *price change*
        # Note: paper uses comma notation ,i(t) which is confusing, assume i(t) is self.price_momentum
        self.price_momentum = self.gamma * self.price_momentum + (1.0 - self.gamma) * price_delta

        # Calculate the new *price* using the momentum update (basis of Eq 16)
        new_shout_price = current_shout_price_f + self.price_momentum

        # --- Convert New Price back to New Margin μ (Equation 12/16) ---
        # μ(t+1) = (new_price / λ) - 1
        if limit_price_f == 0:
            # Avoid division by zero; if limit is 0, margin is ill-defined.
            # Maybe keep margin unchanged or set to a default? Let's keep unchanged.
            # self.logger.debug("Limit price is 0, cannot calculate new margin. Keeping current.")
            new_margin = self.margin
        else:
            new_margin = (new_shout_price / limit_price_f) - 1.0

        # Clamp margin based on role
        if self.is_buyer:
            new_margin = np.clip(new_margin, -1.0, 0.0) # Buyer margin mu is in [-1, 0]
        else:
            new_margin = max(0.0, new_margin)          # Seller margin mu is in [0, inf)

        # Log the update details (optional)
        # self.logger.debug(f"Update Triggered ({'Inc' if increase_margin else 'Dec'} Margin): "
        #                   f"q={event_price_q_f:.2f}, p={current_shout_price_f:.2f}, τ={target_price_tau:.2f}, "
        #                   f"Δ={price_delta:.4f}, Mom={self.price_momentum:.4f}, "
        #                   f"NewP={new_shout_price:.2f}, λ={limit_price_f:.2f}, Oldμ={self.margin:.4f} -> Newμ={new_margin:.4f}")

        self.margin = new_margin

    def _trigger_margin_update(self, event_type, event_price, step_outcome):
        """ Determines if margin should be updated based on market event. """

        my_last_shout_p = self.last_shout_price # Get the price this agent last calculated
        if my_last_shout_p is None: # Cannot compare if we didn't have a price
            # self.logger.debug("No last shout price, skipping margin update trigger.")
            return

        try:
            p = float(my_last_shout_p)
            q = float(event_price)
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid price types p={my_last_shout_p}, q={event_price}. Cannot trigger update.")
            return

        increase = None # None = no change, True = increase, False = decrease
        trade_info = step_outcome.get('last_trade_info') if step_outcome else None

        # --- Logic based on Figure 27 / Section 4.1 ---
        if event_type == 'accepted' and trade_info:
            trade_was_bid = trade_info['type'] == 'sell_accepts_bid'
            trade_was_offer = trade_info['type'] == 'buy_accepts_ask'
            i_was_buyer = self.is_buyer and trade_info.get('buyer') == self
            i_was_seller = (not self.is_buyer) and trade_info.get('seller') == self
            i_was_involved = i_was_buyer or i_was_seller

            # Condition 1 (Raise Margin)
            if not self.is_buyer and p <= q: # Seller
                increase = True
            if self.is_buyer and p >= q: # Buyer
                increase = True

            # Condition 2 (Lower Margin - Active traders only)
            if self.can_trade(): # Only active traders lower margin
                # Seller lowers if trade was accepted BID and p >= q
                if not self.is_buyer and trade_was_bid and p >= q:
                    increase = False
                # Buyer lowers if trade was accepted OFFER and p <= q
                if self.is_buyer and trade_was_offer and p <= q:
                    increase = False

        # Updates based on rejections are harder without explicit rejection signal.
        # Cliff/Bruten paper relies on knowing if the last shout was bid/offer and accepted/rejected.
        # We only reliably know about accepted shouts here.
        # else: # Event was some form of rejection (inferred)
        #     last_shout_was_bid = ??? # Need info from auction/agent state
        #     last_shout_was_offer = ???
        #
        #     if last_shout_was_offer: # Rejected Offer
        #         # Seller lowers if active and p >= q
        #          if not self.is_buyer and self.can_trade() and p >= q: increase = False
        #     elif last_shout_was_bid: # Rejected Bid
        #         # Buyer lowers if active and p <= q
        #          if self.is_buyer and self.can_trade() and p <= q: increase = False

        # --- Apply update if needed ---
        if increase is not None:
            # self.logger.debug(f"Event: {event_type} @ {q:.2f}. My last_p={p:.2f}. Triggering margin update (Increase={increase}).")
            self._calculate_and_apply_update(q, increase)
        # else:
        #     self.logger.debug(f"Event: {event_type} @ {q:.2f}. My last_p={p:.2f}. No margin update triggered.")


    def observe_reward(self, last_state, action_idx, reward, next_state, done, step_outcome=None):
        """
        Called by the Auction after each step. Use this to trigger margin updates.
        """
        # --- ZIP Logic: Update margin based on step outcome ---
        if step_outcome:
            event_type = None
            event_price = None
            trade_info = step_outcome.get('last_trade_info') # Check if a trade occurred this step

            if trade_info:
                event_type = 'accepted'
                event_price = trade_info.get('price')
            else:
                # TODO: Infer rejection based on step_outcome if needed for full C&B logic
                # Requires knowing if a bid/offer was active and didn't transact.
                pass # No reliable rejection info passed currently

            if event_type and event_price is not None:
                self._trigger_margin_update(event_type, event_price, step_outcome)

        # --- RL Logic (if this class were combined with RL) ---
        # BaseTrader's observe_reward is empty, so no super() call needed unless mixing.
        pass


    # --- Acceptance logic remains simple ---
    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept ask only if it's profitable. """
        if not self.can_trade() or current_offer_price is None: return False
        value = self.get_next_value_cost()
        if value is None: return False
        try: offer_price_f = float(current_offer_price)
        except (ValueError, TypeError): return False
        accept = (offer_price_f <= value)
        if accept: self._clear_rl_step_state() # Clear any RL temp state if accepting
        return accept

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept bid only if it's profitable. """
        if not self.can_trade() or current_bid_price is None: return False
        cost = self.get_next_value_cost()
        if cost is None: return False
        try: bid_price_f = float(current_bid_price)
        except (ValueError, TypeError): return False
        accept = (bid_price_f >= cost)
        if accept: self._clear_rl_step_state() # Clear any RL temp state if accepting
        return accept

    # --- record_trade just calls super ---
    def record_trade(self, price):
        """ Record the trade using BaseTrader method. No margin update here. """
        return super().record_trade(price)


# --- Concrete Buyer/Seller Classes ---
class ZIPBuyer(ZIPTrader):
    def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, True, private_values, **kwargs) # is_buyer = True

class ZIPSeller(ZIPTrader):
     def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, False, private_values, **kwargs) # is_buyer = False