# traders/base.py
import logging
import random
import numpy as np

class BaseTrader:
    """
    Base class for all trading agents in the SFI Double Auction.
    Refactored to align more closely with attributes available in the Java SRobotSkeleton.
    """
    def __init__(self, name, is_buyer, private_values, strategy="base", **kwargs):
        self.name = name
        try: # Create a numeric ID for seeding, fallback to hash
            numeric_part = name[1:]
            self.id_numeric = int(numeric_part) if len(name) > 1 and numeric_part.isdigit() else hash(name) % (10**8)
        except ValueError:
            self.id_numeric = hash(name) % (10**8)

        self.strategy = strategy
        self.is_buyer = is_buyer            # Corresponds to Java 'role' (1=buyer, 2=seller)
        self.logger = logging.getLogger(f'trader.{self.name}') # Logger for each agent instance

        # --- Agent's Private Valuation ---
        self._original_private_values = tuple(private_values) # Store original values
        self.private_values = list(private_values) # Working copy (Java: token[])

        # Ensure private values are integers and sorted correctly
        if not isinstance(self.private_values, (list, np.ndarray)) or not all(isinstance(v, (int, float, np.number)) for v in self.private_values):
             self.logger.error(f"Invalid private values: {private_values}")
             self.private_values = []
             self._original_private_values = tuple()
        elif not all(isinstance(v, int) for v in self.private_values):
             self.private_values = [int(round(v)) for v in self.private_values]
             self._original_private_values = tuple(self.private_values) # Store int version
        else:
             self.private_values = list(self.private_values) # Ensure it's a list copy

        if is_buyer: self.private_values.sort(reverse=True)
        else: self.private_values.sort()
        self._original_private_values = tuple(self.private_values) # Store sorted tuple

        self.max_tokens = len(self.private_values) # Corresponds to Java 'ntokens'

        # --- Game/Market Parameters (Set via update_game_params) ---
        self.nrounds = 0                    # Java: nrounds
        self.nperiods = 0                   # Java: nperiods
        self.ntimes = 0                     # Java: ntimes (steps per period)
        self.min_price = 1                  # Java: minprice
        self.max_price = 1000               # Java: maxprice
        self.nbuyers = 0                    # Java: nbuyers
        self.nsellers = 0                   # Java: nsellers
        self.gametype = 0                   # Java: gametype

        # --- Current State Tracking ---
        self.current_round = -1             # Java: r (0-based in Python)
        self.current_period = -1            # Java: p (0-based in Python)
        self.current_step = -1              # Java: t (0-based in Python)
        self.current_substep = ""           # Python specific for context

        # --- Period-Specific State (Reset in reset_for_new_period) ---
        self.tokens_left = 0                # Calculated based on mytrades_period
        self.mytrades_period = 0            # Java: mytrades (trades *this period*)
        self.mylasttime_period = 0          # Java: mylasttime (step of my last trade *this period*, 0 if none)
        self.current_period_profit = 0.0    # Java: pprofit (profit *this period*)

        # --- Round-Specific State (Reset in reset_for_new_round) ---
        self.current_round_profit = 0.0     # Java: rprofit (profit *this round*)
        self.tradelist_round = []           # Java: tradelist[] (trades per period *this round*)
        self.profitlist_round = []          # Java: profitlist[] (profit per period *this round*)

        # --- Game-Specific State ---
        self.total_game_profit = 0.0        # Java: gprofit (accumulated across rounds)

        # --- Previous Period State (for strategies like Kaplan) ---
        self.prev_period_trade_prices = [] # Stores trade prices from the last completed period
        self.prev_period_min_price = None  # Min trade price from last completed period
        self.prev_period_max_price = None  # Max trade price from last completed period
        self.prev_period_avg_price = None  # Avg trade price from last completed period

        # --- RL Specific State (Python additions) ---
        self._current_step_state = None
        self._current_step_action = None
        self._current_step_log_prob = None
        self._current_step_value = None

        # self.logger.debug(f"Initialized BaseTrader {self.name}")

    def update_game_params(self, params):
        """ Stores game/market parameters provided by the Auction environment. """
        self.nrounds = params.get('num_rounds', self.nrounds)
        self.nperiods = params.get('num_periods', self.nperiods)
        self.ntimes = params.get('num_steps', self.ntimes) # steps per period
        self.min_price = params.get('min_price', self.min_price)
        self.max_price = params.get('max_price', self.max_price)
        self.nbuyers = params.get('num_buyers', self.nbuyers)
        self.nsellers = params.get('num_sellers', self.nsellers)
        self.gametype = params.get('gametype', self.gametype)
        # self.logger.debug(f"Updated game params: Periods={self.nperiods}, Steps={self.ntimes}, PriceRange=[{self.min_price},{self.max_price}]")


    def reset_for_new_round(self):
        """
        FULL Reset for a new round (new market conditions / RL episode).
        Corresponds roughly to Java SRobotSkeleton's roundStart logic for agent state.
        """
        # self.logger.debug(f"Resetting for new round {self.current_round + 1}.")
        # Accumulate total profit before resetting round profit
        self.total_game_profit += self.current_round_profit

        # Reset round-specific accumulators
        self.current_round_profit = 0.0
        # Initialize lists based on nperiods (ensure update_game_params was called)
        if self.nperiods > 0:
            self.tradelist_round = [0] * self.nperiods
            self.profitlist_round = [0.0] * self.nperiods
        else:
            self.logger.warning("nperiods not set before reset_for_new_round, initializing lists empty.")
            self.tradelist_round = []
            self.profitlist_round = []

        # Reset period-specific state as well for the first period of the round
        self.current_period_profit = 0.0
        self.mytrades_period = 0
        self.mylasttime_period = 0
        self.tokens_left = self.max_tokens # Should be reset in period reset anyway
        self.current_period = -1 # Reset current period index for the new round

        # Clear previous period price stats
        self.prev_period_trade_prices = []
        self.prev_period_min_price = None
        self.prev_period_max_price = None
        self.prev_period_avg_price = None

        # Reset strategy-specific learning state (implemented by subclasses)
        self._reset_learning_state()
        # Clear temporary RL step state
        self._clear_rl_step_state()


    def reset_for_new_period(self, round_idx, period_idx):
        """
        Reset only PERIOD-specific state. Keep learned state and round accumulators.
        Corresponds roughly to Java SRobotSkeleton's periodStart logic for agent state.
        """
        # Store results from the completed period into round lists
        if self.current_period >= 0 and self.current_period < len(self.profitlist_round):
            # self.logger.debug(f"Storing results for finished R{self.current_round}P{self.current_period}: Profit={self.current_period_profit}, Trades={self.mytrades_period}")
            self.profitlist_round[self.current_period] = self.current_period_profit
            self.tradelist_round[self.current_period] = self.mytrades_period
        elif self.current_period >= len(self.profitlist_round) and len(self.profitlist_round)>0:
             self.logger.warning(f"Attempted to store period {self.current_period} results, but list size is {len(self.profitlist_round)}")

        # Update current time indices
        self.current_round = round_idx
        self.current_period = period_idx
        self.current_step = -1 # Reset step counter for the period
        self.current_substep = ""

        # Reset period-specific counters/state
        self.tokens_left = self.max_tokens # Reset tokens available for the period
        self.mytrades_period = 0            # Reset trades made *this period*
        self.mylasttime_period = 0          # Reset time of my last trade *this period*
        self.current_period_profit = 0.0    # Reset profit earned *this period*

        # Reset previous period price stats (will be updated in update_end_of_period)
        self.prev_period_trade_prices = []
        self.prev_period_min_price = None
        self.prev_period_max_price = None
        self.prev_period_avg_price = None

        # DO NOT reset round accumulators (current_round_profit, profitlist_round, tradelist_round)
        # DO NOT reset total_game_profit
        # DO NOT reset learned parameters or persistent state (like LSTM hidden state) here

        self._clear_rl_step_state() # Clear temporary step state only
        # self.logger.debug(f"Reset for R{round_idx} P{period_idx}. Tokens={self.tokens_left}, PeriodProfit={self.current_period_profit}, PeriodTrades={self.mytrades_period}")


    def _reset_learning_state(self):
        """ Abstract method for strategy-specific reset of learned parameters/state. """
        # Called by reset_for_new_round()
        pass # Base class does nothing, subclasses override if they learn

    def _clear_rl_step_state(self):
         """ Helper to clear internal RL state related to a single step's action choice. """
         self._current_step_state = None
         self._current_step_action = None
         self._current_step_log_prob = None
         self._current_step_value = None

    def can_trade(self):
        """ Check if the trader has tokens remaining for the current period. """
        # Based on number of trades made this period vs max tokens
        can = self.mytrades_period < self.max_tokens
        # if not can: self.logger.debug(f"can_trade=False (trades={self.mytrades_period}, max={self.max_tokens})")
        return can

    def get_next_value_cost(self):
        """
        Get the value/cost for the next available token IN THIS PERIOD.
        Uses mytrades_period as the index into the sorted private_values list.
        """
        if not self.can_trade():
            # self.logger.warning(f"get_next_value_cost called but cannot trade (trades={self.mytrades_period})")
            return None
        # Index based on tokens *used* in this period
        idx = self.mytrades_period # 0-based index for the next trade
        if 0 <= idx < len(self.private_values):
            # Return the original value/cost for this token index
            # EL strategy will use its internal reservation_prices instead
            return self.private_values[idx]
        else:
            self.logger.error(f"Invalid index {idx} accessing private_values (len={len(self.private_values)}, trades_period={self.mytrades_period}).")
            return None

    def record_trade(self, price):
        """
        Records a trade, updates profits & token count for the CURRENT PERIOD/ROUND.
        Returns profit_increment or None if trade is invalid.
        Assumes current_step is correctly set externally before this is called.
        """
        current_step = self.current_step # Get current step index from internal state

        if not self.can_trade():
            self.logger.error(f"Trade recorded at R{self.current_round}P{self.current_period}S{current_step} but cannot trade (trades={self.mytrades_period})!")
            return None # Indicate failure

        # Use the specific reservation price logic for EL, otherwise use base value/cost
        # EL needs to override this method or provide _get_current_reservation_price
        if hasattr(self, '_get_current_reservation_price'):
             val_cost_basis = self._get_current_reservation_price()
             if val_cost_basis is None:
                  self.logger.error(f"Could not get reservation price basis for trade R{self.current_round}P{self.current_period}S{current_step}")
                  val_cost_basis = self.get_next_value_cost() # Fallback to base value
                  if val_cost_basis is None: return None # Fail if base value also missing
        else:
             val_cost_basis = self.get_next_value_cost()
             if val_cost_basis is None:
                  self.logger.error(f"Trade recorded at R{self.current_round}P{self.current_period}S{current_step} but no value/cost found!")
                  return None

        # Ensure price is valid numeric type
        try: price_int = int(round(float(price)))
        except (ValueError, TypeError):
             self.logger.warning(f"Received non-numeric price {price} for trade. Cannot record.")
             return None

        # Calculate profit increment based on the determined basis
        profit_increment = (val_cost_basis - price_int) if self.is_buyer else (price_int - val_cost_basis)

        # Update state *after* getting necessary info
        self.mytrades_period += 1
        self.tokens_left = self.max_tokens - self.mytrades_period # Update derived state
        self.mylasttime_period = self.current_step # Record step (0-based) of this trade
        self.current_period_profit += profit_increment
        self.current_round_profit += profit_increment
        # self.total_game_profit gets updated in reset_for_new_round

        # Mark success for EL agent if applicable (implementation in EL class)
        if hasattr(self, 'trade_success') and 0 <= self.mytrades_period - 1 < len(self.trade_success):
            self.trade_success[self.mytrades_period - 1] = True # Use trade index

        # self.logger.debug(f"TRADE Recorded @ R{self.current_round}P{self.current_period}S{current_step}: Price={price_int}. Basis={val_cost_basis}, Inc={profit_increment:.2f}. PeriodTrades={self.mytrades_period}, PeriodProfit={self.current_period_profit:.2f}, RoundProfit={self.current_round_profit:.2f}, MyLastTime={self.mylasttime_period}")
        return profit_increment


    # --- Abstract & Optional Methods ---
    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """
        Agent logic to determine bid/ask price.
        Args:
            current_bid_info (dict or None): {'price': P, 'agent': agent_obj} of current best bid.
            current_ask_info (dict or None): {'price': P, 'agent': agent_obj} of current best ask.
            phibid (int): Highest bid observed EVER in the period.
            phiask (int): Lowest ask observed EVER in the period.
            market_history (dict): Contains additional info like:
                'last_trade_info_for_period' (dict or None): Info on the last trade.
                'lasttime' (int): Step index (0-based) of the last market trade, or -1 if none.
                'all_bids_this_step' (list): Bids submitted in the current BA substep.
                'all_asks_this_step' (list): Asks submitted in the current BA substep.
        Returns:
            int or None: The price to bid/ask, or None to submit nothing.
        """
        raise NotImplementedError(f"'make_bid_or_ask' not implemented for {self.strategy}")

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """
        Buyer logic to decide whether to accept the current_offer_price (which is current_ask).
        Only called if this agent holds the current_bid and quotes cross.
        Args:
            current_offer_price (int): The price of the current best ask.
            current_bid_info (dict): Info about my own winning bid.
            current_ask_info (dict): Info about the ask I might accept.
            phibid (int): Highest bid observed EVER in the period.
            phiask (int): Lowest ask observed EVER in the period.
            market_history (dict): Additional market context (incl. 'lasttime').
        Returns:
            bool: True to accept the offer, False otherwise.
        """
        raise NotImplementedError(f"'request_buy' not implemented for {self.strategy}")

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """
        Seller logic to decide whether to accept the current_bid_price.
        Only called if this agent holds the current_ask and quotes cross.
        Args:
            current_bid_price (int): The price of the current best bid.
            current_bid_info (dict): Info about the bid I might accept.
            current_ask_info (dict): Info about my own winning ask.
            phibid (int): Highest bid observed EVER in the period.
            phiask (int): Lowest ask observed EVER in the period.
            market_history (dict): Additional market context (incl. 'lasttime').
        Returns:
            bool: True to accept the bid, False otherwise.
        """
        raise NotImplementedError(f"'request_sell' not implemented for {self.strategy}")

    # --- Optional methods used by specific strategies / RL agents ---
    def update_end_of_period(self, period_trade_prices):
        """
        Hook called by Auction at the end of each period.
        Corresponds roughly to Java playerPeriodEnd() for agent updates.
        Args:
             period_trade_prices (list): List of trade prices that occurred in the period just ended.
        """
        # Example: Calculate and store stats from the completed period
        self.prev_period_trade_prices = list(period_trade_prices) # Store the prices
        if self.prev_period_trade_prices:
            try:
                self.prev_period_min_price = int(round(np.min(self.prev_period_trade_prices)))
                self.prev_period_max_price = int(round(np.max(self.prev_period_trade_prices)))
                self.prev_period_avg_price = float(np.mean(self.prev_period_trade_prices))
                # self.logger.debug(f"End P{self.current_period}: Stored prev period prices. Min={self.prev_period_min_price}, Max={self.prev_period_max_price}, Avg={self.prev_period_avg_price:.2f}")
            except Exception as e:
                 self.logger.warning(f"Could not calculate stats for prev period prices {self.prev_period_trade_prices}: {e}")
                 self.prev_period_min_price = None
                 self.prev_period_max_price = None
                 self.prev_period_avg_price = None
        else:
            self.prev_period_min_price = None
            self.prev_period_max_price = None
            self.prev_period_avg_price = None
            # self.logger.debug(f"End P{self.current_period}: No trades in previous period.")
        pass # Subclasses can override to add specific logic (e.g., EL updates)

    def set_mode(self, training=True):
        """ Set agent mode (e.g., training vs evaluation for RL). """
        pass

    def observe_reward(self, last_state, action_idx, reward, next_state, done, step_outcome=None):
        """ Provide observation data to RL agent's logic or adaptive heuristics like ZIP. """
        pass

    def get_last_episode_stats(self):
        """ Retrieve training stats for the last completed episode (round). """
        return {}

    def save_model(self, path_prefix):
        """ Save learned model parameters (for RL agents). """
        self.logger.warning(f"Save model not implemented for strategy {self.strategy}")

    def load_model(self, path_prefix):
        """ Load learned model parameters (for RL agents). """
        self.logger.warning(f"Load model not implemented for strategy {self.strategy}")
