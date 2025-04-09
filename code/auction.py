# auction.py
import random
import numpy as np
import logging
import time
from collections import defaultdict
import os
import inspect
from tqdm import tqdm

# Assuming utils.py and traders/registry.py are in the correct path
try:
    from utils import (compute_equilibrium, generate_sfi_components,
                       calculate_sfi_values_for_participant)
    from traders.registry import get_trader_class
    from traders.base import BaseTrader # Make sure BaseTrader has new reset logic
except ImportError as e:
     print(f"Error importing modules in auction.py: {e}")
     print("Ensure utils.py, traders/registry.py, and traders/base.py are accessible.")
     raise

class Auction:
    """
    Implements the SFI Double Auction mechanism.
    Modified to treat a ROUND as a single RL episode.
    Handles rounds, periods, steps, quote updates, trade execution,
    and round-based RL training.
    """
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('auction')
        self.num_rounds = config["num_rounds"]
        self.num_periods = config.get("num_periods", 1)
        self.num_steps = config["num_steps"]
        self.min_price = int(config["min_price"])
        self.max_price = int(config["max_price"])
        self.gametype = config.get("gametype", 0)
        self.num_buyers = config['num_buyers']
        self.num_sellers = config['num_sellers']
        self.buyer_specs = config['buyers']
        self.seller_specs = config['sellers']
        self.num_training_rounds = config.get("num_training_rounds", 0)
        self.rl_config = config # Pass entire config

        self.round_stats = []       # Stores summary dict for each round
        self.all_step_logs = []     # Stores detailed dict for every step across all rounds
        self.rl_training_logs = []  # Stores RL training stats dict after each round's update

        # Market state variables (reset per period)
        self.current_bid_info = None
        self.current_ask_info = None
        self.phibid = self.min_price
        self.phiask = self.max_price
        self.quote_cleared_in_last_buy_sell = True
        self.last_trade_info_for_period = None

        # RNG Setup
        seed_values = config.get("rng_seed_values", int(time.time()))
        seed_auction = config.get("rng_seed_auction", int(time.time()) + 1)
        self.value_rng = random.Random(seed_values)
        self.auction_rng = random.Random(seed_auction)
        self.logger.info(f"RNG Seeds: Values={seed_values}, Auction={seed_auction}")

        self.current_round = -1
        self.current_period = -1
        self.current_step = -1

    def _create_traders_for_round(self, r):
        """ Instantiate agents. Called at the start of each round. """
        buyers = []; sellers = []
        N = self.config.get('num_tokens', 1)
        # Generate SFI components ONCE per round
        sfi_components = generate_sfi_components(self.gametype, self.min_price, self.max_price, self.value_rng)

        # Create Buyers
        for i in range(self.num_buyers):
            spec_idx = i % len(self.buyer_specs)
            spec = self.buyer_specs[spec_idx]
            trader_type = spec["type"]
            init_args = spec.get("init_args", {})
            name = f"B{i}"
            role_l = 1 # Role label for buyer

            # Calculate values for this buyer
            values = calculate_sfi_values_for_participant(name, role_l, N, sfi_components, self.min_price, self.max_price, self.value_rng)

            TraderCls = get_trader_class(trader_type, True) # is_buyer = True
            sig = inspect.signature(TraderCls.__init__)

            # Instantiate trader, passing rl_config if expected
            if 'rl_config' in sig.parameters:
                trader_instance = TraderCls(name, True, values, rl_config=self.rl_config, **init_args)
            else:
                trader_instance = TraderCls(name, True, values, **init_args)

            trader_instance.update_market_params(self.min_price, self.max_price)
            buyers.append(trader_instance)

        # Create Sellers
        for j in range(self.num_sellers):
            spec_idx = j % len(self.seller_specs)
            spec = self.seller_specs[spec_idx]
            trader_type = spec["type"]
            init_args = spec.get("init_args", {})
            name = f"S{j}"
            role_l = 2 # Role label for seller

            # Calculate costs for this seller
            costs = calculate_sfi_values_for_participant(name, role_l, N, sfi_components, self.min_price, self.max_price, self.value_rng)

            TraderCls = get_trader_class(trader_type, False) # is_buyer = False
            sig = inspect.signature(TraderCls.__init__)

            if 'rl_config' in sig.parameters:
                trader_instance = TraderCls(name, False, costs, rl_config=self.rl_config, **init_args)
            else:
                trader_instance = TraderCls(name, False, costs, **init_args)

            trader_instance.update_market_params(self.min_price, self.max_price)
            sellers.append(trader_instance)

        # Log agent creation summary occasionally
        if r == 0:
            self.logger.info(f"Round {r}: Created {len(buyers)} buyers ({[t.strategy for t in buyers]}), {len(sellers)} sellers ({[t.strategy for t in sellers]}).")
        elif r > 0 and r % 500 == 0:
            self.logger.info(f"Round {r}: Created agents.")

        all_traders = buyers + sellers
        current_round_rl_agents = []
        round_initial_lstm_states = {} # Store initial states for this round's RL agents

        # --- ROUND START ---
        # Reset agents for the new round (full reset including learned state)
        for t in all_traders:
            t.reset_for_new_round() # BaseTrader method handles common reset

            # --- IMPLEMENTED FIX ---
            # Explicitly call agent-specific learning state reset if it exists
            # Crucial for agents like GD (history) and EL (reservations)
            if hasattr(t, '_reset_learning_state') and callable(getattr(t, '_reset_learning_state')):
                t.logger.debug(f"Calling _reset_learning_state for {t.name}")
                t._reset_learning_state()
            # --- END FIX ---

            # Identify RL agents and set mode
            if hasattr(t, 'logic') and hasattr(t.logic, 'is_training'):
                if hasattr(t, 'set_mode'):
                    # Assuming `is_training_round` is defined correctly based on `r`
                    is_training_round = r < self.num_training_rounds
                    t.set_mode(training=is_training_round)
                current_round_rl_agents.append(t)
                # Store initial LSTM state for BPTT (after agent reset)
                if hasattr(t, 'current_lstm_state') and t.current_lstm_state is not None:
                     round_initial_lstm_states[t] = (t.current_lstm_state[0].clone().detach(), t.current_lstm_state[1].clone().detach())
                else:
                     # Handle cases where RL agent might not have LSTM state yet (e.g., first round)
                     if hasattr(t, 'logic') and hasattr(t.logic, 'agent') and hasattr(t.logic.agent, 'get_initial_state'):
                          initial_state = t.logic.agent.get_initial_state(batch_size=1, device=t.logic.device)
                          round_initial_lstm_states[t] = (initial_state[0].clone().detach(), initial_state[1].clone().detach())
                          t.logger.debug(f"Generated initial LSTM state for {t.name} at round start.")
                     else:
                          self.logger.warning(f"Could not get/store initial LSTM state for RL agent {t.name}")

        # Make sure to return the RL agents and their initial states if needed elsewhere
        # (currently they are used directly in run_auction)
        return buyers, sellers, all_traders, current_round_rl_agents, round_initial_lstm_states

    def run_auction(self):
        """ Main auction loop. Each round is one RL episode. """
        self.logger.info(f"Starting Auction: {self.config['experiment_name']}")
        self.logger.info(f"Params: R={self.num_rounds}, P={self.num_periods}, S={self.num_steps}, TrainR={self.num_training_rounds}")

        unique_rl_traders_all_rounds = set()

        use_tqdm = self.logger.getEffectiveLevel() >= logging.INFO
        rounds_iterable = range(self.num_rounds)
        rounds_iterator = tqdm(rounds_iterable, desc="Auction Rounds", unit="round", dynamic_ncols=True, disable=not use_tqdm)

        for r in rounds_iterator:
            self.current_round = r
            is_training_round = r < self.num_training_rounds
            mode = "Training" if is_training_round else "Evaluation"

            # Create agents and perform round start resets (including specific learning state resets)
            buyers, sellers, all_traders, current_round_rl_agents, round_initial_lstm_states = \
                self._create_traders_for_round(r)

            # Add identified RL agents to the set tracking unique RL agents across all rounds
            for agent in current_round_rl_agents:
                 unique_rl_traders_all_rounds.add(agent)

            # Calculate Equilibrium for the round
            all_buyer_values = sorted([v for b in buyers for v in b.private_values], reverse=True)
            all_seller_costs = sorted([c for s in sellers for c in s.private_values]) # Corrected variable name
            eq_q, eq_p_mid, eq_surplus = compute_equilibrium(all_buyer_values, all_seller_costs)
            # self.logger.debug(f"Round {r}: Eq Q={eq_q}, Eq Price={eq_p_mid:.2f}, Eq Surplus={eq_surplus:.2f}")

            round_total_trades = 0; round_trade_prices = []; round_step_logs = []
            last_step_final_market_info = {} # Store market state after very last step

            # --- Period Loop ---
            for p in range(self.num_periods):
                self.current_period = p
                # Reset only period-specific state (tokens, step counters)
                # Crucially: Does NOT reset learned params or LSTM state now
                for t in all_traders:
                     t.reset_for_new_period(self.num_steps, r, p) # BaseTrader handles common period reset

                # Reset period-specific market state
                self.current_bid_info = None; self.current_ask_info = None
                self.phibid = self.min_price; self.phiask = self.max_price
                self.quote_cleared_in_last_buy_sell = True
                self.last_trade_info_for_period = None # Reset for this period

                # Run steps for the period
                period_trades, period_profit, period_prices, period_step_logs, last_step_final_market_info = \
                    self._run_period_steps(r, p, buyers, sellers, current_round_rl_agents, round_initial_lstm_states)

                # Accumulate round totals
                round_total_trades += period_trades
                round_trade_prices.extend(period_prices)
                round_step_logs.extend(period_step_logs)

                # --- Inter-Period Updates (e.g., for EL) ---
                # Gather period summary stats if needed by agents
                period_summary_stats = {"trades": period_trades, "profit": period_profit, "prices": period_prices} # Example
                for t in all_traders:
                     if hasattr(t, 'update_end_of_period') and callable(getattr(t, 'update_end_of_period')): # Check if agent needs update
                          t.update_end_of_period(period_summary_stats)

            # --- End of Round (End of RL Episode) ---
            round_rl_stats = [] # Collect stats generated by learn() call

            # --- Trigger Learning for RL Agents ---
            # Note: Learning is now triggered *within* observe_reward when done=True at the end of the round
            # This section primarily collects the stats *after* learning has occurred for the round.
            if current_round_rl_agents:
                 for agent in current_round_rl_agents:
                      if hasattr(agent, 'get_last_episode_stats'):
                           stats = agent.get_last_episode_stats()
                           if stats: # Store if learning happened (stats dict is not empty)
                                stats['round'] = r; stats['period'] = -1 # Indicate round-level stats
                                stats['agent_name'] = agent.name
                                round_rl_stats.append(stats)
                                self.logger.debug(f"Collected RL stats for {agent.name} in R{r}: {stats}")
                           # else: self.logger.debug(f"No learning stats returned for {agent.name} in R{r}.")

            # Add collected RL stats to the main log
            if round_rl_stats:
                 self.rl_training_logs.extend(round_rl_stats)

            # --- Calculate Round Summary Stats ---
            round_total_profit = sum(t.current_round_profit for t in all_traders)
            # Handle zero surplus case carefully
            if abs(eq_surplus) < 1e-9:
                efficiency = 1.0 if abs(round_total_profit) < 1e-9 else 0.0
                if abs(round_total_profit) > 1e-9:
                    self.logger.warning(f"R{r}: Non-zero profit ({round_total_profit:.2f}) with zero theoretical surplus.")
            else:
                efficiency = round_total_profit / eq_surplus

            avg_p = np.mean(round_trade_prices) if round_trade_prices else None
            adiff_p = abs(avg_p - eq_p_mid) if avg_p is not None and eq_p_mid is not None else None
            adiff_q = abs(round_total_trades - eq_q) if eq_q is not None else None
            bot_details_round = []; role_strat_perf_round = defaultdict(lambda: {"profit": 0.0, "count": 0, "trades": 0})

            for t in all_traders:
                role = "buyer" if t.is_buyer else "seller"
                strat = t.strategy
                profit = t.current_round_profit
                # Calculate trades accurately based on tokens used
                num_trades_agent = t.max_tokens - t.tokens_left # Trades this round
                bot_details_round.append({"name": t.name, "role": role, "strategy": strat, "profit": profit, "trades": num_trades_agent})
                role_strat_perf_round[(role, strat)]["profit"] += profit
                role_strat_perf_round[(role, strat)]["count"] += 1
                role_strat_perf_round[(role, strat)]["trades"] += num_trades_agent

            role_strat_perf_dict = {str(k): v for k, v in role_strat_perf_round.items()}
            rstats = {
                "round": r, "mode": mode, "num_periods": self.num_periods, "num_steps": self.num_steps,
                "eq_q": eq_q, "eq_p": eq_p_mid, "eq_surplus": eq_surplus,
                "actual_trades": round_total_trades, "actual_total_profit": round_total_profit,
                "market_efficiency": efficiency, "avg_price": avg_p,
                "abs_diff_price": adiff_p, "abs_diff_quantity": adiff_q,
                "buyer_vals": all_buyer_values, "seller_vals": all_seller_costs,
                "role_strat_perf": role_strat_perf_dict, "bot_details": bot_details_round
            }
            self.round_stats.append(rstats)
            self.all_step_logs.extend(round_step_logs) # Add logs from this round

            # --- TQDM Update ---
            rl_stats_for_tqdm = {}
            round_total_profit_rl = sum(t.current_round_profit for t in current_round_rl_agents)
            if current_round_rl_agents:
                 # Extract latest stats collected for this round
                 latest_round_stats = [s for s in round_rl_stats if s.get('round') == r]
                 avg_pl = np.nanmean([s.get('avg_policy_loss', np.nan) for s in latest_round_stats]) if latest_round_stats else np.nan
                 avg_vl = np.nanmean([s.get('avg_value_loss', np.nan) for s in latest_round_stats]) if latest_round_stats else np.nan
                 rl_stats_for_tqdm = {"Mode": mode[0], "RLProfit": f"{round_total_profit_rl:.1f}",
                                      "AvgPL": f"{avg_pl:.3f}" if not np.isnan(avg_pl) else "N/A",
                                      "AvgVL": f"{avg_vl:.3f}" if not np.isnan(avg_vl) else "N/A"}

            if use_tqdm:
                combined_stats = {"Eff": f"{efficiency:.3f}"}
                combined_stats.update(rl_stats_for_tqdm)
                rounds_iterator.set_postfix(combined_stats, refresh=False)

        if use_tqdm: rounds_iterator.close()

        # --- Save Models (End of All Rounds) ---
        if self.config.get("save_rl_model", False) and unique_rl_traders_all_rounds:
             save_dir = os.path.join(self.config.get("experiment_dir", "experiments"), # Ensure dir exists
                                      self.config["experiment_name"], "models")
             os.makedirs(save_dir, exist_ok=True)
             self.logger.info(f"Saving final RL models for {len(unique_rl_traders_all_rounds)} unique agents to {save_dir}...")
             saved_count = 0
             for agent_trader in unique_rl_traders_all_rounds:
                 # Use strategy and name for unique file prefix
                 model_path_prefix = os.path.join(save_dir, f"{agent_trader.strategy}_agent_{agent_trader.name}")
                 try:
                     if hasattr(agent_trader, 'save_model') and callable(getattr(agent_trader, 'save_model')):
                         agent_trader.save_model(model_path_prefix)
                         saved_count += 1
                     else:
                         self.logger.warning(f"Agent {agent_trader.name} has no save_model method.")
                 except Exception as e:
                     self.logger.error(f"Save failed for {agent_trader.name}: {e}", exc_info=True)
             self.logger.info(f"Attempted to save models for {saved_count} agents.")
        elif self.config.get("save_rl_model", False):
             self.logger.warning("RL model saving enabled, but no RL agents were found/tracked.")

        self.logger.info("===== SFI Auction Run Finished =====")


    def _run_period_steps(self, r, p, buyers, sellers, current_round_rl_agents, round_initial_lstm_states):
        """ Runs steps within a period. Returns period stats and final market info. """
        period_step_logs = []; period_trades = 0; period_profit = 0.0; period_prices = []
        market_history_for_agents = { # Store state passed between steps within period
            'last_trade_info_for_period': self.last_trade_info_for_period,
            'all_bids_this_step': [], # Initialize for first step
            'all_asks_this_step': []
        }
        last_step_market_info = {} # Store info after the very last step

        all_traders = buyers + sellers

        for st in range(self.num_steps):
            self.current_step = st
            for t_agent in all_traders: t_agent.current_step = st # Update agent's internal step

            # --- Action Phase ---
            # Pass current market history (e.g., last trade) to agents
            # This also captures the submitted quotes for logging and next step's state
            submitted_quotes = self._run_bid_offer_substep(r, p, st, buyers, sellers, market_history_for_agents)
            trade_info = self._run_buy_sell_substep(r, p, st, buyers, sellers, market_history_for_agents)

            # --- Reward Calculation & Update Period Stats ---
            step_rewards = defaultdict(float)
            if trade_info:
                 period_trades += 1
                 b_inc = trade_info.get('buyer_profit_inc', 0.0) or 0.0 # Handle None
                 s_inc = trade_info.get('seller_profit_inc', 0.0) or 0.0 # Handle None
                 period_profit += b_inc + s_inc
                 period_prices.append(trade_info['price'])
                 step_rewards[trade_info['buyer']] = b_inc
                 step_rewards[trade_info['seller']] = s_inc
                 self.last_trade_info_for_period = trade_info # Update tracker

            # --- Update Market History for Next Step ---
            market_history_for_agents['last_trade_info_for_period'] = self.last_trade_info_for_period
            # Store quotes submitted *this* step for use in the *next* step's state calc
            # Ensure bids/asks are lists of (agent, price) or similar structure PPO expects
            # The substep returns dicts {name: price}, convert them
            bids_list = list(sorted([(name, price) for name, price in submitted_quotes['bids'].items()], key=lambda item: item[1], reverse=True))
            asks_list = list(sorted([(name, price) for name, price in submitted_quotes['asks'].items()], key=lambda item: item[1]))
            market_history_for_agents['all_bids_this_step'] = bids_list
            market_history_for_agents['all_asks_this_step'] = asks_list

            # --- Prepare Info Dict for s_{t+1} (State observed AFTER step st) ---
            market_info_for_next_state = {
                 "step": st, "total_steps": self.num_steps, "period": p, "total_periods": self.num_periods,
                 "current_bid_info": self.current_bid_info, # Market state AFTER BA/BS substeps
                 "current_ask_info": self.current_ask_info,
                 "phibid": self.phibid, "phiask": self.phiask,
                 "last_trade_info": self.last_trade_info_for_period, # Current last trade
                 "all_bids": market_history_for_agents['all_bids_this_step'], # Quotes submitted in st
                 "all_asks": market_history_for_agents['all_asks_this_step'] # Quotes submitted in st
            }
            # Store info after the very last step for round-end calculations / final state
            if st == self.num_steps - 1:
                 last_step_market_info = market_info_for_next_state.copy()

            # --- Calculate Next State (s_{t+1}) for ALL agents ---
            # Important: calculate next state *before* calling observe_reward
            state_after_step = {}
            for agent in all_traders:
                 if hasattr(agent, '_get_state') and callable(getattr(agent, '_get_state')):
                     try:
                         # Use the finalized market info AFTER the step resolved
                         next_state = agent._get_state(market_info_for_next_state)
                         state_after_step[agent] = next_state
                     except Exception as e:
                         self.logger.error(f"Error getting state for agent {agent.name}: {e}", exc_info=True)
                         state_after_step[agent] = None # Or a default zero state

            # --- RL Observation and Potential Learning Trigger ---
            # `done` is True only at the very end of the ROUND (last period, last step)
            is_round_done = (p == self.num_periods - 1) and (st == self.num_steps - 1)
            for agent in current_round_rl_agents:
                 # Retrieve state and action taken *in this step* (st)
                 # These should have been stored inside agent during make_bid_or_ask
                 last_state = agent._current_step_state
                 action_idx = agent._current_step_action
                 reward = step_rewards.get(agent, 0.0) # Reward received *after* action
                 next_state = state_after_step.get(agent) # The s_{t+1} calculated above

                 if next_state is None and hasattr(agent, 'state_dim'):
                      # Provide a default zero state if calculation failed
                      next_state = np.zeros(agent.state_dim, dtype=np.float32)

                 # Check if agent actually took an action based on policy in this step
                 if last_state is not None and action_idx is not None:
                     if hasattr(agent, 'observe_reward') and callable(getattr(agent, 'observe_reward')):
                          # Pass the initial LSTM state for this round if needed by observe_reward
                          # (Though PPOTrader handles it internally)
                          agent.observe_reward(last_state=last_state, action_idx=action_idx,
                                               reward=reward, next_state=next_state,
                                               done=is_round_done, # Pass correct round-level done flag
                                               step_outcome=market_info_for_next_state) # Pass context if needed
                     else:
                         self.logger.warning(f"RL Agent {agent.name} is missing observe_reward method.")
                 # else: Agent didn't take a policy action this step (e.g., couldn't trade)

            # --- Logging Step ---
            log_row = {
                 "round": r, "period": p, "step": st,
                 "phibid_start": market_history_for_agents.get('phibid_start_step', self.phibid), # Placeholder if not stored
                 "phiask_start": market_history_for_agents.get('phiask_start_step', self.phiask), # Placeholder
                 "bids_submitted": submitted_quotes['bids'], # Dict {name: price}
                 "asks_submitted": submitted_quotes['asks'], # Dict {name: price}
                 "phibid_updated": self.phibid, # Value after BA substep
                 "phiask_updated": self.phiask, # Value after BA substep
                 "current_bid_price": self.current_bid_info['price'] if self.current_bid_info else None,
                 "current_bidder": self.current_bid_info['agent'].name if self.current_bid_info else None,
                 "current_ask_price": self.current_ask_info['price'] if self.current_ask_info else None,
                 "current_asker": self.current_ask_info['agent'].name if self.current_ask_info else None,
                 "trade_executed": 1 if trade_info else 0,
                 "trade_price": trade_info.get('price') if trade_info else None,
                 "trade_buyer": trade_info.get('buyer').name if trade_info and trade_info.get('buyer') else None,
                 "trade_seller": trade_info.get('seller').name if trade_info and trade_info.get('seller') else None,
                 "trade_type": trade_info.get('type') if trade_info else None,
                 "buyer_profit_inc": trade_info.get('buyer_profit_inc') if trade_info else None,
                 "seller_profit_inc": trade_info.get('seller_profit_inc') if trade_info else None,
            }
            period_step_logs.append(log_row)

        # Return results for the period
        return period_trades, period_profit, period_prices, period_step_logs, last_step_market_info

    def _run_bid_offer_substep(self, r, p, st, buyers, sellers, market_history):
        """ Agents submit bids/asks. Returns dict of submitted quotes. """
        all_traders = buyers + sellers
        for t_agent in all_traders: t_agent.current_substep = "bid_offer" # Inform agent

        # Store market state *before* quotes are submitted (for agent decisions)
        prev_bid_info = self.current_bid_info
        prev_ask_info = self.current_ask_info
        prev_phibid = self.phibid
        prev_phiask = self.phiask

        # Determine if quotes should persist from previous BS step
        # If a trade occurred, quotes were cleared. If no trade, they persist.
        prev_bid_price = prev_bid_info['price'] if prev_bid_info and not self.quote_cleared_in_last_buy_sell else None
        prev_ask_price = prev_ask_info['price'] if prev_ask_info and not self.quote_cleared_in_last_buy_sell else None

        submitted_bids = {}; potential_bids = []; submitted_asks = {}; potential_asks = []

        # Pass the relevant market history dict to agents
        history_for_make = market_history.copy() # Avoid modifying the dict passed between steps

        for agent in all_traders:
             if agent.can_trade():
                 # Call agent's decision logic, passing necessary info
                 price = agent.make_bid_or_ask(prev_bid_info, prev_ask_info, prev_phibid, prev_phiask, history_for_make)
                 if price is not None and isinstance(price, (int, float, np.number)) and self.min_price <= price <= self.max_price:
                      price_int = int(round(price))
                      if agent.is_buyer:
                          potential_bids.append((price_int, agent)) # Store (price, agent_obj)
                          submitted_bids[agent.name] = price_int # Store {name: price} for logging
                      else:
                          potential_asks.append((price_int, agent))
                          submitted_asks[agent.name] = price_int

        # --- Update Market State (phibid/phiask, current best) ---
        if submitted_bids:
            self.phibid = max(self.phibid, max(submitted_bids.values()))
        if submitted_asks:
            self.phiask = min(self.phiask, min(submitted_asks.values()))

        # Determine new best bid (must beat previous if persistent)
        valid_bids = [(pr, ag) for pr, ag in potential_bids if prev_bid_price is None or pr > prev_bid_price]
        if valid_bids:
            max_bid_price = max(b[0] for b in valid_bids)
            tied_bids = [b for b in valid_bids if b[0] == max_bid_price]
            chosen_p, chosen_a = self.auction_rng.choice(tied_bids)
            self.current_bid_info = {'price': chosen_p, 'agent': chosen_a}
        elif self.quote_cleared_in_last_buy_sell: # If no valid new bid AND quotes were cleared, clear best bid
            self.current_bid_info = None
        # Else: No valid new bid, but quotes were persistent -> keep old self.current_bid_info

        # Determine new best ask (must beat previous if persistent)
        valid_asks = [(pr, ag) for pr, ag in potential_asks if prev_ask_price is None or pr < prev_ask_price]
        if valid_asks:
            min_ask_price = min(a[0] for a in valid_asks)
            tied_asks = [a for a in valid_asks if a[0] == min_ask_price]
            chosen_p, chosen_a = self.auction_rng.choice(tied_asks)
            self.current_ask_info = {'price': chosen_p, 'agent': chosen_a}
        elif self.quote_cleared_in_last_buy_sell:
            self.current_ask_info = None
        # Else: keep old self.current_ask_info

        # Reset the flag for the next BS step
        self.quote_cleared_in_last_buy_sell = False

        # Return dict of submitted quotes {name: price} for logging
        return {"bids": submitted_bids, "asks": submitted_asks}


    def _run_buy_sell_substep(self, r, p, st, buyers, sellers, market_history):
        """ Agents holding best quotes decide to accept. Returns trade info dict or None. """
        for t_agent in buyers + sellers: t_agent.current_substep = "buy_sell" # Inform agent

        bid_info = self.current_bid_info
        ask_info = self.current_ask_info

        bid_price = bid_info['price'] if bid_info else None
        bidder = bid_info['agent'] if bid_info else None
        ask_price = ask_info['price'] if ask_info else None
        asker = ask_info['agent'] if ask_info else None

        # Check if trade is possible (quotes exist and cross)
        if bid_price is None or ask_price is None or bid_price < ask_price:
            # No trade possible, quotes persist
            self.quote_cleared_in_last_buy_sell = False # Ensure flag is false if no trade
            return None

        # Get acceptance decisions from agents holding quotes
        buy_requested, sell_requested = False, False
        # Pass the relevant market history dict to agents
        history_for_request = market_history.copy()

        if bidder and bidder.can_trade():
             # Buyer holding best bid decides whether to accept the current ask_price
             if bidder.request_buy(ask_price, bid_info, ask_info, self.phibid, self.phiask, history_for_request):
                 buy_requested = True
        if asker and asker.can_trade():
             # Seller holding best ask decides whether to accept the current bid_price
             if asker.request_sell(bid_price, bid_info, ask_info, self.phibid, self.phiask, history_for_request):
                 sell_requested = True

        # --- Resolve Trade ---
        trade_info = None; win_type = None; buyer, seller, price, exec_type = None, None, None, None

        if buy_requested and not sell_requested: win_type = 'buy_accepts_ask'
        elif not buy_requested and sell_requested: win_type = 'sell_accepts_bid'
        elif buy_requested and sell_requested:
             # Simultaneous acceptance: Randomly pick one winner (AURORA tie-break)
             win_type = self.auction_rng.choice(['buy_accepts_ask', 'sell_accepts_bid'])
             self.logger.debug(f"R{r}P{p}S{st}: Simultaneous accept! Tie break: {win_type}")

        # Determine trade parameters based on winning acceptance type
        if win_type == 'buy_accepts_ask':
            if bidder and asker: # Ensure both agents still exist
                buyer, seller, price, exec_type = bidder, asker, ask_price, win_type # Trade at ask price
        elif win_type == 'sell_accepts_bid':
            if bidder and asker:
                buyer, seller, price, exec_type = bidder, asker, bid_price, win_type # Trade at bid price

        # Execute trade if parameters are valid
        if buyer and seller and price is not None:
            # Final check: ensure both agents *still* can trade (in case state changed unexpectedly)
            if buyer.can_trade() and seller.can_trade():
                # Record trade with BOTH agents
                # record_trade returns profit increment or None if failed
                b_inc = buyer.record_trade(p, st, price)
                s_inc = seller.record_trade(p, st, price)

                # Check if both recordings were successful
                if b_inc is not None and s_inc is not None:
                    trade_info = {
                        "buyer": buyer, "seller": seller, "price": price,
                        "buyer_profit_inc": b_inc, "seller_profit_inc": s_inc,
                        "type": exec_type
                    }
                    # Trade successful: Clear quotes for next BA step
                    self.current_bid_info = None
                    self.current_ask_info = None
                    self.quote_cleared_in_last_buy_sell = True # Set flag for next BA step
                    self.logger.debug(f"    Trade successful @ {price:.2f}. B_profit={b_inc:.2f}, S_profit={s_inc:.2f}. Quotes cleared.")
                else:
                    self.logger.error(f"R{r}P{p}S{st}: Trade recording failed for B={buyer.name} (inc:{b_inc}) or S={seller.name} (inc:{s_inc}) at P={price}. State may be inconsistent.")
                    # Decide how to handle failed recording - e.g., revert state? Log error and continue?
                    # Current: Log and continue, quotes might persist incorrectly.
                    self.quote_cleared_in_last_buy_sell = False # Assume quotes persist if record failed
            else:
                self.logger.warning(f"R{r}P{p}S{st}: Trade failed final can_trade check: B={buyer.name}({buyer.tokens_left}), S={seller.name}({seller.tokens_left})")
                self.quote_cleared_in_last_buy_sell = False # No trade, quotes persist
        else:
            # No acceptance or invalid trade setup
            self.quote_cleared_in_last_buy_sell = False # No trade, quotes persist

        return trade_info