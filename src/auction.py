# auction.py
import random
import numpy as np
import logging
import time
from collections import defaultdict
import os
import inspect
from tqdm import tqdm
import csv

# Assuming utils.py and traders/base.py are in the correct path
try:
    from .utils import (compute_equilibrium, generate_sfi_components,
                       calculate_sfi_values_for_participant)
    # <<< REMOVED registry import >>>
    from .traders.base import BaseTrader # Import the updated BaseTrader
except ImportError as e:
     print(f"Error importing modules in auction.py: {e}")
     print("Ensure utils.py and traders/base.py are accessible.") # Removed registry from msg
     raise

class Auction:
    """
    Implements the SFI Double Auction mechanism.
    Modified to treat a ROUND as a single RL episode and use the updated BaseTrader.
    Handles rounds, periods, steps, quote updates, trade execution,
    and round-based RL training.
    """
    def __init__(self, config, step_log_file=None):
        """ Initializes the auction simulation environment. """
        self.config = config
        self.logger = logging.getLogger('auction')
        
        # CSV writer for streaming step logs
        self.step_log_file = step_log_file
        self.csv_writer = None
        
        # Persistent agents storage
        self.persistent_agents_created = False
        self.persistent_buyers = None
        self.persistent_sellers = None
        self.persistent_all_traders = None
        self.persistent_rl_agents = None

        # Core simulation parameters
        self.num_rounds = int(config.get("num_rounds", 10)) # Ensure integer
        self.num_periods = int(config.get("num_periods", 1))
        self.num_steps = int(config.get("num_steps", 25))
        self.num_training_rounds = int(config.get("num_training_rounds", 0))

        # Market parameters
        self.min_price = int(config.get("min_price", 1))
        self.max_price = int(config.get("max_price", 2000))
        self.gametype = config.get("gametype", 0) # For SFI value generation

        # Agent configuration (Should contain resolved classes from main.py)
        self.num_buyers = int(config.get('num_buyers', 0))
        self.num_sellers = int(config.get('num_sellers', 0))
        self.buyer_specs = config.get('buyers', []) # List of buyer specs with 'class' key
        self.seller_specs = config.get('sellers', []) # List of seller specs with 'class' key
        self.num_tokens = int(config.get('num_tokens', 1)) # Tokens per agent per period

        # RL configuration passed to agents
        self.rl_config = config # Pass entire config dict

        # Data logging structures
        self.round_stats = []       # Stores summary dict for each round
        # self.all_step_logs = []    # REMOVED - now streamed to CSV
        self.rl_training_logs = []  # Stores RL training stats dict after each round's update
        
        # Initialize CSV writer if file handle provided
        if self.step_log_file:
            # Define headers for step log CSV
            headers = [
                "round", "period", "step", "phibid_updated", "phiask_updated",
                "current_bid_price", "current_bidder", "current_ask_price", "current_asker",
                "trade_executed", "trade_price", "trade_buyer", "trade_seller", 
                "trade_type", "buyer_profit_inc", "seller_profit_inc", "last_mkt_trade_step"
            ]
            self.csv_writer = csv.DictWriter(self.step_log_file, fieldnames=headers, extrasaction='ignore')
            self.csv_writer.writeheader()

        # Market state variables (reset per period)
        self.current_bid_info = None # {'price': P, 'agent': agent_obj}
        self.current_ask_info = None # {'price': P, 'agent': agent_obj}
        self.phibid = self.min_price # Highest bid observed EVER in the period
        self.phiask = self.max_price # Lowest ask observed EVER in the period
        self.quote_cleared_in_last_buy_sell = True # Flag for quote persistence rule
        self.last_trade_info_for_period = None # Stores info of the last trade in the current period
        self.last_market_trade_step_period = -1 # Step index of last market trade (-1 if none)

        # RNG Setup
        seed_values = config.get("rng_seed_values", int(time.time()))
        seed_auction = config.get("rng_seed_auction", int(time.time()) + 1)
        self.value_rng = random.Random(seed_values)
        self.auction_rng = random.Random(seed_auction)
        self.logger.info(f"RNG Seeds: Values={seed_values}, Auction={seed_auction}")

        # Track current simulation state
        self.current_round = -1
        self.current_period = -1
        self.current_step = -1

    def _create_persistent_traders(self):
        """Create agents once for the entire simulation."""
        if self.persistent_agents_created:
            return
            
        self.logger.info("Creating persistent agents for entire simulation...")
        self.persistent_buyers = []
        self.persistent_sellers = []
        sfi_components = generate_sfi_components(self.gametype, self.min_price, self.max_price, self.value_rng)
        game_params = {
             "num_rounds": self.num_rounds, "num_periods": self.num_periods,
             "num_steps": self.num_steps, "min_price": self.min_price,
             "max_price": self.max_price, "num_buyers": self.num_buyers,
             "num_sellers": self.num_sellers, "gametype": self.gametype,
             "num_tokens": self.num_tokens,
        }

        # Create Buyers
        if self.num_buyers > 0 and self.buyer_specs:
            for i in range(self.num_buyers):
                spec_idx = i % len(self.buyer_specs)
                spec = self.buyer_specs[spec_idx]
                # --- <<< USE PRE-RESOLVED CLASS >>> ---
                TraderCls = spec.get("class") # Get class object
                if not TraderCls:
                    self.logger.error(f"Config error: Buyer spec {i} missing resolved 'class'. Spec: {spec}")
                    raise ValueError(f"Missing trader class for buyer {i}")
                # --- <<< END CHANGE >>> ---
                init_args = spec.get("init_args", {})
                name = f"B{i}"
                role_l = 1
                values = calculate_sfi_values_for_participant(name, role_l, self.num_tokens, sfi_components, self.min_price, self.max_price, self.value_rng)

                try:
                    sig = inspect.signature(TraderCls.__init__)
                    merged_args = init_args.copy()
                    if 'rl_config' in sig.parameters: merged_args['rl_config'] = self.rl_config
                    trader_instance = TraderCls(name=name, is_buyer=True, private_values=values, **merged_args)
                    trader_instance.update_game_params(game_params)
                    self.persistent_buyers.append(trader_instance)
                except Exception as e:
                    self.logger.error(f"Failed to create buyer {name} (type: {TraderCls.__name__}): {e}", exc_info=True) # Use class name in log
                    raise

        # Create Sellers (Similar changes)
        if self.num_sellers > 0 and self.seller_specs:
            for j in range(self.num_sellers):
                spec_idx = j % len(self.seller_specs)
                spec = self.seller_specs[spec_idx]
                # --- <<< USE PRE-RESOLVED CLASS >>> ---
                TraderCls = spec.get("class")
                if not TraderCls:
                    self.logger.error(f"Config error: Seller spec {j} missing resolved 'class'. Spec: {spec}")
                    raise ValueError(f"Missing trader class for seller {j}")
                # --- <<< END CHANGE >>> ---
                init_args = spec.get("init_args", {})
                name = f"S{j}"
                role_l = 2
                costs = calculate_sfi_values_for_participant(name, role_l, self.num_tokens, sfi_components, self.min_price, self.max_price, self.value_rng)
                try:
                    sig = inspect.signature(TraderCls.__init__)
                    merged_args = init_args.copy()
                    if 'rl_config' in sig.parameters: merged_args['rl_config'] = self.rl_config
                    trader_instance = TraderCls(name=name, is_buyer=False, private_values=costs, **merged_args)
                    trader_instance.update_game_params(game_params)
                    self.persistent_sellers.append(trader_instance)
                except Exception as e:
                    self.logger.error(f"Failed to create seller {name} (type: {TraderCls.__name__}): {e}", exc_info=True)
                    raise

        # Log agent creation summary
        buyer_strats = [t.strategy for t in self.persistent_buyers] if self.persistent_buyers else []
        seller_strats = [t.strategy for t in self.persistent_sellers] if self.persistent_sellers else []
        self.logger.info(f"Created PERSISTENT agents: {len(self.persistent_buyers)} buyers ({buyer_strats}), {len(self.persistent_sellers)} sellers ({seller_strats}).")
        
        self.persistent_all_traders = self.persistent_buyers + self.persistent_sellers
        self.persistent_rl_agents = [t for t in self.persistent_all_traders if hasattr(t, 'logic')]
        self.persistent_agents_created = True
        
    def _reset_traders_for_round(self, r):
        """Reset existing persistent agents with new values for the round."""
        # Generate new SFI values for this round
        sfi_components = generate_sfi_components(self.gametype, self.min_price, self.max_price, self.value_rng)
        
        # Update buyer values
        for i, buyer in enumerate(self.persistent_buyers):
            name = buyer.name
            role_l = 1
            values = calculate_sfi_values_for_participant(name, role_l, self.num_tokens, sfi_components, self.min_price, self.max_price, self.value_rng)
            buyer.private_values = values
            
        # Update seller costs
        for j, seller in enumerate(self.persistent_sellers):
            name = seller.name
            role_l = 2
            costs = calculate_sfi_values_for_participant(name, role_l, self.num_tokens, sfi_components, self.min_price, self.max_price, self.value_rng)
            seller.private_values = costs
            
        if r == 0 or r % 100 == 0:
            self.logger.debug(f"Round {r}: Reset values for persistent agents")
        
        current_round_rl_agents = self.persistent_rl_agents
        round_initial_lstm_states = {}

        # --- ROUND START RESETS ---
        for t in self.persistent_all_traders:
            t.reset_for_new_round() # BaseTrader method now handles resetting round lists, profits etc.

            # Identify RL agents and set mode / store initial LSTM state
            if hasattr(t, 'logic') and hasattr(t.logic, 'is_training'): # Check if it's an RL agent
                is_training_round = r < self.num_training_rounds
                if hasattr(t, 'set_mode'): t.set_mode(training=is_training_round)
                current_round_rl_agents.append(t)
                # Store initial LSTM state (logic remains same)
                if hasattr(t, 'current_lstm_state') and t.current_lstm_state is not None:
                     round_initial_lstm_states[t] = (t.current_lstm_state[0].clone().detach(), t.current_lstm_state[1].clone().detach())
                     # self.logger.debug(f"Stored initial LSTM state for {t.name} at R{r} start.")
                else: # Handle case where state might need generation
                     if hasattr(t, 'logic') and hasattr(t.logic, 'agent') and hasattr(t.logic.agent, 'get_initial_state'):
                          try:
                              initial_state = t.logic.agent.get_initial_state(batch_size=1, device=t.logic.device)
                              round_initial_lstm_states[t] = (initial_state[0].clone().detach(), initial_state[1].clone().detach())
                              t.current_lstm_state = initial_state
                              # self.logger.debug(f"Generated initial LSTM state for {t.name} at R{r} start.")
                          except Exception as lstm_e: self.logger.error(f"Failed to get initial LSTM state for {t.name}: {lstm_e}")
                     # else: self.logger.warning(f"Could not get/store initial LSTM state for RL agent {t.name} at R{r} start.")

        return self.persistent_buyers, self.persistent_sellers, self.persistent_all_traders, current_round_rl_agents, round_initial_lstm_states

    def run_auction(self):
        """ Main auction loop. Each round is one RL episode. """
        self.logger.info(f"Starting Auction: {self.config['experiment_name']}")
        self.logger.info(f"Params: R={self.num_rounds}, P={self.num_periods}, S={self.num_steps}, TrainR={self.num_training_rounds}")

        unique_rl_traders_all_rounds = set() # Keep track of unique RL agents for saving

        use_tqdm = self.logger.getEffectiveLevel() >= logging.INFO
        rounds_iterable = range(self.num_rounds)
        rounds_iterator = tqdm(rounds_iterable, desc="Auction Rounds", unit="round", dynamic_ncols=True, disable=not use_tqdm)

        # Create persistent agents on first call
        if not self.persistent_agents_created:
            self._create_persistent_traders()
        
        # --- Main Round Loop ---
        for r in rounds_iterator:
            self.current_round = r
            is_training_round = r < self.num_training_rounds
            mode = "Training" if is_training_round else "Evaluation"

            # Reset agent values for new round (but keep neural networks)
            buyers, sellers, all_traders, current_round_rl_agents, round_initial_lstm_states = \
                self._reset_traders_for_round(r)

            for agent in current_round_rl_agents: unique_rl_traders_all_rounds.add(agent)

            # Calculate Equilibrium for the round
            all_buyer_values = sorted([v for b in buyers for v in b.private_values], reverse=True)
            all_seller_costs = sorted([c for s in sellers for c in s.private_values])
            eq_q, eq_p_mid, eq_surplus = compute_equilibrium(all_buyer_values, all_seller_costs)
            self.logger.debug(f"R{r}: Equilibrium Calculated - Q={eq_q}, P_mid={eq_p_mid:.2f}, Max Theoretical Surplus={eq_surplus:.2f}")

            # Initialize round statistics collectors
            round_total_trades = 0
            round_trade_prices = []
            round_step_logs = []
            last_step_final_market_info = {}

            # --- Period Loop ---
            for p in range(self.num_periods):
                self.current_period = p
                # Reset agent period-specific state (BaseTrader handles common parts)
                for t in all_traders:
                     # Pass current round and period index to reset
                     t.reset_for_new_period(r, p) # <<< UPDATED CALL

                # Reset period-specific market state variables
                self.current_bid_info = None
                self.current_ask_info = None
                self.phibid = self.min_price
                self.phiask = self.max_price
                self.quote_cleared_in_last_buy_sell = True
                self.last_trade_info_for_period = None
                self.last_market_trade_step_period = -1 # Reset last market trade step tracker

                # Run steps for the period
                period_trades, period_profit, period_prices_this_period, period_step_logs, last_step_final_market_info = \
                    self._run_period_steps(r, p, buyers, sellers, current_round_rl_agents, round_initial_lstm_states)

                # Accumulate round totals
                round_total_trades += period_trades
                round_trade_prices.extend(period_prices_this_period)
                # Don't accumulate step logs in memory anymore
                # round_step_logs.extend(period_step_logs)

                # --- Inter-Period Updates (Call agent hook) ---
                # Pass the list of trade prices from the period that just ended
                for t in all_traders:
                     t.update_end_of_period(period_prices_this_period) # <<< CALL AGENT HOOK

            # --- End of Round (End of RL Episode) ---
            # Collect RL stats (logic remains same)
            round_rl_stats = []
            if current_round_rl_agents:
                 for agent in current_round_rl_agents:
                      if hasattr(agent, 'get_last_episode_stats'):
                           stats = agent.get_last_episode_stats()
                           if stats:
                                stats['round'] = r; stats['period'] = -1; stats['agent_name'] = agent.name
                                round_rl_stats.append(stats)
            if round_rl_stats: self.rl_training_logs.extend(round_rl_stats)

            # --- Calculate Round Summary Stats ---
            self.logger.debug(f"--- R{r}: Calculating End-of-Round Stats ---")
            round_total_profit = 0.0
            bot_details_round = []
            role_strat_perf_round = defaultdict(lambda: {"profit": 0.0, "count": 0, "trades": 0})

            for t in all_traders:
                role = "buyer" if t.is_buyer else "seller"
                strat = t.strategy
                # Use agent's internal round profit accumulator
                agent_round_profit = t.current_round_profit
                round_total_profit += agent_round_profit
                # Calculate total trades across all periods in the round
                num_trades_agent_round = sum(t.tradelist_round) if t.tradelist_round else 0
                # self.logger.debug(f" R{r}: Agent {t.name} ({strat}, {role}) - Final Round Profit: {agent_round_profit:.2f}, RoundTrades: {num_trades_agent_round} (From list: {t.tradelist_round})")

                bot_details_round.append({
                    "name": t.name, "role": role, "strategy": strat,
                    "profit": agent_round_profit,
                    "trades": num_trades_agent_round, # Use round total trades
                    "values_costs": list(t.private_values) # Store original values/costs
                })
                role_strat_perf_round[(role, strat)]["profit"] += agent_round_profit
                role_strat_perf_round[(role, strat)]["count"] += 1
                role_strat_perf_round[(role, strat)]["trades"] += num_trades_agent_round

            # --- Efficiency Calculation (remains same conceptually) ---
            total_theoretical_surplus_round = eq_surplus * self.num_periods
            calculated_efficiency = 0.0
            if abs(total_theoretical_surplus_round) < 1e-9:
                calculated_efficiency = 1.0 if abs(round_total_profit) < 1e-9 else 0.0
                # Add logging if desired
            else:
                calculated_efficiency = round_total_profit / total_theoretical_surplus_round
                # Add warning logs if desired
            efficiency = calculated_efficiency
            # self.logger.debug(f"R{r}: Final Calculated Efficiency (vs round total surplus): {efficiency:.4f}")

            # Other round stats
            avg_p = np.mean(round_trade_prices) if round_trade_prices else None
            adiff_p = abs(avg_p - eq_p_mid) if avg_p is not None and eq_p_mid is not None else None
            # Compare round total trades vs equilibrium quantity * num periods? Or just vs eq_q?
            # Let's keep it vs eq_q for consistency with original intention, though maybe less meaningful.
            adiff_q = abs(round_total_trades - eq_q) if eq_q is not None else None
            role_strat_perf_dict = {str(k): v for k, v in role_strat_perf_round.items()} # Convert keys

            # Store round statistics
            rstats = {
                "round": r, "mode": mode, "num_periods": self.num_periods, "num_steps": self.num_steps,
                "eq_q": eq_q, "eq_p": eq_p_mid, "eq_surplus": eq_surplus,
                "actual_trades": round_total_trades, "actual_total_profit": round_total_profit,
                "market_efficiency": efficiency,
                "avg_price": avg_p,
                "abs_diff_price": adiff_p, "abs_diff_quantity": adiff_q,
                "buyer_vals": all_buyer_values, "seller_vals": all_seller_costs,
                "role_strat_perf": role_strat_perf_dict,
                "bot_details": bot_details_round
            }
            self.round_stats.append(rstats)
            # Step logs are now streamed to CSV, not stored in memory
            # self.all_step_logs.extend(round_step_logs) # Append step logs

            # --- TQDM Update (logic remains same) ---
            if use_tqdm:
                rl_stats_for_tqdm = {}
                round_total_profit_rl = sum(t.current_round_profit for t in current_round_rl_agents)
                if current_round_rl_agents:
                     latest_round_stats = [s for s in round_rl_stats if s.get('round') == r]
                     avg_pl = np.nanmean([s.get('avg_policy_loss', np.nan) for s in latest_round_stats]) if latest_round_stats else np.nan
                     avg_vl = np.nanmean([s.get('avg_value_loss', np.nan) for s in latest_round_stats]) if latest_round_stats else np.nan
                     rl_stats_for_tqdm = {"Mode": mode[0], "RLProfit": f"{round_total_profit_rl:.1f}",
                                          "AvgPL": f"{avg_pl:.3f}" if not np.isnan(avg_pl) else "N/A",
                                          "AvgVL": f"{avg_vl:.3f}" if not np.isnan(avg_vl) else "N/A"}
                display_eff = np.clip(efficiency, 0.0, 1.0)
                combined_stats = {"Eff": f"{display_eff:.3f}"}
                combined_stats.update(rl_stats_for_tqdm)
                rounds_iterator.set_postfix(combined_stats, refresh=False)

            # --- EXPLICIT MEMORY CLEANUP ---
            # Clear buffers for all RL agents after round is complete
            if current_round_rl_agents:
                for agent in current_round_rl_agents:
                    if hasattr(agent, 'logic') and hasattr(agent.logic, 'clear_buffer'):
                        agent.logic.clear_buffer()
                        self.logger.debug(f"Cleared buffer for agent {agent.name}")
            
        # --- End of Round Loop ---
        if use_tqdm: rounds_iterator.close()

        # --- Save Models (logic remains same) ---
        if self.config.get("save_rl_model", False) and unique_rl_traders_all_rounds:
             save_dir = os.path.join(self.config.get("experiment_dir", "experiments"),
                                      self.config["experiment_name"], "models")
             os.makedirs(save_dir, exist_ok=True)
             self.logger.info(f"Saving final RL models for {len(unique_rl_traders_all_rounds)} unique agents to {save_dir}...")
             saved_count = 0
             for agent_trader in unique_rl_traders_all_rounds:
                 model_path_prefix = os.path.join(save_dir, f"{agent_trader.strategy}_agent_{agent_trader.name}")
                 try:
                     if hasattr(agent_trader, 'save_model') and callable(getattr(agent_trader, 'save_model')):
                         agent_trader.save_model(model_path_prefix)
                         saved_count += 1
                     else: self.logger.warning(f"Agent {agent_trader.name} has no save_model method.")
                 except Exception as e: self.logger.error(f"Save failed for {agent_trader.name}: {e}", exc_info=True)
             self.logger.info(f"Attempted to save models for {saved_count} agents.")
        elif self.config.get("save_rl_model", False): self.logger.warning("RL model saving enabled, but no RL agents tracked.")

        self.logger.info("===== SFI Auction Run Finished =====")


    def _run_period_steps(self, r, p, buyers, sellers, current_round_rl_agents, round_initial_lstm_states):
        """ Runs steps within a period. Returns period stats and final market info. """
        period_step_logs = []
        period_trades = 0
        period_profit = 0.0
        period_prices = []
        # Define market_history dict structure *before* loop
        market_history_for_agents = {
            'last_trade_info_for_period': None, # Updated if trade occurs
            'lasttime': -1,                     # Step index of last market trade
            'all_bids_this_step': [],
            'all_asks_this_step': []
        }
        last_step_final_market_info = {}
        all_traders = buyers + sellers

        if self.num_steps <= 0:
            self.logger.warning(f"R{r}P{p}: num_steps is {self.num_steps}. Skipping step loop.")
            return period_trades, period_profit, period_prices, period_step_logs, last_step_final_market_info

        # --- Main Step Loop ---
        for st in range(self.num_steps):
            self.current_step = st
            # Update agent's internal step counter
            for t_agent in all_traders: t_agent.current_step = st # Ensure agent knows current step

            # --- Action Phase ---
            # Update history with last trade step *before* passing to agents
            market_history_for_agents['lasttime'] = self.last_market_trade_step_period
            submitted_quotes = self._run_bid_offer_substep(r, p, st, buyers, sellers, market_history_for_agents)
            trade_info = self._run_buy_sell_substep(r, p, st, buyers, sellers, market_history_for_agents)

            # --- Reward Calculation & Update Period Stats ---
            step_rewards = defaultdict(float)
            if trade_info: # If a trade occurred
                 period_trades += 1
                 b_inc = trade_info.get('buyer_profit_inc', 0.0) or 0.0
                 s_inc = trade_info.get('seller_profit_inc', 0.0) or 0.0
                 period_profit += b_inc + s_inc
                 period_prices.append(trade_info['price'])
                 step_rewards[trade_info['buyer']] = b_inc
                 step_rewards[trade_info['seller']] = s_inc
                 # Update market state trackers
                 self.last_trade_info_for_period = trade_info
                 self.last_market_trade_step_period = st # Update last market trade step

            # --- Update Market History for *Next* Step ---
            market_history_for_agents['last_trade_info_for_period'] = self.last_trade_info_for_period
            # market_history_for_agents['lasttime'] updated at start of next loop iteration
            bids_list = list(sorted([(name, price) for name, price in submitted_quotes['bids'].items()], key=lambda item: item[1], reverse=True))
            asks_list = list(sorted([(name, price) for name, price in submitted_quotes['asks'].items()], key=lambda item: item[1]))
            market_history_for_agents['all_bids_this_step'] = bids_list
            market_history_for_agents['all_asks_this_step'] = asks_list

            # --- Prepare Info Dict for s_{t+1} ---
            # This is the context passed to RL agent's observe_reward
            market_info_for_next_state = {
                 "step": st, "total_steps": self.num_steps, "period": p, "total_periods": self.num_periods,
                 "current_bid_info": self.current_bid_info, # Market state AFTER BA/BS substeps
                 "current_ask_info": self.current_ask_info,
                 "phibid": self.phibid, "phiask": self.phiask,
                 "last_trade_info": self.last_trade_info_for_period, # Current last trade
                 "lasttime": self.last_market_trade_step_period, # Include last market trade step
                 "all_bids": market_history_for_agents['all_bids_this_step'],
                 "all_asks": market_history_for_agents['all_asks_this_step']
            }
            if st == self.num_steps - 1:
                 last_step_final_market_info = market_info_for_next_state.copy()

            # --- Calculate Next State (s_{t+1}) for ALL agents ---
            state_after_step = {}
            for agent in all_traders:
                 if hasattr(agent, '_get_state') and callable(getattr(agent, '_get_state')):
                     try:
                         next_state = agent._get_state(market_info_for_next_state)
                         state_after_step[agent] = next_state
                     except Exception as e:
                         self.logger.error(f"Error getting state for agent {agent.name} at R{r}P{p}S{st}: {e}", exc_info=True)
                         state_after_step[agent] = None

            # --- RL Observation and Potential Learning Trigger ---
            is_round_done = (p == self.num_periods - 1) and (st == self.num_steps - 1)
            for agent in all_traders: # Send observation to ALL agents (ZIP needs it too)
                 last_state = agent._current_step_state
                 action_idx = agent._current_step_action
                 reward = step_rewards.get(agent, 0.0)
                 next_state = state_after_step.get(agent)
                 if next_state is None and hasattr(agent, 'state_dim'): next_state = np.zeros(agent.state_dim, dtype=np.float32)

                 # Call observe_reward if the agent implements it (used by ZIP and RL)
                 if hasattr(agent, 'observe_reward') and callable(getattr(agent, 'observe_reward')):
                      # Note: ZIP doesn't use last_state/action_idx/reward/next_state for learning,
                      # but RL agents might. step_outcome is primary for ZIP.
                      agent.observe_reward(last_state=last_state, action_idx=action_idx,
                                           reward=reward, next_state=next_state,
                                           done=is_round_done,
                                           step_outcome=market_info_for_next_state) # Pass context

            # --- Logging Step ---
            log_row = {
                 "round": r, "period": p, "step": st,
                 "phibid_updated": self.phibid, "phiask_updated": self.phiask,
                 "bids_submitted": submitted_quotes['bids'], "asks_submitted": submitted_quotes['asks'],
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
                 "last_mkt_trade_step": self.last_market_trade_step_period, # Log the last market trade step
            }
            period_step_logs.append(log_row)
            
            # Stream to CSV if writer available
            if self.csv_writer:
                self.csv_writer.writerow(log_row)
                # Flush periodically to ensure data is written
                if st % 100 == 0:
                    self.step_log_file.flush()
        # --- End of Step Loop ---

        return period_trades, period_profit, period_prices, period_step_logs, last_step_final_market_info

    # --- _run_bid_offer_substep (remains largely the same, uses market_history) ---
    def _run_bid_offer_substep(self, r, p, st, buyers, sellers, market_history):
        """ Agents submit bids/asks. Returns dict of submitted quotes. """
        all_traders = buyers + sellers
        for t_agent in all_traders: t_agent.current_substep = "bid_offer"

        prev_bid_info = self.current_bid_info
        prev_ask_info = self.current_ask_info
        prev_phibid = self.phibid
        prev_phiask = self.phiask

        prev_bid_price = prev_bid_info['price'] if prev_bid_info and not self.quote_cleared_in_last_buy_sell else None
        prev_ask_price = prev_ask_info['price'] if prev_ask_info and not self.quote_cleared_in_last_buy_sell else None

        submitted_bids = {} # {name: price}
        potential_bids = [] # [(price, agent_obj)]
        submitted_asks = {} # {name: price}
        potential_asks = [] # [(price, agent_obj)]

        # Pass the *current* market history to agents for decision making
        history_for_make = market_history.copy()

        for agent in all_traders:
             if agent.can_trade():
                 price = agent.make_bid_or_ask(prev_bid_info, prev_ask_info, prev_phibid, prev_phiask, history_for_make) # Pass history
                 if price is not None and isinstance(price, (int, float, np.number)) and self.min_price <= price <= self.max_price:
                      price_int = int(round(price))
                      if agent.is_buyer:
                          potential_bids.append((price_int, agent))
                          submitted_bids[agent.name] = price_int
                      else: # Seller
                          potential_asks.append((price_int, agent))
                          submitted_asks[agent.name] = price_int

        # --- Update Market State (phibid/phiask, current best) ---
        if submitted_bids: self.phibid = max(self.phibid, max(submitted_bids.values()))
        if submitted_asks: self.phiask = min(self.phiask, min(submitted_asks.values()))

        valid_bids = [(pr, ag) for pr, ag in potential_bids if prev_bid_price is None or pr > prev_bid_price]
        if valid_bids:
            max_bid_price = max(b[0] for b in valid_bids)
            tied_bids = [b for b in valid_bids if b[0] == max_bid_price]
            chosen_p, chosen_a = self.auction_rng.choice(tied_bids)
            self.current_bid_info = {'price': chosen_p, 'agent': chosen_a}
        elif self.quote_cleared_in_last_buy_sell: self.current_bid_info = None

        valid_asks = [(pr, ag) for pr, ag in potential_asks if prev_ask_price is None or pr < prev_ask_price]
        if valid_asks:
            min_ask_price = min(a[0] for a in valid_asks)
            tied_asks = [a for a in valid_asks if a[0] == min_ask_price]
            chosen_p, chosen_a = self.auction_rng.choice(tied_asks)
            self.current_ask_info = {'price': chosen_p, 'agent': chosen_a}
        elif self.quote_cleared_in_last_buy_sell: self.current_ask_info = None

        self.quote_cleared_in_last_buy_sell = False # Reset flag

        return {"bids": submitted_bids, "asks": submitted_asks}


    # --- _run_buy_sell_substep (Pass market_history, Update record_trade call) ---
    def _run_buy_sell_substep(self, r, p, st, buyers, sellers, market_history):
        """ Agents holding best quotes decide to accept. Returns trade info dict or None. """
        for t_agent in buyers + sellers: t_agent.current_substep = "buy_sell"

        bid_info = self.current_bid_info
        ask_info = self.current_ask_info
        bid_price = bid_info['price'] if bid_info else None
        bidder = bid_info['agent'] if bid_info else None
        ask_price = ask_info['price'] if ask_info else None
        asker = ask_info['agent'] if ask_info else None

        if bid_price is None or ask_price is None or bid_price < ask_price:
            self.quote_cleared_in_last_buy_sell = False
            return None

        buy_requested, sell_requested = False, False
        # Pass the *current* market history to agents
        history_for_request = market_history.copy()

        if bidder and bidder.can_trade():
             if bidder.request_buy(ask_price, bid_info, ask_info, self.phibid, self.phiask, history_for_request): # Pass history
                 buy_requested = True
        if asker and asker.can_trade():
             if asker.request_sell(bid_price, bid_info, ask_info, self.phibid, self.phiask, history_for_request): # Pass history
                 sell_requested = True

        # --- Resolve Trade ---
        trade_info = None; win_type = None; buyer, seller, price, exec_type = None, None, None, None

        if buy_requested and not sell_requested: win_type = 'buy_accepts_ask'
        elif not buy_requested and sell_requested: win_type = 'sell_accepts_bid'
        elif buy_requested and sell_requested:
             win_type = self.auction_rng.choice(['buy_accepts_ask', 'sell_accepts_bid'])
             self.logger.debug(f"R{r}P{p}S{st}: Simultaneous accept! Tie break: {win_type}")

        if win_type == 'buy_accepts_ask':
            if bidder and asker: buyer, seller, price, exec_type = bidder, asker, ask_price, win_type
        elif win_type == 'sell_accepts_bid':
            if bidder and asker: buyer, seller, price, exec_type = bidder, asker, bid_price, win_type

        if buyer and seller and price is not None:
            if buyer.can_trade() and seller.can_trade():
                # --- UPDATED record_trade CALL ---
                b_inc = buyer.record_trade(price)
                s_inc = seller.record_trade(price)
                # --- END UPDATE ---

                if b_inc is not None and s_inc is not None:
                    trade_info = {
                        "buyer": buyer, "seller": seller, "price": price,
                        "buyer_profit_inc": b_inc, "seller_profit_inc": s_inc,
                        "type": exec_type,
                        "step": st # Add step index to trade info
                    }
                    self.current_bid_info = None; self.current_ask_info = None
                    self.quote_cleared_in_last_buy_sell = True
                    # self.logger.debug(f"    Trade successful @ {price:.2f}. B={buyer.name}(+{b_inc:.2f}), S={seller.name}(+{s_inc:.2f}). Quotes cleared.")
                else:
                    self.logger.error(f"R{r}P{p}S{st}: Trade recording failed for B={buyer.name} (inc:{b_inc}) or S={seller.name} (inc:{s_inc}) at P={price}. State inconsistent.")
                    self.quote_cleared_in_last_buy_sell = False
            else:
                self.logger.warning(f"R{r}P{p}S{st}: Trade failed final can_trade check: B={buyer.name}(can_trade={buyer.can_trade()}), S={seller.name}(can_trade={seller.can_trade()})")
                self.quote_cleared_in_last_buy_sell = False
        else:
            self.quote_cleared_in_last_buy_sell = False

        return trade_info