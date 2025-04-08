# auction.py
import random
import numpy as np
import logging
import time
from collections import defaultdict
import os
import inspect # To check constructor signature
from tqdm import tqdm # Import tqdm

# Assuming utils.py, traders/registry.py, traders/base.py exist in correct relative paths
from utils import (compute_equilibrium, generate_sfi_components,
                   calculate_sfi_values_for_participant)
from traders.registry import get_trader_class
from traders.base import BaseTrader # Import BaseTrader for type hints
from traders.ql import DQNLogic # Specific import to check agent type

class Auction:
    """
    Implements the simplified AURORA Double Auction mechanism based on SFI rules.
    Manages rounds, periods, steps, quote updates, and trade execution.
    Handles episodic RL training across multi-period rounds with tqdm monitoring.
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
        self.rl_config = config # Pass the whole config for RL params etc.

        self.round_stats = []
        self.all_step_logs = []

        # Market state variables
        self.current_bid_info = None # Stores {'price': int, 'agent': BaseTrader}
        self.current_ask_info = None # Stores {'price': int, 'agent': BaseTrader}
        self.phibid = self.min_price # Highest bid seen in the period
        self.phiask = self.max_price # Lowest ask seen in the period
        self.quote_cleared_in_last_buy_sell = True # Track if quotes need refreshing

        # RNG Setup
        seed_values = config.get("rng_seed_values", int(time.time()))
        seed_auction = config.get("rng_seed_auction", int(time.time()) + 1)
        self.value_rng = random.Random(seed_values)
        self.auction_rng = random.Random(seed_auction) # For tie-breaking etc.
        self.logger.info(f"RNG Seeds: Values={seed_values}, Auction={seed_auction}")
        # Note: We don't seed the global `random` here, rely on instances

        self.current_round = -1
        self.current_period = -1
        self.current_step = -1
        self.market_history_for_agents = None # Placeholder for potential future history features

    def _create_traders_for_round(self, r):
        """Instantiate buyer and seller agents for the start of a round."""
        buyers = []
        sellers = []
        N = self.config['num_tokens'] # Tokens per agent per period

        # Generate common SFI value components for this round
        sfi_components = generate_sfi_components(self.gametype, self.min_price, self.max_price, self.value_rng)

        # Create Buyers
        for i in range(self.num_buyers):
            spec = self.buyer_specs[i % len(self.buyer_specs)] # Cycle through specs if needed
            trader_type = spec["type"]
            init_args = spec.get("init_args", {})
            name = f"B{i}"
            # Use name to derive ID for SFI value calc, trader gets name
            role_l = 1 # SFI role index for buyer

            values = calculate_sfi_values_for_participant(name, role_l, N, sfi_components, self.min_price, self.max_price, self.value_rng)
            TraderCls = get_trader_class(trader_type, True) # Get Buyer class

            # Instantiate with RL config if the class expects it
            sig = inspect.signature(TraderCls.__init__)
            if 'rl_config' in sig.parameters:
                trader_instance = TraderCls(name, True, values, rl_config=self.rl_config, **init_args)
            else:
                trader_instance = TraderCls(name, True, values, **init_args)

            trader_instance.update_market_params(self.min_price, self.max_price)
            buyers.append(trader_instance)

        # Create Sellers
        for j in range(self.num_sellers):
            spec = self.seller_specs[j % len(self.seller_specs)]
            trader_type = spec["type"]
            init_args = spec.get("init_args", {})
            name = f"S{j}"
            # Use name for ID
            role_l = 2 # SFI role index for seller

            costs = calculate_sfi_values_for_participant(name, role_l, N, sfi_components, self.min_price, self.max_price, self.value_rng)
            TraderCls = get_trader_class(trader_type, False) # Get Seller class

            sig = inspect.signature(TraderCls.__init__)
            if 'rl_config' in sig.parameters:
                trader_instance = TraderCls(name, False, costs, rl_config=self.rl_config, **init_args)
            else:
                trader_instance = TraderCls(name, False, costs, **init_args)

            trader_instance.update_market_params(self.min_price, self.max_price)
            sellers.append(trader_instance)

        # Reduced logging: Log only once per N rounds, or if DEBUG is enabled
        if r % 100 == 0 or self.logger.isEnabledFor(logging.DEBUG):
             self.logger.info(f"Round {r}: Created {len(buyers)} buyers, {len(sellers)} sellers using Gametype {self.gametype}.")
        return buyers, sellers

    def run_auction(self):
        """Main loop to run the auction for the configured number of rounds."""
        self.logger.info(f"Starting SFI Auction Replication: {self.config['experiment_name']}")
        self.logger.info(f"Params: Rounds={self.num_rounds}, Periods={self.num_periods}, Steps={self.num_steps}")
        self.logger.info(f"RL Settings: Training Rounds={self.num_training_rounds}, Agent Type={self.config.get('rl_agent_type', 'N/A')}")

        rl_agents_for_saving = [] # Track unique RL agents *after* training for final save

        # Wrap the rounds loop with tqdm for progress monitoring
        rounds_iterator = tqdm(range(self.num_rounds), desc="Auction Rounds", unit="round", dynamic_ncols=True)
        for r in rounds_iterator:
            self.current_round = r
            is_training_round = r < self.num_training_rounds
            mode = "Training" if is_training_round else "Evaluation"
            # self.logger.info(f"\n===== Round {r} ({mode}) Start =====") # Reduced logging

            buyers, sellers = self._create_traders_for_round(r)
            all_traders = buyers + sellers
            current_round_rl_agents = [] # RL agents participating in this round
            unique_rl_traders_this_round = [] # Track unique agents for stats reporting

            # Reset agents for the new round and set training/evaluation mode
            for t in all_traders:
                t.reset_for_new_round()
                # Identify RL agents specifically (check for DQNLogic or similar)
                if hasattr(t, 'agent') and isinstance(t.agent, DQNLogic):
                    if hasattr(t, 'set_mode'): t.set_mode(training=is_training_round)
                    current_round_rl_agents.append(t) # Active RL agents this round
                    if t not in unique_rl_traders_this_round:
                        unique_rl_traders_this_round.append(t)
                    # Collect unique agents post-training for saving model
                    if not is_training_round and t not in rl_agents_for_saving:
                         rl_agents_for_saving.append(t)

            # Calculate equilibrium for the round (based on initial values)
            all_buyer_values = sorted([v for b in buyers for v in b.private_values], reverse=True)
            all_seller_costs = sorted([v for s in sellers for v in s.private_values])
            eq_q, eq_p_mid, eq_surplus = compute_equilibrium(all_buyer_values, all_seller_costs)
            # self.logger.info(f"Round {r}: Eq Q={eq_q}, Eq Price={eq_p_mid:.2f}, Eq Surplus={eq_surplus:.2f}")

            round_total_trades = 0
            round_trade_prices = []
            round_step_logs = []
            round_total_profit_rl = 0.0 # Track RL profit specifically for tqdm monitoring

            # Loop through periods within the round
            for p in range(self.num_periods):
                self.current_period = p
                # self.logger.info(f"--- Period {p} Start (R{r}) ---") # Reduced logging
                for t in all_traders: t.reset_for_new_period(self.num_steps, r, p)

                # Reset market state for the period
                self.current_bid_info = None; self.current_ask_info = None
                self.phibid = self.min_price; self.phiask = self.max_price
                self.quote_cleared_in_last_buy_sell = True

                # Run steps within the period
                period_trades, period_profit, period_prices, period_step_logs_for_period = \
                    self._run_period_steps(r, p, buyers, sellers, is_training_round, current_round_rl_agents)

                # Accumulate period results to round totals
                round_total_trades += period_trades
                round_trade_prices.extend(period_prices)
                round_step_logs.extend(period_step_logs_for_period)
                # self.logger.info(f"--- Period {p} End (R{r}). Trades: {period_trades}, Period Profit: {period_profit:.2f} ---") # Reduced logging

            # End of Round: Signal 'done' to RL agents and gather stats for monitoring
            rl_stats_for_tqdm = {}
            if unique_rl_traders_this_round: # If there were RL agents in this round
                agent_avg_rewards = []
                agent_epsilons = []
                for rl_agent in unique_rl_traders_this_round: # Use unique list
                    # Ensure final state observation and 'done' signal is sent
                    if hasattr(rl_agent, 'observe_reward'):
                        final_state = None
                        if hasattr(rl_agent, '_get_state'):
                            final_state = rl_agent._get_state(self.current_bid_info, self.current_ask_info, self.phibid, self.phiask)
                        # Pass None for last_state/action as it's just a 'done' signal
                        rl_agent.observe_reward(last_state=None, action_idx=None, reward=0.0, next_state=final_state, done=True)

                    # Get stats for tqdm monitoring
                    if hasattr(rl_agent, 'get_last_episode_stats'):
                        stats = rl_agent.get_last_episode_stats() # This also clears internal reward list
                        agent_avg_rewards.append(stats.get('avg_reward', 0.0))
                        agent_epsilons.append(stats.get('epsilon', -1.0))

                    round_total_profit_rl += rl_agent.current_round_profit # Sum actual profit achieved by RL agents

                avg_reward = np.mean(agent_avg_rewards) if agent_avg_rewards else 0.0
                current_eps = np.mean(agent_epsilons) if agent_epsilons and any(e >= 0 for e in agent_epsilons) else -1.0 # Avg epsilon if multiple learners
                rl_stats_for_tqdm = {
                    "Mode": mode[0], # T for Training, E for Eval
                    "AvgRLRew": f"{avg_reward:.3f}",
                    "RLProfit": f"{round_total_profit_rl:.1f}", # Display total RL profit this round
                    "Eps": f"{current_eps:.3f}" if current_eps >= 0 else "N/A"
                }

            # Round Summary Calculation (for saving to CSV)
            round_total_profit = sum(t.current_round_profit for t in all_traders)
            efficiency = round_total_profit / eq_surplus if eq_surplus > 1e-9 else 0.0
            avg_p = np.mean(round_trade_prices) if round_trade_prices else None
            adiff_p = abs(avg_p - eq_p_mid) if avg_p is not None and eq_p_mid is not None else None
            adiff_q = abs(round_total_trades - eq_q) if eq_q is not None else None

            # Collect bot details for logging/analysis
            bot_details_round = []
            role_strat_perf_round = defaultdict(lambda: {"profit": 0.0, "count": 0})
            for t in all_traders:
                role = "buyer" if t.is_buyer else "seller"
                strat = t.strategy
                profit = t.current_round_profit
                bot_details_round.append({"name": t.name, "role": role, "strategy": strat, "profit": profit})
                role_strat_perf_round[(role, strat)]["profit"] += profit
                role_strat_perf_round[(role, strat)]["count"] += 1

            # Store comprehensive round stats
            rstats = {
                "round": r, "mode": mode, "num_periods": self.num_periods, "num_steps": self.num_steps,
                "eq_q": eq_q, "eq_p": eq_p_mid, "eq_surplus": eq_surplus,
                "actual_trades": round_total_trades, "actual_total_profit": round_total_profit,
                "market_efficiency": efficiency, "avg_price": avg_p,
                "abs_diff_price": adiff_p, "abs_diff_quantity": adiff_q,
                "buyer_vals": all_buyer_values, "seller_vals": all_seller_costs,
                "role_strat_perf": dict(role_strat_perf_round), # Convert defaultdict
                "bot_details": bot_details_round
            }
            self.round_stats.append(rstats)
            self.all_step_logs.extend(round_step_logs) # Append logs from all periods

            # Update tqdm postfix with RL stats and overall efficiency
            combined_stats = {"Eff": f"{efficiency:.3f}"}
            combined_stats.update(rl_stats_for_tqdm)
            rounds_iterator.set_postfix(combined_stats, refresh=True)
            # Minimal logging for round end
            # self.logger.info(f"===== Round {r} ({mode}) End. Trades={round_total_trades}, Profit={round_total_profit:.2f}, Efficiency={efficiency:.4f} =====")

        # End of Auction: Save Models if configured
        if self.config.get("save_rl_model", False) and rl_agents_for_saving:
             save_dir = os.path.join(self.config["experiment_dir"], self.config["experiment_name"], "models")
             os.makedirs(save_dir, exist_ok=True)
             self.logger.info(f"Saving final RL models to {save_dir}...")
             for agent_trader in rl_agents_for_saving:
                 # Use a consistent naming scheme
                 model_path = os.path.join(save_dir, f"{agent_trader.strategy}_agent_{agent_trader.name}_final.pth")
                 try:
                     if hasattr(agent_trader, 'agent') and hasattr(agent_trader.agent, 'save_model'):
                         agent_trader.agent.save_model(model_path)
                     else:
                         self.logger.warning(f"Agent {agent_trader.name} ({agent_trader.strategy}) has no save_model method on its 'agent' attribute.")
                 except Exception as e:
                     self.logger.error(f"Failed to save model for {agent_trader.name}: {e}", exc_info=True)

        self.logger.info("===== SFI Auction Run Finished =====")


    def _run_period_steps(self, r, p, buyers, sellers, is_training_round, current_round_rl_agents):
        """Runs the Bid/Offer and Buy/Sell substeps for a single period."""
        period_step_logs = []
        period_trades = 0
        period_profit = 0.0
        period_prices = []

        # Track state/action across steps for RL reward assignment
        last_observed_state_by_agent = {} # State observed at the START of the previous step
        last_action_idx_by_agent = {}   # Action index chosen IN the previous step

        for st in range(self.num_steps):
            self.current_step = st
            self.logger.debug(f"\n--- R{r} P{p} Step {st} ---")
            self.logger.debug(f"State BEFORE Step: Bid={self.current_bid_info}, Ask={self.current_ask_info}, PHIBID={self.phibid}, PHIASK={self.phiask}, Cleared={self.quote_cleared_in_last_buy_sell}")

            # --- State Observation (for RL) ---
            current_step_states = {} # State observed at START of THIS step
            if is_training_round:
                for agent in current_round_rl_agents:
                     # Store state if agent is active and RL-based
                     if agent.can_trade() and hasattr(agent, '_get_state'):
                         current_step_states[agent] = agent._get_state(self.current_bid_info, self.current_ask_info, self.phibid, self.phiask)

            # --- Action Phase ---
            # Dictionary to store action index CHOSEN in THIS step by RL agents
            current_step_action_indices = {}
            submitted_quotes = self._run_bid_offer_substep(r, p, st, buyers, sellers, current_step_states, current_step_action_indices)
            trade_info = self._run_buy_sell_substep(r, p, st, buyers, sellers, current_step_states, current_step_action_indices)

            # --- Post-Action State Observation (for RL) ---
            state_after_step = {} # State observed at END of THIS step (after actions resolve)
            if is_training_round:
                for agent in current_round_rl_agents:
                    if hasattr(agent, '_get_state'):
                        state_after_step[agent] = agent._get_state(self.current_bid_info, self.current_ask_info, self.phibid, self.phiask)

            # --- Reward Calculation & RL Observation ---
            step_rewards = defaultdict(float) # Rewards earned *as a result of this step's actions*
            if trade_info:
                 period_trades += 1
                 # Calculate total profit from this trade
                 trade_profit = trade_info['buyer_profit_inc'] + trade_info['seller_profit_inc']
                 period_profit += trade_profit # Accumulate total period profit
                 period_prices.append(trade_info['price'])
                 # Assign rewards based on individual profit increments
                 step_rewards[trade_info['buyer']] = trade_info['buyer_profit_inc']
                 step_rewards[trade_info['seller']] = trade_info['seller_profit_inc']
                 self.logger.debug(f"    Trade generated rewards: B={trade_info['buyer_profit_inc']:.2f}, S={trade_info['seller_profit_inc']:.2f}")

            # Assign rewards to RL agents based on the *previous* step's state/action
            if is_training_round:
                 for agent, last_state in last_observed_state_by_agent.items(): # Iterate agents who had a state last step
                     action_idx = last_action_idx_by_agent.get(agent) # Get action from PREVIOUS step
                     if action_idx is not None: # Only observe if an action was recorded last step
                         # Reward is the outcome (profit/loss) observed *now* due to the *last* action
                         reward = step_rewards.get(agent, 0.0)
                         # Next state is the state observed *now* (after this step's actions resolved)
                         next_state = state_after_step.get(agent)
                         if hasattr(agent, 'observe_reward'):
                             # Pass done=False, 'done' signal is handled at end-of-round
                             agent.observe_reward(last_state=last_state, action_idx=action_idx, reward=reward, next_state=next_state, done=False)
                     # else: self.logger.debug(f"Agent {agent.name} had state last step, but no action recorded.")


            # Update trackers for the *next* step's reward assignment cycle
            # The state observed now becomes the 'last_state' for the next iteration
            last_observed_state_by_agent = current_step_states
            # The action chosen now becomes the 'last_action' for the next iteration
            last_action_idx_by_agent = current_step_action_indices

            # --- Logging Step ---
            log_row = {
                "round": r, "period": p, "step": st,
                "bids_submitted": submitted_quotes['bids'],
                "asks_submitted": submitted_quotes['asks'],
                "phibid_updated": self.phibid, "phiask_updated": self.phiask,
                "current_bid_price": self.current_bid_info['price'] if self.current_bid_info else None,
                "current_bidder": self.current_bid_info['agent'].name if self.current_bid_info else None,
                "current_ask_price": self.current_ask_info['price'] if self.current_ask_info else None,
                "current_asker": self.current_ask_info['agent'].name if self.current_ask_info else None,
                "trade_executed": 1 if trade_info else 0,
                "trade_price": trade_info['price'] if trade_info else None,
                "trade_buyer": trade_info['buyer'].name if trade_info else None,
                "trade_seller": trade_info['seller'].name if trade_info else None,
                "trade_type": trade_info['type'] if trade_info else None
            }
            period_step_logs.append(log_row)
            self.logger.debug(f"State AFTER Step: Bid={self.current_bid_info}, Ask={self.current_ask_info}, Trade={bool(trade_info)}")

        return period_trades, period_profit, period_prices, period_step_logs


    def _run_bid_offer_substep(self, r, p, st, buyers, sellers, current_rl_states, current_rl_action_indices):
        """ AURORA Bid-Offer Sub-Step. Agents submit quotes. """
        self.logger.debug(f"  Bid-Offer Substep: Start")
        all_traders = buyers + sellers
        for t_agent in all_traders: # Update agent's internal clock
            t_agent.current_step = st
            t_agent.current_substep = "bid_offer"

        # Determine if quotes were cleared by a trade in the previous buy/sell substep
        # If they were cleared, previous bid/ask should be None for improvement check
        prev_bid_price = None if self.quote_cleared_in_last_buy_sell else (self.current_bid_info['price'] if self.current_bid_info else None)
        prev_ask_price = None if self.quote_cleared_in_last_buy_sell else (self.current_ask_info['price'] if self.current_ask_info else None)
        self.logger.debug(f"    Improvement check baseline: prev_bid={prev_bid_price}, prev_ask={prev_ask_price}")

        submitted_bids_this_step = {} # Track actual submissions {name: price}
        potential_new_bids = []       # Track submissions for determining new best bid [(price, agent)]
        submitted_asks_this_step = {}
        potential_new_asks = []

        # Ask all agents for bids/asks
        for agent in all_traders:
             if agent.can_trade():
                 # RL Trader's make_bid_or_ask internally calls choose_action and sets _last_action_idx
                 price = agent.make_bid_or_ask(current_bid_info=self.current_bid_info, current_ask_info=self.current_ask_info, phibid=self.phibid, phiask=self.phiask, market_history=self.market_history_for_agents)

                 # If RL agent chose an action, retrieve its index and store it for this step
                 if agent in current_rl_states and hasattr(agent, '_last_action_idx') and agent._last_action_idx is not None:
                      action_idx_chosen = agent._last_action_idx
                      current_rl_action_indices[agent] = action_idx_chosen # Store action chosen *in this step*
                      agent._last_action_idx = None # Clear internal tracker after storing for this step

                 # Process valid submitted prices
                 if price is not None and self.min_price <= price <= self.max_price:
                      # Ensure price is int
                      price_int = int(round(price))
                      if agent.is_buyer:
                          potential_new_bids.append((price_int, agent))
                          submitted_bids_this_step[agent.name] = price_int
                      else:
                          potential_new_asks.append((price_int, agent))
                          submitted_asks_this_step[agent.name] = price_int
                 # else: Agent chose 'pass', 'accept', or submitted invalid price

        self.logger.debug(f"    Submitted: Bids={submitted_bids_this_step}, Asks={submitted_asks_this_step}")

        # Update market extremes (phibid, phiask) based on submissions *this step*
        if submitted_bids_this_step: self.phibid = max(self.phibid, max(submitted_bids_this_step.values()))
        if submitted_asks_this_step: self.phiask = min(self.phiask, min(submitted_asks_this_step.values()))
        self.logger.debug(f"    Updated Extremes: PHIBID={self.phibid}, PHIASK={self.phiask}")

        # Determine the new standing bid, considering only *improving* bids
        valid_improving_bids = [(price, agent) for price, agent in potential_new_bids if prev_bid_price is None or price > prev_bid_price]
        self.logger.debug(f"    Valid Improving Bids: {len(valid_improving_bids)}")
        if valid_improving_bids:
            max_bid_price = max(b[0] for b in valid_improving_bids)
            tied_bidders = [b for b in valid_improving_bids if b[0] == max_bid_price]
            # Randomly choose among tied bidders
            chosen_price, chosen_agent = self.auction_rng.choice(tied_bidders)
            self.current_bid_info = {'price': chosen_price, 'agent': chosen_agent}
            self.logger.debug(f"    New Best Bid: {self.current_bid_info['price']} by {self.current_bid_info['agent'].name}")
        elif self.quote_cleared_in_last_buy_sell:
             self.current_bid_info = None # Explicitly clear if no improving bid and quotes were reset
             self.logger.debug("    No improving bid, quotes were cleared. Current bid remains None.")
        # else: Keep the previous bid if no improving bid was submitted and quotes weren't cleared

        # Determine the new standing ask, considering only *improving* asks
        valid_improving_asks = [(price, agent) for price, agent in potential_new_asks if prev_ask_price is None or price < prev_ask_price]
        self.logger.debug(f"    Valid Improving Asks: {len(valid_improving_asks)}")
        if valid_improving_asks:
            min_ask_price = min(a[0] for a in valid_improving_asks)
            tied_askers = [a for a in valid_improving_asks if a[0] == min_ask_price]
            # Randomly choose among tied askers
            chosen_price, chosen_agent = self.auction_rng.choice(tied_askers)
            self.current_ask_info = {'price': chosen_price, 'agent': chosen_agent}
            self.logger.debug(f"    New Best Ask: {self.current_ask_info['price']} by {self.current_ask_info['agent'].name}")
        elif self.quote_cleared_in_last_buy_sell:
            self.current_ask_info = None # Explicitly clear if no improving ask and quotes were reset
            self.logger.debug("    No improving ask, quotes were cleared. Current ask remains None.")
        # else: Keep the previous ask if no improving ask was submitted and quotes weren't cleared


        # Reset the quote cleared flag *after* updating quotes for the current step
        self.quote_cleared_in_last_buy_sell = False
        self.logger.debug(f"  Bid-Offer Substep: End. Current Bid={self.current_bid_info}, Current Ask={self.current_ask_info}")
        return {"bids": submitted_bids_this_step, "asks": submitted_asks_this_step}


    def _run_buy_sell_substep(self, r, p, st, buyers, sellers, current_rl_states, current_rl_action_indices):
        """AURORA Buy-Sell Sub-Step. Eligible agents decide to accept standing quotes."""
        self.logger.debug(f"  Buy-Sell Substep: Start")
        all_traders = buyers + sellers
        for t_agent in all_traders: # Update agent's internal clock
            t_agent.current_step = st
            t_agent.current_substep = "buy_sell"

        # Get current market state
        bid_info = self.current_bid_info
        ask_info = self.current_ask_info
        bid_price = bid_info['price'] if bid_info else None
        bidder = bid_info['agent'] if bid_info else None
        ask_price = ask_info['price'] if ask_info else None
        asker = ask_info['agent'] if ask_info else None

        # Check if a trade is even possible (bid >= ask)
        if bid_price is None or ask_price is None or bid_price < ask_price:
            self.logger.debug(f"    No trade possible (Bid {bid_price} / Ask {ask_price}).")
            return None # No trade

        buy_requesting_agents = []
        sell_requesting_agents = []

        # --- Collect BUY requests (from eligible buyers accepting the current ask) ---
        if ask_price is not None and asker is not None: # Must be a valid ask to accept
            # AURORA Rules: Only the holder of the current bid can accept the ask
            eligible_buyers = []
            if bidder and bidder.can_trade():
                eligible_buyers = [bidder]

            self.logger.debug(f"    Eligible Buyers to accept Ask({ask_price}) by {asker.name}: {[b.name for b in eligible_buyers]}")
            for b in eligible_buyers:
                # RL agent's request_buy internally calls choose_action and sets _last_action_idx
                accepts = b.request_buy(ask_price, self.current_bid_info, self.current_ask_info, self.phibid, self.phiask, self.market_history_for_agents)
                if accepts:
                     buy_requesting_agents.append(b)
                     self.logger.debug(f"    Buyer {b.name} requests BUY at {ask_price}")
                     # If RL agent accepted, store the action index for this step
                     if b in current_rl_states and hasattr(b, '_last_action_idx') and b._last_action_idx is not None:
                          action_idx_chosen = b._last_action_idx
                          current_rl_action_indices[b] = action_idx_chosen
                          b._last_action_idx = None # Clear internal tracker


        # --- Collect SELL requests (from eligible sellers accepting the current bid) ---
        if bid_price is not None and bidder is not None: # Must be a valid bid to accept
            # AURORA Rules: Only the holder of the current ask can accept the bid
            eligible_sellers = []
            if asker and asker.can_trade():
                eligible_sellers = [asker]

            self.logger.debug(f"    Eligible Sellers to accept Bid({bid_price}) by {bidder.name}: {[s.name for s in eligible_sellers]}")
            for s in eligible_sellers:
                # RL agent's request_sell internally calls choose_action and sets _last_action_idx
                accepts = s.request_sell(bid_price, self.current_bid_info, self.current_ask_info, self.phibid, self.phiask, self.market_history_for_agents)
                if accepts:
                     sell_requesting_agents.append(s)
                     self.logger.debug(f"    Seller {s.name} requests SELL at {bid_price}")
                     # If RL agent accepted, store the action index for this step
                     if s in current_rl_states and hasattr(s, '_last_action_idx') and s._last_action_idx is not None:
                          action_idx_chosen = s._last_action_idx
                          current_rl_action_indices[s] = action_idx_chosen
                          s._last_action_idx = None # Clear internal tracker


        # --- Resolve Requests and Execute Trade (if any) ---
        trade_info = None
        winning_request_type = None
        final_buyer, final_seller, final_price = None, None, None
        executed_trade_type = None

        has_buy_request = bool(buy_requesting_agents)
        has_sell_request = bool(sell_requesting_agents)

        if not has_buy_request and not has_sell_request:
            self.logger.debug("    No acceptances received.")
            pass # No trade
        elif has_buy_request and not has_sell_request:
            winning_request_type = 'buy' # Bidder accepted the ask
        elif not has_buy_request and has_sell_request:
            winning_request_type = 'sell' # Asker accepted the bid
        else: # Both holder accepted simultaneously (unlikely but possible if both are same agent?)
            self.logger.debug(f"    Simultaneous acceptances: Buyer={bidder.name}, Seller={asker.name}")
            # Randomly choose which acceptance prevails (SFI tie-break rule)
            winning_request_type = self.auction_rng.choice(['buy', 'sell'])
            self.logger.debug(f"    Randomly chose winner type: {winning_request_type}")

        # Determine participants based on winning request type
        if winning_request_type == 'buy':
            # Buyer (bidder) accepted the Ask. Seller is the asker.
            if bidder and asker: # Ensure both original quote holders exist
                 final_buyer, final_seller, final_price = bidder, asker, ask_price
                 executed_trade_type = 'buy_request' # Buyer accepted Ask
                 self.logger.debug(f"    Buy request wins. Buyer={bidder.name}, Seller={asker.name}, Price={ask_price}")
            else: self.logger.warning("Buy request won, but bidder or asker is invalid.")
        elif winning_request_type == 'sell':
            # Seller (asker) accepted the Bid. Buyer is the bidder.
            if bidder and asker: # Ensure both original quote holders exist
                final_buyer, final_seller, final_price = bidder, asker, bid_price
                executed_trade_type = 'sell_request' # Seller accepted Bid
                self.logger.debug(f"    Sell request wins. Buyer={bidder.name}, Seller={asker.name}, Price={bid_price}")
            else: self.logger.warning("Sell request won, but bidder or asker is invalid.")

        # Execute the trade if participants and price are determined
        if final_buyer and final_seller and final_price is not None:
            # Final check if both selected participants can still trade
            if final_buyer.can_trade() and final_seller.can_trade():
                # <<< Log changed to DEBUG >>>
                self.logger.debug(f"    TRADE Executing: Type={executed_trade_type.upper()}, Price={final_price}, Buyer={final_buyer.name}, Seller={final_seller.name}")
                # Record the trade and get profit increments
                buyer_profit_inc = final_buyer.record_trade(p, st, final_price)
                seller_profit_inc = final_seller.record_trade(p, st, final_price)

                trade_info = {
                    "buyer": final_buyer, "seller": final_seller, "price": final_price,
                    "buyer_profit_inc": buyer_profit_inc, "seller_profit_inc": seller_profit_inc,
                    "type": executed_trade_type
                }

                # Clear the standing quotes after a successful trade
                self.logger.debug("    Clearing quotes due to trade.")
                self.current_bid_info = None
                self.current_ask_info = None
                self.quote_cleared_in_last_buy_sell = True # Set flag for next bid/offer step
            else:
                # Should ideally not happen if can_trade checks were done before requests, but possible in race conditions?
                self.logger.warning(f"    Trade failed token check at execution: B={final_buyer.name}({final_buyer.tokens_left}), S={final_seller.name}({final_seller.tokens_left})")
                trade_info = None # Abort trade
        else:
             if winning_request_type: self.logger.debug(f"    No valid trade participants determined despite win type '{winning_request_type}'.")
             else: self.logger.debug("    No trade executed in this substep.")
             trade_info = None

        self.logger.debug(f"  Buy-Sell Substep: End. Trade executed: {bool(trade_info)}")
        return trade_info