"""
Tournament Engine.

Runs a multi-round, multi-period tournament with configured agents.
"""

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from engine.event_logger import EventLogger
from engine.market import Market
from engine.token_generator import TokenGenerator, UniformTokenGenerator
from engine.agent_factory import create_agent
from engine.efficiency import (
    extract_trades_from_orderbook,
    calculate_v_inefficiency,
    calculate_em_inefficiency,
    calculate_actual_surplus,
    calculate_max_surplus,
    calculate_allocative_efficiency,
    calculate_equilibrium_profits,
    get_transaction_prices,
    calculate_price_std_dev
)

class Tournament:
    """
    Manages the execution of a tournament.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.results = []

        # Event logger for Market Heartbeat visualization (optional)
        self.event_logger: EventLogger | None = None
        if config.get("log_events", False):
            log_dir = Path(config.get("log_dir", "logs"))
            exp_id = config.get("experiment_id", "exp")
            event_log_path = log_dir / f"{exp_id}_events.jsonl"
            self.event_logger = EventLogger(event_log_path)
            self.logger.info(f"Event logging enabled: {event_log_path}")
        
    def run(self) -> pd.DataFrame:
        """Run the tournament and return results."""
        
        # Setup Token Generator
        # Use "uniform" mode for Gode & Sunder replication (simple uniform random)
        # Use "santafe" mode (default) for Santa Fe tournament gametype-based generation
        token_mode = self.config.market.get("token_mode", "santafe")

        if token_mode == "uniform":
            token_gen = UniformTokenGenerator(
                self.config.market.num_tokens,
                self.config.market.min_price,
                self.config.market.max_price,
                self.config.experiment.rng_seed_values,
                num_buyers=len(self.config.agents.buyer_types),
                num_sellers=len(self.config.agents.seller_types),
            )
            self.logger.info(f"Using Gode & Sunder style overlapping supply/demand")
        else:
            token_gen = TokenGenerator(
                self.config.market.gametype,
                self.config.market.num_tokens,
                self.config.experiment.rng_seed_values
            )
        
        # Create Agents
        agents = []

        # Market composition (needed by some traders like Lin for weighting formulas)
        buyer_types = self.config.agents.buyer_types
        seller_types = self.config.agents.seller_types
        num_buyers = len(buyer_types)
        num_sellers = len(seller_types)

        # Prepare kwargs for LLM agents
        agent_kwargs = {}
        if hasattr(self.config, 'llm_output_dir'):
            agent_kwargs['output_dir'] = self.config.llm_output_dir

        # Buyers
        for i, agent_type in enumerate(buyer_types):
            player_id = i + 1
            agents.append(create_agent(
                agent_type,
                player_id,
                True,
                self.config.market.num_tokens,
                [0] * self.config.market.num_tokens, # Valuations set later
                seed=self.config.experiment.rng_seed_auction + player_id,
                num_times=self.config.market.num_steps,
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                price_min=self.config.market.min_price,
                price_max=self.config.market.max_price,
                **agent_kwargs
            ))

        # Sellers
        for i, agent_type in enumerate(seller_types):
            player_id = num_buyers + i + 1
            agents.append(create_agent(
                agent_type,
                player_id,
                False,
                self.config.market.num_tokens,
                [0] * self.config.market.num_tokens, # Costs set later
                seed=self.config.experiment.rng_seed_auction + player_id,
                num_times=self.config.market.num_steps,
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                price_min=self.config.market.min_price,
                price_max=self.config.market.max_price,
                **agent_kwargs
            ))
            
        self.logger.info(f"Initialized {len(agents)} agents: {buyer_types} vs {seller_types}")
        
        # Run Rounds
        for r in range(1, self.config.experiment.num_rounds + 1):
            self.logger.info(f"Starting Round {r}")
            token_gen.new_round()
            
            # Assign tokens
            all_buyer_values = []
            all_seller_costs = []
            
            # Keep track of valuations per agent for efficiency metrics
            buyer_valuations_dict = {}
            seller_costs_dict = {}
            
            for agent in agents:
                tokens = token_gen.generate_tokens(agent.is_buyer)
                # CRITICAL FIX: Use start_round() to properly reset agent state
                # This ensures num_trades, profit tracking, and other state are reset
                agent.start_round(tokens)
                if agent.is_buyer:
                    all_buyer_values.extend(tokens)
                    buyer_valuations_dict[agent.player_id] = tokens
                else:
                    all_seller_costs.extend(tokens)
                    # Use LOCAL ID for sellers to match OrderBook (1..num_sellers)
                    local_seller_id = agent.player_id - len(buyer_types)
                    seller_costs_dict[local_seller_id] = tokens
            
            # Calculate Equilibrium Profit & Max Trades
            # Matches engine/metrics.py logic but also counts trades
            b_vals = sorted([v for v in all_buyer_values if v > 0], reverse=True)
            s_costs = sorted([c for c in all_seller_costs if c > 0])
            
            eq_profit = 0
            max_trades = 0
            n = min(len(b_vals), len(s_costs))
            for i in range(n):
                if b_vals[i] > s_costs[i]:
                    eq_profit += b_vals[i] - s_costs[i]
                    max_trades += 1
                else:
                    break

            # Calculate equilibrium price for visualization
            # CE price = midpoint of marginal buyer/seller pair
            if max_trades > 0:
                ce_price = (b_vals[max_trades - 1] + s_costs[max_trades - 1]) // 2
            else:
                ce_price = (b_vals[0] + s_costs[0]) // 2 if b_vals and s_costs else 0

            # Run Periods
            for p in range(1, self.config.market.num_periods + 1):
                self.logger.info(f"Starting Period {p}")
                
                # Create Market
                market = Market(
                    num_buyers=len(buyer_types),
                    num_sellers=len(seller_types),
                    price_min=self.config.market.min_price,
                    price_max=self.config.market.max_price,
                    num_times=self.config.market.num_steps,
                    buyers=[a for a in agents if a.is_buyer],
                    sellers=[a for a in agents if not a.is_buyer],
                    seed=self.config.experiment.rng_seed_auction + r*1000 + p,
                    deadsteps=self.config.market.get("deadsteps", 0),
                    event_logger=self.event_logger,
                )
                market.set_period(r, p)

                # Log period start with equilibrium info for visualization
                if self.event_logger:
                    self.event_logger.log_period_start(r, p, ce_price, eq_profit)

                # Notify agents start period
                for a in agents:
                    a.start_period(p)
                    
                # Run Market
                while market.current_time < market.num_times:
                    market.run_time_step()
                    # Check for early termination due to deadsteps
                    if market.orderbook.should_terminate_early():
                        break
                    
                # Notify agents end period
                for a in agents:
                    a.end_period()
                    
                # Extract trades first (needed for all metrics)
                trades = extract_trades_from_orderbook(market.orderbook, market.num_times)
                actual_trades = len(trades)

                # Build buyer and seller lists for max surplus calculation
                buyer_vals_list = [buyer_valuations_dict[a.player_id] for a in agents if a.is_buyer]
                seller_costs_list = [seller_costs_dict[a.player_id - len(buyer_types)] for a in agents if not a.is_buyer]

                # Calculate efficiency using surplus-based method
                actual_surplus = calculate_actual_surplus(trades, buyer_valuations_dict, seller_costs_dict)
                max_surplus = calculate_max_surplus(buyer_vals_list, seller_costs_list)
                efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)

                # DIAGNOSTIC: Print first few trades to check valuations
                if (r == 1 or r == 3) and p == 1 and len(trades) > 0:
                    self.logger.info(f"\n=== DIAGNOSTIC: Round {r} Period {p} ===")
                    self.logger.info(f"Trades: {len(trades)}, Actual Surplus: {actual_surplus}, Max Surplus: {max_surplus}")
                    self.logger.info(f"buyer_valuations_dict keys: {list(buyer_valuations_dict.keys())}")
                    self.logger.info(f"seller_costs_dict keys: {list(seller_costs_dict.keys())}")

                    # Print ALL valuations for Round 3
                    if r == 3:
                        self.logger.info(f"ALL BUYER VALUATIONS:")
                        for bid, vals in buyer_valuations_dict.items():
                            self.logger.info(f"  Buyer {bid}: {vals}")
                        self.logger.info(f"ALL SELLER COSTS:")
                        for sid, costs in seller_costs_dict.items():
                            self.logger.info(f"  Seller {sid}: {costs}")
                        # Also print what agents think their valuations are
                        self.logger.info(f"AGENT VALUATIONS (from agent objects):")
                        for agent in agents:
                            if agent.is_buyer:
                                self.logger.info(f"  Agent {agent.player_id} (buyer): {agent.valuations}")
                            else:
                                self.logger.info(f"  Agent {agent.player_id} (seller): {agent.valuations}")
                    for i, (bid, sid, price, bunit) in enumerate(trades[:5]):
                        buyer_val = buyer_valuations_dict[bid][bunit]
                        # Track seller unit
                        seller_unit = sum(1 for b2, s2, p2, u2 in trades[:i] if s2 == sid)
                        seller_cost = seller_costs_dict[sid][seller_unit]
                        surplus = buyer_val - seller_cost
                        self.logger.info(
                            f"  Trade {i}: B{bid}[{bunit}]={buyer_val} + S{sid}[{seller_unit}]={seller_cost} "
                            f"@ ${price} → surplus={surplus}"
                        )
                    # Check agent profits
                    total_buyer_profit = sum(a.period_profit for a in agents if a.is_buyer)
                    total_seller_profit = sum(a.period_profit for a in agents if not a.is_buyer)
                    self.logger.info(f"Agent profits: buyers={total_buyer_profit}, sellers={total_seller_profit}, total={total_buyer_profit+total_seller_profit}")

                    # DIAGNOSTIC: Compare to manual calculation
                    if r == 3:
                        manual_surplus = 0
                        for trade_idx, (bid, sid, price, bunit) in enumerate(trades):
                            bval = buyer_valuations_dict[bid][bunit]
                            # Track seller unit manually
                            sunit = sum(1 for b2, s2, p2, u2 in trades[:trade_idx] if s2 == sid)
                            scost = seller_costs_dict[sid][sunit]
                            trade_surplus = bval - scost
                            manual_surplus += trade_surplus
                            if trade_idx < 10:
                                self.logger.info(f"    Manual Trade {trade_idx}: B{bid}[{bunit}]={bval}, S{sid}[{sunit}]={scost}, price={price}, surplus={trade_surplus}")
                        self.logger.info(f"  Manual surplus total: {manual_surplus}")
                        self.logger.info(f"  calculate_actual_surplus returned: {actual_surplus}")
                        self.logger.info(f"  Agent profit total: {total_buyer_profit + total_seller_profit}")
                    self.logger.info(f"=== END DIAGNOSTIC ===\n")

                # Calculate Decomposition Metrics
                v_inefficiency = calculate_v_inefficiency(max_trades, actual_trades)
                em_inefficiency = calculate_em_inefficiency(trades, buyer_valuations_dict, seller_costs_dict)

                # Calculate equilibrium profits per agent
                # Use average transaction price as proxy for equilibrium price P0
                transaction_prices = get_transaction_prices(market.orderbook, market.num_times)
                equilibrium_price = int(sum(transaction_prices) / len(transaction_prices)) if transaction_prices else 0

                # Calculate price volatility metrics
                price_std_dev = calculate_price_std_dev(transaction_prices)
                price_mean = float(sum(transaction_prices) / len(transaction_prices)) if transaction_prices else 0.0
                price_volatility_pct = (price_std_dev / price_mean * 100) if price_mean > 0 else 0.0

                buyer_eq_profits, seller_eq_profits = calculate_equilibrium_profits(
                    buyer_vals_list, seller_costs_list, equilibrium_price
                )

                self.logger.info(
                    f"Round {r} Period {p}: Eff {efficiency:.2f}% "
                    f"V-Ineff {v_inefficiency} EM-Ineff {em_inefficiency}"
                )

                # Log individual agent performance
                for agent in agents:
                    # Get agent's equilibrium profit
                    if agent.is_buyer:
                        agent_eq_profit = buyer_eq_profits.get(agent.player_id, 0)
                    else:
                        # Seller IDs in eq_profits dict are local (1..num_sellers)
                        local_seller_id = agent.player_id - len(buyer_types)
                        agent_eq_profit = seller_eq_profits.get(local_seller_id, 0)

                    # Calculate profit deviation from equilibrium
                    profit_deviation = agent.period_profit - agent_eq_profit

                    self.results.append({
                        "round": r,
                        "period": p,
                        "agent_id": agent.player_id,
                        "agent_type": agent.__class__.__name__,
                        "is_buyer": agent.is_buyer,
                        "num_trades": agent.num_trades,
                        "period_profit": agent.period_profit,
                        "agent_eq_profit": agent_eq_profit,
                        "profit_deviation": profit_deviation,
                        "efficiency": efficiency,
                        "eq_profit": eq_profit,
                        "v_inefficiency": v_inefficiency,
                        "em_inefficiency": em_inefficiency,
                        "price_std_dev": price_std_dev,
                        "price_mean": price_mean,
                        "price_volatility_pct": price_volatility_pct
                    })
                
        # Close event logger if enabled
        if self.event_logger is not None:
            self.event_logger.close()
            self.logger.info("Event log saved")

        return pd.DataFrame(self.results)
