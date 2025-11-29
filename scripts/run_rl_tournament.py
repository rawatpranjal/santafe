#!/usr/bin/env python3
"""
Run the Santa Fe Tournament with the trained RL agent.
"""

import argparse
import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from engine.market import Market
from engine.agent_factory import create_agent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RLTournamentRunner:
    def __init__(self, model_path: str, output_dir: str = None):
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results/tournament_rl_{timestamp}"
            
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path
        
        # Define trader types including PPO
        self.trader_types = [
            "PPO", "ZIC", "Kaplan", "ZIP", "GD", "ZI2",
            "Lin", "Perry", "Jacobson", "Skeleton"
        ]
        
        # Define environments (subset for quick evaluation, or full)
        # Using a representative subset for now to get quick feedback
        self.environments = {
            "BASE": {"buyers": 4, "sellers": 4, "tokens": 4, "periods": 10, "steps": 100},
            "EQL": {"buyers": 4, "sellers": 4, "tokens": 4, "periods": 10, "steps": 100, "symmetric": True},
            "LAD": {"buyers": 6, "sellers": 2, "tokens": 4, "periods": 10, "steps": 100}
        }

    def run_tournament(self):
        logger.info("=" * 60)
        logger.info(f"RL TOURNAMENT (Model: {self.model_path})")
        logger.info("=" * 60)
        
        all_results = []
        
        for env_name, env_config in self.environments.items():
            logger.info(f"\n--- Environment: {env_name} ---")
            
            trader_profits = {trader: [] for trader in self.trader_types}
            efficiency_values = []
            
            # Run 50 rounds per environment
            for round_num in range(50):
                # Create agents first
                from engine.token_generator import TokenGenerator
                token_gen = TokenGenerator(1111 + round_num, env_config.get("tokens", 4), None)
                token_gen.new_round()
                
                # Create mixed market
                # Strategy: 1 PPO agent, rest are random legacy agents
                round_agents = ["PPO"] # Ensure PPO is in
                
                # Fill rest with random other agents
                other_types = [t for t in self.trader_types if t != "PPO"]
                
                num_needed = env_config["buyers"] + env_config["sellers"] - 1
                round_agents.extend(np.random.choice(other_types, num_needed, replace=True))
                np.random.shuffle(round_agents)
                
                buyers = []
                sellers = []
                
                for i in range(env_config["buyers"]):
                    atype = round_agents[i]
                    tokens = token_gen.generate_tokens(True)
                    kwargs = {}
                    if atype == "PPO":
                        kwargs["model_path"] = self.model_path
                        
                    agent = create_agent(atype, i+1, True, env_config.get("tokens", 4), tokens, **kwargs)
                    buyers.append(agent)
                    
                for i in range(env_config["sellers"]):
                    atype = round_agents[env_config["buyers"] + i]
                    tokens = token_gen.generate_tokens(False)
                    kwargs = {}
                    if atype == "PPO":
                        kwargs["model_path"] = self.model_path
                        
                    agent = create_agent(atype, env_config["buyers"]+i+1, False, env_config.get("tokens", 4), tokens, **kwargs)
                    sellers.append(agent)

                # Initialize market with agents
                market = Market(
                    num_buyers=env_config["buyers"],
                    num_sellers=env_config["sellers"],
                    num_times=env_config.get("steps", 100),
                    price_min=0,
                    price_max=1000,
                    buyers=buyers,
                    sellers=sellers,
                    seed=round_num
                )
                
                # Initialize agents for the period
                for agent in buyers + sellers:
                    agent.start_period(1)
                
                # Run market steps
                for _ in range(env_config.get("steps", 100)):
                    if not market.run_time_step():
                        break
                        
                # Calculate efficiency
                # We need to calculate equilibrium profit manually here or use market method if available
                # Market doesn't have get_efficiency() in the file I viewed? 
                # Ah, run_full_tournament used market.get_efficiency(). Let's assume it exists or I missed it.
                # Wait, I viewed market.py and it DID NOT have get_efficiency().
                # I need to implement it or import it.
                
                # Let's use the metric function I fixed earlier
                from engine.metrics import calculate_equilibrium_profit
                
                all_vals = []
                all_costs = []
                for b in buyers: all_vals.extend(b.valuations)
                for s in sellers: all_costs.extend(s.valuations)
                
                max_profit = calculate_equilibrium_profit(all_vals, all_costs)
                
                actual_profit = 0
                for a in buyers + sellers:
                    actual_profit += a.period_profit # Note: period_profit might need accumulation if multiple periods
                    
                    # Track per-trader profit (normalized by max possible for that agent?)
                    # Or just raw profit. Raw profit is biased by token allocation.
                    # But over many rounds it averages out.
                    trader_profits[type(a).__name__.replace("Agent", "").replace("Trader", "")].append(a.period_profit)
                    # Note: PPOAgent class name is PPOAgent, others are like ZIC, Kaplan
                    # My factory returns ZIC, Kaplan instances.
                    # PPO returns PPOAgent.
                    
                    # Fix key name
                    t_name = type(a).__name__
                    if t_name == "PPOAgent": t_name = "PPO"
                    if t_name not in trader_profits: trader_profits[t_name] = [] # Should be init
                    
                
                if max_profit > 0:
                    efficiency = actual_profit / max_profit
                else:
                    efficiency = 0
                efficiency_values.append(efficiency)
                
            # Summarize Environment
            mean_eff = np.mean(efficiency_values)
            logger.info(f"Efficiency: {mean_eff:.1f}%")
            
            # Calculate average profit per agent type
            avg_profits = {}
            for t, profits in trader_profits.items():
                if profits:
                    avg_profits[t] = np.mean(profits)
                else:
                    avg_profits[t] = 0.0
            
            # Rank agents
            ranking = sorted(avg_profits.items(), key=lambda x: x[1], reverse=True)
            logger.info("Ranking:")
            for rank, (name, profit) in enumerate(ranking, 1):
                logger.info(f"  {rank}. {name}: {profit:.1f}")
                
            # Store PPO rank
            ppo_rank = next((i for i, (n, _) in enumerate(ranking, 1) if n == "PPO"), -1)
            
            all_results.append({
                "environment": env_name,
                "efficiency": mean_eff,
                "ppo_rank": ppo_rank,
                "ppo_profit": avg_profits.get("PPO", 0),
                "top_agent": ranking[0][0]
            })
            
        # Overall Summary
        df = pd.DataFrame(all_results)
        print("\n" + "="*60)
        print("TOURNAMENT RESULTS")
        print("="*60)
        print(df.to_string(index=False))
        
        avg_rank = df["ppo_rank"].mean()
        print(f"\nAverage PPO Rank: {avg_rank:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to PPO model zip")
    args = parser.parse_args()
    
    runner = RLTournamentRunner(args.model)
    runner.run_tournament()
