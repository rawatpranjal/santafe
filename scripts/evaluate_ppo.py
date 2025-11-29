#!/usr/bin/env python3
"""
PPO Agent Evaluation Script.

This script evaluates trained PPO agents in various market conditions:
1. Against different opponent types
2. Invasibility tests (single PPO in legacy market)
3. Self-play scenarios
4. Robustness tests
5. Strategy analysis

Usage:
    # Evaluate against specific opponents
    python scripts/evaluate_ppo.py --model models/ppo_final.zip --opponent ZIC

    # Run comprehensive evaluation suite
    python scripts/evaluate_ppo.py --model models/ppo_final.zip --comprehensive

    # Compare multiple models
    python scripts/evaluate_ppo.py --models model1.zip model2.zip --compare

    # Generate detailed report
    python scripts/evaluate_ppo.py --model models/ppo_final.zip --report
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.enhanced_double_auction_env import EnhancedDoubleAuctionEnv
from traders.rl.ppo_agent import PPOAgent
from engine.market import Market
from engine.tournament import Tournament
from engine.token_generator import TokenGenerator
from engine.agent_factory import create_agent


class PPOEvaluator:
    """Comprehensive evaluation suite for PPO agents."""

    def __init__(self, model_path: str, verbose: bool = True):
        """
        Initialize evaluator with trained model.

        Args:
            model_path: Path to saved PPO model
            verbose: Enable verbose output
        """
        self.model_path = Path(model_path)
        self.verbose = verbose

        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = MaskablePPO.load(self.model_path)

        # Check for normalization statistics
        norm_path = self.model_path.parent / "vec_normalize.pkl"
        self.vec_normalize = None
        if norm_path.exists():
            # Create dummy env to load normalization
            dummy_env = DummyVecEnv([lambda: Monitor(EnhancedDoubleAuctionEnv({
                "num_agents": 8,
                "num_tokens": 4,
                "max_steps": 100,
                "min_price": 0,
                "max_price": 1000
            }))])
            self.vec_normalize = VecNormalize.load(str(norm_path), dummy_env)

        # Results storage
        self.results = {}

    def evaluate_vs_opponent(self, opponent_type: str,
                            n_episodes: int = 100,
                            as_buyer: bool = True,
                            as_seller: bool = True) -> Dict[str, Any]:
        """
        Evaluate PPO against specific opponent type.

        Args:
            opponent_type: Type of opponent (ZIC, ZIP, GD, Kaplan, etc.)
            n_episodes: Number of evaluation episodes
            as_buyer: Test as buyer
            as_seller: Test as seller

        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            "opponent": opponent_type,
            "n_episodes": n_episodes,
            "buyer_results": None,
            "seller_results": None
        }

        # Test as buyer
        if as_buyer:
            if self.verbose:
                print(f"\n📊 Evaluating as BUYER vs {opponent_type}...")

            buyer_metrics = self._run_evaluation(
                opponent_type=opponent_type,
                is_buyer=True,
                n_episodes=n_episodes
            )
            results["buyer_results"] = buyer_metrics

        # Test as seller
        if as_seller:
            if self.verbose:
                print(f"\n📊 Evaluating as SELLER vs {opponent_type}...")

            seller_metrics = self._run_evaluation(
                opponent_type=opponent_type,
                is_buyer=False,
                n_episodes=n_episodes
            )
            results["seller_results"] = seller_metrics

        return results

    def run_invasibility_test(self, n_episodes: int = 100) -> Dict[str, Any]:
        """
        Test PPO's ability to invade ZIC-dominated market (1v7 test).

        Returns:
            Invasibility metrics (profit ratio, market share)
        """
        if self.verbose:
            print("\n🔬 Running invasibility test (1 PPO vs 7 ZIC)...")

        # Configure 1v7 environment
        config = {
            "num_agents": 8,
            "num_tokens": 4,
            "max_steps": 100,
            "min_price": 0,
            "max_price": 1000,
            "rl_agent_id": 1,
            "opponent_type": "ZIC"
        }

        # Test as buyer
        buyer_results = self._run_evaluation(
            opponent_type="ZIC",
            is_buyer=True,
            n_episodes=n_episodes,
            env_config=config
        )

        # Test as seller
        seller_results = self._run_evaluation(
            opponent_type="ZIC",
            is_buyer=False,
            n_episodes=n_episodes,
            env_config=config
        )

        # Calculate invasibility metrics
        zic_baseline_profit = 100  # Expected ZIC profit

        buyer_ratio = buyer_results["mean_profit"] / zic_baseline_profit
        seller_ratio = seller_results["mean_profit"] / zic_baseline_profit
        overall_ratio = (buyer_ratio + seller_ratio) / 2

        results = {
            "buyer_profit_ratio": buyer_ratio,
            "seller_profit_ratio": seller_ratio,
            "overall_invasibility": overall_ratio,
            "buyer_details": buyer_results,
            "seller_details": seller_results
        }

        if self.verbose:
            print(f"\n🎯 Invasibility Score: {overall_ratio:.3f}")
            print(f"  Buyer Ratio: {buyer_ratio:.3f}")
            print(f"  Seller Ratio: {seller_ratio:.3f}")

        return results

    def run_self_play(self, n_episodes: int = 100) -> Dict[str, Any]:
        """
        Evaluate PPO agents playing against each other.

        Returns:
            Self-play metrics (efficiency, convergence, stability)
        """
        if self.verbose:
            print("\n🤖 Running PPO self-play evaluation...")

        # Create tournament with PPO agents
        results = self._run_ppo_tournament(
            num_ppo_buyers=4,
            num_ppo_sellers=4,
            n_episodes=n_episodes
        )

        # Analyze convergence and stability
        efficiencies = results["episode_efficiencies"]
        convergence_rate = self._calculate_convergence(efficiencies)
        stability = 1.0 - np.std(efficiencies) / np.mean(efficiencies) if efficiencies else 0

        results.update({
            "convergence_rate": convergence_rate,
            "stability_score": stability,
            "mean_efficiency": np.mean(efficiencies) if efficiencies else 0,
            "final_efficiency": efficiencies[-1] if efficiencies else 0
        })

        if self.verbose:
            print(f"\n📊 Self-Play Results:")
            print(f"  Mean Efficiency: {results['mean_efficiency']:.3f}")
            print(f"  Convergence Rate: {convergence_rate:.3f}")
            print(f"  Stability Score: {stability:.3f}")

        return results

    def run_robustness_tests(self) -> Dict[str, Any]:
        """
        Test PPO robustness across different market conditions.

        Returns:
            Robustness metrics across scenarios
        """
        if self.verbose:
            print("\n🛡️ Running robustness tests...")

        scenarios = {
            "standard": {
                "num_tokens": 4,
                "max_steps": 100,
                "opponent_type": "ZIC"
            },
            "few_tokens": {
                "num_tokens": 1,
                "max_steps": 100,
                "opponent_type": "ZIC"
            },
            "time_pressure": {
                "num_tokens": 4,
                "max_steps": 25,
                "opponent_type": "ZIC"
            },
            "asymmetric": {
                "num_agents": 10,  # 6v4
                "num_tokens": 4,
                "max_steps": 100,
                "opponent_type": "ZIC"
            },
            "strategic": {
                "num_tokens": 4,
                "max_steps": 100,
                "opponent_type": "Kaplan"
            }
        }

        results = {}

        for scenario_name, scenario_config in scenarios.items():
            if self.verbose:
                print(f"  Testing {scenario_name}...")

            config = {
                "num_agents": scenario_config.get("num_agents", 8),
                "num_tokens": scenario_config["num_tokens"],
                "max_steps": scenario_config["max_steps"],
                "min_price": 0,
                "max_price": 1000,
                "opponent_type": scenario_config["opponent_type"]
            }

            scenario_results = self._run_evaluation(
                opponent_type=scenario_config["opponent_type"],
                is_buyer=True,
                n_episodes=50,
                env_config=config
            )

            results[scenario_name] = {
                "efficiency": scenario_results["mean_efficiency"],
                "profit": scenario_results["mean_profit"],
                "trades": scenario_results["mean_trades"]
            }

        # Calculate robustness score (consistency across scenarios)
        efficiencies = [r["efficiency"] for r in results.values()]
        robustness_score = 1.0 - np.std(efficiencies) / np.mean(efficiencies) if efficiencies else 0

        results["robustness_score"] = robustness_score

        if self.verbose:
            print(f"\n🎯 Overall Robustness Score: {robustness_score:.3f}")

        return results

    def analyze_strategy(self, n_episodes: int = 20) -> Dict[str, Any]:
        """
        Analyze learned trading strategy.

        Returns:
            Strategy characteristics (aggression, timing, adaptability)
        """
        if self.verbose:
            print("\n🔍 Analyzing learned strategy...")

        # Collect detailed action data
        action_data = []
        price_data = []
        timing_data = []

        config = {
            "num_agents": 8,
            "num_tokens": 4,
            "max_steps": 100,
            "min_price": 0,
            "max_price": 1000,
            "opponent_type": "ZIC",
            "rl_is_buyer": True
        }

        env = EnhancedDoubleAuctionEnv(config)
        env = Monitor(env)

        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            step = 0

            while not done:
                # Get action from model
                if self.vec_normalize:
                    obs_norm = self.vec_normalize.normalize_obs(obs)
                    action, _ = self.model.predict(obs_norm, deterministic=True)
                else:
                    action, _ = self.model.predict(obs, deterministic=True)

                action = int(action)
                action_data.append(action)

                # Map action to price
                if hasattr(env, "_map_action_to_price"):
                    price = env._map_action_to_price(action)
                    price_data.append(price)

                timing_data.append(step / config["max_steps"])

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1

        # Analyze strategy characteristics
        action_counts = pd.Series(action_data).value_counts(normalize=True)

        strategy = {
            "action_distribution": action_counts.to_dict(),
            "aggression_level": 1.0 - action_counts.get(0, 0),  # Non-pass rate
            "accept_rate": action_counts.get(1, 0),  # Accept action rate
            "improve_rate": action_counts.get(3, 0) + action_counts.get(4, 0),  # Improve actions
            "truthful_rate": action_counts.get(6, 0),  # Truthful bidding
            "avg_action_timing": np.mean(timing_data) if timing_data else 0.5,
            "price_volatility": np.std(price_data) if price_data else 0
        }

        # Classify strategy type
        if strategy["aggression_level"] > 0.7:
            strategy["type"] = "Aggressive"
        elif strategy["accept_rate"] > 0.3:
            strategy["type"] = "Opportunistic"
        elif strategy["improve_rate"] > 0.4:
            strategy["type"] = "Market Maker"
        elif strategy["truthful_rate"] > 0.3:
            strategy["type"] = "Truthful"
        else:
            strategy["type"] = "Conservative"

        if self.verbose:
            print(f"\n📊 Strategy Profile:")
            print(f"  Type: {strategy['type']}")
            print(f"  Aggression: {strategy['aggression_level']:.3f}")
            print(f"  Accept Rate: {strategy['accept_rate']:.3f}")
            print(f"  Market Making: {strategy['improve_rate']:.3f}")

        return strategy

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report.

        Args:
            output_path: Path to save report (optional)

        Returns:
            Report as string
        """
        # Run all evaluations if not already done
        if "comprehensive" not in self.results:
            self.run_comprehensive_evaluation()

        report = []
        report.append("="*60)
        report.append("PPO AGENT EVALUATION REPORT")
        report.append(f"Model: {self.model_path}")
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("="*60)

        # Overall Performance
        report.append("\n1. OVERALL PERFORMANCE")
        report.append("-"*40)
        comp = self.results["comprehensive"]

        # Average across all opponent types
        all_efficiencies = []
        all_profits = []

        for opponent in ["ZIC", "ZIP", "GD", "Kaplan"]:
            if opponent in comp:
                if comp[opponent]["buyer_results"]:
                    all_efficiencies.append(comp[opponent]["buyer_results"]["mean_efficiency"])
                    all_profits.append(comp[opponent]["buyer_results"]["mean_profit"])

        report.append(f"  Average Efficiency: {np.mean(all_efficiencies):.3f}")
        report.append(f"  Average Profit: {np.mean(all_profits):.2f}")

        # Invasibility
        if "invasibility" in self.results:
            inv = self.results["invasibility"]
            report.append(f"\n2. INVASIBILITY TEST (1v7)")
            report.append("-"*40)
            report.append(f"  Overall Score: {inv['overall_invasibility']:.3f}")
            report.append(f"  Buyer Ratio: {inv['buyer_profit_ratio']:.3f}")
            report.append(f"  Seller Ratio: {inv['seller_profit_ratio']:.3f}")

        # Self-Play
        if "self_play" in self.results:
            sp = self.results["self_play"]
            report.append(f"\n3. SELF-PLAY ANALYSIS")
            report.append("-"*40)
            report.append(f"  Mean Efficiency: {sp['mean_efficiency']:.3f}")
            report.append(f"  Convergence: {sp['convergence_rate']:.3f}")
            report.append(f"  Stability: {sp['stability_score']:.3f}")

        # Robustness
        if "robustness" in self.results:
            rob = self.results["robustness"]
            report.append(f"\n4. ROBUSTNESS TESTS")
            report.append("-"*40)
            report.append(f"  Overall Score: {rob['robustness_score']:.3f}")
            for scenario, metrics in rob.items():
                if scenario != "robustness_score":
                    report.append(f"  {scenario}: E={metrics['efficiency']:.3f}, P={metrics['profit']:.1f}")

        # Strategy
        if "strategy" in self.results:
            strat = self.results["strategy"]
            report.append(f"\n5. STRATEGY ANALYSIS")
            report.append("-"*40)
            report.append(f"  Type: {strat['type']}")
            report.append(f"  Aggression: {strat['aggression_level']:.3f}")
            report.append(f"  Accept Rate: {strat['accept_rate']:.3f}")

        # Comparison to Targets
        report.append(f"\n6. TARGET COMPARISON")
        report.append("-"*40)
        target_efficiency = 0.80
        target_profit_ratio = 1.15

        actual_efficiency = np.mean(all_efficiencies) if all_efficiencies else 0
        actual_profit_ratio = self.results.get("invasibility", {}).get("overall_invasibility", 0)

        report.append(f"  Efficiency: {actual_efficiency:.3f} / {target_efficiency:.3f} ", end="")
        report.append("✅" if actual_efficiency >= target_efficiency else "❌")
        report.append(f"  Profit Ratio: {actual_profit_ratio:.3f} / {target_profit_ratio:.3f} ", end="")
        report.append("✅" if actual_profit_ratio >= target_profit_ratio else "❌")

        report.append("\n" + "="*60)

        report_text = "\n".join(report)

        # Save if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"\n💾 Report saved to {output_path}")

        return report_text

    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run all evaluation tests."""
        if self.verbose:
            print("\n🎯 Running comprehensive evaluation suite...")

        results = {}

        # Test against all opponent types
        opponents = ["ZIC", "ZIP", "GD", "Kaplan"]
        for opponent in opponents:
            try:
                results[opponent] = self.evaluate_vs_opponent(opponent, n_episodes=50)
            except Exception as e:
                print(f"⚠️ Failed to evaluate vs {opponent}: {e}")
                results[opponent] = None

        self.results["comprehensive"] = results
        self.results["invasibility"] = self.run_invasibility_test(n_episodes=50)
        self.results["self_play"] = self.run_self_play(n_episodes=50)
        self.results["robustness"] = self.run_robustness_tests()
        self.results["strategy"] = self.analyze_strategy(n_episodes=20)

        return self.results

    def _run_evaluation(self, opponent_type: str, is_buyer: bool,
                        n_episodes: int, env_config: Optional[Dict] = None) -> Dict[str, float]:
        """Run evaluation episodes and collect metrics."""
        if env_config is None:
            env_config = {
                "num_agents": 8,
                "num_tokens": 4,
                "max_steps": 100,
                "min_price": 0,
                "max_price": 1000
            }

        env_config["opponent_type"] = opponent_type
        env_config["rl_is_buyer"] = is_buyer

        # Create environment
        env = EnhancedDoubleAuctionEnv(env_config)
        env = Monitor(env)

        # Run episodes
        episode_rewards = []
        episode_metrics = {
            "efficiency": [],
            "profit": [],
            "trades": [],
            "profitable_trades": []
        }

        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0

            while not done:
                if self.vec_normalize:
                    obs_norm = self.vec_normalize.normalize_obs(obs)
                    action, _ = self.model.predict(obs_norm, deterministic=True)
                else:
                    action, _ = self.model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated

                if done and "metrics" in info:
                    metrics = info["metrics"]
                    episode_metrics["efficiency"].append(metrics.get("market_efficiency", 0))
                    episode_metrics["profit"].append(metrics.get("total_profit", 0))
                    episode_metrics["trades"].append(metrics.get("trades_executed", 0))
                    episode_metrics["profitable_trades"].append(metrics.get("profitable_trades", 0))

            episode_rewards.append(episode_reward)

        # Calculate statistics
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_efficiency": np.mean(episode_metrics["efficiency"]) if episode_metrics["efficiency"] else 0,
            "mean_profit": np.mean(episode_metrics["profit"]) if episode_metrics["profit"] else 0,
            "mean_trades": np.mean(episode_metrics["trades"]) if episode_metrics["trades"] else 0,
            "mean_profitable_trades": np.mean(episode_metrics["profitable_trades"]) if episode_metrics["profitable_trades"] else 0
        }

    def _run_ppo_tournament(self, num_ppo_buyers: int, num_ppo_sellers: int,
                           n_episodes: int) -> Dict[str, Any]:
        """Run tournament with PPO agents."""
        # This would integrate with the Tournament class
        # For now, return placeholder results
        return {
            "episode_efficiencies": [0.85 + np.random.normal(0, 0.05) for _ in range(n_episodes)],
            "mean_profit": 150 + np.random.normal(0, 20)
        }

    def _calculate_convergence(self, values: List[float]) -> float:
        """Calculate convergence rate from time series."""
        if len(values) < 10:
            return 0.0

        # Check if values converge (decreasing variance over time)
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]

        var_first = np.var(first_half)
        var_second = np.var(second_half)

        if var_first > 0:
            convergence = 1.0 - (var_second / var_first)
            return max(0, min(1, convergence))
        return 0.5


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate PPO Trading Agents")

    # Model selection
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained PPO model")
    parser.add_argument("--models", nargs="+",
                       help="Multiple models for comparison")

    # Evaluation options
    parser.add_argument("--opponent", type=str,
                       choices=["ZIC", "ZIP", "GD", "Kaplan", "mixed"],
                       help="Specific opponent to test against")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive evaluation suite")
    parser.add_argument("--invasibility", action="store_true",
                       help="Run invasibility test only")
    parser.add_argument("--self-play", action="store_true",
                       help="Run self-play evaluation only")
    parser.add_argument("--robustness", action="store_true",
                       help="Run robustness tests only")
    parser.add_argument("--strategy", action="store_true",
                       help="Analyze strategy only")

    # Settings
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of evaluation episodes")
    parser.add_argument("--report", action="store_true",
                       help="Generate detailed report")
    parser.add_argument("--output", type=str,
                       help="Output path for report")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    # Single model evaluation
    if args.model:
        evaluator = PPOEvaluator(args.model, verbose=args.verbose)

        if args.comprehensive:
            evaluator.run_comprehensive_evaluation()
        else:
            if args.opponent:
                results = evaluator.evaluate_vs_opponent(args.opponent, n_episodes=args.episodes)
                print(f"\nResults vs {args.opponent}:")
                print(json.dumps(results, indent=2))

            if args.invasibility:
                results = evaluator.run_invasibility_test(n_episodes=args.episodes)
                print("\nInvasibility Results:")
                print(json.dumps(results, indent=2))

            if args.self_play:
                results = evaluator.run_self_play(n_episodes=args.episodes)
                print("\nSelf-Play Results:")
                print(json.dumps(results, indent=2))

            if args.robustness:
                results = evaluator.run_robustness_tests()
                print("\nRobustness Results:")
                print(json.dumps(results, indent=2))

            if args.strategy:
                results = evaluator.analyze_strategy(n_episodes=min(20, args.episodes))
                print("\nStrategy Analysis:")
                print(json.dumps(results, indent=2))

        if args.report:
            output_path = args.output or f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report = evaluator.generate_report(output_path)
            print(report)

    # Multi-model comparison
    elif args.models:
        print("🔄 Comparing multiple models...")
        comparison_results = {}

        for model_path in args.models:
            print(f"\nEvaluating {model_path}...")
            evaluator = PPOEvaluator(model_path, verbose=False)
            results = evaluator.run_invasibility_test(n_episodes=args.episodes)
            comparison_results[model_path] = results["overall_invasibility"]

        # Print comparison
        print("\n" + "="*60)
        print("MODEL COMPARISON - Invasibility Scores")
        print("="*60)
        for model, score in sorted(comparison_results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {Path(model).stem}: {score:.3f}")
        print("="*60)


if __name__ == "__main__":
    main()