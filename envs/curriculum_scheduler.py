"""
Curriculum Learning Scheduler for Progressive PPO Training.

This module implements a curriculum learning system that:
1. Manages staged difficulty progression
2. Monitors performance and advances stages
3. Adjusts environment and PPO parameters
4. Tracks curriculum metrics
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from collections import deque

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum stage."""
    name: str
    timesteps: int
    env_config: Dict[str, Any]
    ppo_config: Dict[str, Any]
    success_criteria: Dict[str, float]

    # Tracking
    started_at: int = 0
    completed_at: int = 0
    best_efficiency: float = 0.0
    best_profit_ratio: float = 0.0
    attempts: int = 0


class CurriculumScheduler:
    """
    Manages curriculum learning progression for PPO training.

    Automatically advances through stages based on performance criteria.
    """

    def __init__(self,
                 config: Dict[str, Any],
                 base_env_fn: callable,
                 verbose: int = 1):
        """
        Initialize curriculum scheduler.

        Args:
            config: Curriculum configuration dictionary
            base_env_fn: Function to create environment with given config
            verbose: Logging verbosity
        """
        self.config = config
        self.base_env_fn = base_env_fn
        self.verbose = verbose

        # Parse stages
        self.stages = []
        for stage_config in config["curriculum"]["stages"]:
            stage = CurriculumStage(**stage_config)
            self.stages.append(stage)

        # State tracking
        self.current_stage_idx = 0
        self.current_stage = self.stages[0]
        self.total_timesteps = 0
        self.stage_history = []

        # Performance tracking
        self.eval_buffer = deque(maxlen=10)  # Last 10 evaluations
        self.efficiency_history = []
        self.profit_history = []

        # Settings
        self.auto_advance = config["curriculum"].get("auto_advance", True)
        self.patience = config["training"].get("patience", 100_000)
        self.warmup_steps = config["training"].get("warmup_steps", 10_000)
        self.eval_freq = config["training"].get("curriculum_eval_freq", 25_000)

    def get_current_env_config(self) -> Dict[str, Any]:
        """Get environment configuration for current stage."""
        base_config = self.config["env"].copy()
        base_config.update(self.current_stage.env_config)
        return base_config

    def get_current_ppo_config(self) -> Dict[str, Any]:
        """Get PPO configuration for current stage."""
        base_config = self.config["ppo"].copy()
        base_config.update(self.current_stage.ppo_config)
        return base_config

    def should_advance(self, eval_results: Dict[str, float]) -> bool:
        """
        Check if current stage success criteria are met.

        Args:
            eval_results: Dictionary with evaluation metrics

        Returns:
            True if should advance to next stage
        """
        if not self.auto_advance:
            return False

        # Don't advance during warmup
        stage_steps = self.total_timesteps - self.current_stage.started_at
        if stage_steps < self.warmup_steps:
            return False

        # Check if spent too long in stage (patience exceeded)
        if stage_steps > self.current_stage.timesteps + self.patience:
            if self.verbose:
                print(f"⚠️ Stage '{self.current_stage.name}' patience exceeded")
            return True

        # Check success criteria
        criteria = self.current_stage.success_criteria

        efficiency = eval_results.get("efficiency", 0.0)
        profit_ratio = eval_results.get("profit_ratio", 0.0)

        # Update best scores
        self.current_stage.best_efficiency = max(
            self.current_stage.best_efficiency, efficiency
        )
        self.current_stage.best_profit_ratio = max(
            self.current_stage.best_profit_ratio, profit_ratio
        )

        # Check if criteria met
        efficiency_met = efficiency >= criteria.get("min_efficiency", 0.0)
        profit_met = profit_ratio >= criteria.get("min_profit_ratio", 0.0)

        if efficiency_met and profit_met:
            if self.verbose:
                print(f"✅ Stage '{self.current_stage.name}' success criteria met!")
                print(f"   Efficiency: {efficiency:.3f} >= {criteria['min_efficiency']:.3f}")
                print(f"   Profit Ratio: {profit_ratio:.3f} >= {criteria['min_profit_ratio']:.3f}")
            return True

        return False

    def advance_stage(self) -> bool:
        """
        Advance to next curriculum stage.

        Returns:
            True if advanced, False if at final stage
        """
        # Complete current stage
        self.current_stage.completed_at = self.total_timesteps
        self.stage_history.append(asdict(self.current_stage))

        # Check if more stages
        if self.current_stage_idx >= len(self.stages) - 1:
            if self.verbose:
                print(f"🎓 Curriculum complete! All {len(self.stages)} stages finished.")
            return False

        # Advance to next stage
        self.current_stage_idx += 1
        self.current_stage = self.stages[self.current_stage_idx]
        self.current_stage.started_at = self.total_timesteps
        self.current_stage.attempts += 1

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📚 ADVANCING TO STAGE {self.current_stage_idx + 1}/{len(self.stages)}: {self.current_stage.name}")
            print(f"   Opponents: {self.current_stage.env_config.get('opponent_mix', self.current_stage.env_config.get('opponent_type'))}")
            print(f"   Difficulty: {self.current_stage.env_config.get('difficulty', 'medium')}")
            print(f"   Duration: {self.current_stage.timesteps:,} timesteps")
            print(f"{'='*60}\n")

        # Clear evaluation buffer for new stage
        self.eval_buffer.clear()

        return True

    def get_stage_progress(self) -> Dict[str, Any]:
        """Get current stage progress information."""
        stage_steps = self.total_timesteps - self.current_stage.started_at
        stage_progress = stage_steps / self.current_stage.timesteps

        return {
            "stage_name": self.current_stage.name,
            "stage_idx": self.current_stage_idx,
            "total_stages": len(self.stages),
            "stage_steps": stage_steps,
            "stage_progress": stage_progress,
            "total_timesteps": self.total_timesteps,
            "best_efficiency": self.current_stage.best_efficiency,
            "best_profit_ratio": self.current_stage.best_profit_ratio,
            "attempts": self.current_stage.attempts
        }

    def save_curriculum_state(self, path: Path) -> None:
        """Save curriculum state to file."""
        state = {
            "current_stage_idx": self.current_stage_idx,
            "total_timesteps": self.total_timesteps,
            "stage_history": self.stage_history,
            "efficiency_history": self.efficiency_history,
            "profit_history": self.profit_history,
            "stages": [asdict(stage) for stage in self.stages]
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_curriculum_state(self, path: Path) -> None:
        """Load curriculum state from file."""
        with open(path, 'r') as f:
            state = json.load(f)

        self.current_stage_idx = state["current_stage_idx"]
        self.total_timesteps = state["total_timesteps"]
        self.stage_history = state["stage_history"]
        self.efficiency_history = state["efficiency_history"]
        self.profit_history = state["profit_history"]

        # Reconstruct stages
        self.stages = []
        for stage_dict in state["stages"]:
            self.stages.append(CurriculumStage(**stage_dict))

        self.current_stage = self.stages[self.current_stage_idx]


class CurriculumCallback(BaseCallback):
    """
    Callback for managing curriculum progression during training.

    Evaluates performance and advances stages when criteria are met.
    """

    def __init__(self,
                 scheduler: CurriculumScheduler,
                 eval_env: VecEnv,
                 eval_freq: int = 10_000,
                 n_eval_episodes: int = 100,
                 verbose: int = 1):
        """
        Initialize curriculum callback.

        Args:
            scheduler: Curriculum scheduler instance
            eval_env: Environment for evaluation
            eval_freq: Frequency of evaluation
            n_eval_episodes: Number of episodes for evaluation
            verbose: Logging verbosity
        """
        super().__init__(verbose)
        self.scheduler = scheduler
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.last_eval = 0

    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Update scheduler timestep count
        self.scheduler.total_timesteps = self.num_timesteps

        # Check if time to evaluate
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self.last_eval = self.num_timesteps

            # Evaluate current policy
            eval_results = self._evaluate_policy()

            # Check if should advance stage
            if self.scheduler.should_advance(eval_results):
                # Save model before advancing
                if self.scheduler.config.get("save_stage_models", True):
                    stage_name = self.scheduler.current_stage.name
                    model_path = Path(self.scheduler.config["output"]["model_save_path"]) / f"stage_{stage_name}.zip"
                    self.model.save(model_path)
                    if self.verbose:
                        print(f"💾 Saved stage model to {model_path}")

                # Advance to next stage
                if self.scheduler.advance_stage():
                    # Update environment configuration
                    self._update_environment()

                    # Update PPO hyperparameters
                    self._update_model_params()
                else:
                    # Curriculum complete
                    if self.verbose:
                        print("🎯 Curriculum learning complete!")
                    return False  # Stop training

        # Log curriculum progress
        if self.num_timesteps % 10000 == 0:
            progress = self.scheduler.get_stage_progress()
            self.logger.record("curriculum/stage", progress["stage_idx"])
            self.logger.record("curriculum/stage_progress", progress["stage_progress"])
            self.logger.record("curriculum/best_efficiency", progress["best_efficiency"])
            self.logger.record("curriculum/best_profit_ratio", progress["best_profit_ratio"])

        return True  # Continue training

    def _evaluate_policy(self) -> Dict[str, float]:
        """
        Evaluate current policy and return metrics.

        Returns:
            Dictionary with efficiency and profit ratio
        """
        # Run evaluation episodes
        episode_rewards = []
        episode_lengths = []
        episode_metrics = {
            "efficiency": [],
            "profit": [],
            "trades": []
        }

        # Run evaluation episodes
        episode_rewards = []
        episode_lengths = []
        episode_metrics = {
            "efficiency": [],
            "profit": [],
            "trades": []
        }

        obs = self.eval_env.reset()
        n_envs = self.eval_env.num_envs
        current_rewards = np.zeros(n_envs)
        current_lengths = np.zeros(n_envs, dtype=int)
        episode_counts = 0

        while episode_counts < self.n_eval_episodes:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = self.eval_env.step(action)
            
            current_rewards += rewards
            current_lengths += 1
            
            for i, done in enumerate(dones):
                if done:
                    episode_counts += 1
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    
                    # Extract metrics from info
                    if "metrics" in infos[i]:
                        metrics = infos[i]["metrics"]
                        episode_metrics["efficiency"].append(metrics.get("market_efficiency", 0))
                        episode_metrics["profit"].append(metrics.get("total_profit", 0))
                        episode_metrics["trades"].append(metrics.get("trades_executed", 0))
                    
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    
                    if episode_counts >= self.n_eval_episodes:
                        break

        # Calculate statistics
        mean_reward = np.mean(episode_rewards)
        mean_efficiency = np.mean(episode_metrics["efficiency"]) if episode_metrics["efficiency"] else 0
        mean_profit = np.mean(episode_metrics["profit"]) if episode_metrics["profit"] else 0
        mean_trades = np.mean(episode_metrics["trades"]) if episode_metrics["trades"] else 0

        # Calculate profit ratio (vs baseline expected profit)
        baseline_profit = 100  # Expected random agent profit
        profit_ratio = mean_profit / baseline_profit if baseline_profit > 0 else 0

        results = {
            "mean_reward": mean_reward,
            "efficiency": mean_efficiency,
            "profit": mean_profit,
            "profit_ratio": profit_ratio,
            "trades": mean_trades
        }

        # Update scheduler's tracking
        self.scheduler.efficiency_history.append(mean_efficiency)
        self.scheduler.profit_history.append(profit_ratio)
        self.scheduler.eval_buffer.append(results)

        if self.verbose:
            print(f"\n📊 Evaluation at {self.num_timesteps:,} steps:")
            print(f"   Mean Reward: {mean_reward:.2f}")
            print(f"   Efficiency: {mean_efficiency:.3f}")
            print(f"   Profit Ratio: {profit_ratio:.3f}")
            print(f"   Trades/Episode: {mean_trades:.1f}")

        return results

    def _update_environment(self) -> None:
        """Update environment configuration for new stage."""
        # Get new environment config
        new_config = self.scheduler.get_current_env_config()

        # Update training environments
        # We use env_method to call update_config on all sub-environments
        if hasattr(self.model.env, "env_method"):
            self.model.env.env_method("update_config", new_config)
        elif hasattr(self.model.env, "envs"):
            for env in self.model.env.envs:
                if hasattr(env, "update_config"):
                    env.update_config(new_config)
        
        # Update evaluation environments
        if hasattr(self.eval_env, "env_method"):
            self.eval_env.env_method("update_config", new_config)
        elif hasattr(self.eval_env, "envs"):
            for env in self.eval_env.envs:
                if hasattr(env, "update_config"):
                    env.update_config(new_config)

        if self.verbose:
            print(f"🔄 Updated environment configuration for stage '{self.scheduler.current_stage.name}'")

    def _update_model_params(self) -> None:
        """Update PPO hyperparameters for new stage."""
        new_params = self.scheduler.get_current_ppo_config()

        # Update learning rate
        if "learning_rate" in new_params:
            self.model.learning_rate = new_params["learning_rate"]

        # Update entropy coefficient
        if "ent_coef" in new_params:
            self.model.ent_coef = new_params["ent_coef"]

        # Update other parameters
        if "clip_range" in new_params:
            self.model.clip_range = new_params["clip_range"]
        if "n_epochs" in new_params:
            self.model.n_epochs = new_params["n_epochs"]

        if self.verbose:
            print(f"🔧 Updated PPO parameters for stage '{self.scheduler.current_stage.name}'")
            print(f"   Learning Rate: {self.model.learning_rate}")
            print(f"   Entropy Coef: {self.model.ent_coef}")