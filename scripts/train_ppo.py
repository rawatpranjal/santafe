"""
PPO Training Script for Double Auction Market RL Agent.

This script sets up vectorized environments, initializes PPO with action masking,
and runs the training loop with W&B logging and checkpointing.

Usage:
    python train_ppo.py                              # Use default configs
    python train_ppo.py ppo.learning_rate=0.0001     # Override specific param
    python train_ppo.py vectorization.n_envs=8       # Use 8 parallel envs
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.logger import configure


class EntropyScheduleCallback(BaseCallback):
    """
    Callback that decays the entropy coefficient from start to end over training.

    This encourages broad exploration early (high entropy) and exploitation later (low entropy).
    """

    def __init__(
        self,
        start_ent_coef: float = 0.1,
        end_ent_coef: float = 0.01,
        total_timesteps: int = 200_000,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.start_ent_coef = start_ent_coef
        self.end_ent_coef = end_ent_coef
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Linear decay
        progress = self.num_timesteps / self.total_timesteps
        progress = min(progress, 1.0)  # Clamp to [0, 1]

        new_ent_coef = self.start_ent_coef + (self.end_ent_coef - self.start_ent_coef) * progress
        self.model.ent_coef = new_ent_coef

        # Log periodically
        if self.verbose > 0 and self.num_timesteps % 10000 == 0:
            print(f"  [EntropySchedule] Step {self.num_timesteps}: ent_coef={new_ent_coef:.4f}")

        return True

# W&B integration
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from envs.vec_env_utils import make_vec_env, make_eval_vec_env


def setup_wandb(cfg: DictConfig) -> Optional[Any]:
    """
    Initialize Weights & Biases logging.

    Args:
        cfg: Hydra config

    Returns:
        WandB run object or None if disabled/unavailable
    """
    if not WANDB_AVAILABLE or not cfg.rl.wandb.enabled:
        print("W&B logging disabled")
        return None

    # Initialize W&B
    run = wandb.init(
        project=cfg.rl.wandb.project,
        entity=cfg.rl.wandb.entity,
        name=cfg.rl.wandb.name,
        tags=list(cfg.rl.wandb.tags),
        notes=cfg.rl.wandb.notes,
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=True,  # Sync SB3 tensorboard logs
        monitor_gym=True,  # Log environment stats
        save_code=True
    )

    print(f"W&B run initialized: {run.name}")
    return run


def create_training_env(cfg: DictConfig) -> VecEnv:
    """
    Create vectorized training environment.

    Args:
        cfg: Hydra config

    Returns:
        Vectorized environment for training
    """
    env_kwargs = {
        "num_buyers": cfg.rl.env.num_buyers,
        "num_sellers": cfg.rl.env.num_sellers,
        "num_tokens_per_agent": cfg.rl.env.num_tokens_per_agent,
        "max_timesteps": cfg.rl.env.max_timesteps,
        "price_min": cfg.rl.env.price_min,
        "price_max": cfg.rl.env.price_max,
        "rl_agent_type": cfg.rl.env.rl_agent_type,
        "opponent_type": cfg.rl.env.opponent_type,
        "use_enhanced_env": cfg.rl.env.get("use_enhanced_env", True),
        "pure_profit_mode": cfg.rl.env.get("pure_profit_mode", True),
    }

    vec_env = make_vec_env(
        n_envs=cfg.rl.n_envs,
        start_method=cfg.rl.start_method,
        env_kwargs=env_kwargs,
        seed=cfg.rl.seed
    )

    print(f"Created {cfg.rl.n_envs} parallel training environments")
    return vec_env


def create_eval_env(cfg: DictConfig) -> VecEnv:
    """
    Create vectorized evaluation environment.

    Args:
        cfg: Hydra config

    Returns:
        Vectorized environment for evaluation
    """
    env_kwargs = {
        "num_buyers": cfg.rl.env.num_buyers,
        "num_sellers": cfg.rl.env.num_sellers,
        "num_tokens_per_agent": cfg.rl.env.num_tokens_per_agent,
        "max_timesteps": cfg.rl.env.max_timesteps,
        "price_min": cfg.rl.env.price_min,
        "price_max": cfg.rl.env.price_max,
        "rl_agent_type": cfg.rl.env.rl_agent_type,
        "opponent_type": cfg.rl.env.opponent_type,
        "use_enhanced_env": cfg.rl.env.get("use_enhanced_env", True),
        "pure_profit_mode": cfg.rl.env.get("pure_profit_mode", True),
    }

    eval_vec_env = make_eval_vec_env(
        n_envs=cfg.rl.eval.n_envs,
        start_method=cfg.rl.start_method,
        env_kwargs=env_kwargs,
        seed=cfg.rl.eval.seed
    )

    print(f"Created {cfg.rl.eval.n_envs} parallel evaluation environments")
    return eval_vec_env


def create_callbacks(cfg: DictConfig, eval_env: VecEnv) -> CallbackList:
    """
    Create training callbacks for checkpointing, evaluation, and logging.

    Args:
        cfg: Hydra config
        eval_env: Vectorized evaluation environment

    Returns:
        List of callbacks
    """
    callbacks = []

    # Entropy schedule callback - decay from 0.15 to 0.005 over training
    entropy_callback = EntropyScheduleCallback(
        start_ent_coef=cfg.rl.ent_coef,  # Initial value from config (0.15)
        end_ent_coef=0.005,  # Final value for sharp exploitation
        total_timesteps=cfg.rl.total_timesteps,
        verbose=1
    )
    callbacks.append(entropy_callback)

    # Checkpoint callback - save model periodically
    checkpoint_dir = Path(cfg.rl.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.rl.save_freq,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_double_auction",
        save_replay_buffer=cfg.rl.save_replay_buffer,
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback - periodic evaluation on held-out envs
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best_model"),
        log_path=str(checkpoint_dir / "eval_logs"),
        eval_freq=cfg.rl.eval_freq,
        n_eval_episodes=cfg.rl.eval_episodes,
        deterministic=cfg.rl.eval_deterministic,
        render=False
    )
    callbacks.append(eval_callback)

    # W&B callback - log to Weights & Biases
    if WANDB_AVAILABLE and cfg.rl.wandb.enabled:
        wandb_callback = WandbCallback(
            gradient_save_freq=1000,
            model_save_path=str(checkpoint_dir / "wandb_model"),
            verbose=2
        )
        callbacks.append(wandb_callback)

    return CallbackList(callbacks)


def create_ppo_model(cfg: DictConfig, env: VecEnv) -> MaskablePPO:
    """
    Create MaskablePPO model with action masking and hyperparameters from config.

    Args:
        cfg: Hydra config
        env: Training environment

    Returns:
        Initialized MaskablePPO model
    """
    # Parse policy kwargs
    policy_kwargs = {}
    if "policy_kwargs" in cfg.rl:
        policy_kwargs = OmegaConf.to_container(cfg.rl.policy_kwargs, resolve=True)
        # Convert activation_fn string to actual class
        if "activation_fn" in policy_kwargs:
            act_fn_str = policy_kwargs["activation_fn"]
            if act_fn_str == "torch.nn.ReLU":
                policy_kwargs["activation_fn"] = torch.nn.ReLU
            elif act_fn_str == "torch.nn.Tanh":
                policy_kwargs["activation_fn"] = torch.nn.Tanh
            elif act_fn_str == "torch.nn.LeakyReLU":
                policy_kwargs["activation_fn"] = torch.nn.LeakyReLU

    # Create MaskablePPO model (supports action masking from info["action_mask"])
    model = MaskablePPO(
        policy=cfg.rl.policy,
        env=env,
        learning_rate=cfg.rl.learning_rate,
        n_steps=cfg.rl.n_steps,
        batch_size=cfg.rl.batch_size,
        n_epochs=cfg.rl.n_epochs,
        gamma=cfg.rl.gamma,
        gae_lambda=cfg.rl.gae_lambda,
        clip_range=cfg.rl.clip_range,
        clip_range_vf=cfg.rl.clip_range_vf,
        normalize_advantage=cfg.rl.normalize_advantage,
        ent_coef=cfg.rl.ent_coef,
        vf_coef=cfg.rl.vf_coef,
        max_grad_norm=cfg.rl.max_grad_norm,
        # use_sde not supported by MaskablePPO
        policy_kwargs=policy_kwargs,
        verbose=cfg.rl.verbose,
        seed=cfg.rl.seed,
        device=cfg.rl.device,
        tensorboard_log=cfg.rl.tensorboard_log,
    )

    print(f"Created MaskablePPO model with policy: {cfg.rl.policy}")
    print(f"  Learning rate: {cfg.rl.learning_rate}")
    print(f"  Batch size: {cfg.rl.batch_size}")
    print(f"  N steps: {cfg.rl.n_steps}")
    print(f"  Action masking: ENABLED (reads from info['action_mask'])")

    return model


@hydra.main(version_base=None, config_path="../conf", config_name="train_config")
def main(cfg: DictConfig) -> None:
    """
    Main training loop.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("PPO TRAINING - Double Auction Market")
    print("=" * 80)
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    print("=" * 80)

    # Set random seeds for reproducibility
    np.random.seed(cfg.rl.seed)
    torch.manual_seed(cfg.rl.seed)
    if cfg.rl.get("deterministic_cudnn", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Initialize W&B
    wandb_run = setup_wandb(cfg)

    # Create environments
    print("\n[1/4] Creating vectorized environments...")
    train_env = create_training_env(cfg)
    eval_env = create_eval_env(cfg)

    # Create PPO model
    print("\n[2/4] Initializing PPO model...")
    model = create_ppo_model(cfg, train_env)

    # Create callbacks
    print("\n[3/4] Setting up callbacks...")
    callbacks = create_callbacks(cfg, eval_env)

    # Train model
    print("\n[4/4] Starting training...")
    print(f"Total timesteps: {cfg.rl.total_timesteps:,}")
    print(f"Parallel environments: {cfg.rl.n_envs}")
    print(f"Effective episodes: ~{cfg.rl.total_timesteps // cfg.rl.env.max_timesteps:,}")
    print("-" * 80)

    try:
        model.learn(
            total_timesteps=cfg.rl.total_timesteps,
            callback=callbacks,
            log_interval=cfg.rl.log_interval,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # Save final model
    print("\n" + "=" * 80)
    print("Training complete!")
    final_model_path = Path(cfg.rl.checkpoint_dir) / "final_model"
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # Close environments
    train_env.close()
    eval_env.close()

    # Finish W&B run
    if wandb_run is not None:
        wandb.finish()

    print("=" * 80)


if __name__ == "__main__":
    # Required for multiprocessing on Windows and with forkserver/spawn
    main()
