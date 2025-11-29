"""
Vectorized Environment Utilities for Double Auction RL Training.

This module provides helper functions for creating vectorized environments
using Stable-Baselines3's SubprocVecEnv for parallel training.
"""

import numpy as np
from typing import Callable, Optional, Dict, Any, List
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from stable_baselines3.common.utils import set_random_seed

from envs.double_auction_env import DoubleAuctionEnv
from envs.enhanced_double_auction_env import EnhancedDoubleAuctionEnv

try:
    from sb3_contrib.common.wrappers import ActionMasker
    HAS_ACTION_MASKER = True
except ImportError:
    HAS_ACTION_MASKER = False


def make_env(
    rank: int,
    num_buyers: int = 5,
    num_sellers: int = 5,
    num_tokens_per_agent: int = 3,
    max_timesteps: int = 100,
    price_min: int = 0,
    price_max: int = 100,
    rl_agent_type: str = "buyer",
    opponent_type: str = "ZIC",
    seed: int = 0,
    use_enhanced_env: bool = True,
    pure_profit_mode: bool = False
) -> Callable[[], DoubleAuctionEnv]:
    """
    Create environment factory for vectorization.

    This function returns a callable that creates a DoubleAuctionEnv instance.
    Each parallel environment gets a unique seed (base_seed + rank) for
    reproducibility while maintaining diversity across processes.

    Args:
        rank: Index of the environment (0 to n_envs-1)
        num_buyers: Number of buyer agents
        num_sellers: Number of seller agents
        num_tokens_per_agent: Tokens allocated per agent
        max_timesteps: Maximum timesteps per episode
        price_min: Minimum allowed price
        price_max: Maximum allowed price
        rl_agent_type: "buyer" or "seller"
        opponent_type: Type of opponent agents (e.g., "ZIC")
        seed: Base random seed
        use_enhanced_env: If True, use EnhancedDoubleAuctionEnv (default)
        pure_profit_mode: If True, use raw profit as reward (no shaping)

    Returns:
        Callable that creates and returns a configured environment
    """
    def _init() -> DoubleAuctionEnv:
        """Initialize environment with rank-specific seed."""
        # Build config dict for environment
        config = {
            "num_agents": num_buyers + num_sellers,
            "num_tokens": num_tokens_per_agent,
            "max_steps": max_timesteps,
            "min_price": price_min,
            "max_price": price_max,
            "rl_agent_id": 1,
            "rl_is_buyer": rl_agent_type == "buyer",
            "opponent_type": opponent_type,
            "pure_profit_mode": pure_profit_mode,
        }

        if use_enhanced_env:
            env = EnhancedDoubleAuctionEnv(config)
        else:
            env = DoubleAuctionEnv(config)

        # Wrap with ActionMasker for MaskablePPO support
        if HAS_ACTION_MASKER:
            def mask_fn(env) -> np.ndarray:
                """Extract action mask from environment."""
                return env._get_action_mask()
            env = ActionMasker(env, mask_fn)

        # Reset with rank-specific seed for deterministic initialization
        env.reset(seed=seed + rank)
        return env

    # Set random seed for this process
    set_random_seed(seed + rank)
    return _init


def make_vec_env(
    n_envs: int = 16,
    start_method: Optional[str] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 0
) -> VecEnv:
    """
    Create vectorized environments using SubprocVecEnv.

    Creates n_envs parallel environments, each running in its own process.
    This enables true parallelization for faster training on multi-core CPUs.

    Args:
        n_envs: Number of parallel environments (default: 16)
        start_method: Multiprocessing start method
            - None: Use platform default (fork on Linux, spawn on Windows)
            - "fork": Fast but not thread-safe (Linux only)
            - "spawn": Slower startup, thread-safe (cross-platform)
            - "forkserver": Balanced option, thread-safe (recommended)
        env_kwargs: Keyword arguments passed to DoubleAuctionEnv
        seed: Base random seed (each env gets seed + rank)

    Returns:
        SubprocVecEnv with n_envs parallel environments

    Example:
        >>> vec_env = make_vec_env(
        ...     n_envs=16,
        ...     start_method="forkserver",
        ...     env_kwargs={"num_buyers": 5, "num_sellers": 5},
        ...     seed=42
        ... )
        >>> obs = vec_env.reset()
        >>> obs.shape
        (16, 9)
    """
    if env_kwargs is None:
        env_kwargs = {}

    # Create list of environment factory functions
    env_fns: List[Callable[[], DoubleAuctionEnv]] = [
        make_env(rank=i, seed=seed, **env_kwargs)
        for i in range(n_envs)
    ]

    # Create vectorized environment
    vec_env = SubprocVecEnv(
        env_fns,
        start_method=start_method
    )

    return vec_env


def make_eval_vec_env(
    n_envs: int = 4,
    start_method: Optional[str] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 1000
) -> VecEnv:
    """
    Create vectorized environments for evaluation.

    Similar to make_vec_env but with different default seed to ensure
    evaluation happens on different market configurations than training.

    Args:
        n_envs: Number of parallel eval environments (default: 4)
        start_method: Multiprocessing start method
        env_kwargs: Keyword arguments passed to DoubleAuctionEnv
        seed: Base random seed for eval (default: 1000, different from training)

    Returns:
        SubprocVecEnv for evaluation
    """
    return make_vec_env(
        n_envs=n_envs,
        start_method=start_method,
        env_kwargs=env_kwargs,
        seed=seed
    )


def get_default_env_kwargs(
    curriculum_stage: str = "zic",
    rl_agent_type: str = "buyer",
    pure_profit_mode: bool = True
) -> Dict[str, Any]:
    """
    Get default environment configuration for curriculum learning.

    Args:
        curriculum_stage: Training stage
            - "zic": Train against ZIC agents (easiest)
            - "kaplan": Train against Kaplan agents (harder)
            - "mixed": Train against mixed agent types (hardest)
        rl_agent_type: "buyer" or "seller"
        pure_profit_mode: If True, use raw profit as reward (default True for Chen et al. style)

    Returns:
        Dictionary of environment kwargs
    """
    base_kwargs = {
        "num_buyers": 5,
        "num_sellers": 5,
        "num_tokens_per_agent": 3,
        "max_timesteps": 100,
        "price_min": 0,
        "price_max": 100,
        "rl_agent_type": rl_agent_type,
        "use_enhanced_env": True,
        "pure_profit_mode": pure_profit_mode,
    }

    # Curriculum-specific settings
    if curriculum_stage == "zic":
        base_kwargs["opponent_type"] = "ZIC"
    elif curriculum_stage == "kaplan":
        base_kwargs["opponent_type"] = "Kaplan"
    elif curriculum_stage == "mixed":
        base_kwargs["opponent_type"] = "Mixed"
    else:
        raise ValueError(f"Unknown curriculum stage: {curriculum_stage}")

    return base_kwargs
