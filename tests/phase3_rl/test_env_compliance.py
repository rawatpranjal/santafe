import pytest
import numpy as np
from stable_baselines3.common.env_checker import check_env
from envs.double_auction_env import DoubleAuctionEnv

def test_env_compliance():
    """Verify DoubleAuctionEnv follows Gymnasium API standards."""
    config = {
        "num_agents": 4,
        "num_tokens": 4,
        "max_steps": 10,
        "min_price": 0,
        "max_price": 100,
        "rl_agent_id": 1,
        "rl_is_buyer": True,
        "opponent_type": "ZIC"
    }
    
    env = DoubleAuctionEnv(config)
    
    # SB3 check_env runs a battery of tests:
    # - Observation space shape/dtype
    # - Action space shape/dtype
    # - Reset returns (obs, info)
    # - Step returns (obs, reward, terminated, truncated, info)
    # - Random actions work
    check_env(env, warn=True)

def test_reset_and_step():
    """Manual verification of reset and step loop."""
    config = {
        "num_agents": 4,
        "num_tokens": 2,
        "max_steps": 5,
        "min_price": 0,
        "max_price": 100,
        "rl_agent_id": 1,
        "rl_is_buyer": True,
        "opponent_type": "ZIC"
    }
    
    env = DoubleAuctionEnv(config)
    obs, info = env.reset(seed=42)
    
    assert obs.shape == (12,)
    assert isinstance(info, dict)
    
    # Run a few steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (12,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        
        if terminated or truncated:
            break
            
    assert terminated # Should terminate after max_steps
