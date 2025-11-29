#!/usr/bin/env python3
"""Debug evaluation to see what info dict contains."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from envs.enhanced_double_auction_env import EnhancedDoubleAuctionEnv

# Create environment
config = {
    "num_agents": 8,
    "num_tokens": 4,
    "max_steps": 100,
    "min_price": 0,
    "max_price": 1000,
    "rl_agent_id": 1,
    "rl_is_buyer": True,
    "opponent_type": "ZIC",
    "profit_weight": 1.0,
    "market_making_weight": 0.5,
    "exploration_weight": 0.05,
    "invalid_penalty": -0.01,
    "efficiency_bonus_weight": 0.2,
    "bid_submission_bonus": 0.02,
    "surplus_capture_weight": 0.1,
    "normalize_rewards": False
}

def make_env():
    env = EnhancedDoubleAuctionEnv(config)
    env = Monitor(env)
    return env

# Create vectorized environment
eval_env = DummyVecEnv([make_env])

# Wrap in VecNormalize
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

# Load model and normalization
model = PPO.load("models/ppo_curriculum/stage_basics.zip", env=eval_env)
norm_path = "models/ppo_curriculum/vec_normalize.pkl"
eval_env = VecNormalize.load(norm_path, eval_env)
eval_env.training = False
eval_env.norm_reward = False

print("Running one episode to debug info dict...")
obs = eval_env.reset()
done = False
step_count = 0

while not (done[0] if isinstance(done, np.ndarray) else done):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)

    step_count += 1

    # Check info structure
    if step_count == 1:
        print(f"\nStep {step_count}:")
        print(f"  Info type: {type(info)}")
        if isinstance(info, list):
            print(f"  Info is list, length: {len(info)}")
            print(f"  Info[0] type: {type(info[0])}")
            print(f"  Info[0] keys: {list(info[0].keys())}")
            if "metrics" in info[0]:
                print(f"  Metrics: {info[0]['metrics']}")
        elif isinstance(info, dict):
            print(f"  Info keys: {list(info.keys())}")
            if "metrics" in info:
                print(f"  Metrics: {info['metrics']}")

    # Check when done
    episode_done = done[0] if isinstance(done, np.ndarray) else done
    if episode_done:
        print(f"\nEpisode ended at step {step_count}:")
        print(f"  Done type: {type(done)}, value: {done}")
        print(f"  Info type: {type(info)}")
        if isinstance(info, list):
            print(f"  Info is list, length: {len(info)}")
            info_dict = info[0]
        else:
            info_dict = info

        print(f"  Info dict keys: {list(info_dict.keys())}")
        print(f"  Has 'metrics': {'metrics' in info_dict}")

        if "metrics" in info_dict:
            metrics = info_dict["metrics"]
            print(f"  Metrics keys: {list(metrics.keys())}")
            print(f"  Efficiency: {metrics.get('market_efficiency', 'NOT FOUND')}")
            print(f"  Total profit: {metrics.get('total_profit', 'NOT FOUND')}")
            print(f"  Trades: {metrics.get('trades_executed', 'NOT FOUND')}")
        else:
            print("  ⚠️ NO METRICS IN INFO DICT!")
            print(f"  Info dict content: {info_dict}")

        break

print("\n✅ Debug complete")
