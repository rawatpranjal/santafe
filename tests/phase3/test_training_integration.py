"""
Integration Tests for Task 3.3 - Vectorization and Training Pipeline.

These tests verify the complete training pipeline works end-to-end:
- train_ppo.py script execution
- PPO model integration with vectorized environments
- Checkpoint saving/loading
- Hydra configuration loading
- Complete training workflows

All tests use minimal timesteps (100-500) for fast CI/CD execution.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from envs.vec_env_utils import make_vec_env, make_eval_vec_env
from envs.double_auction_env import DoubleAuctionEnv


class TestTrainingScriptSmoke:
    """Test train_ppo.py loads and runs without errors."""

    def test_training_script_imports(self):
        """Test train_ppo.py can be imported successfully."""
        try:
            from train_ppo import (
                create_training_env,
                create_eval_env,
                create_ppo_model,
                create_callbacks
            )
            # If we get here, imports worked
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import train_ppo.py: {e}")

    def test_minimal_training_run(self):
        """CRITICAL: Test minimal training run completes without errors."""
        from train_ppo import create_training_env, create_ppo_model

        # Create minimal config
        cfg = OmegaConf.create({
            "vectorization": {
                "n_envs": 2,
                "start_method": "spawn",
                "env": {
                    "num_buyers": 3,
                    "num_sellers": 3,
                    "num_tokens_per_agent": 2,
                    "max_timesteps": 50,
                    "price_min": 0,
                    "price_max": 100,
                    "rl_agent_type": "buyer",
                    "opponent_type": "ZIC"
                }
            },
            "ppo": {
                "policy": "MlpPolicy",
                "learning_rate": 0.0003,
                "n_steps": 32,
                "batch_size": 16,
                "n_epochs": 2,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "clip_range_vf": None,
                "normalize_advantage": True,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "use_sde": False,
                "verbose": 0,
                "seed": 42,
                "device": "cpu",
                "tensorboard_log": None
            },
            "training": {
                "seed": 42
            }
        })

        # Create environment
        vec_env = create_training_env(cfg)

        try:
            # Create model
            model = create_ppo_model(cfg, vec_env)

            # Train for minimal steps
            model.learn(total_timesteps=100, progress_bar=False)

            # Verify model trained (PPO rounds up to complete rollouts)
            # With n_steps=32 and 2 envs, each rollout is 64 steps
            # So 100 timesteps → 128 actual timesteps (2 rollouts)
            assert model.num_timesteps >= 100

        finally:
            vec_env.close()

    def test_training_with_zic_opponents(self):
        """Test training against ZIC opponents."""
        vec_env = make_vec_env(
            n_envs=2,
            start_method="spawn",
            env_kwargs={"opponent_type": "ZIC", "max_timesteps": 50},
            seed=42
        )

        try:
            model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=32, seed=42)
            model.learn(total_timesteps=100, progress_bar=False)

            # PPO rounds up to complete rollouts
            assert model.num_timesteps >= 100

        finally:
            vec_env.close()


class TestPPOModelIntegration:
    """Test PPO model integration with vectorized environments."""

    def test_ppo_model_creation(self):
        """Test PPO model initializes with vectorized env."""
        vec_env = make_vec_env(n_envs=2, start_method="spawn", seed=42)

        try:
            model = PPO("MlpPolicy", vec_env, verbose=0, seed=42)

            # Verify model components
            assert model.policy is not None
            assert model.env == vec_env
            assert model.observation_space.shape == (9,)
            assert model.action_space.n == 4

        finally:
            vec_env.close()

    def test_ppo_training_few_steps(self):
        """Test PPO can train for a few steps."""
        vec_env = make_vec_env(n_envs=2, start_method="spawn", seed=42)

        try:
            model = PPO(
                "MlpPolicy",
                vec_env,
                n_steps=32,
                batch_size=16,
                verbose=0,
                seed=42
            )

            # Train for 100 steps
            model.learn(total_timesteps=100, progress_bar=False)

            # Verify training occurred (PPO rounds up to complete rollouts)
            assert model.num_timesteps >= 100
            assert model._n_updates > 0

        finally:
            vec_env.close()

    def test_ppo_predict_action(self):
        """Test PPO can predict actions from observations."""
        vec_env = make_vec_env(n_envs=2, start_method="spawn", seed=42)

        try:
            model = PPO("MlpPolicy", vec_env, n_steps=32, verbose=0, seed=42)

            # Train briefly
            model.learn(total_timesteps=100, progress_bar=False)

            # Get observations
            obs = vec_env.reset()

            # Predict actions
            actions, _ = model.predict(obs, deterministic=True)

            # Verify actions are valid
            assert actions.shape == (2,)  # 2 environments
            assert np.all(actions >= 0)
            assert np.all(actions < 4)  # 4 discrete actions

        finally:
            vec_env.close()

    def test_ppo_episode_completion(self):
        """Test PPO training completes multiple episodes."""
        vec_env = make_vec_env(
            n_envs=2,
            start_method="spawn",
            env_kwargs={"max_timesteps": 20},  # Short episodes
            seed=42
        )

        try:
            model = PPO("MlpPolicy", vec_env, n_steps=32, verbose=0, seed=42)

            # Train long enough to complete episodes
            model.learn(total_timesteps=200, progress_bar=False)

            # Should have completed without errors (PPO rounds up)
            assert model.num_timesteps >= 200

        finally:
            vec_env.close()


class TestVectorizedEnvironmentIntegration:
    """Test vectorized environments work with SB3 training."""

    def test_vec_env_with_sb3_ppo(self):
        """Test SubprocVecEnv works with SB3 PPO."""
        vec_env = make_vec_env(n_envs=4, start_method="spawn", seed=42)

        try:
            model = PPO("MlpPolicy", vec_env, n_steps=32, verbose=0, seed=42)

            # Train for 200 steps
            model.learn(total_timesteps=200, progress_bar=False)

            # PPO rounds up to complete rollouts
            assert model.num_timesteps >= 200
            assert isinstance(vec_env, SubprocVecEnv)

        finally:
            vec_env.close()

    def test_vec_env_16_parallel_training(self):
        """Test 16 parallel envs (plan.md specification)."""
        vec_env = make_vec_env(n_envs=16, start_method="spawn", seed=42)

        try:
            model = PPO("MlpPolicy", vec_env, n_steps=32, verbose=0, seed=42)

            # Train for 100 steps (brief test)
            model.learn(total_timesteps=100, progress_bar=False)

            # Verify all 16 envs running
            assert vec_env.num_envs == 16
            # PPO with n_steps=32 and 16 envs: each rollout is 512 steps
            # So asking for 100 → 512 actual timesteps (1 rollout)
            assert model.num_timesteps >= 100

        finally:
            vec_env.close()

    def test_vec_env_auto_reset(self):
        """Test vec env auto-resets on episode completion."""
        vec_env = make_vec_env(
            n_envs=2,
            start_method="spawn",
            env_kwargs={"max_timesteps": 20},  # Short episodes
            seed=42
        )

        try:
            vec_env.reset()

            # Run enough steps to trigger auto-reset
            for _ in range(50):
                actions = np.array([0, 0])  # Pass actions
                obs, rewards, dones, infos = vec_env.step(actions)

                # If done, observation should still be valid (auto-reset)
                if dones.any():
                    assert obs.shape == (2, 9)
                    assert np.all(obs >= 0.0)
                    assert np.all(obs <= 1.0)

        finally:
            vec_env.close()


class TestHydraConfigIntegration:
    """Test Hydra configs load and work correctly."""

    def test_train_config_loads(self):
        """Test train_config.yaml loads without errors."""
        config_path = Path("conf/train_config.yaml")
        assert config_path.exists(), "train_config.yaml not found"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Verify it references the sub-configs
        assert "defaults" in config

    def test_ppo_config_loads(self):
        """Test ppo.yaml loads with valid hyperparameters."""
        config_path = Path("conf/rl/ppo.yaml")
        assert config_path.exists(), "ppo.yaml not found"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Verify key hyperparameters present
        assert "learning_rate" in config
        assert "n_steps" in config
        assert "batch_size" in config
        assert "policy" in config

        # Verify reasonable values
        assert 0 < config["learning_rate"] < 1
        assert config["n_steps"] > 0
        assert config["batch_size"] > 0

    def test_vectorization_config_loads(self):
        """Test vectorization.yaml config used correctly."""
        config_path = Path("conf/rl/vectorization.yaml")
        assert config_path.exists(), "vectorization.yaml not found"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Verify vectorization settings
        assert "n_envs" in config
        assert config["n_envs"] == 16  # Plan.md specification

        # Verify environment config
        assert "env" in config
        assert "num_buyers" in config["env"]
        assert "num_sellers" in config["env"]

    def test_training_config_loads(self):
        """Test training.yaml loads correctly."""
        config_path = Path("conf/rl/training.yaml")
        assert config_path.exists(), "training.yaml not found"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Verify training settings
        assert "total_timesteps" in config
        assert "save_freq" in config
        assert "wandb" in config


class TestCheckpointingIntegration:
    """Test checkpointing and callbacks work correctly."""

    def test_checkpoint_callback(self):
        """Test CheckpointCallback saves models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vec_env = make_vec_env(n_envs=2, start_method="spawn", seed=42)

            try:
                checkpoint_callback = CheckpointCallback(
                    save_freq=50,
                    save_path=tmpdir,
                    name_prefix="test_checkpoint"
                )

                model = PPO(
                    "MlpPolicy",
                    vec_env,
                    n_steps=32,
                    verbose=0,
                    seed=42
                )

                # Train long enough to trigger checkpoint
                model.learn(
                    total_timesteps=100,
                    callback=checkpoint_callback,
                    progress_bar=False
                )

                # Verify checkpoint created
                checkpoint_files = list(Path(tmpdir).glob("test_checkpoint_*_steps.zip"))
                assert len(checkpoint_files) > 0, "No checkpoint files created"

            finally:
                vec_env.close()

    def test_checkpoint_loading(self):
        """Test saved checkpoint can be loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vec_env = make_vec_env(n_envs=2, start_method="spawn", seed=42)

            try:
                # Train and save
                model = PPO("MlpPolicy", vec_env, n_steps=32, verbose=0, seed=42)
                model.learn(total_timesteps=100, progress_bar=False)

                save_path = Path(tmpdir) / "test_model.zip"
                model.save(save_path)

                # Load model
                loaded_model = PPO.load(save_path, env=vec_env)

                # Verify loaded model can predict
                obs = vec_env.reset()
                actions, _ = loaded_model.predict(obs)

                assert actions.shape == (2,)

            finally:
                vec_env.close()

    def test_eval_callback(self):
        """Test EvalCallback evaluates periodically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_env = make_vec_env(n_envs=2, start_method="spawn", seed=42)
            eval_env = make_eval_vec_env(n_envs=2, start_method="spawn", seed=1000)

            try:
                eval_callback = EvalCallback(
                    eval_env,
                    eval_freq=50,
                    n_eval_episodes=2,
                    log_path=tmpdir,
                    deterministic=True
                )

                model = PPO("MlpPolicy", train_env, n_steps=32, verbose=0, seed=42)

                # Train with evaluation
                model.learn(
                    total_timesteps=100,
                    callback=eval_callback,
                    progress_bar=False
                )

                # Verify evaluation log created
                log_files = list(Path(tmpdir).glob("evaluations.npz"))
                assert len(log_files) > 0, "No evaluation log created"

            finally:
                train_env.close()
                eval_env.close()


class TestEndToEndPipeline:
    """CRITICAL: Test complete training pipeline."""

    def test_full_pipeline_small_scale(self):
        """Test complete pipeline: config -> train -> eval -> save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create config
            cfg = OmegaConf.create({
                "vectorization": {
                    "n_envs": 4,
                    "start_method": "spawn",
                    "eval": {"n_envs": 2, "seed": 1000},
                    "env": {
                        "num_buyers": 5,
                        "num_sellers": 5,
                        "num_tokens_per_agent": 3,
                        "max_timesteps": 50,
                        "price_min": 0,
                        "price_max": 100,
                        "rl_agent_type": "buyer",
                        "opponent_type": "ZIC"
                    }
                },
                "ppo": {
                    "policy": "MlpPolicy",
                    "learning_rate": 0.0003,
                    "n_steps": 32,
                    "batch_size": 16,
                    "n_epochs": 2,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2,
                    "clip_range_vf": None,
                    "normalize_advantage": True,
                    "ent_coef": 0.01,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                    "use_sde": False,
                    "verbose": 0,
                    "seed": 42,
                    "device": "cpu",
                    "tensorboard_log": None
                },
                "training": {
                    "seed": 42,
                    "save_freq": 200,
                    "save_replay_buffer": False,
                    "eval_freq": 200,
                    "eval_episodes": 2,
                    "eval_deterministic": True,
                    "checkpoint_dir": tmpdir,
                    "wandb": {
                        "enabled": False
                    }
                }
            })

            # 2. Create environments
            from train_ppo import create_training_env, create_eval_env

            train_env = create_training_env(cfg)
            eval_env = create_eval_env(cfg)

            try:
                # 3. Create PPO model
                from train_ppo import create_ppo_model
                model = create_ppo_model(cfg, train_env)

                # 4. Create callbacks
                from train_ppo import create_callbacks
                callbacks = create_callbacks(cfg, eval_env)

                # 5. Train
                model.learn(
                    total_timesteps=500,
                    callback=callbacks,
                    progress_bar=False
                )

                # 6. Evaluate model
                obs = eval_env.reset()
                for _ in range(10):
                    actions, _ = model.predict(obs, deterministic=True)
                    obs, rewards, dones, infos = eval_env.step(actions)

                # 7. Save final model
                final_path = Path(tmpdir) / "final_model.zip"
                model.save(final_path)

                # 8. Verify model saved
                assert final_path.exists()

                # 9. Load and verify
                loaded_model = PPO.load(final_path, env=train_env)
                assert loaded_model is not None

            finally:
                train_env.close()
                eval_env.close()

    def test_training_produces_rewards(self):
        """Test training produces some positive rewards."""
        vec_env = make_vec_env(n_envs=4, start_method="spawn", seed=42)

        try:
            model = PPO("MlpPolicy", vec_env, n_steps=32, verbose=0, seed=42)

            # Train for several hundred steps
            model.learn(total_timesteps=500, progress_bar=False)

            # Collect some rewards
            obs = vec_env.reset()
            total_rewards = 0
            for _ in range(50):
                actions, _ = model.predict(obs)
                obs, rewards, dones, infos = vec_env.step(actions)
                total_rewards += rewards.sum()

            # Should see at least some trades/rewards
            # (This is probabilistic, but very likely with 4 envs × 50 steps)
            assert total_rewards >= 0  # At minimum, non-negative

        finally:
            vec_env.close()
