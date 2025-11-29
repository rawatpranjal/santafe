"""
Unit tests for vectorized environment setup.

Tests the vectorization utilities, SubprocVecEnv creation, and parallel
environment functionality for PPO training.
"""

import pytest
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from envs.vec_env_utils import make_env, make_vec_env, make_eval_vec_env, get_default_env_kwargs
from envs.double_auction_env import DoubleAuctionEnv


class TestEnvironmentFactory:
    """Test environment factory functions."""

    def test_make_env_returns_callable(self):
        """Test make_env returns a callable."""
        env_fn = make_env(rank=0, seed=42)
        assert callable(env_fn)

    def test_make_env_creates_environment(self):
        """Test environment factory creates valid environment."""
        env_fn = make_env(rank=0, seed=42)
        env = env_fn()

        assert isinstance(env, DoubleAuctionEnv)
        assert env.action_space.n == 4
        assert env.observation_space.shape == (9,)

        env.close()

    def test_make_env_unique_seeds(self):
        """Test each rank gets unique seed."""
        env_fn_0 = make_env(rank=0, seed=42)
        env_fn_1 = make_env(rank=1, seed=42)

        env_0 = env_fn_0()
        env_1 = env_fn_1()

        # Reset and get observations
        obs_0, _ = env_0.reset()
        obs_1, _ = env_1.reset()

        # Different seeds should produce different initial observations
        # (with very high probability)
        assert not np.allclose(obs_0, obs_1)

        env_0.close()
        env_1.close()

    def test_make_env_deterministic_with_same_seed(self):
        """Test same rank and seed produces identical environments."""
        env_fn_1 = make_env(rank=0, seed=42)
        env_fn_2 = make_env(rank=0, seed=42)

        env_1 = env_fn_1()
        env_2 = env_fn_2()

        obs_1, _ = env_1.reset()
        obs_2, _ = env_2.reset()

        # Same seed should produce identical initial observations
        np.testing.assert_array_almost_equal(obs_1, obs_2)

        env_1.close()
        env_2.close()

    def test_make_env_respects_kwargs(self):
        """Test environment factory respects custom kwargs."""
        env_fn = make_env(
            rank=0,
            num_buyers=10,
            num_sellers=8,
            max_timesteps=200,
            seed=42
        )
        env = env_fn()

        assert env.num_buyers == 10
        assert env.num_sellers == 8
        assert env.max_timesteps == 200

        env.close()


class TestVectorizedEnvironments:
    """Test vectorized environment creation."""

    def test_make_vec_env_creates_subproc(self):
        """Test make_vec_env creates SubprocVecEnv."""
        vec_env = make_vec_env(n_envs=4, seed=42)

        assert isinstance(vec_env, SubprocVecEnv)
        assert vec_env.num_envs == 4

        vec_env.close()

    def test_vec_env_observation_space(self):
        """Test vectorized env has correct observation space."""
        vec_env = make_vec_env(n_envs=4, seed=42)

        assert vec_env.observation_space.shape == (9,)
        assert vec_env.observation_space.dtype == np.float32

        vec_env.close()

    def test_vec_env_action_space(self):
        """Test vectorized env has correct action space."""
        vec_env = make_vec_env(n_envs=4, seed=42)

        assert vec_env.action_space.n == 4

        vec_env.close()

    def test_vec_env_reset(self):
        """Test vectorized env reset returns correct shape."""
        n_envs = 8
        vec_env = make_vec_env(n_envs=n_envs, seed=42)

        obs = vec_env.reset()

        # Should return (n_envs, obs_dim) array
        assert obs.shape == (n_envs, 9)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

        vec_env.close()

    def test_vec_env_step(self):
        """Test vectorized env step works correctly."""
        n_envs = 4
        vec_env = make_vec_env(n_envs=n_envs, seed=42)

        vec_env.reset()
        actions = np.array([0, 1, 2, 3])  # Different action per env

        obs, rewards, dones, infos = vec_env.step(actions)

        # Check shapes
        assert obs.shape == (n_envs, 9)
        assert rewards.shape == (n_envs,)
        assert dones.shape == (n_envs,)
        assert len(infos) == n_envs

        # Check types
        assert obs.dtype == np.float32
        assert isinstance(rewards, np.ndarray)
        assert isinstance(dones, np.ndarray)

        vec_env.close()

    def test_vec_env_parallel_execution(self):
        """Test multiple steps execute correctly in parallel."""
        n_envs = 4
        vec_env = make_vec_env(n_envs=n_envs, seed=42)

        vec_env.reset()

        # Run 10 steps
        for _ in range(10):
            actions = np.array([vec_env.action_space.sample() for _ in range(n_envs)])
            obs, rewards, dones, infos = vec_env.step(actions)

            # Verify all return values have correct shape
            assert obs.shape == (n_envs, 9)
            assert rewards.shape == (n_envs,)
            assert dones.shape == (n_envs,)

        vec_env.close()

    def test_vec_env_with_kwargs(self):
        """Test vectorized env respects environment kwargs."""
        env_kwargs = {
            "num_buyers": 10,
            "num_sellers": 8,
            "max_timesteps": 50
        }

        vec_env = make_vec_env(n_envs=2, env_kwargs=env_kwargs, seed=42)

        # We can't directly inspect env params in SubprocVecEnv,
        # but we can verify it doesn't crash
        vec_env.reset()
        vec_env.step(np.array([0, 0]))

        vec_env.close()


class TestEvaluationEnvironments:
    """Test evaluation environment utilities."""

    def test_make_eval_vec_env_creates_subproc(self):
        """Test make_eval_vec_env creates SubprocVecEnv."""
        eval_env = make_eval_vec_env(n_envs=2, seed=1000)

        assert isinstance(eval_env, SubprocVecEnv)
        assert eval_env.num_envs == 2

        eval_env.close()

    def test_eval_env_different_seed_from_training(self):
        """Test eval env uses different seed from training env."""
        train_env = make_vec_env(n_envs=2, seed=42)
        eval_env = make_eval_vec_env(n_envs=2, seed=1000)

        train_obs = train_env.reset()
        eval_obs = eval_env.reset()

        # Different seeds should produce different observations
        assert not np.allclose(train_obs, eval_obs)

        train_env.close()
        eval_env.close()


class TestDeterministicSeeding:
    """Test deterministic seeding for reproducibility."""

    def test_vec_env_deterministic_reset(self):
        """Test vec env reset is deterministic with same seed."""
        vec_env_1 = make_vec_env(n_envs=4, seed=42)
        vec_env_2 = make_vec_env(n_envs=4, seed=42)

        obs_1 = vec_env_1.reset()
        obs_2 = vec_env_2.reset()

        # Should produce identical observations
        np.testing.assert_array_almost_equal(obs_1, obs_2)

        vec_env_1.close()
        vec_env_2.close()

    def test_vec_env_different_seeds_produce_different_obs(self):
        """Test different seeds produce different observations."""
        vec_env_1 = make_vec_env(n_envs=4, seed=42)
        vec_env_2 = make_vec_env(n_envs=4, seed=123)

        obs_1 = vec_env_1.reset()
        obs_2 = vec_env_2.reset()

        # Different seeds should produce different observations
        assert not np.allclose(obs_1, obs_2)

        vec_env_1.close()
        vec_env_2.close()


class TestCurriculumHelpers:
    """Test curriculum learning helper functions."""

    def test_get_default_env_kwargs_zic(self):
        """Test default kwargs for ZIC curriculum stage."""
        kwargs = get_default_env_kwargs(curriculum_stage="zic", rl_agent_type="buyer")

        assert kwargs["opponent_type"] == "ZIC"
        assert kwargs["rl_agent_type"] == "buyer"
        assert kwargs["num_buyers"] == 5
        assert kwargs["num_sellers"] == 5

    def test_get_default_env_kwargs_kaplan(self):
        """Test default kwargs for Kaplan curriculum stage."""
        kwargs = get_default_env_kwargs(curriculum_stage="kaplan", rl_agent_type="seller")

        assert kwargs["opponent_type"] == "Kaplan"
        assert kwargs["rl_agent_type"] == "seller"

    def test_get_default_env_kwargs_mixed(self):
        """Test default kwargs for mixed curriculum stage."""
        kwargs = get_default_env_kwargs(curriculum_stage="mixed")

        assert kwargs["opponent_type"] == "Mixed"

    def test_get_default_env_kwargs_invalid_stage(self):
        """Test invalid curriculum stage raises error."""
        with pytest.raises(ValueError, match="Unknown curriculum stage"):
            get_default_env_kwargs(curriculum_stage="invalid_stage")


class TestPerformance:
    """Performance and stress tests for vectorization."""

    def test_vec_env_16_environments(self):
        """Test creating 16 parallel environments (as in plan.md)."""
        n_envs = 16
        vec_env = make_vec_env(n_envs=n_envs, seed=42)

        assert vec_env.num_envs == 16

        # Verify reset works
        obs = vec_env.reset()
        assert obs.shape == (16, 9)

        # Verify step works
        actions = np.zeros(16, dtype=np.int64)  # All pass
        obs, rewards, dones, infos = vec_env.step(actions)
        assert obs.shape == (16, 9)

        vec_env.close()

    def test_vec_env_multiple_episodes(self):
        """Test running multiple episodes in parallel."""
        vec_env = make_vec_env(n_envs=4, seed=42)
        vec_env.reset()

        episode_count = 0
        max_steps = 500  # Should complete several episodes

        for _ in range(max_steps):
            actions = np.array([vec_env.action_space.sample() for _ in range(4)])
            obs, rewards, dones, infos = vec_env.step(actions)

            episode_count += dones.sum()

            # If any environment is done, it auto-resets
            if dones.any():
                # Check auto-reset worked (obs still valid)
                assert obs.shape == (4, 9)

        # Should have completed at least a few episodes
        assert episode_count > 0

        vec_env.close()


class TestStartMethods:
    """Test different multiprocessing start methods."""

    @pytest.mark.skipif(
        not hasattr(pytest, "param"),
        reason="Parameterization not available"
    )
    def test_forkserver_start_method(self):
        """Test forkserver start method (recommended for PyTorch)."""
        vec_env = make_vec_env(n_envs=2, start_method="forkserver", seed=42)

        obs = vec_env.reset()
        assert obs.shape == (2, 9)

        vec_env.close()

    def test_spawn_start_method(self):
        """Test spawn start method (Windows compatible)."""
        vec_env = make_vec_env(n_envs=2, start_method="spawn", seed=42)

        obs = vec_env.reset()
        assert obs.shape == (2, 9)

        vec_env.close()
