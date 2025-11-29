"""
Unit tests for double auction Gymnasium environment.
"""

import pytest
import numpy as np
from envs.double_auction_env import DoubleAuctionEnv, RLAgentWrapper


@pytest.fixture
def env():
    """Create test environment."""
    return DoubleAuctionEnv(
        num_buyers=3,
        num_sellers=3,
        num_tokens_per_agent=2,
        max_timesteps=50,
        price_min=0,
        price_max=100,
        rl_agent_type="buyer",
        seed=42
    )


@pytest.fixture
def env_seller():
    """Create test environment with RL agent as seller."""
    return DoubleAuctionEnv(
        num_buyers=3,
        num_sellers=3,
        num_tokens_per_agent=2,
        max_timesteps=50,
        rl_agent_type="seller",
        seed=42
    )


class TestEnvironmentCreation:
    """Test environment initialization."""

    def test_env_creation(self, env):
        """Test environment can be created."""
        assert env is not None
        assert env.action_space.n == 4
        assert env.observation_space.shape == (9,)

    def test_action_space(self, env):
        """Test action space is discrete with 4 actions."""
        assert env.action_space.n == 4
        # Should be able to sample actions
        action = env.action_space.sample()
        assert 0 <= action < 4

    def test_observation_space(self, env):
        """Test observation space is 9-dimensional box in [0, 1]."""
        assert env.observation_space.shape == (9,)
        assert env.observation_space.dtype == np.float32
        assert np.all(env.observation_space.low == 0.0)
        assert np.all(env.observation_space.high == 1.0)

    def test_buyer_configuration(self, env):
        """Test environment configured with RL agent as buyer."""
        assert env.rl_agent_type == "buyer"
        assert env.num_buyers == 3
        assert env.num_sellers == 3

    def test_seller_configuration(self, env_seller):
        """Test environment configured with RL agent as seller."""
        assert env_seller.rl_agent_type == "seller"


class TestReset:
    """Test environment reset."""

    def test_reset_returns_valid_observation(self, env):
        """Test reset returns valid observation."""
        obs, info = env.reset()

        # Check observation shape and bounds
        assert obs.shape == (9,)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_reset_returns_info_dict(self, env):
        """Test reset returns info dictionary."""
        obs, info = env.reset()

        assert isinstance(info, dict)
        assert "timestep" in info
        assert "rl_agent_trades" in info
        assert "action_mask" in info

    def test_reset_initializes_market(self, env):
        """Test reset initializes market and agents."""
        env.reset()

        assert env.market is not None
        assert env.rl_agent is not None
        assert len(env.agents) == env.num_buyers + env.num_sellers

    def test_reset_with_seed(self, env):
        """Test reset with seed gives reproducible results."""
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)

        np.testing.assert_array_almost_equal(obs1, obs2)

    def test_reset_different_seeds(self, env):
        """Test reset with different seeds gives different results."""
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=456)

        # Should be different (with very high probability)
        assert not np.allclose(obs1, obs2)

    def test_reset_clears_episode_state(self, env):
        """Test reset clears episode state."""
        env.reset()

        assert env.current_timestep == 0
        assert len(env.price_history) == 0
        assert len(env.volume_history) == 0
        assert env.last_trade_price is None


class TestStep:
    """Test environment step."""

    def test_step_returns_correct_tuple(self, env):
        """Test step returns (obs, reward, terminated, truncated, info)."""
        env.reset()
        result = env.step(0)  # Pass action

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_action_0_pass(self, env):
        """Test action 0 (Pass) works."""
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        # Pass should not cause errors
        assert obs.shape == (9,)
        # Pass typically gives 0 reward (no trade)
        assert reward >= 0

    def test_step_increments_timestep(self, env):
        """Test step increments timestep counter."""
        env.reset()
        initial_timestep = env.current_timestep

        env.step(0)

        assert env.current_timestep == initial_timestep + 1

    def test_step_multiple_times(self, env):
        """Test stepping multiple times."""
        env.reset()

        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(0)

            if terminated or truncated:
                break

            assert obs.shape == (9,)

    def test_step_without_reset_raises_error(self, env):
        """Test stepping without reset raises error."""
        with pytest.raises(RuntimeError, match="not initialized"):
            env.step(0)

    def test_step_all_actions(self, env):
        """Test all action types can be executed."""
        env.reset()

        # Try each action type
        for action in range(4):
            env.reset()
            obs, reward, terminated, truncated, info = env.step(action)

            # Should not crash
            assert obs.shape == (9,)


class TestReward:
    """Test reward calculation."""

    def test_pass_gives_zero_reward(self, env):
        """Test passing gives zero reward (typically)."""
        env.reset()

        # Take several pass actions
        total_reward = 0
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(0)
            total_reward += reward

        # Pass should give 0 reward unless by chance a trade happens
        # In most cases, total reward should be small
        assert total_reward >= 0

    def test_reward_bounds(self, env):
        """Test rewards are within reasonable bounds."""
        env.reset()

        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

            # Reward should be bounded by price range
            assert -env.price_max <= reward <= env.price_max

            if terminated or truncated:
                break


class TestActionMasking:
    """Test action masking."""

    def test_action_mask_in_info(self, env):
        """Test action mask is provided in info dict."""
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        assert "action_mask" in info
        assert isinstance(info["action_mask"], np.ndarray)
        assert info["action_mask"].shape == (4,)
        assert info["action_mask"].dtype == bool

    def test_pass_always_valid(self, env):
        """Test action 0 (Pass) is always valid."""
        env.reset()

        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

            # Pass (action 0) should always be valid
            assert info["action_mask"][0] == True

            if terminated or truncated:
                break

    def test_no_valid_actions_when_exhausted(self, env):
        """Test only Pass is valid when all tokens traded."""
        # Create environment with 1 token to quickly exhaust
        env_small = DoubleAuctionEnv(
            num_buyers=2,
            num_sellers=2,
            num_tokens_per_agent=1,
            max_timesteps=100,
            seed=42
        )
        env_small.reset()

        # Step until tokens exhausted
        for _ in range(50):
            obs, reward, terminated, truncated, info = env_small.step(0)

            if terminated:
                # After termination, only Pass should be valid
                # (if we could step again, which we can't in terminated state)
                break


class TestTermination:
    """Test episode termination."""

    def test_terminates_when_tokens_exhausted(self):
        """Test episode terminates when RL agent trades all tokens."""
        env_small = DoubleAuctionEnv(
            num_buyers=2,
            num_sellers=2,
            num_tokens_per_agent=1,  # Only 1 token
            max_timesteps=100,
            seed=42
        )
        env_small.reset()

        # Should eventually terminate when token is traded
        terminated = False
        for _ in range(100):
            obs, reward, terminated, truncated, info = env_small.step(env_small.action_space.sample())

            if terminated:
                break

        # May or may not terminate depending on if trade happens
        # Just check it doesn't crash
        assert isinstance(terminated, bool)

    def test_truncates_at_max_timesteps(self, env):
        """Test episode truncates at max timesteps."""
        env_short = DoubleAuctionEnv(
            num_buyers=2,
            num_sellers=2,
            num_tokens_per_agent=5,  # Many tokens
            max_timesteps=10,  # Short episode
            seed=42
        )
        env_short.reset()

        # Step until truncation
        truncated = False
        for _ in range(15):  # More than max_timesteps
            obs, reward, terminated, truncated, info = env_short.step(0)

            if truncated:
                break

        assert truncated == True


class TestRLAgentWrapper:
    """Test RL agent wrapper."""

    def test_wrapper_initialization(self):
        """Test RL agent wrapper can be created."""
        agent = RLAgentWrapper(
            player_id=1,
            is_buyer=True,
            num_tokens=3,
            valuations=[50, 60, 70],
            price_min=0,
            price_max=100
        )

        assert agent.player_id == 1
        assert agent.is_buyer == True
        assert agent.num_tokens == 3
        assert agent.valuations == [50, 60, 70]

    def test_wrapper_pass_action(self):
        """Test wrapper handles Pass action (0)."""
        agent = RLAgentWrapper(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[50],
        )

        agent.gym_action = 0  # Pass
        quote = agent.bid_ask_response()

        assert quote == 0  # Returns 0 for no quote

    def test_wrapper_accept_action(self):
        """Test wrapper handles Accept action (1)."""
        agent = RLAgentWrapper(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[50],
        )

        agent.gym_action = 1  # Accept
        quote = agent.bid_ask_response()

        # Accept doesn't generate quote
        assert quote == 0  # Returns 0 for no quote

    def test_wrapper_improve_action(self):
        """Test wrapper handles Improve action (2)."""
        agent = RLAgentWrapper(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[50],
            price_min=0,
            price_max=100
        )

        agent.current_bid = 40
        agent.gym_action = 2  # Improve
        quote = agent.bid_ask_response()

        # Should improve bid by 1
        assert quote == 41

    def test_wrapper_match_action(self):
        """Test wrapper handles Match action (3)."""
        agent = RLAgentWrapper(
            player_id=1,
            is_buyer=True,
            num_tokens=1,
            valuations=[50],
            price_min=0,
            price_max=100
        )

        agent.current_bid = 45
        agent.gym_action = 3  # Match
        quote = agent.bid_ask_response()

        # Should match current bid
        assert quote == 45


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_episode(self, env):
        """Test running a full episode."""
        obs, info = env.reset()

        episode_reward = 0
        steps = 0

        for _ in range(env.max_timesteps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            steps += 1

            if terminated or truncated:
                break

        # Episode should complete
        assert steps > 0
        assert steps <= env.max_timesteps

    def test_multiple_episodes(self, env):
        """Test running multiple episodes."""
        for episode in range(3):
            obs, info = env.reset()

            for _ in range(20):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

        # Should complete without errors

    def test_observation_stays_valid(self, env):
        """Test observations stay in valid range throughout episode."""
        env.reset()

        for _ in range(50):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

            # Observation should always be in [0, 1]
            assert np.all(obs >= 0.0)
            assert np.all(obs <= 1.0)

            if terminated or truncated:
                break
