"""
Unit tests for double auction feature extraction.

Tests all 9 features across normal cases and edge cases.
"""

import pytest
import numpy as np
from envs.features import DoubleAuctionFeatureExtractor


@pytest.fixture
def extractor():
    """Create feature extractor with standard parameters."""
    return DoubleAuctionFeatureExtractor(
        max_price=100,
        max_timesteps=100,
        history_window=5
    )


class TestPrivateFeatures:
    """Test private features (agent's internal state)."""

    def test_valuation_normalization(self, extractor):
        """Test valuation feature normalization."""
        # Mid-range valuation
        assert extractor._normalize_valuation(50.0) == pytest.approx(0.5)

        # Boundary values
        assert extractor._normalize_valuation(0.0) == pytest.approx(0.0)
        assert extractor._normalize_valuation(100.0) == pytest.approx(1.0)

        # Out of bounds (should clip)
        assert extractor._normalize_valuation(-10.0) == pytest.approx(0.0)
        assert extractor._normalize_valuation(150.0) == pytest.approx(1.0)

    def test_held_tokens_normalization(self, extractor):
        """Test remaining tokens feature normalization."""
        # All tokens remaining
        assert extractor._normalize_held_tokens(10, 0) == pytest.approx(1.0)

        # Half tokens remaining
        assert extractor._normalize_held_tokens(10, 5) == pytest.approx(0.5)

        # No tokens remaining
        assert extractor._normalize_held_tokens(10, 10) == pytest.approx(0.0)

        # Edge case: zero tokens total
        assert extractor._normalize_held_tokens(0, 0) == pytest.approx(0.0)

        # Over-traded (should clip to 0)
        assert extractor._normalize_held_tokens(10, 15) == pytest.approx(0.0)

    def test_time_normalization(self, extractor):
        """Test timestep feature normalization."""
        # Start of period
        assert extractor._normalize_time(0) == pytest.approx(0.0)

        # Mid-period
        assert extractor._normalize_time(50) == pytest.approx(0.5)

        # End of period
        assert extractor._normalize_time(100) == pytest.approx(1.0)

        # Over time (should clip)
        assert extractor._normalize_time(150) == pytest.approx(1.0)


class TestMarketFeatures:
    """Test market state features."""

    def test_best_bid_normalization(self, extractor):
        """Test best bid feature normalization."""
        # Normal bid
        assert extractor._normalize_best_bid(50) == pytest.approx(0.5)

        # Boundary bids
        assert extractor._normalize_best_bid(0) == pytest.approx(0.0)
        assert extractor._normalize_best_bid(100) == pytest.approx(1.0)

        # No bid (None)
        assert extractor._normalize_best_bid(None) == pytest.approx(0.5)

    def test_best_ask_normalization(self, extractor):
        """Test best ask feature normalization."""
        # Normal ask
        assert extractor._normalize_best_ask(75) == pytest.approx(0.75)

        # Boundary asks
        assert extractor._normalize_best_ask(0) == pytest.approx(0.0)
        assert extractor._normalize_best_ask(100) == pytest.approx(1.0)

        # No ask (None)
        assert extractor._normalize_best_ask(None) == pytest.approx(0.5)

    def test_spread_normalization(self, extractor):
        """Test bid-ask spread feature normalization."""
        # Normal spread
        assert extractor._normalize_spread(40, 60) == pytest.approx(0.2)

        # Zero spread (crossed market)
        assert extractor._normalize_spread(50, 50) == pytest.approx(0.0)

        # Negative spread (inverted, should return 0)
        assert extractor._normalize_spread(60, 50) == pytest.approx(0.0)

        # Wide spread
        assert extractor._normalize_spread(10, 90) == pytest.approx(0.8)

        # Missing bid
        assert extractor._normalize_spread(None, 60) == pytest.approx(0.0)

        # Missing ask
        assert extractor._normalize_spread(40, None) == pytest.approx(0.0)

        # Both missing
        assert extractor._normalize_spread(None, None) == pytest.approx(0.0)

    def test_last_price_normalization(self, extractor):
        """Test last trade price feature normalization."""
        # Normal price
        assert extractor._normalize_last_price(50) == pytest.approx(0.5)

        # Boundary prices
        assert extractor._normalize_last_price(0) == pytest.approx(0.0)
        assert extractor._normalize_last_price(100) == pytest.approx(1.0)

        # No trades yet (None)
        assert extractor._normalize_last_price(None) == pytest.approx(0.5)


class TestHistoryFeatures:
    """Test historical trend features."""

    def test_price_trend_neutral(self, extractor):
        """Test price trend returns neutral when insufficient history."""
        # Empty history
        assert extractor._normalize_price_trend([]) == pytest.approx(0.5)

        # Insufficient history (need 2 * window = 10)
        assert extractor._normalize_price_trend([50, 50, 50]) == pytest.approx(0.5)

    def test_price_trend_upward(self, extractor):
        """Test price trend detects upward momentum."""
        # Prices increasing from 40 to 60
        price_history = [40, 40, 40, 40, 40, 60, 60, 60, 60, 60]
        trend = extractor._normalize_price_trend(price_history)

        # Should be > 0.5 (positive momentum)
        assert trend > 0.5
        # Trend = (60-40)/100 = 0.2, normalized = (0.2+1)/2 = 0.6
        assert trend == pytest.approx(0.6)

    def test_price_trend_downward(self, extractor):
        """Test price trend detects downward momentum."""
        # Prices decreasing from 60 to 40
        price_history = [60, 60, 60, 60, 60, 40, 40, 40, 40, 40]
        trend = extractor._normalize_price_trend(price_history)

        # Should be < 0.5 (negative momentum)
        assert trend < 0.5
        # Trend = (40-60)/100 = -0.2, normalized = (-0.2+1)/2 = 0.4
        assert trend == pytest.approx(0.4)

    def test_price_trend_stable(self, extractor):
        """Test price trend is neutral when prices stable."""
        # Constant prices
        price_history = [50] * 10
        trend = extractor._normalize_price_trend(price_history)

        # Should be exactly 0.5 (no momentum)
        assert trend == pytest.approx(0.5)

    def test_imbalance_neutral(self, extractor):
        """Test volume imbalance returns neutral when no data."""
        # Empty volume history
        assert extractor._normalize_imbalance([]) == pytest.approx(0.5)

        # Zero volume
        assert extractor._normalize_imbalance([(0, 0), (0, 0)]) == pytest.approx(0.5)

    def test_imbalance_buy_heavy(self, extractor):
        """Test volume imbalance detects buy pressure."""
        # All buy volume, no sell volume
        volume_history = [(10, 0), (10, 0), (10, 0)]
        imbalance = extractor._normalize_imbalance(volume_history)

        # Should be 1.0 (maximum buy pressure)
        # Imbalance = (30-0)/(30+0) = 1, normalized = (1+1)/2 = 1.0
        assert imbalance == pytest.approx(1.0)

    def test_imbalance_sell_heavy(self, extractor):
        """Test volume imbalance detects sell pressure."""
        # All sell volume, no buy volume
        volume_history = [(0, 10), (0, 10), (0, 10)]
        imbalance = extractor._normalize_imbalance(volume_history)

        # Should be 0.0 (maximum sell pressure)
        # Imbalance = (0-30)/(0+30) = -1, normalized = (-1+1)/2 = 0.0
        assert imbalance == pytest.approx(0.0)

    def test_imbalance_balanced(self, extractor):
        """Test volume imbalance is neutral when balanced."""
        # Equal buy and sell volume
        volume_history = [(5, 5), (10, 10), (3, 3)]
        imbalance = extractor._normalize_imbalance(volume_history)

        # Should be 0.5 (balanced)
        # Imbalance = (18-18)/(18+18) = 0, normalized = (0+1)/2 = 0.5
        assert imbalance == pytest.approx(0.5)

    def test_imbalance_window_limit(self, extractor):
        """Test volume imbalance uses only recent window."""
        # Provide more history than window size
        volume_history = [(0, 10)] * 10 + [(10, 0)] * 10
        imbalance = extractor._normalize_imbalance(volume_history)

        # Should only use last 5 entries (all buy)
        assert imbalance == pytest.approx(1.0)


class TestFullFeatureExtraction:
    """Test complete feature extraction."""

    def test_extract_features_shape(self, extractor):
        """Test feature vector has correct shape."""
        features = extractor.extract_features(
            valuation=50.0,
            num_tokens=10,
            num_trades=5,
            current_timestep=25,
            best_bid=45,
            best_ask=55,
            last_price=50,
            price_history=[48, 49, 50, 51, 52],
            volume_history=[(5, 5), (6, 4), (7, 3)]
        )

        assert features.shape == (9,)
        assert features.dtype == np.float32

    def test_extract_features_bounds(self, extractor):
        """Test all features are in [0, 1] range."""
        features = extractor.extract_features(
            valuation=75.0,
            num_tokens=10,
            num_trades=3,
            current_timestep=50,
            best_bid=40,
            best_ask=60,
            last_price=50,
            price_history=[40, 42, 44, 46, 48, 50, 52, 54, 56, 58],
            volume_history=[(10, 2), (8, 4), (6, 6), (4, 8), (2, 10)]
        )

        # All features must be in [0, 1]
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)

    def test_extract_features_empty_market(self, extractor):
        """Test feature extraction with empty market (no bids/asks/trades)."""
        features = extractor.extract_features(
            valuation=50.0,
            num_tokens=10,
            num_trades=0,
            current_timestep=0,
            best_bid=None,
            best_ask=None,
            last_price=None,
            price_history=[],
            volume_history=[]
        )

        # Should return valid features with neutral values for missing data
        assert features.shape == (9,)
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)

        # Check specific neutral values
        assert features[3] == pytest.approx(0.5)  # best_bid_norm
        assert features[4] == pytest.approx(0.5)  # best_ask_norm
        assert features[5] == pytest.approx(0.0)  # spread_norm (no spread when no bids/asks)
        assert features[6] == pytest.approx(0.5)  # last_price_norm
        assert features[7] == pytest.approx(0.5)  # price_trend
        assert features[8] == pytest.approx(0.5)  # imbalance

    def test_extract_features_first_timestep(self, extractor):
        """Test feature extraction at first timestep with minimal data."""
        features = extractor.extract_features(
            valuation=80.0,
            num_tokens=5,
            num_trades=0,
            current_timestep=1,
            best_bid=45,
            best_ask=55,
            last_price=None,
            price_history=[],
            volume_history=[]
        )

        # Private features should work
        assert features[0] == pytest.approx(0.8)  # valuation
        assert features[1] == pytest.approx(1.0)  # all tokens remaining
        assert features[2] == pytest.approx(0.01)  # early timestep

        # Market features should work
        assert features[3] == pytest.approx(0.45)  # best_bid
        assert features[4] == pytest.approx(0.55)  # best_ask
        assert features[5] == pytest.approx(0.1)  # spread

        # No trade/history yet
        assert features[6] == pytest.approx(0.5)  # last_price (neutral)
        assert features[7] == pytest.approx(0.5)  # price_trend (neutral)
        assert features[8] == pytest.approx(0.5)  # imbalance (neutral)

    def test_get_feature_names(self, extractor):
        """Test feature names match dimensionality."""
        names = extractor.get_feature_names()
        assert len(names) == 9
        assert names[0] == "valuation_norm"
        assert names[8] == "imbalance_5_step"

    def test_get_observation_space_size(self, extractor):
        """Test observation space size is correct."""
        assert extractor.get_observation_space_size() == 9


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_max_price(self):
        """Test extractor handles zero max_price gracefully."""
        extractor = DoubleAuctionFeatureExtractor(
            max_price=1,  # Minimum non-zero
            max_timesteps=100,
            history_window=5
        )

        features = extractor.extract_features(
            valuation=0.5,
            num_tokens=10,
            num_trades=0,
            current_timestep=0,
            best_bid=0,
            best_ask=1,
            last_price=None,
            price_history=[],
            volume_history=[]
        )

        # Should not crash and return valid features
        assert features.shape == (9,)
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)

    def test_zero_max_timesteps(self):
        """Test extractor handles zero max_timesteps."""
        extractor = DoubleAuctionFeatureExtractor(
            max_price=100,
            max_timesteps=0,
            history_window=5
        )

        # Time normalization should return 0 when max_timesteps is 0
        assert extractor._normalize_time(50) == pytest.approx(0.0)

    def test_very_long_history(self, extractor):
        """Test extractor handles very long price/volume history."""
        long_price_history = list(range(1000))
        long_volume_history = [(i, 1000-i) for i in range(1000)]

        features = extractor.extract_features(
            valuation=50.0,
            num_tokens=10,
            num_trades=5,
            current_timestep=500,
            best_bid=45,
            best_ask=55,
            last_price=50,
            price_history=long_price_history,
            volume_history=long_volume_history
        )

        # Should only use recent window (5 entries)
        assert features.shape == (9,)
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)

    def test_extreme_valuations(self, extractor):
        """Test extractor clips extreme valuations."""
        # Very high valuation
        high_val = extractor._normalize_valuation(1000.0)
        assert high_val == pytest.approx(1.0)

        # Very low valuation
        low_val = extractor._normalize_valuation(-1000.0)
        assert low_val == pytest.approx(0.0)

    def test_negative_trades(self, extractor):
        """Test extractor handles negative trade counts gracefully."""
        # Should clip to valid range
        result = extractor._normalize_held_tokens(10, -5)
        assert result >= 0.0
        assert result <= 1.0
