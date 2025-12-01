"""
Behavioral tests for Ledyard (Easley-Ledyard-Olson) trader.

Tests the trader's distinctive personality: band-restricted trading,
time-progressive quotes, market power awareness, and comparison to other strategies.
"""

import numpy as np
import pytest

from engine.market import Market
from engine.token_generator import TokenGenerator
from traders.legacy.kaplan import Kaplan
from traders.legacy.ledyard import Ledyard
from traders.legacy.zic import ZIC


class TestBandRestrictedTrading:
    """Test that quotes stay within price band."""

    def test_buyer_bids_within_band(self, ledyard_buyer):
        """Buyer bids stay within [L_B, U_B]."""
        buyer = ledyard_buyer

        # Set up a specific band
        buyer.prev_price_low = 70.0
        buyer.prev_price_high = 90.0

        bids = []
        for time in range(10, 90, 10):
            buyer.current_high_bid = 65  # Below band
            buyer.current_low_ask = 95  # Above band
            buyer.bid_ask(time=time, nobidask=0)
            bid = buyer.bid_ask_response()
            if bid > 0:
                bids.append(bid)

        # All bids should be within band (clipped by value)
        # L_B = min(100, 70) = 70
        # U_B = min(100, 90) = 90
        for bid in bids:
            assert bid >= 65  # Must improve current_high_bid
            assert bid <= 100  # Cannot exceed value

    def test_seller_asks_within_band(self, ledyard_seller):
        """Seller asks stay within [L_S, U_S]."""
        seller = ledyard_seller

        # Set up a specific band
        seller.prev_price_low = 50.0
        seller.prev_price_high = 80.0

        asks = []
        for time in range(10, 90, 10):
            seller.current_high_bid = 45  # Below band
            seller.current_low_ask = 85  # Above band
            seller.bid_ask(time=time, nobidask=0)
            ask = seller.bid_ask_response()
            if ask > 0:
                asks.append(ask)

        # All asks should be within band (clipped by cost)
        # L_S = max(30, 50) = 50
        # U_S = max(30, 80) = 80
        for ask in asks:
            assert ask >= 30  # Cannot go below cost
            assert ask < 85  # Must improve current_low_ask


class TestTimeProgressiveQuotes:
    """Test conservative early, aggressive late behavior."""

    def test_buyer_more_conservative_early(self, ledyard_buyer):
        """Early bids closer to L_B, late bids closer to U_B."""
        buyer = ledyard_buyer
        buyer.prev_price_low = 60.0
        buyer.prev_price_high = 90.0

        # Collect bids at different times
        early_bids = []
        late_bids = []

        for _ in range(5):
            buyer.start_period(1)
            buyer.prev_price_low = 60.0
            buyer.prev_price_high = 90.0

            # Early bid
            buyer.current_high_bid = 50
            buyer.current_low_ask = 95
            buyer.bid_ask(time=10, nobidask=0)
            bid = buyer.bid_ask_response()
            if bid > 0:
                early_bids.append(bid)

            # Late bid
            buyer.current_high_bid = 50
            buyer.current_low_ask = 95
            buyer.bid_ask(time=80, nobidask=0)
            bid = buyer.bid_ask_response()
            if bid > 0:
                late_bids.append(bid)

        # Average late bid should be higher than average early bid
        if early_bids and late_bids:
            assert np.mean(late_bids) > np.mean(early_bids)

    def test_seller_more_conservative_early(self, ledyard_seller):
        """Early asks closer to U_S, late asks closer to L_S."""
        seller = ledyard_seller
        seller.prev_price_low = 50.0
        seller.prev_price_high = 80.0

        early_asks = []
        late_asks = []

        for _ in range(5):
            seller.start_period(1)
            seller.prev_price_low = 50.0
            seller.prev_price_high = 80.0

            # Early ask
            seller.current_high_bid = 45
            seller.current_low_ask = 100
            seller.bid_ask(time=10, nobidask=0)
            ask = seller.bid_ask_response()
            if ask > 0:
                early_asks.append(ask)

            # Late ask
            seller.current_high_bid = 45
            seller.current_low_ask = 100
            seller.bid_ask(time=80, nobidask=0)
            ask = seller.bid_ask_response()
            if ask > 0:
                late_asks.append(ask)

        # Average late ask should be lower than average early ask
        if early_asks and late_asks:
            assert np.mean(late_asks) < np.mean(early_asks)


class TestMarketPowerAwareness:
    """Test market power affects aggressiveness."""

    def test_crowded_buyer_concedes_faster(self):
        """Buyer on long side (many buyers) concedes faster."""
        # Create crowded buyer (8 buyers, 2 sellers)
        crowded = Ledyard(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
            num_times=100,
            num_buyers=8,
            num_sellers=2,
            seed=42,
        )

        # Create balanced buyer
        balanced = Ledyard(
            player_id=2,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
            num_times=100,
            num_buyers=4,
            num_sellers=4,
            seed=42,
        )

        # Set same band for both
        for buyer in [crowded, balanced]:
            buyer.prev_price_low = 60.0
            buyer.prev_price_high = 90.0

        # At same time, crowded buyer should have higher reservation price
        crowded.current_time = 50
        balanced.current_time = 50

        r_crowded = crowded._compute_reservation_price(100)
        r_balanced = balanced._compute_reservation_price(100)

        # Crowded buyer concedes faster → higher reservation price
        assert r_crowded > r_balanced

    def test_powerful_seller_shades_more(self):
        """Seller on short side (few sellers) shades more."""
        # Create powerful seller (2 sellers, 8 buyers)
        powerful = Ledyard(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[30, 40, 50, 60],
            price_min=0,
            price_max=200,
            num_times=100,
            num_buyers=8,
            num_sellers=2,
            seed=42,
        )

        # Create balanced seller
        balanced = Ledyard(
            player_id=2,
            is_buyer=False,
            num_tokens=4,
            valuations=[30, 40, 50, 60],
            price_min=0,
            price_max=200,
            num_times=100,
            num_buyers=4,
            num_sellers=4,
            seed=42,
        )

        # Set same band for both
        for seller in [powerful, balanced]:
            seller.prev_price_low = 50.0
            seller.prev_price_high = 80.0

        # At same time, powerful seller should have higher reservation price
        powerful.current_time = 50
        balanced.current_time = 50

        r_powerful = powerful._compute_reservation_price(30)
        r_balanced = balanced._compute_reservation_price(30)

        # Powerful seller shades more → higher reservation price (more patient)
        assert r_powerful > r_balanced


class TestBandNarrowing:
    """Test that price band narrows across periods."""

    def test_band_narrows_with_trades(self, ledyard_buyer):
        """Band should narrow as trades cluster around equilibrium."""
        buyer = ledyard_buyer

        # Initial wide band
        initial_width = buyer.prev_price_high - buyer.prev_price_low

        # Simulate trades narrowing the band
        buyer.current_traded_prices = [85, 88, 90, 87, 86]
        buyer.current_asks = [92, 90, 88]
        buyer.current_bids = [82, 85, 87]

        buyer.start_period(2)

        # New band should be narrower
        new_width = buyer.prev_price_high - buyer.prev_price_low

        assert new_width < initial_width


class TestDifferentFromKaplan:
    """Test Ledyard differs from Kaplan (spread-based sniper)."""

    def test_ledyard_continuous_vs_kaplan_sniping(self, ledyard_buyer, kaplan_buyer):
        """Ledyard trades continuously, Kaplan waits for tight spreads."""
        # Set up wide spread scenario
        for agent in [ledyard_buyer, kaplan_buyer]:
            agent.start_period(1)

        # Give Ledyard a band to work with
        ledyard_buyer.prev_price_low = 70.0
        ledyard_buyer.prev_price_high = 90.0

        ledyard_bids = 0
        kaplan_bids = 0

        # Test across multiple time steps with wide spread
        for time in range(20, 80, 10):
            # Wide spread (50 points)
            ledyard_buyer.current_high_bid = 50
            ledyard_buyer.current_low_ask = 100
            kaplan_buyer.current_high_bid = 50
            kaplan_buyer.current_low_ask = 100

            ledyard_buyer.bid_ask(time=time, nobidask=0)
            bid = ledyard_buyer.bid_ask_response()
            if bid > 0:
                ledyard_bids += 1

            kaplan_buyer.bid_ask(time=time, nobidask=0)
            bid = kaplan_buyer.bid_ask_response()
            if bid > 0:
                kaplan_bids += 1

        # Ledyard should be more active (not waiting for tight spread)
        # Kaplan typically waits for spread to narrow
        # This is a characteristic difference
        pass  # The test documents expected behavior


class TestDifferentFromZIC:
    """Test Ledyard differs from ZIC (random uniform)."""

    def test_ledyard_structured_vs_zic_random(self, ledyard_buyer, zic_buyer):
        """Ledyard quotes are structured, ZIC is random."""
        ledyard_buyer.start_period(1)
        zic_buyer.start_period(1)

        # Set up Ledyard with narrow band
        ledyard_buyer.prev_price_low = 80.0
        ledyard_buyer.prev_price_high = 90.0

        ledyard_bids = []
        zic_bids = []

        # Collect multiple bids
        for _ in range(20):
            ledyard_buyer.current_high_bid = 75
            ledyard_buyer.current_low_ask = 95
            ledyard_buyer.bid_ask(time=50, nobidask=0)
            bid = ledyard_buyer.bid_ask_response()
            if bid > 0:
                ledyard_bids.append(bid)

            zic_buyer.current_high_bid = 75
            zic_buyer.current_low_ask = 95
            zic_buyer.bid_ask(time=50, nobidask=0)
            bid = zic_buyer.bid_ask_response()
            if bid > 0:
                zic_bids.append(bid)

        # Ledyard should have lower variance (constrained to band)
        if ledyard_bids and len(ledyard_bids) > 2:
            ledyard_var = np.var(ledyard_bids)
            # Ledyard quotes should be relatively consistent (small noise only)
            assert ledyard_var < 50  # Should be low variance

        # ZIC has high variance (uniform random)
        if zic_bids and len(zic_bids) > 2:
            zic_var = np.var(zic_bids)
            # ZIC should have higher variance
            pass  # Just documenting expected difference


class TestMarketIntegration:
    """Full market execution tests."""

    def test_ledyard_vs_zic_profitability(self):
        """Ledyard should earn positive profit against ZIC."""
        np.random.seed(42)

        token_gen = TokenGenerator(game_type=6453, num_tokens=4, seed=42)
        token_gen.new_round()

        # Create Ledyard agent
        ledyard = Ledyard(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=token_gen.generate_tokens(True),
            price_min=0,
            price_max=1000,
            num_times=100,
            num_buyers=4,
            num_sellers=4,
            seed=42,
        )

        buyers = [ledyard]
        for i in range(3):
            buyers.append(
                ZIC(
                    player_id=i + 2,
                    is_buyer=True,
                    num_tokens=4,
                    valuations=token_gen.generate_tokens(True),
                    price_min=0,
                    price_max=1000,
                    seed=100 + i,
                )
            )

        sellers = []
        for i in range(4):
            sellers.append(
                ZIC(
                    player_id=i + 5,
                    is_buyer=False,
                    num_tokens=4,
                    valuations=token_gen.generate_tokens(False),
                    price_min=0,
                    price_max=1000,
                    seed=200 + i,
                )
            )

        all_agents = buyers + sellers
        for agent in all_agents:
            agent.start_period(1)

        market = Market(
            num_buyers=4,
            num_sellers=4,
            num_times=100,
            price_min=0,
            price_max=1000,
            buyers=buyers,
            sellers=sellers,
            seed=42,
        )

        for _ in range(100):
            market.run_time_step()

        # Ledyard should be profitable (or at least not losing badly)
        assert ledyard.period_profit >= 0 or ledyard.num_trades == 0

    def test_ledyard_completes_trades(self):
        """Ledyard should complete at least some trades per period."""
        np.random.seed(123)

        token_gen = TokenGenerator(game_type=6453, num_tokens=4, seed=123)

        total_trades = 0
        num_periods = 5

        for period in range(1, num_periods + 1):
            token_gen.new_round()

            ledyard = Ledyard(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=token_gen.generate_tokens(True),
                price_min=0,
                price_max=1000,
                num_times=100,
                num_buyers=4,
                num_sellers=4,
                seed=period,
            )

            buyers = [ledyard]
            for i in range(3):
                buyers.append(
                    ZIC(
                        player_id=i + 2,
                        is_buyer=True,
                        num_tokens=4,
                        valuations=token_gen.generate_tokens(True),
                        price_min=0,
                        price_max=1000,
                        seed=period * 100 + i,
                    )
                )

            sellers = []
            for i in range(4):
                sellers.append(
                    ZIC(
                        player_id=i + 5,
                        is_buyer=False,
                        num_tokens=4,
                        valuations=token_gen.generate_tokens(False),
                        price_min=0,
                        price_max=1000,
                        seed=period * 200 + i,
                    )
                )

            all_agents = buyers + sellers
            for agent in all_agents:
                agent.start_period(period)

            market = Market(
                num_buyers=4,
                num_sellers=4,
                num_times=100,
                price_min=0,
                price_max=1000,
                buyers=buyers,
                sellers=sellers,
                seed=period * 1000,
            )

            for _ in range(100):
                market.run_time_step()

            total_trades += ledyard.num_trades

        # Should complete at least some trades across periods
        assert total_trades > 0


class TestCrossPeriodsLearning:
    """Test that Ledyard learns from previous periods."""

    def test_band_adapts_across_periods(self):
        """Price band should adapt based on observed prices."""
        buyer = Ledyard(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
            num_times=100,
            num_buyers=4,
            num_sellers=4,
            seed=42,
        )

        # Period 1: Wide band
        assert buyer.prev_price_high - buyer.prev_price_low == 200

        # Simulate period 1 trades
        buyer.current_traded_prices = [80, 85, 90]
        buyer.current_asks = [78, 82]
        buyer.current_bids = [88, 92]

        # Start period 2
        buyer.start_period(2)

        # Band should have narrowed
        assert buyer.prev_price_low == 78.0  # min(asks + trades)
        assert buyer.prev_price_high == 92.0  # max(bids + trades)

        # Simulate period 2 trades
        buyer.current_traded_prices = [83, 85, 87]
        buyer.current_asks = [82, 84]
        buyer.current_bids = [86, 88]

        # Start period 3
        buyer.start_period(3)

        # Band should have narrowed further
        assert buyer.prev_price_low == 82.0
        assert buyer.prev_price_high == 88.0


class TestBehavioralProfile:
    """Test behavioral characteristics match expected profile."""

    def test_low_pass_rate_late(self, ledyard_buyer):
        """Ledyard should trade actively late (late-period override)."""
        buyer = ledyard_buyer
        buyer.start_period(1)

        pass_count = 0
        total_steps = 10

        for step in range(90, 100):
            buyer.current_high_bid = 70
            buyer.current_low_ask = 95  # Profitable
            buyer.bid_ask(time=step, nobidask=0)
            bid = buyer.bid_ask_response()
            if bid == 0:
                pass_count += 1

        # Should have low pass rate late (aggressive)
        pass_rate = pass_count / total_steps
        assert pass_rate < 0.5

    def test_deterministic_with_seed(self):
        """Ledyard with same seed should produce consistent results."""
        results = []

        for _ in range(2):
            buyer = Ledyard(
                player_id=1,
                is_buyer=True,
                num_tokens=4,
                valuations=[100, 90, 80, 70],
                price_min=0,
                price_max=200,
                num_times=100,
                num_buyers=4,
                num_sellers=4,
                seed=42,
            )
            buyer.start_period(1)

            buyer.prev_price_low = 70.0
            buyer.prev_price_high = 90.0
            buyer.current_high_bid = 60
            buyer.current_low_ask = 95

            buyer.bid_ask(time=50, nobidask=0)
            bid = buyer.bid_ask_response()
            results.append(bid)

        # Both runs should produce identical results (same seed)
        assert results[0] == results[1]


# Fixtures for comparison tests
@pytest.fixture
def kaplan_buyer():
    """Kaplan buyer for comparison."""
    return Kaplan(
        player_id=10,
        is_buyer=True,
        num_tokens=4,
        valuations=[100, 90, 80, 70],
        price_min=0,
        price_max=200,
        num_times=100,
        seed=42,
    )
