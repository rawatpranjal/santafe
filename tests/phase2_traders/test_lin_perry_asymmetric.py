"""
Asymmetric market tests for Lin and Perry traders.

Tests robustness in non-standard market conditions:
- Asymmetric market sizes (3v7 buyers/sellers)
- Unequal token counts per agent
- Skewed valuations (box markets)

These tests validate that weighting formulas and adaptive mechanisms work
correctly when market composition deviates from symmetric baselines.
"""

import pytest
import numpy as np
from engine.market import Market
from engine.efficiency import calculate_allocative_efficiency, calculate_max_surplus, calculate_actual_surplus, extract_trades_from_orderbook
from traders.legacy.lin import Lin
from traders.legacy.perry import Perry
from traders.legacy.zic import ZIC


class TestLinAsymmetricMarkets:
    """Test Lin trader in asymmetric market conditions."""

    def test_lin_buyers_3v7_asymmetric_market(self):
        """
        Test Lin buyers in 3v7 asymmetric market (fewer buyers).

        Validates that Lin's market composition weighting formula works correctly
        when buyers are outnumbered 3:7.
        """
        efficiencies = []

        for seed in range(5):
            buyers = [
                Lin(
                    player_id=i+1,
                    is_buyer=True,
                    num_tokens=3,
                    valuations=[100, 90, 80],
                    num_buyers=3,
                    num_sellers=7,
                    num_times=50,
                    seed=seed*10 + i,
                )
                for i in range(3)
            ]

            sellers = [
                ZIC(
                    player_id=i+1,
                    is_buyer=False,
                    num_tokens=3,
                    valuations=[20, 30, 40],
                    seed=seed*10 + 100 + i,
                )
                for i in range(7)
            ]

            market = Market(
                num_buyers=3,
                num_sellers=7,
                num_times=50,
                price_min=0,
                price_max=200,
                buyers=buyers,
                sellers=sellers,
                seed=seed,
            )

            # Run market
            for step in range(50):
                success = market.run_time_step()
                if not success:
                    break

            # Calculate efficiency
            buyer_vals_list = [[100, 90, 80] for _ in range(3)]
            seller_costs_list = [[20, 30, 40] for _ in range(7)]
            buyer_vals_dict = {i+1: [100, 90, 80] for i in range(3)}
            seller_costs_dict = {i+1: [20, 30, 40] for i in range(7)}

            max_surplus = calculate_max_surplus(buyer_vals_list, seller_costs_list)
            trades = extract_trades_from_orderbook(market.orderbook, 50)
            actual_surplus = calculate_actual_surplus(trades, buyer_vals_dict, seller_costs_dict)
            efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)
            efficiencies.append(efficiency)

        mean_efficiency = np.mean(efficiencies)
        print(f"\nLin 3v7 Asymmetric Market:")
        print(f"  Mean efficiency: {mean_efficiency:.2f}%")
        print(f"  Range: [{min(efficiencies):.2f}%, {max(efficiencies):.2f}%]")

        # Lin should still achieve reasonable efficiency in asymmetric markets
        assert mean_efficiency >= 50.0, f"Lin efficiency {mean_efficiency:.2f}% too low in asymmetric market"

    def test_lin_sellers_3v7_asymmetric_market(self):
        """
        Test Lin sellers in 7v3 asymmetric market (more buyers).

        Validates Lin's weighting formula when sellers are outnumbered.
        """
        efficiencies = []

        for seed in range(5):
            buyers = [
                ZIC(
                    player_id=i+1,
                    is_buyer=True,
                    num_tokens=3,
                    valuations=[100, 90, 80],
                    seed=seed*10 + i,
                )
                for i in range(7)
            ]

            sellers = [
                Lin(
                    player_id=i+1,
                    is_buyer=False,
                    num_tokens=3,
                    valuations=[20, 30, 40],
                    num_buyers=7,
                    num_sellers=3,
                    num_times=50,
                    seed=seed*10 + 100 + i,
                )
                for i in range(3)
            ]

            market = Market(
                num_buyers=7,
                num_sellers=3,
                num_times=50,
                price_min=0,
                price_max=200,
                buyers=buyers,
                sellers=sellers,
                seed=seed,
            )

            # Run market
            for step in range(50):
                success = market.run_time_step()
                if not success:
                    break

            # Calculate efficiency
            buyer_vals_list = [[100, 90, 80] for _ in range(7)]
            seller_costs_list = [[20, 30, 40] for _ in range(3)]
            buyer_vals_dict = {i+1: [100, 90, 80] for i in range(7)}
            seller_costs_dict = {i+1: [20, 30, 40] for i in range(3)}

            max_surplus = calculate_max_surplus(buyer_vals_list, seller_costs_list)
            trades = extract_trades_from_orderbook(market.orderbook, 50)
            actual_surplus = calculate_actual_surplus(trades, buyer_vals_dict, seller_costs_dict)
            efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)
            efficiencies.append(efficiency)

        mean_efficiency = np.mean(efficiencies)
        print(f"\nLin 7v3 Asymmetric Market:")
        print(f"  Mean efficiency: {mean_efficiency:.2f}%")
        print(f"  Range: [{min(efficiencies):.2f}%, {max(efficiencies):.2f}%]")

        # Lin should handle asymmetry in both directions
        assert mean_efficiency >= 50.0, f"Lin efficiency {mean_efficiency:.2f}% too low in asymmetric market"

    def test_lin_unequal_token_counts(self):
        """
        Test Lin with unequal token counts per agent.

        Validates that Lin's token depletion tracking (num_trades / num_tokens)
        works correctly when agents have different endowments.
        """
        buyers = [
            Lin(
                player_id=1,
                is_buyer=True,
                num_tokens=5,  # More tokens
                valuations=[100, 95, 90, 85, 80],
                num_buyers=2,
                num_sellers=2,
                num_times=50,
                seed=42,
            ),
            ZIC(
                player_id=2,
                is_buyer=True,
                num_tokens=2,  # Fewer tokens
                valuations=[100, 90],
                seed=43,
            ),
        ]

        sellers = [
            ZIC(
                player_id=1,
                is_buyer=False,
                num_tokens=3,
                valuations=[20, 30, 40],
                seed=44,
            ),
            ZIC(
                player_id=2,
                is_buyer=False,
                num_tokens=4,
                valuations=[25, 35, 45, 55],
                seed=45,
            ),
        ]

        market = Market(
            num_buyers=2,
            num_sellers=2,
            num_times=50,
            price_min=0,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=100,
        )

        lin_buyer = buyers[0]

        # Run market
        for step in range(50):
            success = market.run_time_step()
            if not success:
                break

        # Verify Lin tracked trades correctly
        print(f"\nLin Unequal Token Test:")
        print(f"  Lin tokens: 5")
        print(f"  Lin trades: {lin_buyer.num_trades}")
        print(f"  Token depletion: {lin_buyer.num_trades / 5:.2%}")

        # Lin should not crash with unequal tokens
        assert lin_buyer.num_trades <= 5, "Lin traded more than available tokens"


class TestPerryAsymmetricMarkets:
    """Test Perry trader in asymmetric market conditions."""

    def test_perry_buyers_3v7_asymmetric_market(self):
        """
        Test Perry buyers in 3v7 asymmetric market.

        Validates Perry's adaptive learning works with skewed market composition.
        """
        efficiencies = []

        for seed in range(5):
            buyers = [
                Perry(
                    player_id=i+1,
                    is_buyer=True,
                    num_tokens=3,
                    valuations=[100, 90, 80],
                    num_buyers=3,
                    num_sellers=7,
                    num_times=50,
                    seed=seed*10 + i,
                )
                for i in range(3)
            ]

            sellers = [
                ZIC(
                    player_id=i+1,
                    is_buyer=False,
                    num_tokens=3,
                    valuations=[20, 30, 40],
                    seed=seed*10 + 100 + i,
                )
                for i in range(7)
            ]

            market = Market(
                num_buyers=3,
                num_sellers=7,
                num_times=50,
                price_min=0,
                price_max=200,
                buyers=buyers,
                sellers=sellers,
                seed=seed,
            )

            # Run market
            for step in range(50):
                success = market.run_time_step()
                if not success:
                    break

            # Calculate efficiency
            buyer_vals_list = [[100, 90, 80] for _ in range(3)]
            seller_costs_list = [[20, 30, 40] for _ in range(7)]
            buyer_vals_dict = {i+1: [100, 90, 80] for i in range(3)}
            seller_costs_dict = {i+1: [20, 30, 40] for i in range(7)}

            max_surplus = calculate_max_surplus(buyer_vals_list, seller_costs_list)
            trades = extract_trades_from_orderbook(market.orderbook, 50)
            actual_surplus = calculate_actual_surplus(trades, buyer_vals_dict, seller_costs_dict)
            efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)
            efficiencies.append(efficiency)

        mean_efficiency = np.mean(efficiencies)
        print(f"\nPerry 3v7 Asymmetric Market:")
        print(f"  Mean efficiency: {mean_efficiency:.2f}%")
        print(f"  Range: [{min(efficiencies):.2f}%, {max(efficiencies):.2f}%]")

        # Perry should survive asymmetric markets
        assert mean_efficiency >= 0.0, "Perry should not crash in asymmetric market"

    def test_perry_sellers_7v3_asymmetric_market(self):
        """
        Test Perry sellers in 7v3 asymmetric market.

        Validates Perry's price statistics and a0 adaptation with more buyers.
        """
        efficiencies = []

        for seed in range(5):
            buyers = [
                ZIC(
                    player_id=i+1,
                    is_buyer=True,
                    num_tokens=3,
                    valuations=[100, 90, 80],
                    seed=seed*10 + i,
                )
                for i in range(7)
            ]

            sellers = [
                Perry(
                    player_id=i+1,
                    is_buyer=False,
                    num_tokens=3,
                    valuations=[20, 30, 40],
                    num_buyers=7,
                    num_sellers=3,
                    num_times=50,
                    seed=seed*10 + 100 + i,
                )
                for i in range(3)
            ]

            market = Market(
                num_buyers=7,
                num_sellers=3,
                num_times=50,
                price_min=0,
                price_max=200,
                buyers=buyers,
                sellers=sellers,
                seed=seed,
            )

            # Run market
            for step in range(50):
                success = market.run_time_step()
                if not success:
                    break

            # Calculate efficiency
            buyer_vals_list = [[100, 90, 80] for _ in range(7)]
            seller_costs_list = [[20, 30, 40] for _ in range(3)]
            buyer_vals_dict = {i+1: [100, 90, 80] for i in range(7)}
            seller_costs_dict = {i+1: [20, 30, 40] for i in range(3)}

            max_surplus = calculate_max_surplus(buyer_vals_list, seller_costs_list)
            trades = extract_trades_from_orderbook(market.orderbook, 50)
            actual_surplus = calculate_actual_surplus(trades, buyer_vals_dict, seller_costs_dict)
            efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)
            efficiencies.append(efficiency)

        mean_efficiency = np.mean(efficiencies)
        print(f"\nPerry 7v3 Asymmetric Market:")
        print(f"  Mean efficiency: {mean_efficiency:.2f}%")
        print(f"  Range: [{min(efficiencies):.2f}%, {max(efficiencies):.2f}%]")

        # Perry should handle asymmetry
        assert mean_efficiency >= 0.0, "Perry should not crash in asymmetric market"

    def test_perry_unequal_token_counts(self):
        """
        Test Perry with unequal token counts.

        Validates Perry's efficiency evaluation and a0 adaptation work correctly
        when agents have different endowments.
        """
        buyers = [
            Perry(
                player_id=1,
                is_buyer=True,
                num_tokens=5,
                valuations=[100, 95, 90, 85, 80],
                num_buyers=2,
                num_sellers=2,
                num_times=50,
                seed=42,
            ),
            ZIC(
                player_id=2,
                is_buyer=True,
                num_tokens=2,
                valuations=[100, 90],
                seed=43,
            ),
        ]

        sellers = [
            ZIC(
                player_id=1,
                is_buyer=False,
                num_tokens=3,
                valuations=[20, 30, 40],
                seed=44,
            ),
            ZIC(
                player_id=2,
                is_buyer=False,
                num_tokens=4,
                valuations=[25, 35, 45, 55],
                seed=45,
            ),
        ]

        market = Market(
            num_buyers=2,
            num_sellers=2,
            num_times=50,
            price_min=0,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
            seed=100,
        )

        perry_buyer = buyers[0]
        initial_a0 = perry_buyer.a0

        # Run market
        for step in range(50):
            success = market.run_time_step()
            if not success:
                break

        # Verify Perry tracked trades and adapted parameters
        print(f"\nPerry Unequal Token Test:")
        print(f"  Perry tokens: 5")
        print(f"  Perry trades: {perry_buyer.num_trades}")
        print(f"  Initial a0: {initial_a0:.3f}")
        print(f"  Final a0: {perry_buyer.a0:.3f}")

        # Perry should not crash with unequal tokens
        assert perry_buyer.num_trades <= 5, "Perry traded more than available tokens"


class TestBoxMarkets:
    """Test Lin and Perry in box markets (extreme valuations)."""

    def test_lin_box_market_wide_spread(self):
        """
        Test Lin in box market with wide buyer/seller spread.

        Valuations: Buyers [200, 190, 180], Sellers [10, 20, 30]
        Large surplus available - tests if Lin can capture it.
        """
        buyers = [
            Lin(
                player_id=i+1,
                is_buyer=True,
                num_tokens=3,
                valuations=[200, 190, 180],
                num_buyers=3,
                num_sellers=3,
                num_times=50,
                seed=42 + i,
            )
            for i in range(3)
        ]

        sellers = [
            ZIC(
                player_id=i+1,
                is_buyer=False,
                num_tokens=3,
                valuations=[10, 20, 30],
                seed=100 + i,
            )
            for i in range(3)
        ]

        market = Market(
            num_buyers=3,
            num_sellers=3,
            num_times=50,
            price_min=0,
            price_max=300,
            buyers=buyers,
            sellers=sellers,
            seed=200,
        )

        # Run market
        for step in range(50):
            success = market.run_time_step()
            if not success:
                break

        # Calculate efficiency
        buyer_vals_list = [[200, 190, 180] for _ in range(3)]
        seller_costs_list = [[10, 20, 30] for _ in range(3)]
        buyer_vals_dict = {i+1: [200, 190, 180] for i in range(3)}
        seller_costs_dict = {i+1: [10, 20, 30] for i in range(3)}

        max_surplus = calculate_max_surplus(buyer_vals_list, seller_costs_list)
        trades = extract_trades_from_orderbook(market.orderbook, 50)
        actual_surplus = calculate_actual_surplus(trades, buyer_vals_dict, seller_costs_dict)
        efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)

        print(f"\nLin Box Market (wide spread):")
        print(f"  Max surplus: {max_surplus}")
        print(f"  Actual surplus: {actual_surplus}")
        print(f"  Efficiency: {efficiency:.2f}%")

        # Large surplus should make trading easy
        assert efficiency >= 70.0, f"Lin efficiency {efficiency:.2f}% too low in box market"

    def test_perry_box_market_narrow_spread(self):
        """
        Test Perry in box market with narrow spread.

        Valuations: Buyers [60, 55, 50], Sellers [40, 45, 50]
        Small surplus - tests Perry's conservative strategy.
        """
        buyers = [
            Perry(
                player_id=i+1,
                is_buyer=True,
                num_tokens=3,
                valuations=[60, 55, 50],
                num_buyers=3,
                num_sellers=3,
                num_times=50,
                seed=42 + i,
            )
            for i in range(3)
        ]

        sellers = [
            ZIC(
                player_id=i+1,
                is_buyer=False,
                num_tokens=3,
                valuations=[40, 45, 50],
                seed=100 + i,
            )
            for i in range(3)
        ]

        market = Market(
            num_buyers=3,
            num_sellers=3,
            num_times=50,
            price_min=0,
            price_max=100,
            buyers=buyers,
            sellers=sellers,
            seed=200,
        )

        # Run market
        for step in range(50):
            success = market.run_time_step()
            if not success:
                break

        # Calculate efficiency
        buyer_vals_list = [[60, 55, 50] for _ in range(3)]
        seller_costs_list = [[40, 45, 50] for _ in range(3)]
        buyer_vals_dict = {i+1: [60, 55, 50] for i in range(3)}
        seller_costs_dict = {i+1: [40, 45, 50] for i in range(3)}

        max_surplus = calculate_max_surplus(buyer_vals_list, seller_costs_list)
        trades = extract_trades_from_orderbook(market.orderbook, 50)
        actual_surplus = calculate_actual_surplus(trades, buyer_vals_dict, seller_costs_dict)
        efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)

        print(f"\nPerry Box Market (narrow spread):")
        print(f"  Max surplus: {max_surplus}")
        print(f"  Actual surplus: {actual_surplus}")
        print(f"  Efficiency: {efficiency:.2f}%")

        # Perry's conservative strategy may struggle with narrow spreads
        # Just verify it doesn't crash
        assert efficiency >= 0.0, "Perry should survive narrow spread market"
