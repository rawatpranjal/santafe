"""
Investigate why ZIP has low trading volume in flat supply markets.
"""

import numpy as np
from engine.market import Market
from engine.efficiency import get_transaction_prices
from traders.legacy.zip import ZIP
from traders.legacy.zic import ZIC


def test_why_low_trading_volume():
    """
    Detailed analysis of why there are only 5 trades in 200 timesteps.
    """
    print("\n" + "="*80)
    print("TRADING VOLUME INVESTIGATION")
    print("="*80)

    # Create flat supply market
    P0 = 50
    seller_cost = 50
    num_buyers = 6
    num_sellers = 6

    buyers = []
    sellers = []

    # Create buyers
    for i in range(1, num_buyers + 1):
        val = 100 - (i - 1) * 10  # 100, 90, 80, 70, 60, 50
        buyer = ZIP(
            player_id=i,
            is_buyer=True,
            num_tokens=1,
            valuations=[val],
            price_min=0,
            price_max=100,
            seed=42 + i,
        )
        buyers.append(buyer)
        print(f"Buyer {i}: valuation={val}, initial_margin={buyer.margin:.3f}, "
              f"initial_quote={buyer._calculate_quote()}")

    # Create sellers
    for i in range(1, num_sellers + 1):
        seller = ZIP(
            player_id=i,
            is_buyer=False,
            num_tokens=1,
            valuations=[seller_cost],
            price_min=0,
            price_max=100,
            seed=42 + num_buyers + i,
        )
        sellers.append(seller)
        print(f"Seller {i}: cost={seller_cost}, initial_margin={seller.margin:.3f}, "
              f"initial_quote={seller._calculate_quote()}")

    print("\n" + "-"*80)
    print("SIMULATION: First 20 timesteps")
    print("-"*80)

    market = Market(
        num_buyers=num_buyers,
        num_sellers=num_sellers,
        num_times=20,
        price_min=0,
        price_max=100,
        buyers=buyers,
        sellers=sellers,
        seed=42,
    )

    trades_occurred = []

    for timestep in range(1, 21):
        # Before timestep
        print(f"\nTimestep {timestep}:")

        # Show current quotes
        buyer_quotes = [b._calculate_quote() for b in buyers if b.num_trades < b.num_tokens]
        seller_quotes = [s._calculate_quote() for s in sellers if s.num_trades < s.num_tokens]

        if buyer_quotes and seller_quotes:
            print(f"  Active buyer quotes: {buyer_quotes}")
            print(f"  Active seller quotes: {seller_quotes}")
            print(f"  Max buyer quote: {max(buyer_quotes)}, Min seller quote: {min(seller_quotes)}")

        # Run timestep
        success = market.run_time_step()

        # Check if trade occurred
        trade_price = market.orderbook.trade_price[timestep]
        if trade_price > 0:
            trades_occurred.append(timestep)
            print(f"  ✅ TRADE at price {trade_price}")

            # Check who traded
            for b in buyers:
                if hasattr(b, 'num_trades'):
                    pass  # Would need to track this better

        else:
            print(f"  ❌ NO TRADE")

            # Why no trade? Check acceptance
            high_bid = market.orderbook.high_bid[timestep]
            low_ask = market.orderbook.low_ask[timestep]
            high_bidder = market.orderbook.high_bidder[timestep]
            low_asker = market.orderbook.low_asker[timestep]

            print(f"     high_bid={high_bid} (by buyer {high_bidder})")
            print(f"     low_ask={low_ask} (by seller {low_asker})")

            if high_bid > 0 and low_ask > 0:
                if high_bid >= low_ask:
                    print(f"     Spread CROSSED but no trade - acceptance issue?")

                    # Check what the agents' quotes were
                    if high_bidder > 0 and high_bidder <= len(buyers):
                        buyer_quote = buyers[high_bidder - 1].current_quote
                        print(f"     Buyer {high_bidder} quote={buyer_quote}, would accept ask<={buyer_quote}")
                        print(f"     Actual ask={low_ask}, accept? {low_ask <= buyer_quote}")

                    if low_asker > 0 and low_asker <= len(sellers):
                        seller_quote = sellers[low_asker - 1].current_quote
                        print(f"     Seller {low_asker} quote={seller_quote}, would accept bid>={seller_quote}")
                        print(f"     Actual bid={high_bid}, accept? {high_bid >= seller_quote}")
                else:
                    print(f"     Spread NOT crossed: bid={high_bid} < ask={low_ask}")

        if not success:
            print(f"  Market failed!")
            break

    print("\n" + "="*80)
    print(f"Total trades in 20 timesteps: {len(trades_occurred)}")
    print(f"Trades occurred at timesteps: {trades_occurred}")
    print("="*80)


if __name__ == "__main__":
    test_why_low_trading_volume()
