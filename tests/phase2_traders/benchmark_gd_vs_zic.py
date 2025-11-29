import numpy as np
import matplotlib.pyplot as plt
from engine.market import Market
from traders.legacy.zic import ZIC
from traders.legacy.gd import GD


def run_session(buyers, sellers, num_times=100):
    market = Market(
        num_buyers=len(buyers),
        num_sellers=len(sellers),
        num_times=num_times,
        price_min=0,
        price_max=400,
        buyers=buyers,
        sellers=sellers
    )
    
    # Run market
    for _ in range(num_times):
        market.run_time_step()
        
    # Calculate Efficiency
    # Theoretical Max Surplus
    # 1. Aggregate Demand and Supply
    all_valuations = []
    for b in buyers:
        all_valuations.extend(b.valuations)
    all_valuations.sort(reverse=True)
    
    all_costs = []
    for s in sellers:
        all_costs.extend(s.valuations)
    all_costs.sort()
    
    max_surplus = 0
    for v, c in zip(all_valuations, all_costs):
        if v > c:
            max_surplus += (v - c)
        else:
            break
            
    # Actual Surplus
    actual_surplus = 0
    # We need to track trades. Market doesn't expose trade history easily in this version?
    # We can sum agent profits.
    buyer_profit = sum(b.period_profit for b in buyers)
    seller_profit = sum(s.period_profit for s in sellers)
    actual_surplus = buyer_profit + seller_profit
    
    efficiency = (actual_surplus / max_surplus) * 100 if max_surplus > 0 else 0
    
    return efficiency, buyer_profit, seller_profit

def benchmark(num_sessions=10):
    print(f"Running Benchmark: GD vs ZIC ({num_sessions} sessions each)")
    
    # Configuration
    num_agents = 5
    num_tokens = 5
    price_max = 400
    
    # Scenario 1: GD Buyers vs ZIC Sellers
    eff_1 = []
    gd_profits_1 = []
    zic_profits_1 = []

    for _ in range(num_sessions):
        # Generate random valuations/costs
        valuations = [sorted(np.random.randint(0, price_max, num_tokens), reverse=True) for _ in range(num_agents)]
        costs = [sorted(np.random.randint(0, price_max, num_tokens)) for _ in range(num_agents)]

        buyers = [GD(i+1, True, num_tokens, valuations[i], 0, price_max) for i in range(num_agents)]
        sellers = [ZIC(i+1+num_agents, False, num_tokens, costs[i], 0, price_max) for i in range(num_agents)]

        eff, buyer_profit, seller_profit = run_session(buyers, sellers)
        eff_1.append(eff)
        gd_profits_1.append(buyer_profit)  # GD are buyers
        zic_profits_1.append(seller_profit)  # ZIC are sellers

    avg_gd_1 = np.mean(gd_profits_1)
    avg_zic_1 = np.mean(zic_profits_1)
    dominance_1 = avg_gd_1 / avg_zic_1 if avg_zic_1 > 0 else float('inf')

    print(f"\nScenario 1 (GD Buyers vs ZIC Sellers):")
    print(f"  Avg Efficiency = {np.mean(eff_1):.2f}%")
    print(f"  GD Profit:  {avg_gd_1:.0f}")
    print(f"  ZIC Profit: {avg_zic_1:.0f}")
    print(f"  GD Dominance: {dominance_1:.2f}x")

    # Scenario 2: ZIC Buyers vs GD Sellers
    eff_2 = []
    zic_profits_2 = []
    gd_profits_2 = []

    for _ in range(num_sessions):
        valuations = [sorted(np.random.randint(0, price_max, num_tokens), reverse=True) for _ in range(num_agents)]
        costs = [sorted(np.random.randint(0, price_max, num_tokens)) for _ in range(num_agents)]

        buyers = [ZIC(i+1, True, num_tokens, valuations[i], 0, price_max) for i in range(num_agents)]
        sellers = [GD(i+1+num_agents, False, num_tokens, costs[i], 0, price_max) for i in range(num_agents)]

        eff, buyer_profit, seller_profit = run_session(buyers, sellers)
        eff_2.append(eff)
        zic_profits_2.append(buyer_profit)  # ZIC are buyers
        gd_profits_2.append(seller_profit)  # GD are sellers

    avg_zic_2 = np.mean(zic_profits_2)
    avg_gd_2 = np.mean(gd_profits_2)
    dominance_2 = avg_gd_2 / avg_zic_2 if avg_zic_2 > 0 else float('inf')

    print(f"\nScenario 2 (ZIC Buyers vs GD Sellers):")
    print(f"  Avg Efficiency = {np.mean(eff_2):.2f}%")
    print(f"  ZIC Profit: {avg_zic_2:.0f}")
    print(f"  GD Profit:  {avg_gd_2:.0f}")
    print(f"  GD Dominance: {dominance_2:.2f}x")
    
    # Scenario 3: GD vs GD
    eff_3 = []
    gd_buyer_profits_3 = []
    gd_seller_profits_3 = []

    for _ in range(num_sessions):
        valuations = [sorted(np.random.randint(0, price_max, num_tokens), reverse=True) for _ in range(num_agents)]
        costs = [sorted(np.random.randint(0, price_max, num_tokens)) for _ in range(num_agents)]

        buyers = [GD(i+1, True, num_tokens, valuations[i], 0, price_max) for i in range(num_agents)]
        sellers = [GD(i+1+num_agents, False, num_tokens, costs[i], 0, price_max) for i in range(num_agents)]

        eff, buyer_profit, seller_profit = run_session(buyers, sellers)
        eff_3.append(eff)
        gd_buyer_profits_3.append(buyer_profit)
        gd_seller_profits_3.append(seller_profit)

    avg_gd_buyers = np.mean(gd_buyer_profits_3)
    avg_gd_sellers = np.mean(gd_seller_profits_3)

    print(f"\nScenario 3 (GD vs GD):")
    print(f"  Avg Efficiency = {np.mean(eff_3):.2f}%")
    print(f"  GD Buyer Profit:  {avg_gd_buyers:.0f}")
    print(f"  GD Seller Profit: {avg_gd_sellers:.0f}")
    print(f"  (Balanced competition, both use optimal strategies)")

if __name__ == "__main__":
    benchmark()
