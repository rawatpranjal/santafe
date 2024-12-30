# main.py
import pandas as pd
import random

from config import CONFIG
from traders import (
    ZeroIntelligenceTrader, TruthTellerBuyer, TruthTellerSeller,
    KaplanBuyer, KaplanSeller,
    CreeperBuyer, CreeperSeller
)
from auction import Auction

def create_traders(config):
    rng = random.Random()
    bmin, bmax = config["buyer_valuation_min"], config["buyer_valuation_max"]
    smin, smax = config["seller_cost_min"], config["seller_cost_max"]
    num_tokens = config["num_tokens"]

    buyers, sellers = [], []

    for i, spec in enumerate(config["buyers"]):
        vals = sorted([rng.uniform(bmin, bmax) for _ in range(num_tokens)], reverse=True)
        name = f"Buyer{i}"
        t = None
        if spec["type"] == "zero":
            t = ZeroIntelligenceTrader(name, True, vals)
        elif spec["type"] == "truth":
            t = TruthTellerBuyer(name, True, vals)
        elif spec["type"] == "kaplan":
            t = KaplanBuyer(name, True, vals, margin=config["kaplan_margin"])
        elif spec["type"] == "creeper":
            t = CreeperBuyer(name, True, vals, speed=config["creeper_speed"])
        else:
            raise ValueError(f"Unknown buyer type: {spec['type']}")
        buyers.append(t)

    for j, spec in enumerate(config["sellers"]):
        vals = sorted([rng.uniform(smin, smax) for _ in range(num_tokens)])
        name = f"Seller{j}"
        if spec["type"] == "zero":
            t = ZeroIntelligenceTrader(name, False, vals)
        elif spec["type"] == "truth":
            t = TruthTellerSeller(name, False, vals)
        elif spec["type"] == "kaplan":
            t = KaplanSeller(name, False, vals, margin=config["kaplan_margin"])
        elif spec["type"] == "creeper":
            t = CreeperSeller(name, False, vals, speed=config["creeper_speed"])
        else:
            raise ValueError(f"Unknown seller type: {spec['type']}")
        sellers.append(t)

    return buyers, sellers

def main():
    # random.seed(42)  # optional for reproducibility

    # 1) Create traders
    buyers, sellers = create_traders(CONFIG)

    # 2) Create Auction
    auction = Auction(CONFIG, buyers, sellers)

    # 3) Run
    auction.run_auction()

    # 4) Save logs
    df = pd.DataFrame(auction.logs)
    df.to_csv("log.csv", index=False)

    # 5) Print final profits
    print("\n=== Final Profits ===")
    for b in buyers:
        print(f"{b.name} (Buyer):  {b.profit:.4f}")
    for s in sellers:
        print(f"{s.name} (Seller): {s.profit:.4f}")

    # 6) Print sample logs
    print("\n=== Auction Log (sample) ===")
    print(df.head(10))

if __name__ == "__main__":
    main()
