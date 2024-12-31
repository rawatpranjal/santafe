# config.py

CONFIG = {
    "num_rounds": 100,
    "num_periods": 1,
    "num_steps": 200,
    "num_tokens": 20,

    # Price bounds for valuations/costs
    "buyer_valuation_min": 0.5,
    "buyer_valuation_max": 1.0,
    "seller_cost_min": 0.0,
    "seller_cost_max": 0.5,

    # Example: 3 buyers + 3 sellers
    "buyers": [
        {"type": "random"},            # calls (random, True)->RandomBuyer
        {"type": "kaplan"},            # calls (kaplan, True)->KaplanBuyer
        {"type": "ppobuyer",
         "init_args": {"lr": 1e-4}},   # calls (ppobuyer, True)->PPOBuyer with lr=1e-4
    ],
    "sellers": [
        {"type": "random"},            # calls (random, False)->RandomSeller
        {"type": "kaplan"},            # calls (kaplan, False)->KaplanSeller
        {"type": "random"}          # calls (pposeller, False)->PPOSeller
    ]
}
