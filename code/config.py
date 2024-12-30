# config.py
CONFIG = {
    "num_rounds": 100,
    "num_periods": 1,
    "num_steps": 200,
    "num_tokens": 10,

    "min_price": 0.0,
    "max_price": 1.0,

    "buyer_valuation_min": 0.0,
    "buyer_valuation_max": 1.0,

    "seller_cost_min": 0.0,
    "seller_cost_max": 1.0,

    "sniper_margin": 0.05,

    # For example, mix of kaplan vs random
    "buyers": [
        {"type": "random"},
        {"type": "random"},
        {"type": "random"},
        {"type": "random"},
        {"type": "random"},
        {"type": "random"},
        {"type": "random"},
        {"type": "zipbuyer"},
        {"type": "gdbuyer"},
        {"type": "kaplan"}
    ],
    "sellers": [
        {"type": "random"},
        {"type": "random"},
        {"type": "random"},
        {"type": "random"},
        {"type": "random"},
        {"type": "random"},
        {"type": "random"},
        {"type": "zipseller"},
        {"type": "gdseller"},
        {"type": "kaplan"}
    ]
}
