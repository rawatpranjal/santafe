# config.py

CONFIG = {
    "num_rounds": 500,
    "num_periods": 1,
    "num_steps": 50,
    "num_tokens": 10,

    # Price bounds
    "min_price": 0.0,
    "max_price": 1.0,

    # Valuation bounds
    "buyer_valuation_min": 0.0,
    "buyer_valuation_max": 1.0,

    # Cost bounds
    "seller_cost_min": 0.0,
    "seller_cost_max": 1.0,

    # For Kaplan & Creeper
    "kaplan_margin": 0.1,
    "creeper_speed": 0.1,

    # We have 4 buyer types & 4 seller types
    "buyers": [
        {"type": "creeper"},
        {"type": "zero"},
        {"type": "kaplan"},
        {"type": "truth"}
    ],
    "sellers": [
        {"type": "creeper"},
        {"type": "zero"},
        {"type": "kaplan"},
        {"type": "truth"}
    ]
}
