CONFIG = {
    "num_rounds": 1000,  # for long training
    "num_periods": 1,
    "num_steps": 200,
    "num_tokens": 20,

    "buyer_valuation_min": 0.5,
    "buyer_valuation_max": 1.0,
    "seller_cost_min": 0.0,
    "seller_cost_max": 0.5,

    "buyers": [
        {"type": "random"},
        {"type": "zipbuyer"},
        {"type": "gdbuyer"},
        {"type": "kaplan"},
        {"type": "ppobuyer"}   # <-- Add PPO buyer
    ],
    "sellers": [
        {"type": "random"},
        {"type": "zipseller"},
        {"type": "gdseller"},
        {"type": "kaplan"},
        {"type": "pposeller"}  # <-- Add PPO seller
    ]
}
