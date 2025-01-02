# config.py

CONFIG = {
    "experiment_name": "003_zip_only",
    "experiment_dir": "experiments",

    "num_rounds": 25,
    "num_periods": 5,
    "num_steps": 50,
    "num_tokens": 5,

    # Price bounds for valuations/costs
    "buyer_valuation_min": 0.0,
    "buyer_valuation_max": 1.0,
    "seller_cost_min": 0.0,
    "seller_cost_max": 1.0,

    # All ZIC agents (4 buyers, 4 sellers) for replication
    "buyers": [
        {"type": "zip"},
        {"type": "zip"},
        {"type": "zip"},
        {"type": "zip"},
        {"type": "zip"},
    ],
    "sellers": [
        {"type": "zip"},
        {"type": "zip"},
        {"type": "zip"},
        {"type": "zip"},
        {"type": "zip"},
    ]
}
