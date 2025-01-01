# config.py

CONFIG = {
    "experiment_name": "001_zic_only_gode_sunder",
    "experiment_dir": "experiments",

    "num_rounds": 4,
    "num_periods": 4,
    "num_steps": 50,
    "num_tokens": 4,

    # Price bounds for valuations/costs
    "buyer_valuation_min": 0.3,
    "buyer_valuation_max": 1.0,
    "seller_cost_min": 0.0,
    "seller_cost_max": 0.7,

    # All ZIC agents (4 buyers, 4 sellers) for replication
    "buyers": [
        {"type": "zic"},
        {"type": "zic"},
        {"type": "zic"},
        {"type": "zic"},
    ],
    "sellers": [
        {"type": "zic"},
        {"type": "zic"},
        {"type": "zic"},
        {"type": "gdseller"},
    ]
}
