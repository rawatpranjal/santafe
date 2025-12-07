#!/usr/bin/env python3
"""Generate event logs for P2 Santa Fe traders for curated log creation.

Runs short self-play and easy-play experiments with event logging enabled.

Usage:
    python scripts/generate_p2_event_logs.py          # Generate all logs
    python scripts/generate_p2_event_logs.py self     # Self-play only
    python scripts/generate_p2_event_logs.py easy     # Easy-play only
    python scripts/generate_p2_event_logs.py mixed    # Mixed only
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from engine.tournament import Tournament

LOGS_DIR = Path("logs/p2_curated")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# All Santa Fe traders (12 original + ZIP)
ALL_TRADERS = [
    ("Ringuette", "ring"),
    ("Perry", "perry"),
    ("Kaplan", "kap"),
    ("Skeleton", "skel"),
    ("BGAN", "bgan"),
    ("Jacobson", "jacobson"),
    ("Gamer", "gamer"),
    ("Staecker", "staecker"),
    ("Lin", "lin"),
    ("Breton", "breton"),
    ("Ledyard", "el"),
    ("ZIC", "zic"),
    ("ZIP", "zip"),
]


def run_self_play_with_logging(trader_class: str, trader_short: str) -> None:
    """Run a short self-play experiment with event logging."""
    exp_id = f"p2_self_{trader_short}_base"

    config = OmegaConf.create(
        {
            "experiment": {
                "name": exp_id,
                "num_rounds": 3,  # 3 rounds for curated logs
                "rng_seed_values": 123,
                "rng_seed_auction": 42,
            },
            "market": {
                "min_price": 1,
                "max_price": 2000,
                "num_tokens": 4,
                "num_periods": 1,  # Just period 1
                "num_steps": 75,
                "gametype": 6453,  # BASE environment
                "deadsteps": 0,
            },
            "agents": {
                "buyer_types": [trader_class] * 4,
                "seller_types": [trader_class] * 4,
            },
            "log_events": True,
            "log_dir": str(LOGS_DIR),
            "experiment_id": exp_id,
        }
    )

    print(f"  Running {trader_class} self-play with event logging...")

    try:
        tournament = Tournament(config)
        tournament.run()
        print(f"    Created: {LOGS_DIR / f'{exp_id}_events.jsonl'}")
    except Exception as e:
        print(f"    ERROR: {e}")


def run_easy_play_with_logging(trader_class: str, trader_short: str) -> None:
    """Run easy-play experiment: strategy buyers vs TruthTeller sellers."""
    exp_id = f"p2_easy_{trader_short}_base"

    config = OmegaConf.create(
        {
            "experiment": {
                "name": exp_id,
                "num_rounds": 3,  # 3 rounds for curated logs
                "rng_seed_values": 123,
                "rng_seed_auction": 42,
            },
            "market": {
                "min_price": 1,
                "max_price": 2000,
                "num_tokens": 4,
                "num_periods": 1,  # Just period 1
                "num_steps": 75,
                "gametype": 6453,  # BASE environment
                "deadsteps": 0,
            },
            "agents": {
                "buyer_types": [trader_class] * 4,
                "seller_types": ["TruthTeller"] * 4,
            },
            "log_events": True,
            "log_dir": str(LOGS_DIR),
            "experiment_id": exp_id,
        }
    )

    print(f"  Running {trader_class} easy-play (vs TruthTeller) with event logging...")

    try:
        tournament = Tournament(config)
        tournament.run()
        print(f"    Created: {LOGS_DIR / f'{exp_id}_events.jsonl'}")
    except Exception as e:
        print(f"    ERROR: {e}")


def run_mixed_with_logging() -> None:
    """Run mixed round-robin with all 12 Santa Fe traders."""
    exp_id = "p2_rr_mixed_base"

    # All 12 Santa Fe traders
    buyer_types = ["ZIC", "Skeleton", "Kaplan", "Ringuette", "Gamer", "Perry"]
    seller_types = ["Ledyard", "BGAN", "Staecker", "Jacobson", "Lin", "Breton"]

    config = OmegaConf.create(
        {
            "experiment": {
                "name": exp_id,
                "num_rounds": 3,
                "rng_seed_values": 123,
                "rng_seed_auction": 42,
            },
            "market": {
                "min_price": 1,
                "max_price": 2000,
                "num_tokens": 4,
                "num_periods": 1,
                "num_steps": 75,
                "gametype": 6453,
                "deadsteps": 0,
            },
            "agents": {
                "buyer_types": buyer_types,
                "seller_types": seller_types,
            },
            "log_events": True,
            "log_dir": str(LOGS_DIR),
            "experiment_id": exp_id,
        }
    )

    print("  Running mixed round-robin with event logging...")

    try:
        tournament = Tournament(config)
        tournament.run()
        print(f"    Created: {LOGS_DIR / f'{exp_id}_events.jsonl'}")
    except Exception as e:
        print(f"    ERROR: {e}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    print("Generating P2 event logs for curated log creation...")
    print(f"Output directory: {LOGS_DIR}")
    print(f"Mode: {mode}")
    print()

    if mode in ("all", "self"):
        # Generate self-play logs for each trader
        print("Self-play logs:")
        for trader_class, trader_short in ALL_TRADERS:
            run_self_play_with_logging(trader_class, trader_short)

    if mode in ("all", "easy"):
        # Generate easy-play logs for each trader
        print("\nEasy-play logs (vs TruthTeller):")
        for trader_class, trader_short in ALL_TRADERS:
            run_easy_play_with_logging(trader_class, trader_short)

    if mode in ("all", "mixed"):
        # Generate mixed round-robin log
        print("\nMixed round-robin log:")
        run_mixed_with_logging()

    print("\nDone!")
    print(f"\nGenerated logs in: {LOGS_DIR}")
    print("Run 'python scripts/create_p2_curated_logs.py' to create curated markdown files.")


if __name__ == "__main__":
    main()
