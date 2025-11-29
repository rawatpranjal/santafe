#!/usr/bin/env python3
"""
Scenario runner for trace replay verification.

This script loads a scenario JSON file, executes it through the Python engine,
and outputs a detailed trace for verification.
"""

import json
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.market import Market
from traders.base import Agent


class ScenarioAgent(Agent):
    """
    Test agent that follows a pre-programmed scenario script.

    Actions are provided in the scenario JSON file.
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        scenario_actions: dict[int, dict[str, Any]],
    ) -> None:
        """
        Initialize a scenario-driven agent.

        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations
            scenario_actions: Dict mapping time -> actions for this agent
        """
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.scenario_actions = scenario_actions
        self.current_time = 0

    def bid_ask(self, time: int, nobidask: int) -> None:
        """Record current time and prepare to respond."""
        self.has_responded = False
        self.current_time = time

    def bid_ask_response(self) -> int:
        """Return pre-programmed bid/ask from scenario."""
        self.has_responded = True

        # Get actions for current timestep
        if self.current_time not in self.scenario_actions:
            # No action this timestep - return invalid negative to signal "no bid/ask"
            # This allows existing orders to carry over
            return -999  # Invalid negative (not -1=quit, -2=fail, -3=thinking)

        actions = self.scenario_actions[self.current_time]

        # Get bid (for buyers) or ask (for sellers)
        if self.is_buyer:
            bid: int = actions.get("bid", -999)
            return bid
        else:
            ask: int = actions.get("ask", -999)
            return ask

    def buy_sell(
        self,
        time: int,
        nobuysell: int,
        high_bid: int,
        low_ask: int,
        high_bidder: int,
        low_asker: int,
    ) -> None:
        """Record state and prepare to respond."""
        self.has_responded = False
        self.current_time = time

    def buy_sell_response(self) -> bool:
        """Return pre-programmed accept/reject from scenario."""
        self.has_responded = True

        # Get actions for current timestep
        if self.current_time not in self.scenario_actions:
            return False

        actions = self.scenario_actions[self.current_time]

        # Get accept decision
        accept: bool = actions.get("accept", False)
        return accept


def load_scenario(scenario_path: Path) -> dict[str, Any]:
    """Load scenario from JSON file."""
    with open(scenario_path, "r") as f:
        data: dict[str, Any] = json.load(f)
        return data


def create_agents_from_scenario(scenario: dict[str, Any]) -> tuple[list[Agent], list[Agent]]:
    """
    Create buyer and seller agents from scenario specification.

    Returns:
        Tuple of (buyers, sellers) agent lists
    """
    buyers: list[Agent] = []
    sellers: list[Agent] = []

    for agent_spec in scenario["agents"]:
        player_id = agent_spec["player_id"]
        is_buyer = agent_spec["is_buyer"]
        num_tokens = agent_spec["num_tokens"]
        valuations = agent_spec["valuations"]

        # Build scenario actions for this agent
        scenario_actions: dict[int, dict[str, Any]] = {}

        for timestep_spec in scenario["timesteps"]:
            time = timestep_spec["time"]
            actions = timestep_spec["actions"]

            agent_action: dict[str, Any] = {}

            # Extract bid/ask
            if is_buyer:
                bids = actions.get("bids", {})
                if str(player_id) in bids:
                    agent_action["bid"] = bids[str(player_id)]
            else:
                asks = actions.get("asks", {})
                if str(player_id) in asks:
                    agent_action["ask"] = asks[str(player_id)]

            # Extract accept decision
            if is_buyer:
                buyer_accepts = actions.get("buyer_accepts", {})
                if str(player_id) in buyer_accepts:
                    agent_action["accept"] = buyer_accepts[str(player_id)]
            else:
                seller_accepts = actions.get("seller_accepts", {})
                if str(player_id) in seller_accepts:
                    agent_action["accept"] = seller_accepts[str(player_id)]

            if agent_action:  # Only add if there are actions
                scenario_actions[time] = agent_action

        # Create agent
        agent = ScenarioAgent(
            player_id=player_id,
            is_buyer=is_buyer,
            num_tokens=num_tokens,
            valuations=valuations,
            scenario_actions=scenario_actions,
        )

        if is_buyer:
            buyers.append(agent)
        else:
            sellers.append(agent)

    return buyers, sellers


def run_scenario(scenario_path: Path, verbose: bool = True) -> dict[str, Any]:
    """
    Run a scenario and return the execution trace.

    Args:
        scenario_path: Path to scenario JSON file
        verbose: If True, print detailed trace to stdout

    Returns:
        Dict containing execution trace with actual outcomes
    """
    # Load scenario
    scenario = load_scenario(scenario_path)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Running Scenario: {scenario['description']}")
        print(f"{'='*70}")
        print(f"Purpose: {scenario['purpose']}\n")

    # Create agents
    buyers, sellers = create_agents_from_scenario(scenario)

    # Create market
    setup = scenario["setup"]
    market = Market(
        num_buyers=setup["num_buyers"],
        num_sellers=setup["num_sellers"],
        num_times=setup["num_times"],
        price_min=setup["min_price"],
        price_max=setup["max_price"],
        buyers=buyers,
        sellers=sellers,
        seed=setup["seed"],
    )

    # Run timesteps
    trace: dict[str, Any] = {
        "scenario": scenario["description"],
        "timesteps": [],
    }

    for t in range(1, setup["num_times"] + 1):
        if verbose:
            print(f"\n--- Time {t} ---")

        # Run timestep
        success = market.run_time_step()

        if not success:
            if verbose:
                print("Market failed!")
            break

        # Capture outcomes
        ob = market.orderbook
        timestep_trace = {
            "time": t,
            "high_bidder": int(ob.high_bidder[t]),
            "high_bid": int(ob.high_bid[t]),
            "low_asker": int(ob.low_asker[t]),
            "low_ask": int(ob.low_ask[t]),
            "buyer_accepted": bool(ob.buyer_accepted[t]),
            "seller_accepted": bool(ob.seller_accepted[t]),
            "trade_price": int(ob.trade_price[t]),
        }

        if verbose:
            print(f"  High Bid: {timestep_trace['high_bid']} (Bidder {timestep_trace['high_bidder']})")
            print(f"  Low Ask: {timestep_trace['low_ask']} (Asker {timestep_trace['low_asker']})")
            print(f"  Buyer Accepted: {timestep_trace['buyer_accepted']}")
            print(f"  Seller Accepted: {timestep_trace['seller_accepted']}")
            print(f"  Trade Price: {timestep_trace['trade_price']}")

        trace["timesteps"].append(timestep_trace)

    if verbose:
        print(f"\n{'='*70}\n")

    return trace


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python run_scenario.py <scenario.json>")
        sys.exit(1)

    scenario_path = Path(sys.argv[1])

    if not scenario_path.exists():
        print(f"Error: Scenario file not found: {scenario_path}")
        sys.exit(1)

    trace = run_scenario(scenario_path, verbose=True)

    # Optionally save trace to JSON
    if len(sys.argv) > 2 and sys.argv[2] == "--save":
        output_path = scenario_path.with_suffix(".trace.json")
        with open(output_path, "w") as f:
            json.dump(trace, f, indent=2)
        print(f"Trace saved to: {output_path}")


if __name__ == "__main__":
    main()
