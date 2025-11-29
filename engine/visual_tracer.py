"""
Visual Tracer for Santa Fe Market.

Extracts detailed timestep data from the OrderBook for visualization.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from engine.orderbook import OrderBook


@dataclass
class AgentAction:
    """Record of an agent's action at a specific timestep."""
    agent_id: int
    agent_type: str
    is_buyer: bool
    action: str  # "bid", "ask", "accept", "pass"
    price: int
    result: str  # "winner", "beaten", "tied_lost", "traded", "-"


@dataclass
class TimestepRecord:
    """Complete record of market state at a timestep."""
    time: int
    high_bid: int
    low_ask: int
    spread: int
    trade_occurred: bool
    trade_price: int
    agent_actions: List[AgentAction]


def extract_market_timeline(
    orderbook: OrderBook,
    buyer_types: Dict[int, str],
    seller_types: Dict[int, str],
) -> List[TimestepRecord]:
    """
    Extract complete market timeline from orderbook.

    Args:
        orderbook: The completed orderbook after a period
        buyer_types: Mapping of buyer_id -> agent type name
        seller_types: Mapping of seller_id -> agent type name

    Returns:
        List of TimestepRecord, one per timestep
    """
    records = []

    for t in range(1, orderbook.num_times + 1):
        # Market state
        high_bid = int(orderbook.high_bid[t])
        low_ask = int(orderbook.low_ask[t])
        spread = (low_ask - high_bid) if (high_bid > 0 and low_ask > 0) else 0
        trade_price = int(orderbook.trade_price[t])
        trade_occurred = trade_price > 0

        # Collect agent actions
        actions: List[AgentAction] = []

        # Buyer actions
        for buyer_id in range(1, orderbook.num_buyers + 1):
            bid = int(orderbook.bids[buyer_id, t])
            prev_bid = int(orderbook.bids[buyer_id, t-1]) if t > 1 else 0

            # Determine action and result
            if bid > 0 and bid != prev_bid:
                action = "bid"
                if int(orderbook.high_bidder[t]) == buyer_id:
                    result = "winner"
                elif orderbook.bid_status[buyer_id] == 4:
                    result = "tied_lost"
                else:
                    result = "beaten"
            else:
                action = "pass"
                result = "-"

            # Check if this buyer accepted in buy/sell phase
            if trade_occurred and orderbook.buyer_accepted[t]:
                if int(orderbook.high_bidder[t]) == buyer_id:
                    action = "accept"
                    result = "TRADE"

            actions.append(AgentAction(
                agent_id=buyer_id,
                agent_type=buyer_types.get(buyer_id, "Unknown"),
                is_buyer=True,
                action=action,
                price=bid if action in ("bid", "accept") else 0,
                result=result,
            ))

        # Seller actions
        for seller_id in range(1, orderbook.num_sellers + 1):
            ask = int(orderbook.asks[seller_id, t])
            prev_ask = int(orderbook.asks[seller_id, t-1]) if t > 1 else 0

            # Determine action and result
            if ask > 0 and ask != prev_ask:
                action = "ask"
                if int(orderbook.low_asker[t]) == seller_id:
                    result = "winner"
                elif orderbook.ask_status[seller_id] == 4:
                    result = "tied_lost"
                else:
                    result = "beaten"
            else:
                action = "pass"
                result = "-"

            # Check if this seller accepted in buy/sell phase
            if trade_occurred and orderbook.seller_accepted[t]:
                if int(orderbook.low_asker[t]) == seller_id:
                    action = "accept"
                    result = "TRADE"

            actions.append(AgentAction(
                agent_id=seller_id,
                agent_type=seller_types.get(seller_id, "Unknown"),
                is_buyer=False,
                action=action,
                price=ask if action in ("ask", "accept") else 0,
                result=result,
            ))

        records.append(TimestepRecord(
            time=t,
            high_bid=high_bid,
            low_ask=low_ask,
            spread=spread,
            trade_occurred=trade_occurred,
            trade_price=trade_price,
            agent_actions=actions,
        ))

    return records


def extract_agent_summary(
    agents: List[Any],
    buyer_valuations: Dict[int, List[int]],
    seller_costs: Dict[int, List[int]],
) -> List[Dict[str, Any]]:
    """
    Extract summary statistics for each agent.

    Args:
        agents: List of agent objects after the period
        buyer_valuations: Mapping of buyer_id -> valuations list
        seller_costs: Mapping of seller_id -> costs list

    Returns:
        List of agent summary dictionaries
    """
    summaries = []

    for agent in agents:
        if agent.is_buyer:
            valuations = buyer_valuations.get(agent.player_id, [])
            avg_valuation = sum(valuations) / len(valuations) if valuations else 0
        else:
            valuations = seller_costs.get(agent.player_id, [])
            avg_valuation = sum(valuations) / len(valuations) if valuations else 0

        summaries.append({
            "agent_id": agent.player_id,
            "agent_type": agent.__class__.__name__,
            "is_buyer": agent.is_buyer,
            "num_trades": agent.num_trades,
            "period_profit": agent.period_profit,
            "avg_valuation": avg_valuation,
        })

    return summaries
