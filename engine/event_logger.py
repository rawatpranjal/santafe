"""
Event Logger for Market Heartbeat visualization.

Logs bid/ask events at the timestep level for post-hoc analysis
of trader timing behavior and market dynamics.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO


@dataclass
class BidAskEvent:
    """A single bid or ask event."""

    round: int
    period: int
    step: int
    agent_id: int
    agent_type: str
    is_buyer: bool
    price: int
    status: str  # "winner", "beaten", "pass"


@dataclass
class TradeEvent:
    """A trade execution event with profit information."""

    round: int
    period: int
    step: int
    buyer_id: int
    seller_id: int
    price: int
    buyer_value: int  # Token value for buyer
    seller_cost: int  # Token cost for seller
    buyer_profit: int  # value - price
    seller_profit: int  # price - cost
    buyer_type: str  # Agent class name
    seller_type: str  # Agent class name


@dataclass
class PeriodStartEvent:
    """Period start event with equilibrium info."""

    round: int
    period: int
    equilibrium_price: int
    max_surplus: int


class EventLogger:
    """
    Logs market events to JSONL format for visualization.

    Usage:
        logger = EventLogger(Path("logs/exp_1.11_events.jsonl"))
        logger.log_bid(round=1, period=1, step=5, agent_id=1,
                       agent_type="ZIC", price=450, status="winner")
        logger.close()
    """

    def __init__(self, output_path: Path):
        """
        Initialize the event logger.

        Args:
            output_path: Path to write JSONL file
        """
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO | None = None
        self._open()

    def _open(self) -> None:
        """Open the output file for writing."""
        self._file = open(self.output_path, "w")

    def log_bid(
        self,
        round: int,
        period: int,
        step: int,
        agent_id: int,
        agent_type: str,
        price: int,
        status: str,
    ) -> None:
        """Log a bid event (buyer submitting a price)."""
        event = BidAskEvent(
            round=round,
            period=period,
            step=step,
            agent_id=agent_id,
            agent_type=agent_type,
            is_buyer=True,
            price=price,
            status=status,
        )
        self._write_event(event)

    def log_ask(
        self,
        round: int,
        period: int,
        step: int,
        agent_id: int,
        agent_type: str,
        price: int,
        status: str,
    ) -> None:
        """Log an ask event (seller submitting a price)."""
        event = BidAskEvent(
            round=round,
            period=period,
            step=step,
            agent_id=agent_id,
            agent_type=agent_type,
            is_buyer=False,
            price=price,
            status=status,
        )
        self._write_event(event)

    def log_trade(
        self,
        round: int,
        period: int,
        step: int,
        buyer_id: int,
        seller_id: int,
        price: int,
        buyer_value: int,
        seller_cost: int,
        buyer_profit: int,
        seller_profit: int,
        buyer_type: str,
        seller_type: str,
    ) -> None:
        """Log a trade execution event with profit information."""
        event = TradeEvent(
            round=round,
            period=period,
            step=step,
            buyer_id=buyer_id,
            seller_id=seller_id,
            price=price,
            buyer_value=buyer_value,
            seller_cost=seller_cost,
            buyer_profit=buyer_profit,
            seller_profit=seller_profit,
            buyer_type=buyer_type,
            seller_type=seller_type,
        )
        self._write_event(event)

    def log_period_start(
        self,
        round: int,
        period: int,
        equilibrium_price: int,
        max_surplus: int,
    ) -> None:
        """Log period start with equilibrium info for visualization."""
        event = PeriodStartEvent(
            round=round,
            period=period,
            equilibrium_price=equilibrium_price,
            max_surplus=max_surplus,
        )
        self._write_event(event)

    def _write_event(self, event: BidAskEvent | TradeEvent | PeriodStartEvent) -> None:
        """Write an event to the JSONL file."""
        if self._file is None:
            return

        data = asdict(event)
        if isinstance(event, PeriodStartEvent):
            data["event_type"] = "period_start"
        elif isinstance(event, BidAskEvent):
            data["event_type"] = "bid_ask"
        else:
            data["event_type"] = "trade"
        self._file.write(json.dumps(data) + "\n")

    def flush(self) -> None:
        """Flush the output buffer."""
        if self._file:
            self._file.flush()

    def close(self) -> None:
        """Close the output file."""
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def load_events(log_path: Path) -> list[dict[str, object]]:
    """
    Load events from a JSONL file.

    Args:
        log_path: Path to the JSONL file

    Returns:
        List of event dictionaries
    """
    events = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events
