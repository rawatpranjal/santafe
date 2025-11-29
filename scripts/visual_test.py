#!/usr/bin/env python3
"""
Visual Testing Tool for Santa Fe Traders.

Run small-scale experiments and see rich terminal output showing
trader behavior, price dynamics, and efficiency metrics.

Usage:
    python scripts/visual_test.py --scenario zic_selfplay
    python scripts/visual_test.py --scenario zic_vs_zip --verbose
    python scripts/visual_test.py --all
"""

import argparse
import sys
from typing import Dict, List, Any, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich import box

# Add project root to path
sys.path.insert(0, ".")

from engine.market import Market
from engine.token_generator import TokenGenerator
from engine.agent_factory import create_agent
from engine.visual_tracer import extract_market_timeline, extract_agent_summary, TimestepRecord
from engine.efficiency import (
    extract_trades_from_orderbook,
    calculate_actual_surplus,
    calculate_max_surplus,
    calculate_allocative_efficiency,
)

console = Console()


# =============================================================================
# Scenario Definitions
# =============================================================================

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "zic_selfplay": {
        "name": "ZIC Self-Play",
        "description": "Zero-Intelligence Constrained baseline (~98% efficiency expected)",
        "buyers": ["ZIC", "ZIC", "ZIC", "ZIC"],
        "sellers": ["ZIC", "ZIC", "ZIC", "ZIC"],
    },
    "zip_selfplay": {
        "name": "ZIP Self-Play",
        "description": "Zero-Intelligence Plus with adaptive margins",
        "buyers": ["ZIP", "ZIP", "ZIP", "ZIP"],
        "sellers": ["ZIP", "ZIP", "ZIP", "ZIP"],
    },
    "kaplan_selfplay": {
        "name": "Kaplan Self-Play",
        "description": "Sniper strategy - expect market crash (<60% efficiency)",
        "buyers": ["Kaplan", "Kaplan", "Kaplan", "Kaplan"],
        "sellers": ["Kaplan", "Kaplan", "Kaplan", "Kaplan"],
    },
    "zic_vs_zip": {
        "name": "ZIC vs ZIP",
        "description": "ZIP should profit at ZIC's expense",
        "buyers": ["ZIC", "ZIC", "ZIC", "ZIC"],
        "sellers": ["ZIP", "ZIP", "ZIP", "ZIP"],
    },
    "zic_vs_kaplan": {
        "name": "ZIC vs Kaplan",
        "description": "Kaplan dominates by sniping after ZIC narrows spread",
        "buyers": ["ZIC", "ZIC", "ZIC", "ZIC"],
        "sellers": ["Kaplan", "Kaplan", "Kaplan", "Kaplan"],
    },
    "kaplan_vs_skeleton": {
        "name": "Kaplan vs Skeleton",
        "description": "Sniper strategy vs alpha-weighted bidding",
        "buyers": ["Kaplan", "Kaplan", "Kaplan", "Kaplan"],
        "sellers": ["Skeleton", "Skeleton", "Skeleton", "Skeleton"],
    },
    "gd_vs_zic": {
        "name": "GD vs ZIC",
        "description": "Gjerstad-Dickhaut belief-based trading vs ZIC",
        "buyers": ["GD", "GD", "GD", "GD"],
        "sellers": ["ZIC", "ZIC", "ZIC", "ZIC"],
    },
    "skeleton_selfplay": {
        "name": "Skeleton Self-Play",
        "description": "SRobotExample baseline - alpha-weighted strategy",
        "buyers": ["Skeleton", "Skeleton", "Skeleton", "Skeleton"],
        "sellers": ["Skeleton", "Skeleton", "Skeleton", "Skeleton"],
    },
    "zic_vs_skeleton": {
        "name": "ZIC vs Skeleton",
        "description": "Skeleton's strategic bidding vs ZIC randomness",
        "buyers": ["ZIC", "ZIC", "ZIC", "ZIC"],
        "sellers": ["Skeleton", "Skeleton", "Skeleton", "Skeleton"],
    },
    "mixed": {
        "name": "Mixed Tournament",
        "description": "All major strategies competing (5 types)",
        "buyers": ["ZIC", "ZIP", "Kaplan", "GD", "Skeleton"],
        "sellers": ["ZIC", "ZIP", "Kaplan", "GD", "Skeleton"],
    },
}


# =============================================================================
# Market Execution
# =============================================================================

def run_scenario(
    scenario_name: str,
    num_periods: int = 5,
    num_steps: int = 100,  # Increased from 25 - ZIC needs ~100 steps for ~98% efficiency
    seed: int = 42,
) -> Tuple[List[TimestepRecord], List[Dict[str, Any]], Dict[str, float], Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Run a scenario and return visualization data.

    Returns:
        timeline: List of timestep records
        agent_summaries: List of agent summary dicts
        metrics: Efficiency metrics dict
        buyer_valuations: Dict of buyer_id -> valuations
        seller_costs: Dict of seller_id -> costs
    """
    scenario = SCENARIOS[scenario_name]
    buyer_types = scenario["buyers"]
    seller_types = scenario["sellers"]

    num_buyers = len(buyer_types)
    num_sellers = len(seller_types)
    num_tokens = 4
    price_min = 0
    price_max = 1000
    gametype = 6453

    # Token generation
    token_gen = TokenGenerator(gametype, num_tokens, seed)
    token_gen.new_round()

    # Create agents
    agents = []
    buyer_valuations: Dict[int, List[int]] = {}
    seller_costs: Dict[int, List[int]] = {}

    # Buyers
    for i, agent_type in enumerate(buyer_types):
        player_id = i + 1
        tokens = token_gen.generate_tokens(is_buyer=True)
        buyer_valuations[player_id] = tokens
        agents.append(create_agent(
            agent_type,
            player_id,
            is_buyer=True,
            num_tokens=num_tokens,
            valuations=tokens,
            seed=seed + player_id,
            num_times=num_steps,
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            price_min=price_min,
            price_max=price_max,
        ))

    # Sellers
    for i, agent_type in enumerate(seller_types):
        player_id = num_buyers + i + 1
        tokens = token_gen.generate_tokens(is_buyer=False)
        seller_costs[i + 1] = tokens  # Use local seller ID (1-indexed)
        agents.append(create_agent(
            agent_type,
            player_id,
            is_buyer=False,
            num_tokens=num_tokens,
            valuations=tokens,
            seed=seed + player_id,
            num_times=num_steps,
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            price_min=price_min,
            price_max=price_max,
        ))

    # Initialize agents
    for agent in agents:
        agent.start_round(agent.valuations)
        agent.start_period(1)

    # Create and run market
    buyers = [a for a in agents if a.is_buyer]
    sellers = [a for a in agents if not a.is_buyer]

    market = Market(
        num_buyers=num_buyers,
        num_sellers=num_sellers,
        num_times=num_steps,
        price_min=price_min,
        price_max=price_max,
        buyers=buyers,
        sellers=sellers,
        seed=seed,
    )

    # Run market
    while market.current_time < market.num_times:
        market.run_time_step()

    # End period
    for agent in agents:
        agent.end_period()

    # Extract visualization data
    buyer_type_map = {i+1: buyer_types[i] for i in range(num_buyers)}
    seller_type_map = {i+1: seller_types[i] for i in range(num_sellers)}

    timeline = extract_market_timeline(market.orderbook, buyer_type_map, seller_type_map)
    agent_summaries = extract_agent_summary(agents, buyer_valuations, seller_costs)

    # Calculate efficiency metrics
    trades = extract_trades_from_orderbook(market.orderbook, market.num_times)
    buyer_vals_list = [buyer_valuations[i+1] for i in range(num_buyers)]
    seller_costs_list = [seller_costs[i+1] for i in range(num_sellers)]

    actual_surplus = calculate_actual_surplus(trades, buyer_valuations, seller_costs)
    max_surplus = calculate_max_surplus(buyer_vals_list, seller_costs_list)
    efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)

    metrics = {
        "efficiency": efficiency,
        "actual_surplus": actual_surplus,
        "max_surplus": max_surplus,
        "num_trades": len(trades),
        "max_trades": num_tokens * min(num_buyers, num_sellers),
    }

    return timeline, agent_summaries, metrics, buyer_valuations, seller_costs


# =============================================================================
# Rendering Functions
# =============================================================================

def render_header(scenario_name: str) -> None:
    """Render scenario header."""
    scenario = SCENARIOS[scenario_name]
    console.print()
    console.print(Panel(
        f"[bold cyan]{scenario['name']}[/bold cyan]\n"
        f"[dim]{scenario['description']}[/dim]",
        title="Visual Test",
        border_style="cyan",
    ))


def render_market_timeline(timeline: List[TimestepRecord], max_rows: int = 20) -> None:
    """Render market timeline table."""
    table = Table(
        title="Market Timeline",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Step", style="dim", width=6)
    table.add_column("Bid", justify="right", width=8)
    table.add_column("Ask", justify="right", width=8)
    table.add_column("Spread", justify="right", width=8)
    table.add_column("Trade", justify="center", width=8)
    table.add_column("Price", justify="right", width=10)

    # Show subset of timesteps if too many
    display_records = timeline[:max_rows]
    skipped = len(timeline) - max_rows if len(timeline) > max_rows else 0

    for record in display_records:
        bid_str = str(record.high_bid) if record.high_bid > 0 else "-"
        ask_str = str(record.low_ask) if record.low_ask > 0 else "-"
        spread_str = str(record.spread) if record.spread > 0 else "-"

        if record.trade_occurred:
            trade_str = "[green bold]YES[/green bold]"
            price_str = f"[green]{record.trade_price}[/green]"
        else:
            trade_str = "-"
            price_str = "-"

        table.add_row(
            str(record.time),
            bid_str,
            ask_str,
            spread_str,
            trade_str,
            price_str,
        )

    if skipped > 0:
        table.add_row("...", "...", "...", "...", "...", f"[dim]({skipped} more)[/dim]")

    console.print(table)


def render_agent_trace(timeline: List[TimestepRecord], verbose: bool = False) -> None:
    """Render agent decision trace."""
    table = Table(
        title="Agent Actions (showing trades and significant bids)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold yellow",
    )

    table.add_column("Step", style="dim", width=6)
    table.add_column("Agent", width=8)
    table.add_column("Type", width=10)
    table.add_column("Role", width=8)
    table.add_column("Action", width=10)
    table.add_column("Price", justify="right", width=8)
    table.add_column("Result", width=12)

    row_count = 0
    max_rows = 30 if verbose else 15

    for record in timeline:
        for action in record.agent_actions:
            # Filter to interesting actions (trades, bids, asks, winners)
            if action.action == "pass" and action.result == "-":
                continue
            if not verbose and action.result not in ("winner", "TRADE", "beaten"):
                continue

            role = "Buyer" if action.is_buyer else "Seller"
            agent_label = f"{'B' if action.is_buyer else 'S'}{action.agent_id}"

            # Style based on result
            if action.result == "TRADE":
                result_style = "[green bold]TRADE[/green bold]"
            elif action.result == "winner":
                result_style = "[cyan]winner[/cyan]"
            elif action.result == "beaten":
                result_style = "[red]beaten[/red]"
            else:
                result_style = action.result

            price_str = str(action.price) if action.price > 0 else "-"

            table.add_row(
                str(record.time),
                agent_label,
                action.agent_type,
                role,
                action.action,
                price_str,
                result_style,
            )

            row_count += 1
            if row_count >= max_rows:
                break

        if row_count >= max_rows:
            break

    console.print(table)


def render_efficiency(metrics: Dict[str, float]) -> None:
    """Render efficiency metrics panel."""
    efficiency = metrics["efficiency"]
    actual_surplus = int(metrics["actual_surplus"])
    max_surplus = int(metrics["max_surplus"])
    num_trades = int(metrics["num_trades"])
    max_trades = int(metrics["max_trades"])

    # Create progress bar visualization
    bar_width = 20
    filled = int((efficiency / 100) * bar_width)
    bar = "[green]" + "█" * filled + "[/green]" + "[dim]░[/dim]" * (bar_width - filled)

    # Determine efficiency color
    if efficiency >= 90:
        eff_color = "green"
    elif efficiency >= 70:
        eff_color = "yellow"
    else:
        eff_color = "red"

    content = f"""
[bold]Allocative Efficiency:[/bold]  [{eff_color}]{efficiency:.1f}%[/{eff_color}]  {bar}
[bold]Trades Executed:[/bold]        {num_trades}/{max_trades}  ({100*num_trades/max_trades:.0f}%)
[bold]Actual Surplus:[/bold]         {actual_surplus:,}
[bold]Theoretical Max:[/bold]        {max_surplus:,}
[bold]Surplus Lost:[/bold]           {max_surplus - actual_surplus:,}  ({100*(max_surplus-actual_surplus)/max_surplus:.1f}%)
"""

    console.print(Panel(
        content.strip(),
        title="Efficiency Report",
        border_style="green" if efficiency >= 90 else "yellow" if efficiency >= 70 else "red",
    ))


def render_comparison(agent_summaries: List[Dict[str, Any]]) -> None:
    """Render strategy comparison table."""
    # Group by strategy
    strategy_stats: Dict[str, Dict[str, Any]] = {}

    for summary in agent_summaries:
        strategy = summary["agent_type"]
        if strategy not in strategy_stats:
            strategy_stats[strategy] = {
                "count": 0,
                "total_profit": 0,
                "total_trades": 0,
                "is_buyer": summary["is_buyer"],
            }

        strategy_stats[strategy]["count"] += 1
        strategy_stats[strategy]["total_profit"] += summary["period_profit"]
        strategy_stats[strategy]["total_trades"] += summary["num_trades"]

    table = Table(
        title="Strategy Comparison",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold blue",
    )

    table.add_column("Strategy", width=12)
    table.add_column("Agents", justify="center", width=8)
    table.add_column("Role", width=8)
    table.add_column("Avg Profit", justify="right", width=12)
    table.add_column("Total Trades", justify="right", width=12)

    # Find winner for highlighting
    max_avg_profit = max(
        s["total_profit"] / s["count"] for s in strategy_stats.values()
    ) if strategy_stats else 0

    for strategy, stats in sorted(strategy_stats.items()):
        avg_profit = stats["total_profit"] / stats["count"]
        role = "Buyer" if stats["is_buyer"] else "Seller"

        profit_style = ""
        if avg_profit == max_avg_profit and len(strategy_stats) > 1:
            profit_style = "[green bold]"
            profit_end = "[/green bold] ★"
        else:
            profit_end = ""

        table.add_row(
            strategy,
            str(stats["count"]),
            role,
            f"{profit_style}{avg_profit:.0f}{profit_end}",
            str(stats["total_trades"]),
        )

    console.print(table)

    # Winner announcement
    if len(strategy_stats) > 1:
        winner = max(strategy_stats.items(), key=lambda x: x[1]["total_profit"] / x[1]["count"])
        console.print(f"\n[bold green]Winner: {winner[0]}[/bold green]")


def render_valuations(
    buyer_valuations: Dict[int, List[int]],
    seller_costs: Dict[int, List[int]],
    buyer_types: List[str],
    seller_types: List[str],
) -> None:
    """Render token valuations table."""
    table = Table(
        title="Token Valuations",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold",
    )

    table.add_column("Agent", width=8)
    table.add_column("Type", width=10)
    table.add_column("Role", width=8)
    table.add_column("Token 1", justify="right", width=10)
    table.add_column("Token 2", justify="right", width=10)
    table.add_column("Token 3", justify="right", width=10)
    table.add_column("Token 4", justify="right", width=10)

    for buyer_id, vals in buyer_valuations.items():
        agent_type = buyer_types[buyer_id - 1] if buyer_id <= len(buyer_types) else "?"
        table.add_row(
            f"B{buyer_id}",
            agent_type,
            "Buyer",
            *[str(v) for v in vals],
        )

    for seller_id, costs in seller_costs.items():
        agent_type = seller_types[seller_id - 1] if seller_id <= len(seller_types) else "?"
        table.add_row(
            f"S{seller_id}",
            agent_type,
            "Seller",
            *[str(c) for c in costs],
        )

    console.print(table)


# =============================================================================
# Main Entry Point
# =============================================================================

def run_visual_test(
    scenario_name: str,
    verbose: bool = False,
    show_valuations: bool = False,
    seed: int = 42,
) -> None:
    """Run a visual test for a scenario."""
    if scenario_name not in SCENARIOS:
        console.print(f"[red]Unknown scenario: {scenario_name}[/red]")
        console.print(f"Available: {', '.join(SCENARIOS.keys())}")
        return

    render_header(scenario_name)

    with console.status("[bold green]Running market simulation..."):
        timeline, agent_summaries, metrics, buyer_vals, seller_costs = run_scenario(
            scenario_name, seed=seed
        )

    scenario = SCENARIOS[scenario_name]

    if show_valuations:
        console.print()
        render_valuations(buyer_vals, seller_costs, scenario["buyers"], scenario["sellers"])

    console.print()
    render_market_timeline(timeline)

    console.print()
    render_agent_trace(timeline, verbose=verbose)

    console.print()
    render_efficiency(metrics)

    console.print()
    render_comparison(agent_summaries)

    console.print()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visual Testing Tool for Santa Fe Traders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/visual_test.py --scenario zic_selfplay
  python scripts/visual_test.py --scenario zic_vs_zip --verbose
  python scripts/visual_test.py --all
  python scripts/visual_test.py --list

Available scenarios:
  zic_selfplay     - ZIC baseline (~98% efficiency)
  zip_selfplay     - ZIP adaptive margins
  kaplan_selfplay  - Kaplan sniper (market crash expected)
  skeleton_selfplay- Skeleton alpha-weighted strategy
  zic_vs_zip       - ZIP profit advantage over ZIC
  zic_vs_kaplan    - Kaplan dominance demo
  kaplan_vs_skeleton - Sniper vs alpha-weighted comparison
  zic_vs_skeleton  - Skeleton vs ZIC comparison
  gd_vs_zic        - GD belief-based vs ZIC
  mixed            - All 5 strategies competing
        """,
    )

    parser.add_argument(
        "--scenario", "-s",
        type=str,
        help="Scenario to run",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all scenarios",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available scenarios",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show more detailed agent actions",
    )
    parser.add_argument(
        "--valuations",
        action="store_true",
        help="Show token valuations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    if args.list:
        console.print("\n[bold]Available Scenarios:[/bold]\n")
        for name, scenario in SCENARIOS.items():
            console.print(f"  [cyan]{name:20}[/cyan] - {scenario['description']}")
        console.print()
        return

    if args.all:
        for scenario_name in SCENARIOS:
            run_visual_test(
                scenario_name,
                verbose=args.verbose,
                show_valuations=args.valuations,
                seed=args.seed,
            )
            console.print("─" * 60)
        return

    if args.scenario:
        run_visual_test(
            args.scenario,
            verbose=args.verbose,
            show_valuations=args.valuations,
            seed=args.seed,
        )
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
