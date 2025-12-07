#!/usr/bin/env python3
"""
Generate Introductory Market Visualization Figures.

Creates pedagogical figures for the Market section (04a_market.tex) that introduce:
1. Supply and demand curves with competitive equilibrium
2. Token distribution and private values
3. Market efficiency decomposition

Uses hand-crafted example data (not real market runs) for clarity.

Usage:
    python scripts/generate_market_intro_figures.py
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from paper_style import COLORS, setup_style

# Output directory
FIGURES_DIR = Path("paper/arxiv/figures")
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# Apply shared style settings
setup_style()

# Hand-crafted example data for 4 buyers, 4 sellers, 4 tokens each
# Designed for clear visualization with intersection around Q*=10, P*=145
BUYER_VALUATIONS = [
    [200, 180, 160, 140],  # Buyer 1 (high value)
    [190, 170, 150, 130],  # Buyer 2
    [180, 160, 140, 120],  # Buyer 3
    [170, 150, 130, 110],  # Buyer 4 (low value)
]

SELLER_COSTS = [
    [80, 100, 120, 140],  # Seller 1 (low cost)
    [90, 110, 130, 150],  # Seller 2
    [100, 120, 140, 160],  # Seller 3
    [110, 130, 150, 170],  # Seller 4 (high cost)
]


def build_supply_demand_curves(
    buyer_valuations: list[list[int]], seller_costs: list[list[int]]
) -> tuple[list[int], list[int], int, int, int]:
    """
    Build supply and demand curves from token valuations/costs.

    Returns:
        demand: Sorted valuations (descending)
        supply: Sorted costs (ascending)
        q_star: Equilibrium quantity
        p_star: Equilibrium price
        max_surplus: Maximum possible surplus
    """
    # Flatten and sort
    demand = sorted([v for vals in buyer_valuations for v in vals], reverse=True)
    supply = sorted([c for costs in seller_costs for c in costs])

    # Find equilibrium (where demand crosses supply)
    q_star = 0
    max_surplus = 0
    for i in range(min(len(demand), len(supply))):
        if demand[i] > supply[i]:
            q_star = i + 1
            max_surplus += demand[i] - supply[i]
        else:
            break

    # Equilibrium price is midpoint of marginal pair
    if q_star > 0:
        p_star = (demand[q_star - 1] + supply[q_star - 1]) // 2
    else:
        p_star = (demand[0] + supply[0]) // 2

    return demand, supply, q_star, p_star, max_surplus


def plot_supply_demand_equilibrium() -> None:
    """
    Figure 1: Supply and Demand Curves with Competitive Equilibrium.

    Shows:
    - Step function demand curve (buyer valuations sorted descending)
    - Step function supply curve (seller costs sorted ascending)
    - Equilibrium point (Q*, P*)
    - Total surplus as shaded area
    """
    demand, supply, q_star, p_star, max_surplus = build_supply_demand_curves(
        BUYER_VALUATIONS, SELLER_COSTS
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create step function coordinates
    # For demand: each unit q has value demand[q-1], so we step down
    q_demand = [0]
    p_demand = [demand[0]]
    for i, val in enumerate(demand):
        q_demand.extend([i, i + 1])
        p_demand.extend([val, val])
    q_demand.append(len(demand))
    p_demand.append(demand[-1])

    # For supply: each unit q has cost supply[q-1], so we step up
    q_supply = [0]
    p_supply = [supply[0]]
    for i, cost in enumerate(supply):
        q_supply.extend([i, i + 1])
        p_supply.extend([cost, cost])
    q_supply.append(len(supply))
    p_supply.append(supply[-1])

    # Shade total surplus area (area between curves up to Q*)
    for i in range(q_star):
        ax.fill_between(
            [i, i + 1],
            [supply[i], supply[i]],
            [demand[i], demand[i]],
            color=COLORS["surplus"],
            alpha=0.4,
        )

    # Plot demand curve
    ax.step(
        range(len(demand) + 1),
        [demand[0]] + list(demand),
        where="pre",
        color=COLORS["demand"],
        linewidth=2.5,
        label="Demand D(q)",
    )

    # Plot supply curve
    ax.step(
        range(len(supply) + 1),
        [supply[0]] + list(supply),
        where="pre",
        color=COLORS["supply"],
        linewidth=2.5,
        label="Supply S(q)",
    )

    # Mark equilibrium point
    ax.axhline(p_star, color=COLORS["equilibrium"], linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(q_star, color=COLORS["equilibrium"], linestyle="--", linewidth=1.5, alpha=0.7)

    # Equilibrium marker
    ax.scatter(
        [q_star],
        [p_star],
        color=COLORS["equilibrium"],
        s=150,
        zorder=5,
        marker="o",
        edgecolors="black",
        linewidth=1.5,
    )

    # Annotations
    ax.annotate(
        f"Equilibrium\n$Q^*={q_star}$, $P^*={p_star}$",
        xy=(q_star, p_star),
        xytext=(q_star + 2, p_star + 25),
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
    )

    # Label surplus area
    ax.text(
        q_star / 2,
        (demand[q_star // 2] + supply[q_star // 2]) / 2,
        f"Total Surplus\n= {max_surplus}",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLORS["surplus"], alpha=0.9
        ),
    )

    # Labels and formatting
    ax.set_xlabel("Quantity (units)")
    ax.set_ylabel("Price")
    ax.set_title("Supply and Demand with Competitive Equilibrium")
    ax.set_xlim(-0.5, 17)
    ax.set_ylim(50, 220)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add tick marks for each unit
    ax.set_xticks(range(0, 17, 2))

    plt.tight_layout()

    # Save
    output_path = FIGURES_DIR / "market_supply_demand.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_token_distribution() -> None:
    """
    Figure 2: Token Distribution and Private Values.

    Shows how tokens are assigned to buyers and sellers,
    with descending valuations for buyers and ascending costs for sellers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Buyer valuations
    ax_buyers = axes[0]
    num_buyers = len(BUYER_VALUATIONS)
    num_tokens = len(BUYER_VALUATIONS[0])
    bar_width = 0.18
    x = np.arange(num_tokens)

    for i, vals in enumerate(BUYER_VALUATIONS):
        offset = (i - num_buyers / 2 + 0.5) * bar_width
        bars = ax_buyers.bar(
            x + offset,
            vals,
            bar_width,
            label=f"Buyer {i+1}",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

    ax_buyers.set_xlabel("Token (unit)")
    ax_buyers.set_ylabel("Redemption Value")
    ax_buyers.set_title("(a) Buyer Valuations")
    ax_buyers.set_xticks(x)
    ax_buyers.set_xticklabels([f"Unit {i+1}" for i in range(num_tokens)])
    ax_buyers.legend(loc="upper right")
    ax_buyers.set_ylim(0, 220)
    ax_buyers.grid(True, alpha=0.3, axis="y")

    # Add annotation about descending structure
    ax_buyers.annotate(
        "Values decrease\nwithin each buyer",
        xy=(2.5, 130),
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
    )

    # Right panel: Seller costs
    ax_sellers = axes[1]
    num_sellers = len(SELLER_COSTS)

    for i, costs in enumerate(SELLER_COSTS):
        offset = (i - num_sellers / 2 + 0.5) * bar_width
        bars = ax_sellers.bar(
            x + offset,
            costs,
            bar_width,
            label=f"Seller {i+1}",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

    ax_sellers.set_xlabel("Token (unit)")
    ax_sellers.set_ylabel("Production Cost")
    ax_sellers.set_title("(b) Seller Costs")
    ax_sellers.set_xticks(x)
    ax_sellers.set_xticklabels([f"Unit {i+1}" for i in range(num_tokens)])
    ax_sellers.legend(loc="upper left")
    ax_sellers.set_ylim(0, 220)
    ax_sellers.grid(True, alpha=0.3, axis="y")

    # Add annotation about ascending structure
    ax_sellers.annotate(
        "Costs increase\nwithin each seller",
        xy=(1.5, 180),
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
    )

    plt.tight_layout()

    # Save
    output_path = FIGURES_DIR / "market_token_distribution.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_efficiency_decomposition() -> None:
    """
    Figure 3: Market Efficiency Decomposition.

    Shows:
    - Maximum possible surplus (total area between curves)
    - Example of realized surplus
    - V-inefficiency (missed profitable trades)
    - EM-inefficiency (bad trades)
    """
    demand, supply, q_star, p_star, max_surplus = build_supply_demand_curves(
        BUYER_VALUATIONS, SELLER_COSTS
    )

    # Simulate a scenario with some inefficiency
    # Assume 8 trades occurred (instead of 10), and 1 was a bad trade
    actual_trades = 8
    bad_trade_loss = 20  # Extra-marginal trade at position 12: demand=130, supply=150

    # Calculate actual surplus
    actual_surplus = sum(demand[i] - supply[i] for i in range(actual_trades)) - bad_trade_loss

    # V-inefficiency: missed trades (units 9 and 10)
    v_ineff = sum(demand[i] - supply[i] for i in range(actual_trades, q_star))

    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Theoretical maximum
    ax_max = axes[0]

    # Shade total surplus
    for i in range(q_star):
        ax_max.fill_between(
            [i, i + 1],
            [supply[i], supply[i]],
            [demand[i], demand[i]],
            color=COLORS["surplus"],
            alpha=0.5,
        )

    # Plot curves
    ax_max.step(
        range(len(demand) + 1),
        [demand[0]] + list(demand),
        where="pre",
        color=COLORS["demand"],
        linewidth=2.5,
        label="Demand D(q)",
    )
    ax_max.step(
        range(len(supply) + 1),
        [supply[0]] + list(supply),
        where="pre",
        color=COLORS["supply"],
        linewidth=2.5,
        label="Supply S(q)",
    )

    # Equilibrium lines
    ax_max.axhline(p_star, color=COLORS["equilibrium"], linestyle="--", linewidth=1.5, alpha=0.7)
    ax_max.axvline(q_star, color=COLORS["equilibrium"], linestyle="--", linewidth=1.5, alpha=0.7)

    # Labels
    ax_max.set_xlabel("Quantity")
    ax_max.set_ylabel("Price")
    ax_max.set_title("(a) Maximum Possible Surplus (100% Efficiency)")
    ax_max.set_xlim(-0.5, 14)
    ax_max.set_ylim(50, 220)
    ax_max.legend(loc="upper right")
    ax_max.grid(True, alpha=0.3)

    # Surplus annotation
    ax_max.text(
        q_star / 2,
        p_star,
        f"Max Surplus = {max_surplus}\n({q_star} trades)",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLORS["surplus"]),
    )

    # Right panel: Actual with inefficiencies
    ax_actual = axes[1]

    # Shade realized surplus (first 8 trades)
    for i in range(actual_trades):
        ax_actual.fill_between(
            [i, i + 1],
            [supply[i], supply[i]],
            [demand[i], demand[i]],
            color=COLORS["surplus"],
            alpha=0.5,
        )

    # Shade V-inefficiency (missed trades 9-10)
    for i in range(actual_trades, q_star):
        ax_actual.fill_between(
            [i, i + 1],
            [supply[i], supply[i]],
            [demand[i], demand[i]],
            color=COLORS["v_ineff"],
            alpha=0.5,
            hatch="//",
        )

    # Shade EM-inefficiency (bad trade at position 12 where supply > demand)
    em_q = q_star + 2  # Position 12: demand=130, supply=150, loss=20
    ax_actual.fill_between(
        [em_q, em_q + 1],
        [demand[em_q], demand[em_q]],
        [supply[em_q], supply[em_q]],
        color=COLORS["em_ineff"],
        alpha=0.5,
        hatch="\\\\",
    )

    # Plot curves
    ax_actual.step(
        range(len(demand) + 1),
        [demand[0]] + list(demand),
        where="pre",
        color=COLORS["demand"],
        linewidth=2.5,
        label="Demand D(q)",
    )
    ax_actual.step(
        range(len(supply) + 1),
        [supply[0]] + list(supply),
        where="pre",
        color=COLORS["supply"],
        linewidth=2.5,
        label="Supply S(q)",
    )

    # Equilibrium lines
    ax_actual.axhline(p_star, color=COLORS["equilibrium"], linestyle="--", linewidth=1.5, alpha=0.7)
    ax_actual.axvline(q_star, color=COLORS["equilibrium"], linestyle="--", linewidth=1.5, alpha=0.7)

    # Labels
    ax_actual.set_xlabel("Quantity")
    ax_actual.set_ylabel("Price")

    efficiency = (actual_surplus / max_surplus) * 100
    ax_actual.set_title(f"(b) Actual Outcome ({efficiency:.0f}% Efficiency)")
    ax_actual.set_xlim(-0.5, 14)
    ax_actual.set_ylim(50, 220)
    ax_actual.grid(True, alpha=0.3)

    # Create legend with efficiency components
    legend_elements = [
        mpatches.Patch(
            facecolor=COLORS["surplus"], alpha=0.5, label=f"Realized Surplus ({actual_surplus})"
        ),
        mpatches.Patch(
            facecolor=COLORS["v_ineff"],
            alpha=0.5,
            hatch="//",
            label=f"V-Inefficiency: Missed trades ({v_ineff})",
        ),
        mpatches.Patch(
            facecolor=COLORS["em_ineff"],
            alpha=0.5,
            hatch="\\\\",
            label=f"EM-Inefficiency: Bad trade ({bad_trade_loss})",
        ),
    ]
    ax_actual.legend(handles=legend_elements, loc="upper right")

    # Annotations
    ax_actual.annotate(
        "Missed\nprofitable\ntrades",
        xy=(actual_trades + 0.5, (demand[actual_trades] + supply[actual_trades]) / 2),
        xytext=(actual_trades - 1.5, 80),
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
    )

    ax_actual.annotate(
        "Extra-marginal\ntrade (loss)",
        xy=(em_q + 0.5, (demand[em_q] + supply[em_q]) / 2),
        xytext=(em_q + 1.5, 100),
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
    )

    plt.tight_layout()

    # Save
    output_path = FIGURES_DIR / "market_efficiency_decomposition.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_trader_hierarchy() -> None:
    """
    Figure 4: Budget Constraint and Adaptive Learning.

    Shows the same market with three different trader types:
    - ZI: Unconstrained random (trades anywhere, including bad trades)
    - ZIC: Budget constrained (only profitable trades, may miss some)
    - ZIP: Adaptive learning (learns to trade at equilibrium)

    This visualizes the hierarchy ZI < ZIC < ZIP that drives efficiency differences.
    """
    demand, supply, q_star, p_star, max_surplus = build_supply_demand_curves(
        BUYER_VALUATIONS, SELLER_COSTS
    )

    # Curated trade data for each trader type (same market, different outcomes)
    # ZI: Scattered trades, some in bad regions
    zi_trades = [
        (1, 75),  # Trade 1 at bad price
        (2, 185),  # Trade 2 above equilibrium
        (3, 95),  # Trade 3 below equilibrium
        (4, 165),  # Trade 4
        (5, 60),  # Trade 5 way too low
        (6, 175),  # Trade 6
        (7, 200),  # Trade 7 way too high
        (8, 110),  # Trade 8
        (9, 145),  # Trade 9 near equilibrium
        (10, 80),  # Trade 10 bad price
        (11, 155),  # Trade 11 - extra-marginal (bad)
        (12, 50),  # Trade 12 - extra-marginal (bad)
    ]

    # ZIC: All trades within profitable region, but miss 2 trades
    zic_trades = [
        (1, 155),  # Trade 1
        (2, 148),  # Trade 2
        (3, 142),  # Trade 3
        (4, 138),  # Trade 4
        (5, 145),  # Trade 5
        (6, 140),  # Trade 6
        (7, 135),  # Trade 7
        (8, 142),  # Trade 8 - last trade, miss 2 marginal
    ]

    # ZIP: All trades cluster at equilibrium, captures all 10
    zip_trades = [
        (1, 145),  # Trade 1
        (2, 142),  # Trade 2
        (3, 140),  # Trade 3
        (4, 139),  # Trade 4
        (5, 140),  # Trade 5
        (6, 141),  # Trade 6
        (7, 140),  # Trade 7
        (8, 140),  # Trade 8
        (9, 139),  # Trade 9
        (10, 140),  # Trade 10 - all profitable trades captured
    ]

    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax_idx, (ax, trades, name, color, description) in enumerate(
        zip(
            axes,
            [zi_trades, zic_trades, zip_trades],
            ["ZI (Unconstrained)", "ZIC (Budget Constrained)", "ZIP (Adaptive Learning)"],
            [COLORS["zi"], COLORS["zic"], COLORS["zip"]],
            [
                "Trades anywhere\nIncluding bad trades",
                "Only profitable trades\nMay miss opportunities",
                "Learns equilibrium price\nCaptures all surplus",
            ],
        )
    ):
        # Plot supply and demand curves
        ax.step(
            range(len(demand) + 1),
            [demand[0]] + list(demand),
            where="pre",
            color=COLORS["demand"],
            linewidth=2,
            alpha=0.7,
        )
        ax.step(
            range(len(supply) + 1),
            [supply[0]] + list(supply),
            where="pre",
            color=COLORS["supply"],
            linewidth=2,
            alpha=0.7,
        )

        # Shade profitable region
        for i in range(q_star):
            ax.fill_between(
                [i, i + 1],
                [supply[i], supply[i]],
                [demand[i], demand[i]],
                color=COLORS["surplus"],
                alpha=0.15,
            )

        # Shade unprofitable region (where cost > value)
        for i in range(q_star, min(len(demand), len(supply))):
            ax.fill_between(
                [i, i + 1],
                [demand[i], demand[i]],
                [supply[i], supply[i]],
                color=COLORS["em_ineff"],
                alpha=0.15,
            )

        # Plot equilibrium line
        ax.axhline(p_star, color=COLORS["equilibrium"], linestyle="--", linewidth=1.5, alpha=0.5)

        # Plot trades
        trade_qs = [t[0] for t in trades]
        trade_ps = [t[1] for t in trades]
        ax.scatter(
            trade_qs,
            trade_ps,
            s=100,
            c=color,
            marker="o",
            edgecolors="black",
            linewidth=1,
            zorder=5,
            label=f"{len(trades)} trades",
        )

        # Calculate efficiency for this trader type
        # Surplus from each trade depends on which units traded
        # For simplicity, estimate based on number of trades vs max
        if name.startswith("ZI"):
            # ZI trades extra-marginal units, losing value
            eff = 28  # Typical ZI efficiency
            efficiency_text = f"~{eff}% efficiency"
        elif name.startswith("ZIC"):
            # ZIC misses 2 trades
            eff = 97  # Typical ZIC efficiency
            efficiency_text = f"~{eff}% efficiency"
        else:
            # ZIP captures all
            eff = 99  # Typical ZIP efficiency
            efficiency_text = f"~{eff}% efficiency"

        # Labels
        panel_label = chr(ord("a") + ax_idx)
        ax.set_title(f"({panel_label}) {name}")
        ax.set_xlabel("Quantity")
        if ax_idx == 0:
            ax.set_ylabel("Price")
        ax.set_xlim(-0.5, 14)
        ax.set_ylim(40, 220)
        ax.grid(True, alpha=0.3)

        # Add description box
        ax.text(
            0.5,
            0.02,
            f"{description}\n{efficiency_text}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=color, alpha=0.9),
        )

        # Highlight bad trades for ZI
        if name.startswith("ZI"):
            # Mark extra-marginal trades
            for q, p in trades:
                if q > q_star:
                    ax.scatter(
                        [q], [p], s=150, facecolors="none", edgecolors="red", linewidth=2, zorder=6
                    )

    # Add overall title
    fig.suptitle("The Budget Constraint and Adaptive Learning Hierarchy", y=1.02)

    plt.tight_layout()

    # Save
    output_path = FIGURES_DIR / "market_trader_hierarchy.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_learning_curves() -> None:
    """
    Figure 5: Learning Curves - Efficiency and Price Convergence Over Rounds.

    Shows how each trader type performs over time:
    - ZI: Flat low efficiency, high volatility (no learning)
    - ZIC: Flat high efficiency (constraint provides instant benefit, no learning)
    - ZIP: Converges from lower start to high efficiency (adaptive learning)

    This demonstrates that ZIP's advantage comes from margin adjustment over time.
    """
    # Curated learning curve data (50 rounds)
    rounds = np.arange(1, 51)

    # ZI: Constant low efficiency with high noise (no learning)
    np.random.seed(42)
    zi_efficiency = 28 + np.random.normal(0, 8, 50)
    zi_efficiency = np.clip(zi_efficiency, 5, 50)

    # ZIC: Constant high efficiency with low noise (no learning, just constraint)
    zic_efficiency = 97 + np.random.normal(0, 2, 50)
    zic_efficiency = np.clip(zic_efficiency, 90, 100)

    # ZIP: Starts around 85%, converges to 99% over ~15 rounds (learning curve)
    # Exponential convergence: eff(t) = target - (target - start) * exp(-t/tau)
    zip_start = 85
    zip_target = 99
    zip_tau = 8  # Time constant for learning
    zip_base = zip_target - (zip_target - zip_start) * np.exp(-rounds / zip_tau)
    zip_efficiency = zip_base + np.random.normal(0, 1.5, 50)
    zip_efficiency = np.clip(zip_efficiency, 80, 100)

    # Price deviation from equilibrium (RMSD as % of P*)
    # ZI: High constant deviation
    zi_rmsd = 45 + np.random.normal(0, 5, 50)
    zi_rmsd = np.clip(zi_rmsd, 30, 60)

    # ZIC: Low constant deviation
    zic_rmsd = 8 + np.random.normal(0, 1.5, 50)
    zic_rmsd = np.clip(zic_rmsd, 4, 15)

    # ZIP: Converges from higher to very low
    zip_rmsd_start = 18
    zip_rmsd_target = 5
    zip_rmsd_base = zip_rmsd_target + (zip_rmsd_start - zip_rmsd_target) * np.exp(-rounds / 10)
    zip_rmsd = zip_rmsd_base + np.random.normal(0, 1, 50)
    zip_rmsd = np.clip(zip_rmsd, 2, 25)

    # Create 2-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Efficiency over rounds
    ax_eff = axes[0]
    ax_eff.plot(rounds, zi_efficiency, color=COLORS["zi"], linewidth=1.5, alpha=0.8, label="ZI")
    ax_eff.plot(rounds, zic_efficiency, color=COLORS["zic"], linewidth=1.5, alpha=0.8, label="ZIC")
    ax_eff.plot(rounds, zip_efficiency, color=COLORS["zip"], linewidth=2, label="ZIP")

    # Add smoothed trend lines
    from scipy.ndimage import uniform_filter1d

    window = 5
    ax_eff.plot(
        rounds,
        uniform_filter1d(zi_efficiency, window),
        color=COLORS["zi"],
        linewidth=2.5,
        linestyle="--",
    )
    ax_eff.plot(
        rounds,
        uniform_filter1d(zic_efficiency, window),
        color=COLORS["zic"],
        linewidth=2.5,
        linestyle="--",
    )
    ax_eff.plot(
        rounds,
        uniform_filter1d(zip_efficiency, window),
        color=COLORS["zip"],
        linewidth=2.5,
        linestyle="--",
    )

    ax_eff.set_xlabel("Round")
    ax_eff.set_ylabel("Allocative Efficiency (%)")
    ax_eff.set_title("(a) Efficiency Convergence")
    ax_eff.set_xlim(1, 50)
    ax_eff.set_ylim(0, 105)
    ax_eff.legend(loc="lower right")
    ax_eff.grid(True, alpha=0.3)

    # Add annotations
    ax_eff.annotate(
        "ZIP learns\noptimal margins",
        xy=(10, 92),
        xytext=(20, 70),
        arrowprops=dict(arrowstyle="->", color=COLORS["zip"], lw=1.5),
        color=COLORS["zip"],
    )
    ax_eff.annotate(
        "ZIC: constraint alone\nachieves high efficiency",
        xy=(35, 97),
        xytext=(25, 85),
        ha="center",
        color=COLORS["zic"],
    )
    ax_eff.annotate(
        "ZI: no learning,\nlow efficiency",
        xy=(35, 28),
        xytext=(35, 45),
        ha="center",
        color=COLORS["zi"],
    )

    # Right panel: Price deviation over rounds
    ax_price = axes[1]
    ax_price.plot(rounds, zi_rmsd, color=COLORS["zi"], linewidth=1.5, alpha=0.8, label="ZI")
    ax_price.plot(rounds, zic_rmsd, color=COLORS["zic"], linewidth=1.5, alpha=0.8, label="ZIC")
    ax_price.plot(rounds, zip_rmsd, color=COLORS["zip"], linewidth=2, label="ZIP")

    # Smoothed trend lines
    ax_price.plot(
        rounds, uniform_filter1d(zi_rmsd, window), color=COLORS["zi"], linewidth=2.5, linestyle="--"
    )
    ax_price.plot(
        rounds,
        uniform_filter1d(zic_rmsd, window),
        color=COLORS["zic"],
        linewidth=2.5,
        linestyle="--",
    )
    ax_price.plot(
        rounds,
        uniform_filter1d(zip_rmsd, window),
        color=COLORS["zip"],
        linewidth=2.5,
        linestyle="--",
    )

    ax_price.set_xlabel("Round")
    ax_price.set_ylabel("Price Deviation (RMSD % of P*)")
    ax_price.set_title("(b) Price Convergence")
    ax_price.set_xlim(1, 50)
    ax_price.set_ylim(0, 70)
    ax_price.legend(loc="upper right")
    ax_price.grid(True, alpha=0.3)

    # Add annotations
    ax_price.annotate(
        "ZIP prices\nconverge to P*",
        xy=(15, 8),
        xytext=(25, 25),
        arrowprops=dict(arrowstyle="->", color=COLORS["zip"], lw=1.5),
        color=COLORS["zip"],
    )

    plt.tight_layout()

    # Save
    output_path = FIGURES_DIR / "learning_curves.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main() -> None:
    """Generate all introductory market figures."""
    print("Generating introductory market figures...")
    print(f"Output directory: {FIGURES_DIR}")
    print()

    # Calculate and print example data stats
    demand, supply, q_star, p_star, max_surplus = build_supply_demand_curves(
        BUYER_VALUATIONS, SELLER_COSTS
    )
    print("Example market configuration:")
    print(f"  Buyers: {len(BUYER_VALUATIONS)} x {len(BUYER_VALUATIONS[0])} tokens")
    print(f"  Sellers: {len(SELLER_COSTS)} x {len(SELLER_COSTS[0])} tokens")
    print(f"  Equilibrium: Q* = {q_star}, P* = {p_star}")
    print(f"  Maximum surplus: {max_surplus}")
    print()

    # Generate figures
    print("Figure 1: Supply and Demand Curves...")
    plot_supply_demand_equilibrium()

    print("Figure 2: Token Distribution...")
    plot_token_distribution()

    print("Figure 3: Efficiency Decomposition...")
    plot_efficiency_decomposition()

    print("Figure 4: Trader Hierarchy (ZI < ZIC < ZIP)...")
    plot_trader_hierarchy()

    print("Figure 5: Learning Curves...")
    plot_learning_curves()

    print()
    print("All figures generated successfully!")
    print(f"Files saved to: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
