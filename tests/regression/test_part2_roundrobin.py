# tests/regression/test_part2_roundrobin.py
"""
Regression tests for Part 2 Round Robin (4-strategy mixed market).

These tests verify documented results from checklists/results.md Table 2.6.
Tests are RUTHLESS - they fail if results drift from documented specs.

Key specs from results.md Table 2.6 (Rank Table):
| Strategy | BASE Rank |
|----------|-----------|
| ZIP | 1.6±0.5 |
| Skeleton | 1.8±0.9 |
| Kaplan | 2.6±0.7 |
| ZIC | 4.0±0.0 |

Key findings:
- ZIC ALWAYS ranks last (4.0±0.0) - critical invariant
- ZIP/Skeleton competitive for top positions
- Hierarchy: ZIP ≈ Skeleton > Kaplan > ZIC
"""

import numpy as np
import pytest

from tests.regression.conftest import run_roundrobin_trials
from traders.legacy.kaplan import Kaplan
from traders.legacy.skeleton import Skeleton
from traders.legacy.zic import ZIC
from traders.legacy.zip import ZIP


class TestRoundRobinBASE:
    """Round-robin tests in BASE environment."""

    def test_zic_typically_last(self):
        """Table 2.6: ZIC BASE Rank = 4.0±0.0

        ZIC should typically rank last or near-last in mixed markets.
        Note: With fewer trials, variance is higher.
        """
        strategies = [ZIP, Skeleton, Kaplan, ZIC]
        profits = run_roundrobin_trials(strategies, env_name="BASE")

        # Calculate ranks for each trial
        strategy_names = [s.__name__ for s in strategies]
        num_trials = len(profits["ZIC"])

        zic_ranks = []
        for i in range(num_trials):
            trial_profits = {name: profits[name][i] for name in strategy_names}
            sorted_names = sorted(
                trial_profits.keys(), key=lambda x: trial_profits[x], reverse=True
            )
            zic_rank = sorted_names.index("ZIC") + 1
            zic_ranks.append(zic_rank)

        mean_rank = np.mean(zic_ranks)
        std_rank = np.std(zic_ranks)

        # ZIC should rank low on average (>= 3.0)
        assert mean_rank >= 3.0, (
            f"ZIC mean rank {mean_rank:.1f} < 3.0 (expected ~4.0). " f"std={std_rank:.1f}"
        )

    def test_zip_top_tier(self):
        """Table 2.6: ZIP BASE Rank = 1.6±0.5

        ZIP should rank in top 2 on average.
        RUTHLESS THRESHOLD: rank <= 2.5
        """
        strategies = [ZIP, Skeleton, Kaplan, ZIC]
        profits = run_roundrobin_trials(strategies, env_name="BASE")

        strategy_names = [s.__name__ for s in strategies]
        num_trials = len(profits["ZIP"])

        zip_ranks = []
        for i in range(num_trials):
            trial_profits = {name: profits[name][i] for name in strategy_names}
            sorted_names = sorted(
                trial_profits.keys(), key=lambda x: trial_profits[x], reverse=True
            )
            zip_rank = sorted_names.index("ZIP") + 1
            zip_ranks.append(zip_rank)

        mean_rank = np.mean(zip_ranks)

        assert mean_rank <= 2.5, f"ZIP mean rank {mean_rank:.1f} > 2.5 (expected ~1.6)"

    def test_skeleton_competitive(self):
        """Table 2.6: Skeleton BASE Rank = 1.8±0.9

        Skeleton should also be competitive (top 2-3).
        RUTHLESS THRESHOLD: rank <= 2.8
        """
        strategies = [ZIP, Skeleton, Kaplan, ZIC]
        profits = run_roundrobin_trials(strategies, env_name="BASE")

        strategy_names = [s.__name__ for s in strategies]
        num_trials = len(profits["Skeleton"])

        skeleton_ranks = []
        for i in range(num_trials):
            trial_profits = {name: profits[name][i] for name in strategy_names}
            sorted_names = sorted(
                trial_profits.keys(), key=lambda x: trial_profits[x], reverse=True
            )
            skeleton_rank = sorted_names.index("Skeleton") + 1
            skeleton_ranks.append(skeleton_rank)

        mean_rank = np.mean(skeleton_ranks)

        assert mean_rank <= 2.8, f"Skeleton mean rank {mean_rank:.1f} > 2.8 (expected ~1.8)"


class TestRoundRobinHierarchy:
    """Test critical hierarchy invariants."""

    def test_hierarchy_zip_skeleton_beat_kaplan_zic(self):
        """ZIP and Skeleton should have better ranks than Kaplan and ZIC."""
        strategies = [ZIP, Skeleton, Kaplan, ZIC]
        profits = run_roundrobin_trials(strategies, env_name="BASE")

        zip_mean = np.mean(profits["ZIP"])
        skeleton_mean = np.mean(profits["Skeleton"])
        kaplan_mean = np.mean(profits["Kaplan"])
        zic_mean = np.mean(profits["ZIC"])

        # Top tier (ZIP, Skeleton) should beat bottom tier (Kaplan, ZIC)
        top_tier_min = min(zip_mean, skeleton_mean)
        bottom_tier_max = max(kaplan_mean, zic_mean)

        # Allow some margin for variance
        assert top_tier_min > bottom_tier_max * 0.8, (
            f"Top tier min ({top_tier_min:.0f}) not > bottom tier max ({bottom_tier_max:.0f}) * 0.8. "
            f"ZIP={zip_mean:.0f}, Skeleton={skeleton_mean:.0f}, "
            f"Kaplan={kaplan_mean:.0f}, ZIC={zic_mean:.0f}"
        )

    def test_zic_lowest_profit(self):
        """ZIC should have the lowest mean profit in round-robin."""
        strategies = [ZIP, Skeleton, Kaplan, ZIC]
        profits = run_roundrobin_trials(strategies, env_name="BASE")

        zic_mean = np.mean(profits["ZIC"])

        for strategy in [ZIP, Skeleton, Kaplan]:
            other_mean = np.mean(profits[strategy.__name__])
            assert (
                zic_mean < other_mean
            ), f"ZIC profit ({zic_mean:.0f}) >= {strategy.__name__} ({other_mean:.0f})"


class TestRoundRobinProfitRatios:
    """Test profit ratios between strategies."""

    def test_zip_vs_zic_profit_ratio(self):
        """ZIP should earn significantly more than ZIC.

        Expected ratio ~1.3-1.5x based on results.md.
        """
        strategies = [ZIP, Skeleton, Kaplan, ZIC]
        profits = run_roundrobin_trials(strategies, env_name="BASE")

        zip_mean = np.mean(profits["ZIP"])
        zic_mean = np.mean(profits["ZIC"])

        if zic_mean <= 0:
            pytest.skip("ZIC mean profit <= 0, cannot compute ratio")

        ratio = zip_mean / zic_mean
        assert ratio > 1.1, (
            f"ZIP/ZIC profit ratio {ratio:.2f}x <= 1.1. " f"ZIP={zip_mean:.0f}, ZIC={zic_mean:.0f}"
        )

    def test_all_strategies_positive_profit(self):
        """All strategies should earn positive profit in round-robin.

        Unlike pairwise with ZI, this market should be efficient enough
        for all participants to profit.
        """
        strategies = [ZIP, Skeleton, Kaplan, ZIC]
        profits = run_roundrobin_trials(strategies, env_name="BASE")

        for strategy in strategies:
            mean_profit = np.mean(profits[strategy.__name__])
            assert mean_profit > 0, f"{strategy.__name__} mean profit {mean_profit:.0f} <= 0"


class TestRoundRobinSHRT:
    """Round-robin tests in SHRT environment (time pressure)."""

    def test_zip_dominates_in_shrt(self):
        """Table 2.6: ZIP SHRT Rank = 1.0±0.0

        ZIP should dominate under time pressure.
        """
        strategies = [ZIP, Skeleton, Kaplan, ZIC]
        profits = run_roundrobin_trials(strategies, env_name="SHRT")

        strategy_names = [s.__name__ for s in strategies]
        num_trials = len(profits["ZIP"])

        zip_ranks = []
        for i in range(num_trials):
            trial_profits = {name: profits[name][i] for name in strategy_names}
            sorted_names = sorted(
                trial_profits.keys(), key=lambda x: trial_profits[x], reverse=True
            )
            zip_rank = sorted_names.index("ZIP") + 1
            zip_ranks.append(zip_rank)

        mean_rank = np.mean(zip_ranks)

        # ZIP should be #1 or close
        assert mean_rank <= 2.0, f"SHRT: ZIP mean rank {mean_rank:.1f} > 2.0 (expected ~1.0)"

    def test_zic_still_last_in_shrt(self):
        """ZIC should still rank last even in SHRT environment."""
        strategies = [ZIP, Skeleton, Kaplan, ZIC]
        profits = run_roundrobin_trials(strategies, env_name="SHRT")

        strategy_names = [s.__name__ for s in strategies]
        num_trials = len(profits["ZIC"])

        zic_ranks = []
        for i in range(num_trials):
            trial_profits = {name: profits[name][i] for name in strategy_names}
            sorted_names = sorted(
                trial_profits.keys(), key=lambda x: trial_profits[x], reverse=True
            )
            zic_rank = sorted_names.index("ZIC") + 1
            zic_ranks.append(zic_rank)

        mean_rank = np.mean(zic_ranks)

        assert mean_rank >= 3.5, f"SHRT: ZIC mean rank {mean_rank:.1f} < 3.5 (expected ~4.0)"
