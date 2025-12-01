# tests/regression/test_part2_pairwise.py
"""
Regression tests for Part 2 pairwise matchups.

These tests verify documented pairwise results from checklists/results.md Table 2.3.
Tests are RUTHLESS - they fail if results drift from documented specs.

Key specs from results.md Table 2.3:
| Matchup | Efficiency | Type A Profit | Type B Profit |
|---------|------------|---------------|---------------|
| ZIP vs ZI | 43.6±8.5% | ZIP: 368±9 | ZI: -268±24 |
| ZIP vs ZIC | 96.5±0.3% | ZIP: 124±8 | ZIC: 95±6 |
| ZIC vs ZI | 50.2±7.8% | ZIC: 308±5 | ZI: -193±20 |

Key findings:
- When ZI present, efficiency < 60% with high variance
- ZIP dominates ZIC (~30% more profit)
- ZI loses massively (negative profits)
"""

import numpy as np
import pytest

from tests.regression.conftest import run_pairwise_trials
from traders.legacy.zi import ZI
from traders.legacy.zic import ZIC
from traders.legacy.zip import ZIP


class TestZIPvsZIC:
    """Verify ZIP vs ZIC pairwise matchup (Table 2.3 row 2)."""

    def test_zip_vs_zic_efficiency_96_percent(self):
        """Table 2.3: ZIP vs ZIC efficiency = 96.5±0.3%

        Both strategies are sophisticated, so efficiency remains high.
        RUTHLESS THRESHOLD: [94%, 99%]
        """
        efficiencies, _, _ = run_pairwise_trials(ZIP, ZIC)
        mean_eff = np.mean(efficiencies)
        std_eff = np.std(efficiencies)

        assert 0.94 <= mean_eff <= 0.99, (
            f"ZIP vs ZIC efficiency {mean_eff:.1%} outside spec [94%, 99%]. "
            f"N={len(efficiencies)} periods, std={std_eff:.1%}"
        )

    def test_zip_dominates_zic_profit(self):
        """Table 2.3: ZIP profit 124 vs ZIC profit 95 → ~1.30x ratio.

        ZIP's adaptive learning should earn ~30% more than ZIC.
        RUTHLESS THRESHOLD: ratio in [1.1, 1.6]
        """
        _, zip_profits, zic_profits = run_pairwise_trials(ZIP, ZIC)

        zip_mean = np.mean(zip_profits)
        zic_mean = np.mean(zic_profits)

        # Avoid division by zero
        if zic_mean <= 0:
            pytest.skip("ZIC mean profit <= 0, cannot compute ratio")

        ratio = zip_mean / zic_mean
        assert 1.1 <= ratio <= 1.6, (
            f"ZIP/ZIC profit ratio {ratio:.2f}x outside spec [1.1, 1.6]. "
            f"ZIP={zip_mean:.0f}, ZIC={zic_mean:.0f}"
        )

    def test_both_positive_profits(self):
        """Both ZIP and ZIC should have positive profits against each other."""
        _, zip_profits, zic_profits = run_pairwise_trials(ZIP, ZIC)

        zip_mean = np.mean(zip_profits)
        zic_mean = np.mean(zic_profits)

        assert zip_mean > 0, f"ZIP mean profit {zip_mean:.0f} <= 0"
        assert zic_mean > 0, f"ZIC mean profit {zic_mean:.0f} <= 0"


class TestZIPvsZI:
    """Verify ZIP vs ZI pairwise matchup (Table 2.3 row 1)."""

    def test_zip_vs_zi_efficiency_below_60_percent(self):
        """Table 2.3: ZIP vs ZI efficiency = 43.6±8.5%

        ZI presence destroys efficiency.
        RUTHLESS THRESHOLD: [30%, 60%]
        """
        efficiencies, _, _ = run_pairwise_trials(ZIP, ZI)
        mean_eff = np.mean(efficiencies)
        std_eff = np.std(efficiencies)

        assert 0.30 <= mean_eff <= 0.60, (
            f"ZIP vs ZI efficiency {mean_eff:.1%} outside spec [30%, 60%]. "
            f"N={len(efficiencies)} periods, std={std_eff:.1%}"
        )

    def test_zi_negative_profits(self):
        """Table 2.3: ZI profit = -268±24 (negative).

        ZI trades at a loss, funding ZIP gains.
        """
        _, zip_profits, zi_profits = run_pairwise_trials(ZIP, ZI)

        zi_mean = np.mean(zi_profits)
        zip_mean = np.mean(zip_profits)

        assert zi_mean < 0, f"ZI mean profit {zi_mean:.0f} >= 0 (expected negative)"
        assert zip_mean > 0, f"ZIP mean profit {zip_mean:.0f} <= 0 (expected positive)"

    def test_high_variance_with_zi(self):
        """Table 2.3: ZI markets have high variance (8.5% std).

        ZI introduces randomness that increases outcome variance.
        """
        efficiencies, _, _ = run_pairwise_trials(ZIP, ZI)
        std_eff = np.std(efficiencies)

        # High variance expected (> 5%)
        assert std_eff > 0.05, f"ZIP vs ZI efficiency std {std_eff:.1%} too low (expected >5%)"


class TestZICvsZI:
    """Verify ZIC vs ZI pairwise matchup (Table 2.3 row 3)."""

    def test_zic_vs_zi_efficiency_around_50_percent(self):
        """Table 2.3: ZIC vs ZI efficiency = 50.2±7.8%

        ZI presence keeps efficiency low.
        RUTHLESS THRESHOLD: [35%, 65%]
        """
        efficiencies, _, _ = run_pairwise_trials(ZIC, ZI)
        mean_eff = np.mean(efficiencies)
        std_eff = np.std(efficiencies)

        assert 0.35 <= mean_eff <= 0.65, (
            f"ZIC vs ZI efficiency {mean_eff:.1%} outside spec [35%, 65%]. "
            f"N={len(efficiencies)} periods, std={std_eff:.1%}"
        )

    def test_zic_positive_zi_negative(self):
        """Table 2.3: ZIC profit positive, ZI profit negative.

        ZIC exploits ZI's random trading.
        """
        _, zic_profits, zi_profits = run_pairwise_trials(ZIC, ZI)

        zic_mean = np.mean(zic_profits)
        zi_mean = np.mean(zi_profits)

        assert zic_mean > 0, f"ZIC mean profit {zic_mean:.0f} <= 0"
        assert zi_mean < 0, f"ZI mean profit {zi_mean:.0f} >= 0 (expected negative)"


class TestPairwiseHierarchy:
    """Test the critical hierarchies from pairwise matchups."""

    def test_zi_always_loses(self):
        """ZI should have negative profits in ALL pairwise matchups."""
        for opponent in [ZIP, ZIC]:
            _, opp_profits, zi_profits = run_pairwise_trials(opponent, ZI)
            zi_mean = np.mean(zi_profits)
            opp_name = opponent.__name__

            assert zi_mean < 0, f"ZI vs {opp_name}: ZI mean profit {zi_mean:.0f} >= 0"

    def test_zip_beats_zic_in_profit(self):
        """ZIP should beat ZIC when they face each other."""
        _, zip_profits, zic_profits = run_pairwise_trials(ZIP, ZIC)

        zip_mean = np.mean(zip_profits)
        zic_mean = np.mean(zic_profits)

        assert zip_mean > zic_mean, f"ZIP ({zip_mean:.0f}) <= ZIC ({zic_mean:.0f}) in head-to-head"

    def test_zi_presence_destroys_efficiency(self):
        """When ZI is present, market efficiency drops below 60%."""
        for opponent in [ZIP, ZIC]:
            efficiencies, _, _ = run_pairwise_trials(opponent, ZI)
            mean_eff = np.mean(efficiencies)
            opp_name = opponent.__name__

            assert mean_eff < 0.60, f"{opp_name} vs ZI: efficiency {mean_eff:.1%} >= 60%"
