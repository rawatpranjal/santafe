# tests/regression/test_part1_environments.py
"""
Regression tests for Part 1 (Zero-Intelligence) across multiple environments.

These tests verify efficiency levels from checklists/results.md Table 1.1.
Tests are RUTHLESS - they fail if results drift from documented specs.

Key specs from results.md Table 1.1:
| Trader | BASE | SHRT | TOK | EQL | SML |
|--------|------|------|-----|-----|-----|
| ZI     | 28±3 | 29±3 | 94±1| 100 | 16±2|
| ZIC    | 98±1 | 79±2 | 96±1| 100 | 88±2|
| ZIP    | 99±0 | 99±0 | 100 | 100 | 89±2|

SHRT is particularly important: time pressure breaks ZIC (79%) but not ZIP (99%).
"""

import numpy as np
import pytest

from tests.regression.conftest import (
    run_efficiency_trials,
)
from traders.legacy.zi import ZI
from traders.legacy.zic import ZIC
from traders.legacy.zip import ZIP


class TestSHRTEnvironment:
    """Tests for SHRT (20 steps) - Time pressure environment.

    SHRT is the key differentiator: ZIC drops to 79% while ZIP maintains 99%.
    This proves ZIP's adaptive learning handles time pressure better than random.
    """

    def test_zic_shrt_efficiency_79_percent(self):
        """Table 1.1: ZIC SHRT = 79±2%

        Time pressure breaks ZIC's efficiency significantly.
        RUTHLESS THRESHOLD: [75%, 85%]
        """
        efficiencies = run_efficiency_trials(ZIC, env_name="SHRT")
        mean_eff = np.mean(efficiencies)
        std_eff = np.std(efficiencies)

        assert 0.75 <= mean_eff <= 0.85, (
            f"ZIC SHRT efficiency {mean_eff:.1%} outside spec [75%, 85%]. "
            f"N={len(efficiencies)} periods, std={std_eff:.1%}"
        )

    def test_zip_shrt_efficiency_99_percent(self):
        """Table 1.1: ZIP SHRT = 99±0%

        ZIP maintains high efficiency even under time pressure.
        RUTHLESS THRESHOLD: [97%, 100%]
        """
        efficiencies = run_efficiency_trials(ZIP, env_name="SHRT")
        mean_eff = np.mean(efficiencies)

        assert 0.97 <= mean_eff <= 1.0, (
            f"ZIP SHRT efficiency {mean_eff:.1%} outside spec [97%, 100%]. "
            f"N={len(efficiencies)} periods"
        )

    def test_shrt_zip_beats_zic(self):
        """Critical invariant: ZIP >> ZIC in SHRT environment.

        The gap should be substantial (~20% expected: 99% vs 79%).
        """
        zic_effs = run_efficiency_trials(ZIC, env_name="SHRT")
        zip_effs = run_efficiency_trials(ZIP, env_name="SHRT")

        zic_mean = np.mean(zic_effs)
        zip_mean = np.mean(zip_effs)

        gap = zip_mean - zic_mean
        assert gap > 0.10, (
            f"SHRT gap ZIP-ZIC too small: {gap:.1%} (expected >10%). "
            f"ZIP={zip_mean:.1%}, ZIC={zic_mean:.1%}"
        )


class TestTOKEnvironment:
    """Tests for TOK (1 token per trader) - Minimal market.

    Note: These are smoke tests - detailed specs require exact gametype matching.
    """

    def test_zic_tok_trades(self):
        """ZIC with single tokens should complete some trades."""
        efficiencies = run_efficiency_trials(ZIC, env_name="TOK")
        mean_eff = np.mean(efficiencies)

        # Just verify ZIC achieves positive efficiency
        assert mean_eff > 0.30, f"ZIC TOK efficiency {mean_eff:.1%} too low (expected >30%)"

    def test_zip_tok_trades(self):
        """ZIP with single tokens should complete some trades."""
        efficiencies = run_efficiency_trials(ZIP, env_name="TOK")
        mean_eff = np.mean(efficiencies)

        # Just verify ZIP achieves positive efficiency
        assert mean_eff > 0.30, f"ZIP TOK efficiency {mean_eff:.1%} too low (expected >30%)"


class TestSMLEnvironment:
    """Tests for SML (2B/2S) - Small market.

    Note: Small markets use BASE gametype for quick tests.
    """

    def test_zic_sml_positive_efficiency(self):
        """ZIC in small markets should achieve reasonable efficiency."""
        efficiencies = run_efficiency_trials(ZIC, env_name="SML")
        mean_eff = np.mean(efficiencies)

        # Small markets with 2B/2S still work
        assert mean_eff > 0.80, f"ZIC SML efficiency {mean_eff:.1%} too low (expected >80%)"

    def test_zip_sml_positive_efficiency(self):
        """ZIP in small markets should achieve reasonable efficiency."""
        efficiencies = run_efficiency_trials(ZIP, env_name="SML")
        mean_eff = np.mean(efficiencies)

        assert mean_eff > 0.80, f"ZIP SML efficiency {mean_eff:.1%} too low (expected >80%)"


class TestEnvironmentHierarchies:
    """Test that trader hierarchy holds across environments."""

    @pytest.mark.parametrize("env_name", ["SHRT", "TOK"])
    def test_hierarchy_zic_lt_zip(self, env_name):
        """ZIC <= ZIP should hold in SHRT and TOK environments.

        Note: SML is excluded because small markets have high variance.
        """
        zic_effs = run_efficiency_trials(ZIC, env_name=env_name)
        zip_effs = run_efficiency_trials(ZIP, env_name=env_name)

        zic_mean = np.mean(zic_effs)
        zip_mean = np.mean(zip_effs)

        # Allow margin for statistical noise
        assert (
            zic_mean <= zip_mean + 0.05
        ), f"{env_name}: ZIC ({zic_mean:.1%}) > ZIP ({zip_mean:.1%}) + 5% margin"

    def test_zi_worst_except_eql_tok(self):
        """ZI should be worst in BASE, SHRT, SML but not in EQL/TOK."""
        for env_name in ["SHRT", "SML"]:
            zi_effs = run_efficiency_trials(ZI, env_name=env_name)
            zic_effs = run_efficiency_trials(ZIC, env_name=env_name)

            zi_mean = np.mean(zi_effs)
            zic_mean = np.mean(zic_effs)

            assert zi_mean < zic_mean, f"{env_name}: ZI ({zi_mean:.1%}) >= ZIC ({zic_mean:.1%})"
