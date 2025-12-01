# tests/regression/test_part2_selfplay.py
"""
Regression tests for Part 2 self-play (pure strategy markets).

These tests verify documented results from checklists/results.md Table 2.2.
Tests are RUTHLESS - they fail if results drift from documented specs.

Key specs from results.md Table 2.2 (Self-Play Efficiency):
| Strategy | BASE | SHRT |
|----------|------|------|
| Skeleton | 100±0% | 80±2% |
| ZIC | 98±0% | 81±1% |
| ZIP | 99±0% | 99±0% |
| Kaplan | 100±0% | 66±2% |

Key findings:
- Skeleton/Kaplan achieve 100% in BASE (perfect sniping equilibrium)
- SHRT hurts Kaplan badly (66%) - time pressure breaks sniping
- ZIP robust across environments (99% everywhere)
"""

import numpy as np

from tests.regression.conftest import run_efficiency_trials
from traders.legacy.kaplan import Kaplan
from traders.legacy.skeleton import Skeleton
from traders.legacy.zic import ZIC
from traders.legacy.zip import ZIP


class TestSelfPlayBASE:
    """Self-play tests in BASE environment."""

    def test_skeleton_base_100_percent(self):
        """Table 2.2: Skeleton BASE = 100±0%

        Pure Skeleton self-play achieves near-perfect efficiency.
        RUTHLESS THRESHOLD: [98%, 100%]

        Note: This now works after fixing the initiation bug in skeleton.py.
        The Python implementation incorrectly prevented initiation when both
        cbid and cask were 0, but the original Java SRobotExample does initiate.
        """
        efficiencies = run_efficiency_trials(Skeleton, env_name="BASE")
        mean_eff = np.mean(efficiencies)

        assert 0.98 <= mean_eff <= 1.0, (
            f"Skeleton BASE efficiency {mean_eff:.1%} outside spec [98%, 100%]. "
            f"N={len(efficiencies)} periods"
        )

    def test_kaplan_base_100_percent(self):
        """Table 2.2: Kaplan BASE = 100±0%

        Pure Kaplan markets also achieve near-perfect efficiency.
        RUTHLESS THRESHOLD: [98%, 100%]
        """
        efficiencies = run_efficiency_trials(Kaplan, env_name="BASE")
        mean_eff = np.mean(efficiencies)

        assert 0.98 <= mean_eff <= 1.0, (
            f"Kaplan BASE efficiency {mean_eff:.1%} outside spec [98%, 100%]. "
            f"N={len(efficiencies)} periods"
        )

    def test_zip_base_99_percent(self):
        """Table 2.2: ZIP BASE = 99±0%

        ZIP self-play achieves near-perfect efficiency.
        RUTHLESS THRESHOLD: [97%, 100%]
        """
        efficiencies = run_efficiency_trials(ZIP, env_name="BASE")
        mean_eff = np.mean(efficiencies)

        assert 0.97 <= mean_eff <= 1.0, (
            f"ZIP BASE efficiency {mean_eff:.1%} outside spec [97%, 100%]. "
            f"N={len(efficiencies)} periods"
        )

    def test_zic_base_98_percent(self):
        """Table 2.2: ZIC BASE = 98±0%

        ZIC self-play achieves high efficiency.
        RUTHLESS THRESHOLD: [96%, 100%]
        """
        efficiencies = run_efficiency_trials(ZIC, env_name="BASE")
        mean_eff = np.mean(efficiencies)

        assert 0.96 <= mean_eff <= 1.0, (
            f"ZIC BASE efficiency {mean_eff:.1%} outside spec [96%, 100%]. "
            f"N={len(efficiencies)} periods"
        )


class TestSelfPlaySHRT:
    """Self-play tests in SHRT (20 steps) environment.

    Time pressure reveals differences in strategy robustness.
    """

    def test_kaplan_shrt_66_percent(self):
        """Table 2.2: Kaplan SHRT = 66±2%

        Time pressure BREAKS Kaplan's sniping strategy.
        RUTHLESS THRESHOLD: [60%, 75%]
        """
        efficiencies = run_efficiency_trials(Kaplan, env_name="SHRT")
        mean_eff = np.mean(efficiencies)
        std_eff = np.std(efficiencies)

        assert 0.60 <= mean_eff <= 0.75, (
            f"Kaplan SHRT efficiency {mean_eff:.1%} outside spec [60%, 75%]. "
            f"N={len(efficiencies)} periods, std={std_eff:.1%}"
        )

    def test_skeleton_shrt_high_efficiency(self):
        """Table 2.2: Skeleton SHRT = 80±2% (results.md claim)

        After fixing the initiation bug, Skeleton achieves ~99% in SHRT.
        This is a DISCREPANCY with results.md - see investigation notes.

        RUTHLESS THRESHOLD: [95%, 100%] - based on fixed implementation

        Note: results.md claims 80±2%, but the fixed Skeleton achieves ~99%.
        The 80% value may have been from a different test configuration or
        the buggy implementation produced partial trades somehow.
        """
        efficiencies = run_efficiency_trials(Skeleton, env_name="SHRT")
        mean_eff = np.mean(efficiencies)

        # After fix, Skeleton should achieve high efficiency in SHRT too
        assert 0.95 <= mean_eff <= 1.0, (
            f"Skeleton SHRT efficiency {mean_eff:.1%} outside spec [95%, 100%]. "
            f"N={len(efficiencies)} periods"
        )

    def test_zic_shrt_81_percent(self):
        """Table 2.2: ZIC SHRT = 81±1%

        ZIC struggles under time pressure.
        RUTHLESS THRESHOLD: [76%, 88%]
        """
        efficiencies = run_efficiency_trials(ZIC, env_name="SHRT")
        mean_eff = np.mean(efficiencies)

        assert 0.76 <= mean_eff <= 0.88, (
            f"ZIC SHRT efficiency {mean_eff:.1%} outside spec [76%, 88%]. "
            f"N={len(efficiencies)} periods"
        )

    def test_zip_shrt_99_percent(self):
        """Table 2.2: ZIP SHRT = 99±0%

        ZIP is ROBUST to time pressure - maintains 99% efficiency.
        RUTHLESS THRESHOLD: [96%, 100%]
        """
        efficiencies = run_efficiency_trials(ZIP, env_name="SHRT")
        mean_eff = np.mean(efficiencies)

        assert 0.96 <= mean_eff <= 1.0, (
            f"ZIP SHRT efficiency {mean_eff:.1%} outside spec [96%, 100%]. "
            f"N={len(efficiencies)} periods"
        )


class TestSelfPlayTimePressureComparisons:
    """Compare how strategies handle time pressure (BASE vs SHRT)."""

    def test_zip_robust_to_time_pressure(self):
        """ZIP efficiency should NOT drop significantly in SHRT.

        Gap should be < 5% (99% → 99%).
        """
        base_effs = run_efficiency_trials(ZIP, env_name="BASE")
        shrt_effs = run_efficiency_trials(ZIP, env_name="SHRT")

        base_mean = np.mean(base_effs)
        shrt_mean = np.mean(shrt_effs)
        gap = base_mean - shrt_mean

        assert abs(gap) < 0.05, (
            f"ZIP efficiency gap BASE-SHRT = {gap:.1%} (expected <5%). "
            f"BASE={base_mean:.1%}, SHRT={shrt_mean:.1%}"
        )

    def test_kaplan_hurt_by_time_pressure(self):
        """Kaplan efficiency should DROP significantly in SHRT.

        Gap should be > 25% (100% → 66%).
        """
        base_effs = run_efficiency_trials(Kaplan, env_name="BASE")
        shrt_effs = run_efficiency_trials(Kaplan, env_name="SHRT")

        base_mean = np.mean(base_effs)
        shrt_mean = np.mean(shrt_effs)
        gap = base_mean - shrt_mean

        assert gap > 0.25, (
            f"Kaplan efficiency gap BASE-SHRT = {gap:.1%} (expected >25%). "
            f"BASE={base_mean:.1%}, SHRT={shrt_mean:.1%}"
        )

    def test_zip_beats_kaplan_in_shrt(self):
        """In SHRT, ZIP should massively outperform Kaplan.

        99% vs 66% = 33% gap.
        """
        zip_effs = run_efficiency_trials(ZIP, env_name="SHRT")
        kaplan_effs = run_efficiency_trials(Kaplan, env_name="SHRT")

        zip_mean = np.mean(zip_effs)
        kaplan_mean = np.mean(kaplan_effs)
        gap = zip_mean - kaplan_mean

        assert gap > 0.20, (
            f"SHRT: ZIP-Kaplan gap = {gap:.1%} (expected >20%). "
            f"ZIP={zip_mean:.1%}, Kaplan={kaplan_mean:.1%}"
        )


class TestSelfPlayHierarchy:
    """Test strategy hierarchies in self-play."""

    def test_base_all_high_efficiency(self):
        """In BASE self-play: All sophisticated strategies achieve high efficiency.

        All should be reasonably high (>85%) in self-play.
        """
        kaplan_effs = run_efficiency_trials(Kaplan, env_name="BASE")
        zip_effs = run_efficiency_trials(ZIP, env_name="BASE")
        zic_effs = run_efficiency_trials(ZIC, env_name="BASE")

        kaplan_mean = np.mean(kaplan_effs)
        zip_mean = np.mean(zip_effs)
        zic_mean = np.mean(zic_effs)

        # All should be high (>85%)
        for name, mean in [("Kaplan", kaplan_mean), ("ZIP", zip_mean), ("ZIC", zic_mean)]:
            assert mean > 0.85, f"{name} BASE efficiency {mean:.1%} < 85%"

    def test_shrt_zip_robust(self):
        """In SHRT self-play: ZIP should maintain high efficiency.

        ZIP is designed to handle time pressure well.
        """
        zip_effs = run_efficiency_trials(ZIP, env_name="SHRT")
        zip_mean = np.mean(zip_effs)

        # ZIP should be robust in SHRT
        assert zip_mean > 0.90, f"SHRT: ZIP ({zip_mean:.1%}) not robust (expected >90%)"

    def test_shrt_kaplan_struggles(self):
        """In SHRT self-play: Kaplan should struggle with time pressure.

        Kaplan's sniping strategy needs time to work.
        """
        kaplan_effs = run_efficiency_trials(Kaplan, env_name="SHRT")
        zip_effs = run_efficiency_trials(ZIP, env_name="SHRT")

        kaplan_mean = np.mean(kaplan_effs)
        zip_mean = np.mean(zip_effs)

        # Kaplan should be significantly worse than ZIP
        assert (
            kaplan_mean < zip_mean
        ), f"SHRT: Kaplan ({kaplan_mean:.1%}) not worse than ZIP ({zip_mean:.1%})"
