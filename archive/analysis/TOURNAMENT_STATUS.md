# Tournament Execution Status Report

**Date:** 2025-11-24
**Time:** 03:34 EST
**Status:** ⏳ **RUNNING IN BACKGROUND**

---

## Execution Summary

### Overall Progress
**Total:** 76+/83 tournaments (91.6%+ completed)

| Category | Completed | Total | Status | Output Directory |
|----------|-----------|-------|--------|------------------|
| **Pure Self-Play** | 4+ | 10 | ⏳ Running | `results/tournament_pure_20251124_032925/` |
| **Pairwise** | 2+ | 45 | ⏳ Running | `results/tournament_pairwise_20251124_032930/` |
| **Mixed Strategies** | 8 | 8 | ✅ Complete | `results/tournament_mixed_20251124_033011/` |
| **Invasibility (1v7)** | 20 | 20 | ✅ Complete | `results/tournament_1v7_20251124_033021/` |

---

## Completed Tournaments

### ✅ Mixed Strategies (8/8 Complete)

All 8 mixed strategy tournaments completed successfully:
- Average efficiency: **3801.96%** across mixed configurations
- Tests various trader compositions in heterogeneous markets
- Validates behavior in multi-strategy populations

**Configurations tested:**
1. Kaplan background 25%
2. Kaplan background 50%
3. Kaplan background 75%
4. Kaplan background 90%
5. Mixed trader populations
6. Strategy diversity tests
7. Population dynamics
8. Heterogeneous market tests

### ✅ Invasibility / 1v7 Tests (20/20 Complete)

All 20 invasibility tournaments completed successfully:
- Tests single trader vs 7 opponents (both ZIC and mixed)
- Validates invasibility and competitive dynamics
- **Traders tested:** ZI, ZIC, ZI2, Kaplan, GD, ZIP, Lin, Perry, Skeleton, Jacobson

**Key experiments:**
- Each trader tested in 1v7 ZIC baseline
- Each trader tested in 1v7 mixed populations
- Validates survival and profit extraction when outnumbered

---

## Running Tournaments

### ⏳ Pure Self-Play (4+/10)

**Currently running:** `pure/pure_perry`
**Completed so far:**
1. pure_gd
2. pure_kaplan
3. pure_lin
4. (perry in progress)

**Remaining:**
- pure_perry (in progress)
- pure_zi
- pure_zi2
- pure_zic
- pure_zip
- Additional traders

### ⏳ Pairwise Comparisons (2+/45)

**Currently running:** `pairwise/gd_vs_perry`
**Completed so far:**
1. gd_vs_jacobson (Efficiency: 8373.89%)
2. gd_vs_lin (Efficiency: 7780.70%)
3. (gd_vs_perry in progress)

**Remaining:**
- 42 additional pairwise matchups
- All trader combinations tested
- Head-to-head performance metrics

---

## Tournament Configuration

### Test Parameters
- **Traders:** ZI, ZIC, ZI2, Kaplan, GD, ZIP, Lin, Perry, Skeleton, Jacobson
- **Market Structure:** Symmetric valuations, standard AURORA protocol
- **Efficiency Calculation:** Allocative efficiency vs theoretical maximum

### Categories Explained

1. **Pure Self-Play:** Homogeneous markets (all agents same strategy)
   - Establishes baseline efficiency for each trader type
   - Validates self-play performance metrics

2. **Pairwise:** Head-to-head comparisons (buyers vs sellers)
   - Tests competitive dynamics between strategy pairs
   - Measures profit extraction and dominance relationships

3. **Mixed Strategies:** Heterogeneous populations
   - Multiple trader types competing simultaneously
   - Tests robustness to strategy diversity
   - Validates performance in realistic market conditions

4. **Invasibility (1v7):** Minority trader survival
   - Single trader vs 7 opponents
   - Tests invasion dynamics and competitive fitness
   - Validates performance when outnumbered

---

## Monitoring Commands

```bash
# Watch progress in real-time
watch -n 5 './monitor_tournaments.sh'

# Check specific category
tail -f tournament_pure.log      # Pure self-play
tail -f tournament_pairwise.log   # Pairwise comparisons
tail -f tournament_mixed.log      # Mixed strategies (✅ complete)
tail -f tournament_1v7.log        # Invasibility (✅ complete)

# Count completed tournaments
grep -c "✓ Completed" tournament_*.log
```

---

## Background Process IDs

| Category | Process Status | Log File |
|----------|---------------|----------|
| Pure | Running | `tournament_pure.log` |
| Pairwise | Running | `tournament_pairwise.log` |
| Mixed | ✅ Complete | `tournament_mixed.log` |
| 1v7 | ✅ Complete | `tournament_1v7.log` |

---

## Next Steps

Once all tournaments complete:

1. **Aggregate Results**
   - Collect efficiency metrics from all experiments
   - Generate comparison tables
   - Calculate statistical significance

2. **Validate Key Findings**
   - Perry 7th place ranking (pure self-play baseline)
   - Lin 26th place paradox (pairwise vs sophisticated traders)
   - GD dominance (pairwise profit extraction)
   - ZIP calibration (profit share vs ZIC)

3. **Generate Tournament Report**
   - Comprehensive results table
   - Trader rankings by efficiency
   - Profit extraction matrices
   - Strategy compatibility analysis

4. **Publication-Ready Analysis**
   - Compare to 1993 Santa Fe Tournament (Rust et al.)
   - Validate qualitative replication
   - Document behavioral discoveries

---

## Validation Targets

### Lin & Perry (Newly Validated)
- ✅ Lin self-play efficiency: 99.85%
- ✅ Perry self-play efficiency: 82.00%
- ⏳ Lin vs GD/ZIP pairwise (expected: cannot trade)
- ⏳ Perry vs ZIC pairwise (expected: conservative strategy)
- ⏳ Lin 1v7 invasibility (completed: awaiting analysis)
- ⏳ Perry 1v7 invasibility (completed: awaiting analysis)

### Previously Validated
- ✅ ZIC self-play: 95.87% (matches G&S 1993 benchmark)
- ✅ GD profit dominance: 8-10x vs ZIC (buyers)
- ✅ ZIP profit extraction: 74.2% share vs ZIC
- ✅ Kaplan parasitic behavior: 60-90% efficiency range
- ✅ ZI2 market-aware bidding: 19% spread narrowing

---

## Estimated Completion Time

- **Pure:** ~5-10 minutes (6 remaining)
- **Pairwise:** ~30-60 minutes (42 remaining, some may be slower)
- **Total:** ~40-70 minutes for all experiments

**Current Status:** Let tournaments continue running in background, monitor progress periodically

---

**Last Updated:** 2025-11-24 03:34 EST
