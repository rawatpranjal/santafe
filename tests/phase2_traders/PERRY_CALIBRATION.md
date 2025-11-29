# Perry Hyperparameter Calibration Guide

## Overview

This calibration framework systematically tests 15 parameter configurations across 10 market scenarios to optimize Perry's performance.

**Goal:** Reduce exploitation vulnerability while maintaining elite efficiency (improve from 6th to 4th-5th place)

## Hyperparameters Being Tuned

1. **a0_initial** (default: 2.0)
   - Controls statistical bidding aggressiveness
   - Lower values = more conservative predictions
   - Higher values = more aggressive statistical bidding
   - Range tested: 0.5 to 5.0

2. **desperate_threshold** (default: 0.20)
   - Time fraction when sellers use desperate acceptance
   - Lower values = desperate earlier in the period
   - Higher values = desperate later in the period
   - Range tested: 0.10 to 0.30

3. **desperate_margin** (default: 2)
   - Units above cost sellers accept when desperate
   - Lower values = more aggressive acceptance
   - Higher values = less desperate behavior
   - Range tested: 0 to 4

## Test Configurations (15 total)

- **baseline** - Original Perry (a0=2.0, threshold=0.20, margin=2)
- **conservative_a0** - Less aggressive (a0=0.5)
- **moderate_a0** - Moderate (a0=1.0)
- **aggressive_a0** - More aggressive (a0=3.0)
- **very_aggressive_a0** - Very aggressive (a0=5.0)
- **early_desperate** - Desperate earlier (threshold=0.10)
- **late_desperate** - Desperate later (threshold=0.30)
- **no_desperate** - No desperate acceptance (margin=0)
- **high_desperate** - Very desperate (margin=4)
- **ultra_conservative** - All conservative settings
- **ultra_aggressive** - All aggressive settings
- **balanced_conservative** - Balanced conservative mix
- **balanced_aggressive** - Balanced aggressive mix
- **anti_exploit_1** - Optimized against exploitation
- **anti_exploit_2** - Alternative anti-exploitation

## Test Scenarios (10 total)

1. **symmetric** - Balanced 4v4 market (baseline)
2. **flat** - Homogeneous valuations
3. **asymmetric_buyers** - 6 buyers vs 4 sellers
4. **asymmetric_sellers** - 4 buyers vs 6 sellers
5. **high_competition** - 8v8 market
6. **multi_token** - 5 tokens per agent
7. **tight_equilibrium** - Narrow spread
8. **wide_equilibrium** - Large spread
9. **self_play** - All Perry (5v5)
10. **vs_zic** - 1v7 invasibility test

## Scoring Methodology

**Composite Score = 30% efficiency + 25% profit_share + 20% invasibility + 15% balance + 10% win_rate**

- **Self-Play Efficiency (30%):** Measures cooperation and market stability
- **Profit Share vs ZIC (25%):** Measures ability to extract value from random traders
- **Invasibility Ratio (20%):** Measures competitiveness (1v7 profit ratio)
- **Balance (15%):** Measures fairness (deviation from 50/50 buyer/seller split)
- **Win Rate vs Sophisticated (10%):** Measures robustness (vs GD, Kaplan, ZIP)

## Running the Calibration

### Quick Test (2-3 hours)
```bash
python tests/phase2_traders/calibrate_perry_v1.py --replications 5 --output results/perry_calibration_quick.json
```

### Full Calibration (8-10 hours)
```bash
python tests/phase2_traders/calibrate_perry_v1.py --replications 10 --output results/perry_calibration_v1.json
```

### Custom Seed
```bash
python tests/phase2_traders/calibrate_perry_v1.py --replications 10 --seed 42 --output results/perry_calibration_seed42.json
```

### Quiet Mode (suppress progress)
```bash
python tests/phase2_traders/calibrate_perry_v1.py --replications 10 --quiet
```

## Expected Output

The script will produce:

1. **Console Output:**
   - Progress for each config and scenario
   - Ranked results table (top 10 configurations)
   - Three recommended profiles (Robust, Profit, Cooperative)

2. **JSON Results File:**
   - Detailed metrics for all 15 configurations
   - Results for all 10 scenarios per config
   - Composite scores and rankings

## Interpreting Results

### Robust Profile (Best Overall)
- Highest composite score
- Balanced across all metrics
- **Recommended for tournament play**

### Profit Profile (Best Invasibility)
- Maximum profit extraction vs ZIC
- May sacrifice some efficiency
- Recommended for competitive markets

### Cooperative Profile (Best Efficiency + Balance)
- Highest self-play efficiency
- Lowest buyer/seller imbalance
- Recommended for cooperative scenarios

## Next Steps After Calibration

1. **Identify Top 5 Configs** based on composite score
2. **Run Vulnerability Tests** against GD, Kaplan, ZIP
3. **Select Final Profiles** (robust, profit, cooperative)
4. **Update tracker.md** with results
5. **Document Findings** in calibration report

## Notes

- Each full calibration run tests ~15 configs × 10 scenarios × 10 reps = 1,500+ market simulations
- Computational time: ~8-10 hours for full run (10 replications)
- Results saved to `results/perry_calibration_v1.json`
- Original Perry baseline included for comparison

## Validation

The script has been validated with:
- Baseline config achieving 93.75% efficiency in symmetric markets
- Correct ID mapping for orderbook trade extraction
- Proper handling of all agent interfaces

## References

- Based on ZIP calibration methodology
- Follows experimental design from Phase 2 trader validation
- Scoring weights derived from tournament performance requirements
