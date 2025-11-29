# RL Experiments Log - PPO Profit Maximization

**Date Started:** November 24, 2025
**Ultimate Goal:** Train PPO agent to **beat ALL legacy traders** (ZIC, ZIP, GD, Kaplan) in individual profit

---

## 🎯 Success Criteria

### Primary Objective
- **Beat Kaplan** (best human-designed trader) by >10% profit per episode
- **Beat GD** by >20% profit per episode
- **Beat ZIP** by >30% profit per episode
- **Beat ZIC** by >50% profit per episode

### Secondary Objectives
- Maintain >70% market efficiency (minimum viability)
- Achieve 3-4 trades per episode (active participation)
- Keep invalid actions <20 per episode (acceptable risk)

---

## 📊 Baseline Performance

### Legacy Traders (from Grand Melee results)
| Trader | Avg Profit/Ep | Efficiency | Trades/Ep | Strategy |
|--------|---------------|------------|-----------|----------|
| **Kaplan** | ~5-6 | 75-80% | 2-3 | Strategic sniping |
| **GD** | ~4-5 | 80-85% | 3-4 | Belief-based |
| **ZIP** | ~3-4 | 85-90% | 3-4 | Adaptive margin |
| **ZIC** | ~2-3 | 95-98% | 4-5 | Random constrained |

### Current PPO (Curriculum Model)
| Model | Profit/Ep | Efficiency | Trades/Ep | Invalid/Ep | Status |
|-------|-----------|------------|-----------|------------|--------|
| **ppo_curriculum** | 0.39 | 27.5% | 0.5 | 49.4 | ❌ Broken (train/eval mismatch) |
| **ppo_vs_zic_v2** | Unknown | Unknown | Unknown | Unknown | 🟡 Needs eval |

**Problem:** Current agent achieves 85-89% efficiency during training but only 0.39 profit (10x worse than ZIC!). Agent is "altruistic" - optimizes market good over personal gain.

---

## 🧪 Experiment Series

### Phase 1: Profit-Focused Reward Engineering

---

### Exp-001: Pure Profit Maximization
**Date:** 2025-11-24 | **Status:** 🟡 Planned

**Hypothesis:** Current agent is "altruistic" because efficiency_bonus (0.1) and market_making (0.5) rewards encourage cooperative behavior. Eliminating ALL cooperative signals and using pure profit (100.0 weight) will force selfish profit extraction.

**Configuration Changes:**
```yaml
# Base: conf/rl/experiments/ppo_vs_zic.yaml
# New:  conf/rl/experiments/exp001_pure_profit.yaml

profit_weight: 1.0 → 100.0  # 100x increase - ONLY reward personal profit
efficiency_bonus_weight: 0.2 → 0.0  # Remove market efficiency reward
market_making_weight: 0.5 → 0.0  # Remove liquidity provision reward
bid_submission_bonus: 0.02 → 0.0  # Remove participation reward
exploration_weight: 0.05 → 0.0  # Remove exploration bonus
surplus_capture_weight: 0.1 → 0.0  # Remove surplus bonus
invalid_penalty: -0.01 → -0.5  # Keep safety net (0.5% of max profit)
normalize_rewards: false → false  # Keep raw profit signals

opponent_type: "ZIC"  # Pure ZIC for clean baseline
total_timesteps: 500_000  # Quick validation (2.5 hours)
```

**Training Details:**
- Timesteps: 500K
- Estimated Time: ~2.5 hours (CPU)
- Opponent: Pure ZIC
- Config Path: `conf/rl/experiments/exp001_pure_profit.yaml`
- Model Save: `models/exp001_pure_profit/`

**Target Results:**
| Metric | Target | Current | Improvement Needed |
|--------|--------|---------|-------------------|
| Profit/Episode | >3.0 | 0.39 | 7.7x |
| vs ZIC Profit | +50% | -80% | 2.7x relative |
| Efficiency | >70% | 27.5% | 2.5x |
| Trades/Episode | >3.0 | 0.5 | 6x |
| Invalid Actions | <20 | 49 | -29 |

**Evaluation Plan:**
1. Train for 500K steps (~2.5 hours)
2. Evaluate vs ZIC for 100 episodes
3. Compare: PPO profit vs ZIC avg profit
4. Analyze: reward components, trading patterns, price strategies

**Expected Outcomes:**
- ✅ **Best Case:** Profit/ep >3.0, beats ZIC by 50%+, learns "selfish" trading
- 🟡 **Medium Case:** Profit/ep ~2.0, matches ZIC, but still passive
- ❌ **Worst Case:** Profit/ep <1.0, reward signal too sparse, agent doesn't learn

**Key Questions:**
1. Does pure profit signal provide enough learning gradient?
2. Will agent learn to trade or remain passive?
3. How many invalid actions with loose penalty (-0.5)?

**Next Steps:**
- If profit >3.0: Proceed to Exp-002 (add exploration back)
- If profit 1.5-3.0: Try Exp-001b with profit_weight=50.0
- If profit <1.5: Need dense intermediate rewards (bid submission, etc.)

---

### Exp-002: Profit + Aggressive Exploration
**Date:** TBD | **Status:** 🔵 Waiting on Exp-001

**Hypothesis:** Pure profit (Exp-001) may be too sparse early in training. Adding high entropy (0.15) and exploration bonus (0.5) will help agent discover profitable strategies faster.

**Configuration Changes:**
```yaml
# Base: exp001_pure_profit.yaml
# Changes:
profit_weight: 100.0  # Keep
ent_coef: 0.01 → 0.15  # Very high entropy for exploration
exploration_weight: 0.0 → 0.5  # Reward trying all actions
invalid_penalty: -0.5 → -0.01  # Very permissive (encourage risk)
```

**Target:** Find optimal exploration/exploitation balance for profit

---

### Exp-003a: ZIC Specialist
**Date:** TBD | **Status:** 🔵 Waiting

**Hypothesis:** Training specialized agents per opponent type will yield higher profits than generalist. ZIC is purely random, so agent should learn to exploit predictability.

**Configuration:**
```yaml
opponent_type: "ZIC"  # Pure ZIC
profit_weight: 50.0
ent_coef: 0.10
total_timesteps: 500_000
```

**Target:** Beat ZIC by >50% (profit ~4-5)

---

### Exp-003b: ZIP Specialist
**Status:** 🔵 Waiting

**Hypothesis:** ZIP adapts margins based on market. Agent should learn to front-run ZIP's adaptation or exploit its lag.

**Configuration:**
```yaml
opponent_type: "ZIP"
profit_weight: 50.0
ent_coef: 0.10
total_timesteps: 500_000
```

**Target:** Beat ZIP by >30% (profit ~5-6)

---

### Exp-003c: GD Specialist
**Status:** 🔵 Waiting

**Hypothesis:** GD forms beliefs about equilibrium. Agent should learn to mislead belief formation or exploit GD's optimization.

**Configuration:**
```yaml
opponent_type: "GD"
profit_weight: 50.0
ent_coef: 0.10
total_timesteps: 500_000
```

**Target:** Beat GD by >20% (profit ~6-7)

---

### Exp-003d: Kaplan Specialist
**Status:** 🔵 Waiting

**Hypothesis:** Kaplan is strategic sniper. Agent needs to either out-snipe or bait Kaplan into bad trades.

**Configuration:**
```yaml
opponent_type: "Kaplan"
profit_weight: 50.0
ent_coef: 0.10
total_timesteps: 750_000  # Longer - Kaplan is hard
```

**Target:** Beat Kaplan by >10% (profit ~6.5-7)

---

### Exp-004: Shading Strategy Focus
**Status:** 🔵 Waiting

**Hypothesis:** Actions 7-8 (shade ±5%, ±10%) enable price discrimination. Adding shading_bonus will encourage agent to use these actions to extract surplus.

**Configuration:**
```yaml
profit_weight: 50.0
# Add new reward component:
shading_bonus: 0.5  # Reward for using actions 7-8
ent_coef: 0.10
```

**Target:** Learn when to shade 5% vs 10% for optimal surplus extraction

---

### Exp-005: Tournament Generalist
**Status:** 🔵 Waiting

**Hypothesis:** Training against random opponent mix will develop robust profit strategy that works against all types.

**Configuration:**
```yaml
opponent_mix: ["ZIC", "ZIP", "GD", "Kaplan"]  # Random each episode
profit_weight: 50.0
ent_coef: 0.10
total_timesteps: 2_000_000  # Long - needs to learn all patterns
```

**Target:** Beat average opponent profit by >25%

---

### Phase 2: Hyperparameter Optimization (Exp-006 to Exp-010)

**Grid Search Over:**
- Learning rate: [0.0001, 0.0005, 0.001]
- Entropy: [0.05, 0.10, 0.15]
- Network size: [128x128, 256x256, 512x256]
- Batch size: [32, 64, 128]

**Status:** 🔵 Waiting on Phase 1 results

---

### Phase 3: Champion Training

### Exp-015: Final Champion (5M Steps)
**Status:** 🔵 Waiting

**Configuration:** Best config from Exp 001-010
**Timesteps:** 5,000,000 (~25 hours)
**Opponent:** Tournament mix
**Goal:** Converge to maximum profit performance

---

## 📈 Results Summary

| Exp | Profit/Ep | vs ZIC | vs ZIP | vs GD | vs Kaplan | Efficiency | Trades | Invalid | Status |
|-----|-----------|--------|--------|-------|-----------|------------|--------|---------|---------|
| Baseline (Curriculum) | 0.39 | -80% | -90% | -90% | -93% | 27.5% | 0.5 | 49 | ❌ |
| 001 | - | - | - | - | - | - | - | - | 🟡 Planned |
| 002 | - | - | - | - | - | - | - | - | 🔵 Waiting |
| 003a | - | - | - | - | - | - | - | - | 🔵 Waiting |
| 003b | - | - | - | - | - | - | - | - | 🔵 Waiting |
| 003c | - | - | - | - | - | - | - | - | 🔵 Waiting |
| 003d | - | - | - | - | - | - | - | - | 🔵 Waiting |
| 004 | - | - | - | - | - | - | - | - | 🔵 Waiting |
| 005 | - | - | - | - | - | - | - | - | 🔵 Waiting |

---

## 🔍 Key Insights

### What We've Learned

**From Curriculum Training (3M steps):**
- ✅ **Agent CAN learn constraints:** 0 invalid actions during training
- ✅ **Agent CAN achieve high efficiency:** 85-89% market efficiency
- ❌ **Agent is "altruistic":** Optimizes market over personal profit (0.02 ratio)
- ❌ **Reward balance is critical:** efficiency_bonus (0.1) dominated profit_weight (1.0)
- ❌ **Train/eval mismatch is severe:** 89% efficiency (train) → 27.5% (eval)

**Hypotheses to Test:**
1. **Cooperation Problem:** Efficiency/market-making rewards make agent "nice" → Test with pure profit (Exp-001)
2. **Exploration Problem:** Low entropy (0.01) prevents finding profitable strategies → Test with high entropy (Exp-002)
3. **Generalization Problem:** Can't specialize vs each opponent type → Test specialist models (Exp-003)
4. **Action Space Problem:** Not using shading actions effectively → Test shading bonus (Exp-004)

---

## 📝 Experiment Template

```markdown
### Exp-{ID}: {Name}
**Date:** YYYY-MM-DD | **Status:** 🟡 Running | ✅ Complete | ❌ Failed

**Hypothesis:** {One sentence explaining what we expect to learn}

**Configuration Changes:**
```yaml
# Show only deltas from baseline
parameter: old_value → new_value
```

**Results:**
| Metric | Target | Result | Δ from Baseline |
|--------|--------|--------|-----------------|
| Profit/Episode | >X | X.X | +X.X (+XX%) |
| vs ZIC Profit | +50% | +XX% | - |
| Efficiency | >70% | XX% | +XX% |
| Trades/Episode | 3-4 | X.X | +X.X |
| Invalid Actions | <20 | XX | -XX |

**Key Insight:** {One sentence summary of what we learned}

**Next Step:** {Which experiment to run next based on results}
```

---

## 🎮 Evaluation Protocol

### Standard Evaluation (Every Experiment)
1. **Training:** 500K-2M steps depending on experiment
2. **Evaluation:** 100 episodes per opponent type
3. **Metrics Tracked:**
   - Individual profit per episode (PRIMARY)
   - Profit relative to opponent avg (vs ZIC, ZIP, GD, Kaplan)
   - Market efficiency (secondary)
   - Trades executed
   - Invalid actions
4. **Analysis:** Compare to baseline and previous experiments

### Tournament Evaluation (Best Models)
1. **Grand Melee:** PPO + ZIC + ZIP + GD + Kaplan (all in same market)
2. **1v1 Matchups:** PPO vs each opponent individually (pure)
3. **Profit Ranking:** Where does PPO rank among all traders?
4. **Strategy Analysis:** What strategies does PPO learn?

---

## 📅 Timeline

### Week 1: Reward Engineering
- **Day 1-2:** Exp-001 (Pure Profit) + evaluation
- **Day 3:** Exp-002 (Aggressive Exploration) if needed
- **Day 4-5:** Analyze results, plan Phase 2

### Week 2: Specialist Training
- **Day 1:** Exp-003a (ZIC specialist)
- **Day 2:** Exp-003b (ZIP specialist)
- **Day 3:** Exp-003c (GD specialist)
- **Day 4-5:** Exp-003d (Kaplan specialist)

### Week 3: Optimization & Champion
- **Day 1-3:** Hyperparameter sweep (Exp-006 to 010)
- **Day 4-7:** Exp-015 (Champion 5M steps)

### Week 4: Evaluation & Analysis
- **Day 1-2:** Tournament evaluation
- **Day 3-4:** Strategy analysis
- **Day 5:** Write-up and documentation

---

## 🏆 Success Metrics Dashboard

### Primary Metric: Profit Dominance
```
PPO Profit > Kaplan Profit + 10%
PPO Profit > GD Profit + 20%
PPO Profit > ZIP Profit + 30%
PPO Profit > ZIC Profit + 50%
```

### Secondary Metrics
- Efficiency >70% (viability)
- Trades/Ep 3-4 (active participation)
- Invalid <20 (acceptable risk)

### Stretch Goals
- PPO #1 in Grand Melee by profit
- PPO beats Kaplan 80%+ of time in 1v1
- PPO profit >8.0 per episode (2x Kaplan)

---

## 📚 Resources

**Configurations:** `conf/rl/experiments/exp{ID}_*.yaml`
**Models:** `models/exp{ID}/`
**Training Logs:** `archive/phase4_training/exp{ID}_log.txt`
**Analysis:** `archive/phase4_training/TRAINING_SUMMARY.md`

**Key Files:**
- `scripts/train_ppo_enhanced.py` - Training script
- `envs/enhanced_double_auction_env.py` - Environment with reward shaping
- `engine/tournament.py` - Tournament evaluation

---

**Next Action:** Create `conf/rl/experiments/exp001_pure_profit.yaml` and launch first training run.
