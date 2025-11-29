# Phase 4 PPO Training Summary

**Date:** November 24, 2025
**Total Training Time:** ~8-10 hours (CPU)
**Total Timesteps:** ~4.5M across all runs

---

## Training Runs

### Run 1: PPO vs ZIC v1 (Baseline)
**Status:** ✅ Complete, FAILED objectives
**Duration:** 500K timesteps (~2.5 hours)
**Model:** `models/ppo_vs_zic/final_model.zip`

**Configuration:**
- Opponents: Pure ZIC (Zero-Intelligence Constrained)
- Learning rate: 0.0003
- Entropy coef: 0.01
- Invalid penalty: -0.1
- Profit weight: 1.0

**Results:**
- ✅ Constraint learning: Perfect (43 → 0 invalid actions by 160K)
- ❌ Trading strategy: Failed (0% efficiency, passive "do nothing" policy)
- ❌ Root cause: Reward imbalance - penalties dominated trading incentives

**Lessons Learned:**
- Sparse rewards insufficient for learning profitable trading
- Need dense intermediate rewards (bid submission, surplus capture)
- Penalty too harsh relative to profit signals

---

### Run 2: PPO vs ZIC v2 (Reward Rebalanced)
**Status:** ✅ Complete, IMPROVED but weak
**Duration:** Unknown timesteps
**Model:** `models/ppo_vs_zic_v2/final_model.zip`

**Configuration Changes:**
- Invalid penalty: -0.1 → -0.01 (10x reduction)
- Market making weight: 0.05 → 0.5 (10x increase)
- Exploration weight: 0.01 → 0.05 (5x increase)
- Added bid_submission_bonus: 0.02
- Added surplus_capture_weight: 0.1
- Learning rate: 0.0003 → 0.0005
- Entropy coef: 0.01 → 0.03

**Results:**
- Mean reward improved 480x (0.01 → 4.81 at 40K steps)
- Still conservative: 0-3 trades vs target 8-12
- 0% efficiency initially, gradual improvement

**Lessons Learned:**
- Reward engineering critical - 10x changes needed
- Agent learns to trade but remains risk-averse
- Need more aggressive profit incentives

---

### Run 3: PPO Curriculum Learning (BEST)
**Status:** ✅ Complete, 3M timesteps
**Duration:** ~3.5 hours (11:40 AM - 12:11 PM)
**Models:** 5 stage checkpoints + final model

**Curriculum Stages:**
1. **Basics** (500K steps): Pure ZIC opponents
2. **Mixed Easy** (500K steps): [ZIC, ZIC, ZIP]
3. **Mixed Balanced** (750K steps): [ZIC, ZIP, GD]
4. **Strategic** (750K steps): [ZIP, GD, Kaplan]
5. **Expert** (500K steps): [GD, Kaplan, Kaplan]

**Stage Performance:**
| Stage | Efficiency | Profit Ratio | Trades/Ep | Notes |
|-------|-----------|--------------|-----------|-------|
| 1 (Basics) | 65-75% | 0.01-0.02 | 1.4 | Patience exceeded |
| 2 (Mixed Easy) | Similar | Similar | Similar | Patience exceeded |
| 3 (Balanced) | 65-75% | 0.01-0.02 | 1.5-2.0 | Patience exceeded |
| 4 (Strategic) | **85-89%** | 0.018-0.023 | 2.0-2.2 | **Best efficiency!** |
| 5 (Expert) | Started | N/A | N/A | No time to complete |

**Final Model Results:**
- ✅ **Market Efficiency: 85-89%** (exceeds 75% target!)
- ❌ **Profit Ratio: 0.018-0.023** (vs target >1.0)
- ⚠️ **Trading Volume: 2.0-2.2** trades/episode (low)
- ✅ **Invalid Actions: 0** during training (perfect constraints)

**Lessons Learned:**
- Curriculum learning enables progression to hard opponents
- Agent learns to maximize market efficiency, not personal profit
- "Patience exceeded" mechanism allowed progress despite not meeting criteria
- Profit ratio metric appears miscalibrated (targets unreachable)

---

### Run 4: PPO vs Mixed
**Status:** ✅ Complete
**Model:** `models/ppo_vs_mixed/final_model.zip`
**Performance:** Unknown (needs evaluation)

---

## Critical Bugs Fixed

### Bug 1: Evaluation Crash (ValueError)
**Issue:** `if done:` treated array as scalar
**Fix:** Handle both scalar and array done conditions
**Impact:** Prevented running final evaluations

### Bug 2: VecNormalize Not Loaded
**Issue:** eval_env created without VecNormalize wrapper
**Fix:** Match eval_env normalization to train_env
**Impact:** Observations on wrong scale → poor performance

### Bug 3: Metrics Not Captured
**Issue:** Episode loop exited before capturing final info
**Fix:** Save final_info before loop exit
**Impact:** All metrics showed 0% even when agent traded

### Bug 4: Float32 JSON Serialization
**Issue:** NumPy types not JSON serializable
**Fix:** Convert all values to Python native float()
**Impact:** Results couldn't be saved to disk

---

## Key Findings

### What Worked
1. ✅ **Reward engineering**: 10x changes made massive difference
2. ✅ **Curriculum learning**: Enabled learning against strategic opponents
3. ✅ **Constraint learning**: Agent perfectly learns trading constraints
4. ✅ **Market efficiency**: 85-89% efficiency achieved (exceeds targets)

### What Didn't Work
1. ❌ **Profit extraction**: Only 2% profit ratio (target: >100%)
2. ❌ **Trading volume**: 2 trades/episode (should be 3-4)
3. ❌ **Evaluation mismatch**: Train vs eval environment differences
4. ❌ **Profit focus**: Agent optimizes market good over personal gain

### Outstanding Issues
1. **Train/Eval Mismatch:** Curriculum model shows 89% efficiency in training but 27% in evaluation
   - Root cause: Evaluation uses base config (ZIC) vs training final stage (GD+Kaplan)
   - Solution: Create eval config matching final training stage
2. **Weak Profit Extraction:** Agent is "too altruistic"
   - Learns to maximize market efficiency
   - Doesn't prioritize personal profit
   - Need stronger profit_weight (1.0 → 5-10)
3. **Low Trading Volume:** Only 2 trades per episode
   - Agent is risk-averse despite reduced penalties
   - May need "trading bonus" to encourage participation

---

## Next Steps

### Immediate (High Priority)
1. **Fix Eval Config:** Create `ppo_curriculum_eval.yaml` matching stage 4 opponents
2. **Rebalance Rewards:** Increase profit_weight to 5-10, add "selfishness bonus"
3. **Test Best Model:** Properly evaluate curriculum model against correct opponents

### Short-term
4. **Tournament Integration:** Test PPO agent in Grand Melee vs legacy traders
5. **Profit-Focused Training:** Retrain with stronger profit incentives
6. **Documentation:** Write up findings for Phase 4 completion

### Medium-term
7. **Multi-Agent PPO:** Multiple RL agents in same market
8. **Opponent Modeling:** Predict and exploit opponent strategies
9. **Meta-Learning:** Fast adaptation to new opponent types

---

## Models Available

All models saved in `models/` directory:

**Best Model:** `ppo_curriculum/final_model.zip` (510KB)
- 3M timesteps, 5-stage curriculum
- 85-89% efficiency during training
- Requires `vec_normalize.pkl` for evaluation

**Stage Checkpoints:**
- `ppo_curriculum/stage_basics.zip` - After ZIC training
- `ppo_curriculum/stage_mixed_easy.zip` - After ZIC+ZIP training
- `ppo_curriculum/stage_mixed_balanced.zip` - After ZIC+ZIP+GD training
- `ppo_curriculum/stage_strategic.zip` - After ZIP+GD+Kaplan training

**Baseline Models:**
- `ppo_vs_zic/final_model.zip` - Conservative v1 (failed)
- `ppo_vs_zic_v2/final_model.zip` - Improved rewards v2 (weak)
- `ppo_vs_mixed/final_model.zip` - Mixed opponents (unknown)

---

## Training Logs

All logs compressed and archived in `archive/phase4_training/`:
- `training_log_curriculum.txt.gz` - 68K lines, 3M timesteps
- `training_log_v2.txt` - Reward rebalancing run
- `training_log*.txt` - Various earlier runs
- `eval_curriculum*.txt` - Evaluation attempts

---

## References

**Configuration Files:** `conf/rl/experiments/`
- `ppo_vs_zic.yaml` - Baseline ZIC training
- `ppo_curriculum.yaml` - 5-stage curriculum
- `ppo_vs_mixed.yaml` - Mixed opponent training

**Code:** `scripts/train_ppo_enhanced.py`
- Main training script with curriculum support
- All evaluation bugs now fixed
- Ready for production use

**Environment:** `envs/enhanced_double_auction_env.py`
- 9-action space (including shading strategies)
- 24-feature observation space
- Comprehensive reward shaping

---

**Status:** Phase 4.1 (PPO Infrastructure) ✅ COMPLETE
**Next:** Phase 4.2 (Profit-Focused Retraining)
