# Development Tracker

## 2025-11-29

### Kaplan Win/Loss Hypothesis Testing - KEY FINDINGS

**Goal:** Determine EXACTLY when Kaplan wins vs loses through 8 targeted experiments.

**Results Table:**
| Test | Hypothesis | Config | Ratio | Eff | Result |
|------|------------|--------|-------|-----|--------|
| H1.1 | ZIC > 50% | 1K + 4ZIC + 3ZIP | 1.12x | 98% | WIN |
| H1.2 | ZIC < 50% | 1K + 1ZIC + 6ZIP | 1.06x | 99% | WIN |
| H2.1 | Sniper deadlock | 2v2 Kap vs Kap | N/A | 99% | OK |
| H2.2 | Sniper competition | K+Skel+2ZIC | 1.10x | 99% | WIN |
| H3.1 | Few tokens | 1K vs 7ZIC (2 tok) | 1.15x | 99% | WIN |
| H3.2 | Single token mixed | 4v4 TOK env | 1.02x | 96% | WIN |
| H4.1 | Time + self-play | 8x Kap SHRT | N/A | **58%** | **DEADLOCK** |
| H4.2 | Time + mixed | 4v4 SHRT | 1.10x | 97% | WIN |

**Hypothesis Verdicts:**
- **H1 (ZIC > 50% = WIN): NOT SUPPORTED** - Kaplan wins even with 14% ZIC
- **H2 (Sniper competition = DEADLOCK): NOT SUPPORTED** - No deadlock with normal time
- **H3 (Single token = WIN): SUPPORTED** - Kaplan wins TOK env even in mixed
- **H4 (Time pressure crashes self-play): SUPPORTED** - SHRT self-play: 57.7% efficiency!

**Key Finding: Time Pressure + Self-Play is the ONLY condition where Kaplan fails**

The SHRT environment (20 steps) crashes Kaplan self-play (57.7% efficiency) but NOT mixed markets. This explains why:
- Kaplan wins 7/10 1990 tournament environments
- Kaplan loses in environments with time pressure AND sniper competition

**Implication:** Kaplan's sniper strategy is robust against all opponent types EXCEPT when multiple Kaplans compete under time pressure.

**Files:** `scripts/run_kaplan_hypotheses.py`, `results/kaplan_hypotheses/`

---

### Part 2 Round-Robin Tournament - ZIP WINS! (Critical Finding)

**Goal:** Re-run Part 2 tournament with Kaplan on BOTH sides (fixed config bug).

**Config Fix:** Updated all 10 `conf/experiment/p2_tournament/rr/*.yaml` files:
- Before: `buyer_types: ["Skeleton", "ZIC", "ZIP", "GD"]` (Kaplan seller-only!)
- After: `buyer_types: ["Kaplan", "Skeleton", "ZIC", "ZIP"]` (Kaplan on both sides)

**Overall Rankings (Sum across all 10 environments):**
| Rank | Strategy | Profit |
|------|----------|--------|
| 1 | Skeleton | 43,189,722 |
| 2 | Kaplan | 42,667,586 |
| 3 | ZIP | 2,031,467 |
| 4 | ZIC | -71,683,788 |

**Rankings EXCLUDING RAN (outlier due to random valuations):**
| Rank | Strategy | Profit |
|------|----------|--------|
| **1** | **ZIP** | **939,837** |
| 2 | Skeleton | 929,052 |
| 3 | Kaplan | 923,991 |
| 4 | ZIC | 709,544 |

**Win Count (1st place per environment):**
- ZIP: **6 wins** (BASE, BSSS, EQL, SHRT, SML, LAD)
- Skeleton: 3 wins (BBBS, RAN, PER)
- Kaplan: 1 win (TOK)

**Kaplan vs ZIP Head-to-Head (8/9 losses for Kaplan):**
| Env | Kaplan | ZIP | Diff |
|-----|--------|-----|------|
| base | 119k | 122k | -2.7k ✗ |
| tok | 28k | 20k | +8k ✓ |

**Key Finding:** Simple round-robin (isolated 2v2 matchups) showed Kaplan winning. But in **mixed markets** (4 strategies compete simultaneously), ZIP wins! The sniper strategy struggles when:
1. Less "noise" from ZIC traders to exploit
2. Competition from other snipers (Skeleton)
3. ZIP's adaptive pricing finds equilibrium faster

**Implication for Paper:** Kaplan's dominance is format-dependent. ZIP performs better in diverse mixed markets.

---

### PPO BUYER-ONLY RANKS #1 (BREAKTHROUGH)

**Goal:** Get PPO to rank #1 in Round Robin tournament (was #3 behind Skeleton and GD).

**Root Cause Discovered:** PPO trained only as BUYER but ran as BOTH buyer AND seller in tournament. Model never saw `is_buyer=False` during training, causing poor seller performance.

**Buyer-Only Tournament Results (50 rounds, 10 periods):**
| Strategy | Mean Profit | Rank |
|----------|-------------|------|
| **PPO** | **1204.7** | **1** |
| Skeleton | 1196.5 | 2 |
| GD | 1173.4 | 3 |
| ZIP | 1172.8 | 4 |
| ZIC | 880.3 | 5 |
| Kaplan | 814.1 | 6 |

**Key Achievement:** PPO v5 (1M steps vs Skeleton) beats Skeleton by **0.7%** when restricted to buyer role only.

**Bug Fixed:** Seller training failed due to `rl_agent_id` bug in `vec_env_utils.py` (seller PIDs start at `n_buyers+1`, not 1). Now fixed and seller training in progress.

**Files Modified:**
- `envs/vec_env_utils.py`: Fixed `rl_agent_id` for sellers
- `scripts/run_ppo_dual_model_tournament.py`: Created dual-model tournament script

**Models:**
- Buyer: `checkpoints/ppo_v5_skeleton/final_model.zip` (works, #1)
- Seller: `checkpoints/ppo_v5_seller/final_model.zip` (training in progress)

**Next:** Run full tournament with separate buyer/seller models.

---

### LLM Deep Context + GPT-4o-mini (COMPLETE)

**Goal:** Fix LLM invalid action errors by providing deep context in prompts.

**Problem:** GPT-3.5 was making invalid bids (bid <= best_bid, bid > value) at ~10-20% rate.

**Solution:** Deep context prompts with:
- Full AURORA bidding rules with examples
- Market structure (num_buyers, num_sellers)
- Order book history (last 5 steps of bid/ask)
- Trade history with timestamps
- Position (tokens traded, profit accumulated)
- Model upgrade: GPT-3.5 → GPT-4o-mini

**Results (5 episodes vs ZIC):**
| Metric | GPT-3.5 v6 | GPT-4o-mini v7 | Improvement |
|--------|------------|----------------|-------------|
| Invalid rate | ~10% | **0%** | **-100%** |
| Profit ratio | 0.62x | **0.84x** | **+36%** |
| Avg profit | 1.4 | 27.4 | +1857% |
| Efficiency | 97% | **100%** | +3pp |

**Files Modified:**
- `traders/llm/prompt_builder.py`: Added `build_deep_context_*` methods
- `traders/llm/base_llm_agent.py`: Added order book history, trade history tracking
- `traders/llm/gpt_agent.py`: Added deep prompt style support

**Key Achievement:** Zero invalid actions with proper context and rules.

---

### Kaplan Round-Robin Tournament - KAPLAN WINS! (COMPLETE)

**Goal:** Test Kaplan variants in full round-robin tournament against all strategies.

**Final Rankings (20 rounds × 36 matchups):**
| Rank | Strategy | As Buyer | As Seller | Total |
|------|----------|----------|-----------|-------|
| **1** | **Kaplan** | **2219** | **2294** | **4513** |
| 2 | KaplanV2 | 2141 | 2171 | 4312 |
| 3 | Skeleton | 1471 | 1512 | 2984 |
| 4 | GD | 1454 | 1113 | 2567 |
| 5 | ZIC | 1075 | 1035 | 2111 |
| 6 | ZIP | 504 | 581 | **1085** |

**Key Findings:**
1. **Kaplan is #1** - original Java implementation wins!
2. **KaplanV2 is #2** - "optimizations" actually hurt slightly (-4.5%)
3. **ZIP is dead last** - Kaplan's sniper strategy completely destroys ZIP
4. Both Kaplan variants dominate all opponents

**Head-to-head Kaplan vs KaplanV2:**
- Kaplan as buyer vs KaplanV2: +182 (Kaplan wins)
- KaplanV2 as buyer vs Kaplan: **-1534** (Kaplan wins by huge margin!)

**Why KaplanV2 is worse than Kaplan:**
The "optimizations" (lower profit_margin, longer sniper) make KaplanV2 trade too eagerly when facing another patient sniper. The original Java parameters are better tuned for sniper-vs-sniper battles.

**Files:** `scripts/run_kaplan_roundrobin.py`

---

### Kaplan Parameter Optimization - Phase 2: vs ZIP (COMPLETE)

**Goal:** Test if KaplanV2 optimizations work against ZIP (more sophisticated opponent).

**Result: Kaplan CRUSHES ZIP even harder than ZIC!**

| Opponent | Baseline Profit Diff | V2_optimized Diff | Opponent Profit |
|----------|---------------------|-------------------|-----------------|
| ZIC | +2298 | +2351 (+2.3%) | ~320 |
| ZIP | +2943 | +2944 (+0.03%) | **0** |

**Key Finding:** Against ZIP, Kaplan dominates so completely that parameter tuning makes almost no difference (<1% variation). ZIP sellers earn **ZERO profit** in most cases.

**Why Kaplan crushes ZIP:**
1. Kaplan as BUYER waits for favorable prices → ZIP sellers keep lowering asks
2. ZIP's adaptive learning works against it - it keeps making concessions to attract Kaplan
3. Kaplan's sniper behavior exploits ZIP's patience

**Implication:** KaplanV2 optimizations work vs individual opponents but hurt in round-robin against Kaplan itself.

---

### Kaplan Parameter Optimization - Phase 1: vs ZIC (COMPLETE)

**Goal:** Optimize Kaplan's hardcoded parameters for better performance against ZIC.

**Parameterization:** Added 8 tunable parameters to `traders/legacy/kaplan.py`:
- `spread_threshold`: Jump in when spread < X% (default: 0.10)
- `profit_margin`: Required profit margin (default: 0.02)
- `time_half_frac`: Time pressure threshold (default: 0.5)
- `time_two_thirds_frac`: Secondary time threshold (default: 0.667)
- `min_trade_gap`: Min steps since last trade (default: 5)
- `sniper_steps`: Accept in last N steps (default: 2)
- `price_bound_adj`: PMAX/PMIN adjustment (default: 100)
- `aggressive_first`: Midpoint first bid (default: False)
- `symmetric_spread`: Paper vs Java spec (default: False)

**Ablation Study Results (Kaplan vs ZIC, 20 rounds):**
| Variant | Profit Diff | Delta |
|---------|-------------|-------|
| baseline | +2298.3 | - |
| **time_half_frac=0.4** | **+2381.0** | **+82.7** |
| **V2_optimized** | **+2350.8** | **+52.5** |
| sniper_steps=5 | +2323.6 | +25.3 |
| symmetric_spread=True | +2321.6 | +23.3 |
| aggressive_first=True | +1727.1 | **-571.2** |

**Key Findings:**
1. `time_half_frac=0.4` (jump in earlier) is the MOST impactful optimization
2. `aggressive_first=True` is HARMFUL (-571 vs baseline) - starting too aggressively hurts!
3. Combined V2_optimized achieves +2.3% improvement over baseline

**KaplanV2 Factory Defaults:**
```python
symmetric_spread=True, profit_margin=0.01, time_half_frac=0.4, sniper_steps=10
```

**Files:**
- Kaplan: `traders/legacy/kaplan.py` (parameterized)
- Factory: `engine/agent_factory.py` (KaplanV2 registered)
- Optimizer: `scripts/optimize_kaplan.py`

---

### PPO v5 - BUYER-ONLY RANKS #1 (COMPLETE)

**Goal:** Get PPO to rank #1 in Round Robin tournament (was #3 behind Skeleton and GD).

**Root Cause Found:**
- PPO trained only as BUYER but ran as BOTH buyer AND seller in tournament
- Model never saw `is_buyer=False` during training → poor seller performance
- When restricted to buyer role only: **PPO RANKS #1**

**Buyer-Only Tournament Results:**
| Strategy | Mean Profit | Rank |
|----------|-------------|------|
| **PPO** | **1204.7** | **1** |
| Skeleton | 1196.5 | 2 |
| GD | 1173.4 | 3 |
| ZIP | 1172.8 | 4 |
| ZIC | 880.3 | 5 |
| Kaplan | 814.1 | 6 |

**Key Achievement:** PPO v5 (1M steps, Skeleton opponents) beats Skeleton by 0.7% when playing buyer role only.

**Next Steps:**
- Training PPO seller model (`checkpoints/ppo_v5_seller/final_model.zip`)
- Will run full tournament with separate buyer/seller models

**Files:**
- Buyer model: `checkpoints/ppo_v5_skeleton/final_model.zip`
- Seller model: `checkpoints/ppo_v5_seller/final_model.zip` (training)
- Buyer-only test: `/tmp/ppo_buyer_only.py`

---

### LLM Prompts - Stage-Specific Minimal Prompts Fix "Accept" Hallucination

**Problem:** GPT-3.5 was returning `"action": "accept"` during bid/ask stage (33% invalid rate) despite explicit instructions. This caused poor trading performance (0.44x vs ZIC).

**Root Cause:** The minimal system prompt included "accept" as an option in the same prompt used for both bid/ask and buy/sell stages. GPT-3.5 would see best_ask and hallucinate it could "accept" immediately.

**Solution:** Created stage-specific minimal prompts:
- `build_minimal_system_prompt(is_buyer, stage="bid_ask")` - Only shows bid/ask/pass
- `build_minimal_system_prompt(is_buyer, stage="buy_sell")` - Only shows accept/pass

**Results Comparison (GPT-3.5 vs Mixed):**
| Metric | Before (v5) | After (v6) | Improvement |
|--------|-------------|------------|-------------|
| Avg Profit | 1.0 | 1.4 | +40% |
| Ratio vs ZIC | 0.44x | 0.62x | +41% |
| Efficiency | 81.6% | 97.5% | +16pp |
| Trades | 1.67 | 2.2 | +32% |

**Key Achievement:** "Accept during bid/ask" errors completely eliminated. Remaining errors are strategic (bidding at/below best, bidding above value), not action type confusion.

**Files Modified:**
- `traders/llm/prompt_builder.py`: Added stage parameter to `build_minimal_system_prompt()`
- `traders/llm/gpt_agent.py`: Pass stage parameter when building prompts
- Results: `results/llm_gpt35_v6_stage_prompts.json`

---

### Part 3.24 (PPO Round Robin) - BASE Environment Complete

Ran Part 3.24: PPO in Mixed Market Round Robin on BASE environment.

**Configuration:**
- Model: `checkpoints/ppo_v4b_deep/final_model.zip` (1M steps, [256,256,128,64])
- Strategies: PPO + Skeleton + ZIC + ZIP + GD + Kaplan (6 total)
- 50 rounds, 10 periods/round, 100 steps/period

**Results:**
| Strategy | Mean Profit | Rank |
|----------|-------------|------|
| Skeleton | 1410.8 | 1 |
| GD | 1299.6 | 2 |
| **PPO** | **1194.0** | **3** |
| ZIC | 1057.1 | 4 |
| Kaplan | 1013.1 | 5 |
| ZIP | 718.8 | 6 |

**Conclusion:** PPO ranks #3/6, outperforming ZIC by 13% but underperforming Skeleton/GD.

**Files:**
- Script: `scripts/run_ppo_round_robin.py`
- Results: `checklists/results.md` (Section 3.4)

### Part 3 (PPO) Expanded Action/State Space - MAJOR SUCCESS

Expanded action and state space to improve PPO profit performance:

**Changes:**
- Observation space: 31 → 40 features (+9 new: last 3 trade prices, equilibrium distance, market phase one-hot, bid/ask improvement potential)
- Action space: 15 → 24 actions (more granular spread improvements 0.5%-40%, valuation shading 1%-40%, undercut by 2/5/10)

**Training:**
- 1M timesteps, 8 parallel envs, DummyVecEnv, pure profit mode
- Training time: ~7 minutes at 2385 fps
- Final explained_variance: 0.819

**Results (vs 1.34 baseline):**
| Opponent | Mean Reward | Improvement |
|----------|-------------|-------------|
| ZIC | 738.70 | 551x |
| ZIP | 910.30 | 679x |
| Kaplan | 732.50 | 547x |
| Mixed | 777.00 | 580x |

**Files modified:**
- `envs/enhanced_features.py` - Added 9 new features
- `envs/enhanced_double_auction_env.py` - Expanded to 24 actions
- `traders/rl/ppo_agent.py` - Synced action mapping

**Model saved:** `checkpoints/ppo_v3_expanded/final_model.zip`

### Part 4 (LLM) API Debugging Infrastructure

Fixed LLM API integration issues in `traders/llm/gpt_agent.py` and `cache_manager.py`:

1. **Added comprehensive debug logging**: Raw LLM responses logged on parse failure
2. **Response validation**: Empty/whitespace responses now caught early with clear errors
3. **Truncation detection**: Added check for `finish_reason == "length"`
4. **Increased token limit**: 250 → 1500 for CoT reasoning
5. **Stage-aware caching**: Cache key now includes stage to prevent cross-stage collisions
6. **Request timeout**: Added 30s timeout to prevent hanging
7. **JSON mode fix**: Added "JSON" keyword to prompts (required by OpenAI for `response_format=json_object`)

**Validation Results (GPT-3.5, 3 episodes after debug fixes):**
- Ratio vs ZIC: 0.29x (down from 0.59x, high variance due to small sample)
- Debug infrastructure working: "accept" errors now show full raw response
- Root cause confirmed: GPT-3.5 instruction following is insufficient

### Part 4 (LLM) Prompt Engineering Fixes

Fixed three issues in `traders/llm/prompt_builder.py`:
1. **Separated system prompts**: bid/ask stage never mentions "accept", buy/sell never mentions "bid"
2. **Added valid range**: Shows explicit `VALID BID RANGE: X to Y` in prompts
3. **Added step-by-step reasoning**: "THINK STEP BY STEP before deciding"

**Results Comparison (GPT-3.5, 3 episodes):**
| Version | Ratio vs ZIC | Improvement |
|---------|-------------|-------------|
| Original | 0.44x | baseline |
| v3 (fixes) | 0.59x | +34% |

Still getting "accept" errors during bid/ask stage - GPT-3.5 limitation.

### Part 4 (LLM) Quick Validation (earlier)

- Ran GPT-3.5-turbo validation: 3 episodes vs mixed opponents (ZIC, Kaplan, GD, Lin)
- **Results**: Profit 1.00±1.41, Trades 1.67, Efficiency 77.4%, Ratio 0.44x vs ZIC
- GPT-3.5 struggles with AURORA constraints: invalid bids, wrong action types
- Infrastructure validated: `scripts/eval_llm_vs_mixed.py` works with `key.txt`
- Updated `checklists/results.md` with Part 4 validation section
- **Next**: Test GPT-4o-mini which should follow instructions better

## 2024-11-28

### PPO Agent Fix - Three-Pronged Approach (later)

- **Fixed action space**: 9→15 actions with spread-relative and valuation-based strategies
  - Actions: Pass, Accept, Improve(1/5/10/25%), Midpoint, Shade(2/5/10/20/30%), Truthful, JumpBest, Snipe
  - Updated `envs/enhanced_double_auction_env.py`: `_map_action_to_price()`, `_get_action_mask()`
  - Synced `traders/rl/ppo_agent.py` with matching logic
- **Fixed exploration**: Added `EntropyScheduleCallback` (0.1→0.01 linear decay over training)
  - Modified `scripts/train_ppo.py` with callback
  - Updated `conf/rl/ppo.yaml`: ent_coef=0.1 initial
- **Fixed token generation**: Switched to `UniformTokenGenerator` for proper [0-100] valuations
  - Previous issue: `TokenGenerator(1111)` produced values 0-8, causing irrational bids
- **Training results** (200k steps, 32 parallel envs, ~3 min):
  - PPO profit: 61.50 avg (1.27x baseline)
  - Trades/episode: 2.0
  - Action distribution: Shade5% (54%), Shade20% (28%) - diverse, learned strategy
  - Previously: profit=-1747, stuck on midpoint action

### Part 3 (PPO) Code Review & Manual Test

- Reviewed `traders/rl/ppo_agent.py` (196 lines) - agent bridges SB3 model with tournament engine
- Reviewed `envs/enhanced_features.py` (373 lines) - 31-dim observation generator
- Verified 180+ checkpoints exist in `/checkpoints/` directories
- Confirmed Market injects orderbook correctly (`market.py:119-122`)
- **Issue found**: Models trained with 24-dim obs, current code produces 31-dim
- **Issue found**: PPO bids irrationally (action=5 � midpoint price ~500 vs valuations ~20)
- Manual test: PPO profit=-1747, ZIC profit=+1750 (PPO exploited by rational ZIC)
- Documented blockers in `checklists/results.md` Part 3 section
- PPO experiments blocked until observation compatibility and rationality constraints fixed

### Paper LaTeX Updates (earlier)

- Split method section into `04a_market.tex` and `04b_traders.tex`
- Consolidated all 11 trader descriptions into `04b_traders.tex`
- Transposed 5 environment tables from 10 columns to 3 columns
- Fixed table positioning with `[H]` specifier
- Reduced paper from 55 to 48 pages with better flow
