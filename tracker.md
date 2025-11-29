# Development Tracker

## 2025-11-29

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
