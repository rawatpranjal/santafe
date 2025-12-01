# Behavioral Metrics Framework

Standard metrics for comparing trading strategy behaviors (PPO, LLM, Kaplan, etc.).

## Core Metrics

### 1. Action Distribution
Percentage breakdown of actions taken:
- **PASS**: No bid/ask submitted
- **Shade**: Bid below valuation (buyer) or ask above cost (seller)
- **Accept**: Accept standing price
- **Jump**: Improve price to narrow spread

### 2. Trade Timing
When trades occur within the 100-step period:
- **Mean trade time**: Average step at which trades complete
- **Early (t<30)**: Percentage of trades in first 30 steps
- **Mid (30-70)**: Percentage of trades in middle 40 steps
- **Late (t>=70)**: Percentage of trades in final 30 steps

### 3. Bid Shading
How aggressively prices are shaded from valuation:
- **Mean shade %**: Average (valuation - bid) / valuation
- **Distribution**: Breakdown by shade range (0-5%, 5-10%, 10-20%, etc.)

### 4. Profit Metrics
- **Total profit**: Sum of (valuation - trade_price) for buyers
- **Profit per trade**: Average profit per completed trade

## Strategy Profiles

### Classical Strategies (Measured: 5 seeds x 5 periods = 25 periods each)

| Strategy | Dominant Action | Mean Trade Time | Early % | Mid % | Late % | Profit/Trade |
|----------|-----------------|-----------------|---------|-------|--------|--------------|
| ZI | PASS (90.9%) | 5.6 | 100.0% | 0.0% | 0.0% | -157.4 |
| ZIC | Shade (50.6%) | 24.2 | 72.5% | 21.6% | 5.9% | 50.9 |
| ZIP | JUMP (37.0%) | 8.3 | 93.7% | 4.8% | 1.6% | 41.9 |
| GD | JUMP (42.1%) | 12.9 | 90.5% | 9.5% | 0.0% | 37.6 |
| Kaplan | PASS (68%) | 50.0 | 12.5% | 70.8% | 16.7% | 54.0 |
| Skeleton | JUMP (36.6%) | 14.4 | 90.0% | 10.0% | 0.0% | 58.1 |
| BGAN | Shade (35.0%) | 70.1 | 16.7% | 14.3% | 69.0% | 66.3 |
| Staecker | PASS (96.7%) | 28.6 | 59.6% | 34.6% | 5.8% | 51.7 |

### Modern AI Strategies

| Strategy | Dominant Action | Mean Trade Time | Early % | Mid % | Late % | Profit/Trade |
|----------|-----------------|-----------------|---------|-------|--------|--------------|
| PPO | Shade (92%) | 7.8 | 98.2% | 1.8% | 0.0% | 39.9 |
| GPT-4.1-mini | Shade (~70%) | ~15 | ~80% | ~15% | ~5% | 1.35x* |
| GPT-4.1 | Anchor ($1) | ~10 | ~90% | ~10% | 0% | 0.93x* |
| GPT-4o | Shade (~60%) | ~20 | ~70% | ~25% | ~5% | 0.85x* |
| GPT-4o-mini | Value (~80%) | ~5 | ~95% | ~5% | 0% | 0.75x* |
| o4-mini | PASS (~60%) | ~40 | ~20% | ~60% | ~20% | 0.51x* |
| GPT-3.5 | Value/Above (~90%) | ~3 | ~98% | ~2% | 0% | 0.59x* |

*LLM profit shown as ratio to opponent average profit (1.0 = break-even with ZIC/ZIP)

## Strategy Archetypes

**ZI (Unconstrained Random)**: Bids randomly in full price range regardless of valuation. Loses money on most trades. Baseline showing that budget constraints are essential for market efficiency.

**ZIC (Constrained Random)**: Uniform random bids within budget constraint. Shades ~31% on average, trades across all time periods. Provides baseline for market efficiency (~98.7%).

**ZIP (Adaptive Learner)**: Learns optimal shading through price observation. Dominant action is JUMP (37%) - aggressively narrows spread. Trades early (93.7%) with minimal shading (1.7%).

**GD (Belief-Based Optimizer)**: Forms probabilistic beliefs from market history. Dominant action is JUMP (42%) - belief-based spread narrowing. Trades early (90.5%) with moderate shading (17.7%).

**Kaplan (Parasitic Sniper)**: Waits for spread to narrow, trades mid-period after other traders reveal information. High PASS rate (68%), highest profit per trade (54.0).

**Skeleton (Simplified Kaplan)**: Stripped-down Kaplan without complex history tracking. Dominant action is JUMP (36.6%) rather than PASS. Trades early (90%) with moderate shading (9.6%). Highest profit per trade among non-Kaplan strategies (58.1).

**BGAN (Bayesian Game Against Nature)**: Belief-based optimizer from Kennet-Friedman 1993 entry. Maintains Normal beliefs over opponent prices and computes reservation prices via Monte Carlo. Very patient early (high option value of waiting), increasingly aggressive as time runs out. Trades very late (69% in final 30 steps), mean trade time 70.1. Highest profit per trade (66.3) due to waiting for best opportunities - but trades less frequently than aggressive strategies.

**Staecker (Predictive Strategy)**: Price-forecasting trader that uses exponential smoothing to predict next high bid and low ask. Trades when predicted prices make deals attractive. Very high PASS rate (96.7%) while building forecasts, but trades earlier than BGAN (mean time 28.6). Intermediate complexity between Skeleton/Kaplan and BGAN. Late-time override (>85%) relaxes prediction constraints. Profit per trade (51.7) competitive with Kaplan-family strategies.

**PPO (Preemptive Sniper)**: Trades immediately at period start with aggressive shading. More trades but lower profit per trade. First-mover advantage.

**GPT-4.1-mini (Profitable Moderate)**: Best-performing LLM. Hybrid approach with moderate shading (~30%) and early-mid trading. Only LLM to consistently beat ZIC baseline.

**GPT-4.1 (Extreme Anchor)**: Bids at $1 regardless of valuation. Extreme conservatism leads to missed opportunities despite occasional high-margin trades.

**GPT-4o (Conservative Bidder)**: Moderate shading but overly cautious. Misses profitable trades by waiting too long.

**GPT-4o-mini (Zero-Margin Bidder)**: Bids at or near valuation, extracting minimal profit per trade. Trades quickly but with no margin.

**o4-mini (Paralyzed Reasoner)**: High PASS rate similar to Kaplan but without the profit. Overthinks decisions and misses trading windows.

**GPT-3.5 (Hallucinator)**: Bids at or above valuation, often producing invalid bids. Fundamental misunderstanding of profit mechanics.

## Analysis Scripts

- `scripts/analyze_strategy_behavior.py` - **Unified script for any classical strategy** (ZI, ZIC, ZIP, GD, Kaplan, Skeleton, BGAN, Staecker)
- `scripts/analyze_kaplan_behavior.py` - Kaplan behavioral metrics (legacy)
- `scripts/analyze_ppo_behavior.py` - PPO behavioral metrics
- `scripts/analyze_llm_behavior.py` - LLM behavioral metrics (parses decision JSONs from llm_outputs/)

## Usage

Run behavioral analysis to compare strategies:
```bash
# Unified script for classical strategies
python scripts/analyze_strategy_behavior.py --strategy ZIC --seeds 5 --periods 5
python scripts/analyze_strategy_behavior.py --strategy GD --seeds 5 --periods 5
python scripts/analyze_strategy_behavior.py --strategy Kaplan --seeds 5 --periods 5
python scripts/analyze_strategy_behavior.py --strategy BGAN --seeds 5 --periods 5
python scripts/analyze_strategy_behavior.py --strategy Staecker --seeds 5 --periods 5

# PPO analysis
python scripts/analyze_ppo_behavior.py --seeds 5 --periods 5

# LLM analysis
python scripts/analyze_llm_behavior.py --model gpt-4.1-mini --seeds 5
```
