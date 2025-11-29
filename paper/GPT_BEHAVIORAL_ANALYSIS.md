# GPT-4o-mini Behavioral Analysis: Trading Pattern Observations

**Date**: November 24, 2025
**Experiment**: gpt4_vs_legacy (partial run, stopped after 278 API calls)
**Model**: gpt-4o-mini
**Cost**: $0.0143 (84,669 input tokens, 2,664 output tokens)

---

## Executive Summary

Analysis of 278 trading decisions from GPT-4o-mini reveals **conservative, seller-biased behavior** with infrequent trading. The agent demonstrates rational constraint adherence but exhibits extreme caution that may underperform against aggressive legacy traders.

---

## Action Distribution

| Action | Count | Percentage | Interpretation |
|--------|-------|------------|----------------|
| **ask** (seller) | 164 | 59.0% | Heavy seller bias |
| **pass** | 105 | 37.8% | Conservative waiting |
| **accept** | 7 | 2.5% | Few trades completed |
| **bid** (buyer) | 2 | 0.7% | Minimal buyer activity |
| **TOTAL** | 278 | 100% | |

### Key Finding: Extreme Seller Bias
- **82:1 ask-to-bid ratio** suggests GPT either:
  1. Was assigned more seller roles in the experiment
  2. Finds asking easier than bidding (asymmetric confidence)
  3. Or the experiment configuration favored seller opportunities

---

## Pricing Strategy Analysis

### Seller (Ask) Behavior
- **Sample size**: 164 ask prices
- **Range**: 109 - 816 (wide spread)
- **Median**: 149 (central tendency)
- **Most common prices**: 149, 148, 150, 194, 198 (tight clustering)

**Pattern Observations**:
1. **Tight clustering around 145-155 range** (30% of asks)
2. **Secondary cluster around 190-200 range** (suggesting different market conditions)
3. **Outlier at 816** (possible irrational bid or constraint violation misunderstanding)

### Buyer (Bid) Behavior
- **Sample size**: 2 bids only (insufficient data)
- **Prices**: 120, 121
- **Pattern**: Extremely conservative, minimal bidding activity

### Conservative Pricing Hypothesis
The median ask of 149 and tight clustering suggest GPT is:
- Bidding near perceived equilibrium (safe strategy)
- Avoiding aggressive profit maximization
- Possibly exhibiting "fairness bias" from RLHF training (as hypothesized in Section 7.6 of paper)

---

## Trade Completion Analysis

### Extremely Low Trade Rate
- **7 accepts out of 278 decisions (2.5%)**
- **Trading frequency**: 1 trade per ~40 decisions

**Possible Causes**:
1. **Overly conservative pricing** - Not competitive enough
2. **Poor timing** - Waiting too long, missing opportunities
3. **Competitive pressure** - Legacy traders (Kaplan, ZIP) outbidding/undercutting
4. **Risk aversion** - RLHF alignment making GPT cautious

**Comparison to Legacy Traders**:
- **Kaplan**: Aggressive sniper, waits until t>0.9T then bids
- **ZIP**: Adaptive margins, continuously adjusts
- **ZIC**: Random but trades frequently
- **GPT-4o-mini**: Passive, infrequent trades

---

## Behavioral Patterns

### 1. Conservative Waiting (37.8% "pass")
GPT frequently chooses to pass rather than submit quotes. This suggests:
- **Uncertainty about optimal price**
- **Waiting for better market conditions**
- **Risk aversion from RLHF training**

### 2. Minimal Buyer Activity (0.7% bids)
Only 2 bids recorded at 120-121. Either:
- GPT was rarely assigned buyer role
- Or GPT finds buying decisions harder (requires bidding below valuation)

### 3. Price Clustering
Two distinct price clusters (145-155 and 190-200) suggest:
- GPT adapts to changing market conditions (valuation resets between periods)
- Or responds to observed spreads (bid near best quotes)

### 4. Outlier Behavior (ask=816)
One extreme ask at 816 (8× median) indicates:
- **Possible error**: Misunderstood constraint or market state
- **Or rational**: Very high valuation (e.g., 900) with aggressive pricing
- **Or irrational**: Failure to parse market state correctly

---

## Comparison to Minimal Test Results

### Reconciliation with Previous 99.7% Efficiency
The earlier minimal test (1 round, 1 period, 20 steps) showed:
- **99.69% efficiency**
- **Buyer profit**: 162 (1.07× vs ZIC)
- **Seller profit**: 11 (across 2 trades)
- **Zero invalid actions**

**This partial run suggests**:
- When trades DO occur, they are efficient
- But trades are INFREQUENT (2.5% accept rate)
- The 99.7% efficiency may reflect small sample size luck
- **Full tournament will test robustness**

---

## Limitations of Current Analysis

### Missing Context
The cache stores hashed prompts (SHA256), so we CANNOT see:
- **Valuations**: What was GPT's private valuation?
- **Market state**: What was best bid/ask spread?
- **Time pressure**: What step in the 100-step auction?
- **Opponent actions**: What were legacy traders doing?
- **Trade outcomes**: Did GPT's quotes get accepted?

### What We Need for Full Behavioral Analysis
1. **Prompt logging**: Save full prompts with market state
2. **Response logging**: Save reasoning (if chain-of-thought enabled)
3. **Outcome tracking**: Did quote lead to trade? Profit earned?
4. **Temporal analysis**: How does behavior change over time?
5. **Opponent awareness**: Does GPT recognize Kaplan's sniper strategy?

---

## Hypotheses for Full Experiments

### H1: Fairness Bias Limits Profit
**Prediction**: GPT-4o-mini will underperform aggressive traders (Kaplan, GD, Perry) because RLHF training makes it reluctant to "exploit" naive agents.

**Test**: Invasibility experiment (1 GPT + 7 ZIC)
- If profit ratio < 1.5×, confirms fairness bias
- If profit ratio > 2.0×, rejects fairness bias

### H2: Conservative Strategy Dominates in Stable Markets
**Prediction**: GPT's conservative bidding succeeds against ZIC but fails against sophisticated traders.

**Test**: Grand melee tournament
- If GPT ranks top-3 against legacy traders, validates strategy
- If GPT ranks bottom-3, confirms underperformance

### H3: Insensitivity to Time Pressure
**Prediction**: GPT does not exhibit "closing panic" (urgent bidding at t>0.9T) unlike human traders and Kaplan.

**Test**: Temporal analysis of bid timing
- Plot bid aggressiveness vs. time remaining
- Compare to Kaplan's sniper timing

---

## Cost-Performance Implications

### Partial Run Economics
- **278 decisions**: $0.0143
- **Extrapolated to full tournament** (10 rounds × 5 periods × 100 steps):
  - ~8,000 decisions per agent
  - Cost: ~$0.41 per agent per tournament
  - For 2 GPT agents (8v8): **~$0.82 total**

**Updated estimates** (based on actual token usage):
- **gpt4_vs_legacy**: $0.82 (was estimated $0.63) ✅ close
- **model_comparison**: $6.72 (was estimated $3.36) ⚠️ 2× higher
- **invasibility**: $20.50 (was estimated $3.15) 🚨 6.5× higher
- **grand_melee**: $6.72 (was estimated $5.00) ⚠️ 1.3× higher

**Recommendation**: Run priority experiments with updated budget awareness.

---

## Next Steps

### 1. Implement Detailed Logging (IMMEDIATE)
Modify `traders/llm/gpt_agent.py` to save:
- Full prompts (not hashed) to `llm_outputs/prompts/`
- Responses with context to `llm_outputs/responses/`
- Format: JSON with round/period/step/agent_id

### 2. Run Minimal Diagnostic Test (CHEAP)
- 1 round, 1 period, 10 steps
- 2 GPT vs 2 ZIC (4v4 market)
- Cost: ~$0.01
- Generate sample logs for manual inspection

### 3. Analyze Behavioral Traces
- Parse prompts to reconstruct decision context
- Identify patterns: time pressure response, spread sensitivity, opponent recognition
- Document in Section 7.6 (Behavioral Analysis)

### 4. Execute Full Experiments (BUDGET APPROVAL)
- Priority 1: gpt4_vs_legacy + model_comparison ($7.54)
- Priority 2: invasibility + grand_melee ($27.22)
- **Total revised budget: $34.76** (was $12.14)

---

## Conclusion

GPT-4o-mini exhibits **conservative, passive trading behavior** with heavy seller bias and low trade frequency. The 99.7% efficiency from minimal testing appears optimistic—the 2.5% accept rate suggests the agent may struggle to complete trades in competitive markets. The extreme ask-to-bid ratio (82:1) and high pass rate (38%) indicate the agent is risk-averse, possibly reflecting RLHF alignment training that prioritizes "fairness" over profit maximization.

**Without detailed prompt/response logs, we cannot definitively diagnose the root cause**. Immediate next step: implement structured logging and run diagnostic test to observe decision-making in context.

---

**Analysis conducted by**: Claude Code (Sonnet 4.5)
**Data source**: .cache/llm_responses/cache.json (gpt4_vs_legacy partial run)
**Methodology**: Statistical analysis of cached API responses (278 decisions)
