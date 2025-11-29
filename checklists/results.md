# Santa Fe Tournament - Experimental Results

> **Purpose:** Track experimental results as experiments complete.
> Structure mirrors `paper.md` experimental design.

---

## Environments Reference

| Code | Name | Key Variation |
|------|------|---------------|
| BASE | Standard | 4B/4S, 4 tokens (baseline) |
| BBBS | Buyer-dominated | 6B/2S (duopsony) |
| BSSS | Seller-dominated | 2B/6S (duopoly) |
| EQL | Equal Endowment | gametype=0 (symmetric token values) |
| RAN | Random | IID uniform token draws (R1=R2=R3=0) |
| PER | Single Period | 1 period per round |
| SHRT | Short/High Pressure | 20 steps per period |
| TOK | Single Token | 1 token per trader |
| SML | Small Market | 2B/2S, gametype=0007 |
| LAD | Low Adaptivity | Same as BASE |

---

## Part 1: Foundational Replication

> References: Smith (1962), Gode & Sunder (1993), Cliff & Bruten (1997)

### Configuration
- 4 tokens per trader
- 100 steps per period (except SHRT: 20 steps)
- 10 periods per round
- Multiple rounds for statistical significance

### Table 1.1: Efficiency Matrix (Trader × Environment)

| Trader | BASE | BBBS | BSSS | EQL | RAN | PER | SHRT | TOK | SML | LAD |
|--------|------|------|------|-----|-----|-----|------|-----|-----|-----|
| **ZI** (Exp 1.1-1.10) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **ZIC** (Exp 1.11-1.20) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **ZIP** (Exp 1.21-1.30) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

### Metrics
- Allocative efficiency (%)
- Price RMSD from equilibrium
- Price autocorrelation (lag-1)
- Trading volume distribution
- Profit dispersion

### Outputs
- [ ] Table 1.1: Efficiency matrix completed
- [ ] Figure 1.1: Efficiency by environment (grouped bar chart)
- [ ] Figure 1.2: Price convergence comparison
- [ ] Figure 1.3: Efficiency distribution (box plots by trader type)

---

## Part 2: Santa Fe Tournament Replication

> Reference: Rust et al. (1994)
> Strategies: Skeleton, ZIC, ZIP, GD, Kaplan

### Configuration
- 4 buyers, 4 sellers (except environment-specific)
- 4 tokens per trader
- 100 steps per period
- 10 periods per round

### 2.1 Against Control (1 Strategy vs 7 ZIC)

| Strategy | BASE | BBBS | BSSS | EQL | RAN | PER | SHRT | TOK | SML | LAD |
|----------|------|------|------|-----|-----|-----|------|-----|-----|-----|
| **Skeleton** (Exp 2.1-10) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **ZIP** (Exp 2.11-20) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **GD** (Exp 2.21-30) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **Kaplan** (Exp 2.31-40) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

### 2.2 Self-Play (All 8 Traders Same Type)

| Strategy | BASE | BBBS | BSSS | EQL | RAN | PER | SHRT | TOK | SML | LAD |
|----------|------|------|------|-----|-----|-----|------|-----|-----|-----|
| **Skeleton** (Exp 2.41-50) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **ZIC** (Exp 2.51-60) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **ZIP** (Exp 2.61-70) | **97.4%** ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **GD** (Exp 2.71-80) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **Kaplan** (Exp 2.81-90) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

#### ZIP Selfplay Details (Exp 2.61)
- **Config**: 8×8 ZIP traders, gametype 6453, 100 rounds, 10 periods
- **Hyperparameters**: β=0.01, γ=0.008, margin∈[0, 0.003] (v4 calibration)
- **Results**:
  - Mean efficiency: **97.4%**
  - Period 1: 99.5% | Period 10: 95.5%
  - Min: 75.1% | Max: 100.0%
- **Date**: 2025-11-28

### 2.3 Pairwise Experiments (4v4 Mixed Markets)

#### ZIP vs ZI (Exp 2.91)
- **Config**: 4 ZIP + 4 ZI per side, 100 rounds, 10 periods
- **Market Efficiency**: 32-43% (ZI causes massive EM-inefficiency)
- **Results**:

| Agent | Total Profit | Avg/Period | Trades/Period |
|-------|-------------|------------|---------------|
| **ZIP** | **+1,967,782** | +246 | 1.2 |
| ZI | -1,385,390 | -173 | 4.0 |

- **Outcome**: ZIP completely dominates. ZI loses money trading randomly.
- **Date**: 2025-11-28

#### ZIP vs ZIC (Exp 2.92)
- **Config**: 4 ZIP + 4 ZIC per side, 100 rounds, 10 periods
- **Market Efficiency**: ~97%
- **Results**:

| Agent | Total Profit | Avg/Period | Trades/Period |
|-------|-------------|------------|---------------|
| **ZIP** | **+934,263** | +117 | 2.3 |
| ZIC | +742,447 | +93 | 1.9 |

- **Outcome**: ZIP earns **25.8% more** profit than ZIC.
- **Date**: 2025-11-28

#### ZIC vs ZI (Exp 2.93)
- **Config**: 4 ZIC + 4 ZI per side, 100 rounds, 10 periods
- **Market Efficiency**: 44.8% (ZI causes massive EM-inefficiency)
- **Results**:

| Agent | Total Profit | Avg/Period | Trades/Period |
|-------|-------------|------------|---------------|
| **ZIC** | **+1,975,372** | +247 | 2.1 |
| ZI | -1,154,671 | -144 | 4.0 |

- **Outcome**: ZIC completely dominates. ZI loses money trading randomly (no budget constraint).
- **Date**: 2025-11-28

#### Pairwise Metrics Summary

| Metric | ZIP vs ZI | ZIP vs ZIC | ZIC vs ZI |
|--------|-----------|------------|-----------|
| **Efficiency (mean)** | 36.0% | 96.9% | 44.8% |
| **Efficiency (std)** | 33.2% | 3.2% | 38.5% |
| **Price Volatility %** | 65.1% | 10.4% | — |
| **EM-Inefficiency** | 655.3 | 0.0 | — |
| **V-Inefficiency** | 1.08 | 0.48 | — |
| **Profit Dispersion (RMS)** | 574.4 | 61.8 | — |
| **Trades/Period** | 21.0 | 16.7 | 24.3 |

**Key Observations:**
1. **ZI destroys efficiency** - When ZI is present, efficiency drops to 36-45% with massive variance
2. **EM-inefficiency** - ZI causes 655 units of extramarginal trading (bad trades that shouldn't happen)
3. **Price volatility** - ZI causes 65% price volatility vs 10% in ZIP-ZIC markets
4. **Profit dispersion** - ZI creates 10x more profit inequality (RMS 574 vs 62)
5. **ZI overtrades** - ZI makes 4 trades/period (random), ZIC/ZIP make 1.2-2.3 (selective)

**Hierarchy Confirmed: ZIP > ZIC > ZI**

### 2.4 ZIP Hyperparameter Tuning (Exp 2.94-2.97)

**Goal**: Address divergences from Cliff & Bruten (1997) claims:
- V-Inefficiency: ZIP 2.51 > ZIC 0.73 (ZIP should have fewer missed trades)
- Price Volatility: ZIP 11.3% > ZIC 8.4% (ZIP should be less volatile)
- Profit Dispersion: ZIP ≈ ZIC (ZIP should be better)

**Methodology**: 100 rounds × 10 periods × 4 configurations (8×8 ZIP selfplay)

#### Results Table

| Config | β (learning) | γ (momentum) | Eff% | Eff_std | V-Ineff | Vol% | ProfDisp | AvgProfit |
|--------|--------------|--------------|------|---------|---------|------|----------|-----------|
| A_high_eff | 0.05 | 0.02 | 93.8 | 8.7 | 1.62 | **11.1** | **73.8** | 105.2 |
| **B_low_vol** | 0.005 | 0.10 | **99.6** | **1.1** | **0.41** | 13.4 | 78.1 | **112.1** |
| C_balanced | 0.02 | 0.03 | 97.9 | 3.9 | 0.99 | 12.4 | 74.8 | 109.7 |
| D_baseline | 0.01 | 0.008 | 99.0 | 2.4 | 0.69 | 12.9 | 76.3 | 111.3 |

#### Key Findings

1. **Winner: B_low_vol (β=0.005, γ=0.10)**
   - V-Inefficiency: 0.41 — **BEATS ZIC's 0.73!** ✅
   - Efficiency: 99.6% — Best of all configs
   - Consistency: Std dev 1.1% (lowest variance)
   - Profits: 112.1/agent (highest)

2. **Trade-off Discovery**:
   - Slower learning (lower β) + higher momentum (higher γ) = better performance
   - Lower β prevents overshoot; higher γ smooths learning trajectory
   - Counterintuitive: "slow and steady" beats "fast learning"

3. **Remaining Divergence**:
   - Volatility 13.4% still > ZIC 8.4%
   - Config A achieves lowest volatility (11.1%) but sacrifices efficiency (93.8%)
   - **Volatility appears fundamental** — ZIP's adaptive learning inherently creates price exploration noise

#### Comparison to Paper Claims

| Metric | Cliff & Bruten Claim | Our Baseline | Our Tuned (B) | Status |
|--------|---------------------|--------------|---------------|--------|
| Efficiency | ZIP ≥ ZIC | 99.0% | 99.6% | ✅ Matches |
| V-Inefficiency | ZIP < ZIC | 0.69 < 0.73 | 0.41 < 0.73 | ✅ **Now beats ZIC** |
| Volatility | ZIP ≤ ZIC | 12.9% > 8.4% | 13.4% > 8.4% | ❌ Still diverges |
| Profit Dispersion | ZIP < ZIC | 76.3 ≈ ZIC | 78.1 ≈ ZIC | ~ Neutral |

**Recommendation**: Adopt B_low_vol (β=0.005, γ=0.10) as new ZIP default.

- **Date**: 2025-11-28

### 2.5 Individual Profit Analysis (ZIP vs ZIC)

**Config**: 4 ZIP + 4 ZIC per side, 50 rounds, 10 periods

#### Per-Agent Breakdown

| Agent | Type | Role | Total Profit | Eq Profit | Deviation | Dev % |
|-------|------|------|--------------|-----------|-----------|-------|
| 1 | ZIP | Buyer | +62,907 | +55,042 | +7,865 | +14.3% |
| 2 | ZIP | Buyer | +60,411 | +53,828 | +6,583 | +12.2% |
| 3 | ZIC | Buyer | +45,223 | +56,742 | -11,519 | -20.3% |
| 4 | ZIC | Buyer | +44,254 | +54,530 | -10,276 | -18.8% |
| 5 | ZIP | Seller | +58,638 | +52,390 | +6,248 | +11.9% |
| 6 | ZIP | Seller | +54,902 | +52,835 | +2,067 | +3.9% |
| 7 | ZIC | Seller | +45,012 | +52,199 | -7,187 | -13.8% |
| 8 | ZIC | Seller | +45,796 | +53,274 | -7,478 | -14.0% |

#### Strategy Summary

| Type | Buyers | Sellers | Total | Advantage |
|------|--------|---------|-------|-----------|
| ZIC | +89,477 | +90,808 | +180,285 | |
| ZIP | +123,318 | +113,540 | +236,858 | **WINNER** |

#### Buyer vs Seller Split

| Side | Profit | Share |
|------|--------|-------|
| Buyers | +212,795 | 51.0% |
| Sellers | +204,348 | 49.0% |

#### Key Findings

1. **Overall Winner**: ZIP earns **31% more** than ZIC (+236k vs +180k)
2. **Buyer Side**: ZIP buyers earn +38% more than ZIC buyers
3. **Seller Side**: ZIP sellers earn +25% more than ZIC sellers
4. **Market Balance**: Buyers capture 51.0% of total surplus (roughly fair)
5. **Deviation Pattern**:
   - All ZIP agents over-earn vs equilibrium (+4% to +14%)
   - All ZIC agents under-earn vs equilibrium (-14% to -20%)

**Interpretation**: ZIP's adaptive learning allows it to extract more than its "fair share" of surplus from the market. ZIC's random pricing leaves value on the table.

- **Date**: 2025-11-28

---

### 2.6 Round Robin Tournament (Mixed Market)

| Environment | Exp # | Efficiency | Top Strategy | 2nd | 3rd | Status |
|-------------|-------|------------|--------------|-----|-----|--------|
| BASE | 2.101 | | | | | ⬜ |
| BBBS | 2.102 | | | | | ⬜ |
| BSSS | 2.103 | | | | | ⬜ |
| EQL | 2.104 | | | | | ⬜ |
| RAN | 2.105 | | | | | ⬜ |
| PER | 2.106 | | | | | ⬜ |
| SHRT | 2.107 | | | | | ⬜ |
| TOK | 2.108 | | | | | ⬜ |
| SML | 2.109 | | | | | ⬜ |
| LAD | 2.110 | | | | | ⬜ |

### Metrics
- Allocative efficiency (%)
- Individual trader efficiency ratios
- Price autocorrelation (lag-1)
- Trading volume by period %
- Bid-ask spread evolution
- Profit rankings

### Outputs
- [ ] Table 2.1: Against Control results (strategy vs 7 ZIC × environment)
- [ ] Table 2.2: Self-play efficiency matrix (5 strategies × 10 environments)
- [ ] Table 2.3: Round Robin tournament results
- [ ] Figure 2.1: Kaplan efficiency (mixed vs pure markets)
- [ ] Figure 2.2: Price autocorrelation by trader type
- [ ] Figure 2.3: Trading volume by period % (closing panic)
- [ ] Figure 2.4: Trader hierarchy chart

---

## Part 3: PPO RL Agents

> Reference: Chen et al. (2010)

### Configuration
- 7,000 trading periods (Chen protocol)
- 25 steps per period
- 4 tokens per trader

### 3.1 Training Curriculum

| Exp # | Opponent | Training Episodes | Final Ratio | Converged? | Status |
|-------|----------|-------------------|-------------|------------|--------|
| 3.1 | ZIC | | | | ⬜ |
| 3.2 | Skeleton | | | | ⬜ |
| 3.3 | Mixed | | | | ⬜ |

### 3.2 Against Control (PPO vs 7 ZIC) — Exp 3.4-3.13

| Environment | Exp # | PPO Ratio | Market Efficiency | Status |
|-------------|-------|-----------|-------------------|--------|
| BASE | 3.4 | | | ⬜ |
| BBBS | 3.5 | | | ⬜ |
| BSSS | 3.6 | | | ⬜ |
| EQL | 3.7 | | | ⬜ |
| RAN | 3.8 | | | ⬜ |
| PER | 3.9 | | | ⬜ |
| SHRT | 3.10 | | | ⬜ |
| TOK | 3.11 | | | ⬜ |
| SML | 3.12 | | | ⬜ |
| LAD | 3.13 | | | ⬜ |

### 3.3 Self-Play (PPO vs PPO) — Exp 3.14-3.23

| Environment | Exp # | Efficiency | Price RMSD | Autocorr | Status |
|-------------|-------|------------|------------|----------|--------|
| BASE | 3.14 | | | | ⬜ |
| BBBS | 3.15 | | | | ⬜ |
| BSSS | 3.16 | | | | ⬜ |
| EQL | 3.17 | | | | ⬜ |
| RAN | 3.18 | | | | ⬜ |
| PER | 3.19 | | | | ⬜ |
| SHRT | 3.20 | | | | ⬜ |
| TOK | 3.21 | | | | ⬜ |
| SML | 3.22 | | | | ⬜ |
| LAD | 3.23 | | | | ⬜ |

### 3.4 Round Robin (PPO in Mixed Market) — Exp 3.24-3.33

| Environment | Exp # | PPO Rank | Market Efficiency | PPO Ratio | Status |
|-------------|-------|----------|-------------------|-----------|--------|
| BASE | 3.24 | /8 | | | ⬜ |
| BBBS | 3.25 | /8 | | | ⬜ |
| BSSS | 3.26 | /8 | | | ⬜ |
| EQL | 3.27 | /8 | | | ⬜ |
| RAN | 3.28 | /8 | | | ⬜ |
| PER | 3.29 | /8 | | | ⬜ |
| SHRT | 3.30 | /8 | | | ⬜ |
| TOK | 3.31 | /8 | | | ⬜ |
| SML | 3.32 | /8 | | | ⬜ |
| LAD | 3.33 | /8 | | | ⬜ |

### Metrics
- PPO efficiency ratio
- Opponent efficiency ratio
- Market efficiency
- PPO profit rank
- Training curves

### Outputs
- [ ] Table 3.1: Against Control results (PPO vs 7 ZIC × environment)
- [ ] Table 3.2: Self-play results (PPO × environment)
- [ ] Table 3.3: Round Robin results (PPO in Mixed × environment)
- [ ] Figure 3.1: PPO training curves
- [ ] Figure 3.2: PPO vs legacy trader comparison

---

## Part 4: LLM Agents

> Zero-shot, no training

### Configuration
- Same as PPO (7,000 periods, 25 steps)
- Zero-shot (no training)

### 4.1 Against Control (LLM vs 7 ZIC) — Exp 4.1-4.10

| Environment | Exp # | LLM Ratio | Market Efficiency | Status |
|-------------|-------|-----------|-------------------|--------|
| BASE | 4.1 | | | ⬜ |
| BBBS | 4.2 | | | ⬜ |
| BSSS | 4.3 | | | ⬜ |
| EQL | 4.4 | | | ⬜ |
| RAN | 4.5 | | | ⬜ |
| PER | 4.6 | | | ⬜ |
| SHRT | 4.7 | | | ⬜ |
| TOK | 4.8 | | | ⬜ |
| SML | 4.9 | | | ⬜ |
| LAD | 4.10 | | | ⬜ |

### 4.2 Self-Play (LLM vs LLM) — Exp 4.11-4.20

| Environment | Exp # | Efficiency | Price RMSD | Autocorr | Status |
|-------------|-------|------------|------------|----------|--------|
| BASE | 4.11 | | | | ⬜ |
| BBBS | 4.12 | | | | ⬜ |
| BSSS | 4.13 | | | | ⬜ |
| EQL | 4.14 | | | | ⬜ |
| RAN | 4.15 | | | | ⬜ |
| PER | 4.16 | | | | ⬜ |
| SHRT | 4.17 | | | | ⬜ |
| TOK | 4.18 | | | | ⬜ |
| SML | 4.19 | | | | ⬜ |
| LAD | 4.20 | | | | ⬜ |

### 4.3 Round Robin (LLM in Mixed Market) — Exp 4.21-4.30

| Environment | Exp # | LLM Rank | Market Efficiency | LLM Ratio | Status |
|-------------|-------|----------|-------------------|-----------|--------|
| BASE | 4.21 | /8 | | | ⬜ |
| BBBS | 4.22 | /8 | | | ⬜ |
| BSSS | 4.23 | /8 | | | ⬜ |
| EQL | 4.24 | /8 | | | ⬜ |
| RAN | 4.25 | /8 | | | ⬜ |
| PER | 4.26 | /8 | | | ⬜ |
| SHRT | 4.27 | /8 | | | ⬜ |
| TOK | 4.28 | /8 | | | ⬜ |
| SML | 4.29 | /8 | | | ⬜ |
| LAD | 4.30 | /8 | | | ⬜ |

### 4.4 Cost Analysis — Exp 4.31

| Model | Avg Cost/Decision | Latency (ms) | Ratio vs ZIC | Status |
|-------|-------------------|--------------|--------------|--------|
| GPT-4 | | | | ⬜ |
| GPT-3.5 | | | | ⬜ |
| Claude | | | | ⬜ |

### Metrics
- LLM efficiency ratio
- Market efficiency
- LLM profit rank
- API cost per decision
- Latency per decision

### Outputs
- [ ] Table 4.1: Against Control results (LLM vs 7 ZIC × environment)
- [ ] Table 4.2: Self-play results (LLM × environment)
- [ ] Table 4.3: Round Robin results (LLM in Mixed × environment)
- [ ] Table 4.4: Cost-benefit summary
- [ ] Figure 4.1: LLM vs legacy trader comparison
- [ ] Figure 4.2: Cost vs performance scatter

---

## Output Artifacts Checklist

### Tables
- [ ] 1.1: Foundational efficiency matrix (ZI/ZIC/ZIP × 10 environments)
- [ ] 2.1: Against Control (strategy vs 7 ZIC × environment)
- [ ] 2.2: Self-play efficiency (5 strategies × 10 environments)
- [ ] 2.3: Round Robin tournament results
- [ ] 3.1: PPO Against Control (PPO vs 7 ZIC × environment)
- [ ] 3.2: PPO Self-play results
- [ ] 3.3: PPO Round Robin results
- [ ] 4.1: LLM Against Control (LLM vs 7 ZIC × environment)
- [ ] 4.2: LLM Self-play results
- [ ] 4.3: LLM Round Robin results
- [ ] 4.4: LLM cost-benefit summary

### Figures
- [ ] 1.1: Efficiency by environment (grouped bar)
- [ ] 1.2: Price convergence comparison
- [ ] 1.3: Efficiency distribution (box plots)
- [ ] 2.1: Kaplan mixed vs pure efficiency
- [ ] 2.2: Price autocorrelation
- [ ] 2.3: Trading volume (closing panic)
- [ ] 2.4: Trader hierarchy chart
- [ ] 3.1: PPO training curves
- [ ] 3.2: PPO vs legacy comparison
- [ ] 4.1: LLM vs legacy comparison
- [ ] 4.2: Cost vs performance scatter

### Statistical Tests
- [ ] Efficiency differences across trader types (ANOVA)
- [ ] Price autocorrelation significance
- [ ] AI vs legacy comparisons
- [ ] Confidence intervals

---

## Configuration Reference

### Part 1 & 2: Classical Experiments
```yaml
market:
  min_price: 1
  max_price: 1000
  num_tokens: 4
  gametype: 6453
  num_periods: 10
  num_steps: 100  # except SHRT: 20 steps
```

### Part 3 & 4: AI Experiments (Chen Protocol)
```yaml
market:
  min_price: 1
  max_price: 1000
  num_tokens: 4
  gametype: 6453
  num_periods: 7000  # trading periods
  num_steps: 25      # steps per period
```

---

## Experiment Count Summary

| Part | Section | Experiments | Status |
|------|---------|-------------|--------|
| 1 | Foundational | 30 (3 traders × 10 env) | ⬜ |
| 2.1 | Against Control | 40 (4 strategies × 10 env) | ⬜ |
| 2.2 | Self-Play | 50 (5 strategies × 10 env) | 1/50 ✅ (ZIP BASE) |
| 2.3 | Pairwise | 45 (all pairs) | 3/45 ✅ (ZIP vs ZI, ZIP vs ZIC, ZIC vs ZI) |
| 2.4 | Round Robin | 10 (10 env) | ⬜ |
| 3.1 | PPO Training | 3 | ⬜ |
| 3.2 | PPO Control | 10 | ⬜ |
| 3.3 | PPO Self-Play | 10 | ⬜ |
| 3.4 | PPO Round Robin | 10 | ⬜ |
| 4.1 | LLM Control | 10 | ⬜ |
| 4.2 | LLM Self-Play | 10 | ⬜ |
| 4.3 | LLM Round Robin | 10 | ⬜ |
| 4.4 | Cost Analysis | 1 | ⬜ |
| **Total** | | **174** | |

---

## Commands Reference

```bash
# Part 1: Foundational Replication (Exp 1.1-1.30)
python scripts/run_experiment.py experiment=part1_zi
python scripts/run_experiment.py experiment=part1_zic
python scripts/run_experiment.py experiment=part1_zip

# Part 2: Santa Fe Replication (Exp 2.1-2.100)
python scripts/run_experiment.py experiment=part2_control
python scripts/run_experiment.py experiment=part2_selfplay
python scripts/run_experiment.py experiment=part2_roundrobin

# Part 3: PPO (Exp 3.1-3.33)
python scripts/run_ai_experiments.py --suite ppo_training
python scripts/run_ai_experiments.py --suite ppo_control
python scripts/run_ai_experiments.py --suite ppo_selfplay
python scripts/run_ai_experiments.py --suite ppo_roundrobin

# Part 4: LLM (Exp 4.1-4.31)
export OPENAI_API_KEY=...
python scripts/run_ai_experiments.py --suite llm_control
python scripts/run_ai_experiments.py --suite llm_selfplay
python scripts/run_ai_experiments.py --suite llm_roundrobin
python scripts/run_ai_experiments.py --suite llm_cost_analysis
```
