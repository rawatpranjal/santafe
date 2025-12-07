# Paper Experiment Plan

All experiments use the 10 Santa Fe environments: BASE, BBBS, BSSS, EQL, RAN, PER, SHRT, TOK, SML, LAD

---

## Part 1: Foundational Replication (Smith, Gode-Sunder, Cliff-Bruten)

References: Smith (1962), Gode & Sunder (1993), Cliff & Bruten (1997)

Total configs: 110 (50 self-play + 50 easy-play + 10 mixed-play)

---

### 1.1 Strategy Descriptions

Hierarchy: ZI (random) → ZIC1 (budget) → ZIC2 (budget + market) → ZIP1 (adaptive) → ZIP2 (adaptive + market)

| Strategy | Description |
|----------|-------------|
| **ZI** | Pure random bidding [MinPrice, MaxPrice], no budget constraint, accepts losses |
| **ZIC1** | Random bidding within budget only: buyers ≤ value, sellers ≥ cost (originally ZIC) |
| **ZIC2** | ZIC1 + market awareness: also constrained by current best bid/ask (originally ZI2) |
| **ZIP1** | Adaptive profit margins via Widrow-Hoff learning rule (originally ZIP) |
| **ZIP2** | ZIP1 + ZIC2 market constraints: passes when target price is infeasible |

#### ZIP Hyperparameter Tuning (10 seeds × 50 rounds, ZIP1 vs ZIC1)

| Config | β (learning) | γ (momentum) |
|--------|--------------|--------------|
| A_high_eff | 0.05 | 0.02 |
| B_low_vol | 0.005 | 0.10 |
| C_balanced | 0.02 | 0.03 |
| D_baseline | 0.01 | 0.008 |

---

### 1.2 Easy-Play Market Metrics (vs TruthTeller)

Goal: Measure search efficiency against naive opponents who ask at true cost.

5 strategies × 10 environments = 50 configs.

| Strategy | Configs |
|----------|---------|
| ZI | `p1_easy_zi_*.yaml` |
| ZIC1 | `p1_easy_zic1_*.yaml` |
| ZIC2 | `p1_easy_zic2_*.yaml` |
| ZIP1 | `p1_easy_zip1_*.yaml` |
| ZIP2 | `p1_easy_zip2_*.yaml` |

**Metrics collected:**
- Allocative Efficiency (%)
- Mean Trade Time (search speed)
- Trades per period

**Analysis: "Search Efficiency" Chart**
- X-Axis: Agent Type (ZI → ZIP2)
- Y-Axis: Mean timestep of first trade
- Hypothesis: ZIP2 will "learn" the markup and trade instantly; ZIC1 will miss frequently

---

### 1.3 Self-Play Market Metrics (Homogeneous)

Goal: Replicate Gode/Sunder (efficiency) and Cliff/Bruten (volatility).

5 strategies × 10 environments = 50 configs.

| Strategy | Configs |
|----------|---------|
| ZI | `p1_self_zi_*.yaml` |
| ZIC1 | `p1_self_zic1_*.yaml` |
| ZIC2 | `p1_self_zic2_*.yaml` |
| ZIP1 | `p1_self_zip1_*.yaml` |
| ZIP2 | `p1_self_zip2_*.yaml` |

**Metrics collected:**
- Allocative Efficiency (%)
- Price Volatility (%)
- V-Inefficiency (missed trades)
- Profit Dispersion (RMS)
- Smith's Alpha (price convergence)
- RMSD from equilibrium
- Trades per period

**Analysis: "Intelligence Ladder" Plot**
- X-Axis: Agent Type (ZI → ZIP2)
- Y-Axis: Price Volatility (StdDev of Price)
- Hypothesis: Should show steep decline. Proves intelligence stabilizes markets.

**Analysis: "Gode-Sunder" Table**
- Compare Efficiency across Envs for ZIC1 vs ZIP2
- If ZIC1 ≈ ZIP2, Gode/Sunder were right (Institution matters most)

---

### 1.4 Mixed-Play Competition (Shark Tank)

Goal: Direct competition. Who extracts surplus from whom?

Setup: 4v4 heterogeneous all-vs-all competition (no ZI - suicide trader distorts analysis).

1 setup × 10 environments = 10 configs: `p1_mixed_*.yaml`

**Buyer lineup (1 each):**
- ZIC1: Baseline constraint ("Dummy")
- ZIC2: Market-aware ("Smart Random")
- ZIP1: Naive adaptive ("Impatient Learner")
- ZIP2: Smart adaptive ("Patient Learner")

**Seller lineup (1 each):** ZIC1, ZIC2, ZIP1, ZIP2

**Metrics:** Avg Rank, Avg Profit, Win Rate, Profit/Trade

**Analysis: "Institutional Blindness" Gap**
- Metric: Profit_ZIP2 - Profit_ZIP1
- If Positive: Proves understanding the "Hold" rule (King of the Hill) is worth money

**Analysis: "Market Awareness" Gap**
- Metric: Profit_ZIC2 - Profit_ZIC1
- If Positive: Proves market awareness beats blind constraints

---

### 1.5 Replication Validation

Compare our results against published benchmarks.

| Paper | Their Result | Our Comparison |
|-------|--------------|----------------|
| Gode-Sunder (1993) | ZIC: 98.7% efficiency | ZIC1: 91-99% |
| Cliff-Bruten (1997) | ZIC fails in asymmetric markets | ZIC1: 66% (SHRT) |
| Cliff-Bruten (1997) | ZIP restores convergence | ZIP1: 99%+ all envs |

---

### 1.6 Strategy Hierarchy

Ranking across three dimensions: efficiency, profit dispersion, robustness.

| Dimension | Ranking |
|-----------|---------|
| Efficiency | ZIP1 = ZIP2 > ZIC2 > ZIC1 >> ZI |
| Profit Dispersion | ZIC1 = ZIC2 < ZIP1 < ZIP2 << ZI |
| Robustness | ZIP1 = ZIP2 >> ZIC2 > ZIC1 = ZI |

---

### 1.7 Behavioral Analysis

Characterizes how each strategy behaves (actions, timing) rather than just outcomes.

#### 1.7.1 Self-Play Behavior
All 8 agents same strategy. 5 seeds × 5 periods.

**Metrics:** Dominant Action, Mean Trade Time, Early%, PASS%, SR, PIR, Profit/Trade

#### 1.7.2 Zero-Play Behavior (Focal Agent)
1 focal buyer vs 7 ZIC1. 5 seeds × 5 periods.

**Metrics:** Same as 1.7.1

---

### 1.8 Inequality Metrics

Measures profit distribution across traders.

**Config:** Self-play, 8 agents same strategy. 3 seeds × 10 rounds × 10 periods. BASE environment.

**Metrics:** Gini, Max/Mean Ratio, Bottom-50% Share, Skewness

---

### 1.9 Deep RL vs Zero-Intelligence

Can PPO exceed ZIP1?

**Config:** BASE environment, 4 buyers (ZI, ZIC1, ZIP1, PPO) vs 4 sellers (ZI, ZIC1, ZIP1, ZIC1). 10 seeds × 50 rounds × 10 periods.

**Model:** `checkpoints/ppo_vs_zi_mix/final_model.zip`

---

### Outputs (In Paper)

| Output | Description |
|--------|-------------|
| table_easyplay.tex | Easy-play efficiency and search time |
| table_foundational.tex | Baseline efficiency in BASE |
| table_efficiency_full.tex | Efficiency across 10 environments |
| table_volatility_full.tex | Price volatility matrix |
| table_vineff_full.tex | V-inefficiency matrix |
| table_dispersion_full.tex | Profit dispersion matrix |
| table_trades_full.tex | Trades per period matrix |
| table_selfplay_behavior.tex | Behavioral signatures |
| table_inequality.tex | Gini and distributional metrics |
| table_mixed_competition.tex | Mixed competition results |
| selfplay_dynamics.pdf | Selfplay market dynamics figure |
| mixed_case_studies.pdf | Mixed competition case studies |
| intelligence_ladder.pdf | Volatility by agent type |
| search_efficiency.pdf | Mean trade time by agent type |

### Additional Generated (Not in Paper)

- learning_curves.pdf
- case_study_zi.pdf
- efficiency_by_environment.pdf
- price_convergence.pdf
- efficiency_boxplots.pdf

---

## Part 2: Ecological Analysis of Heuristic-Based Trading Strategies (Section 6)

**Objective:** To replicate the Santa Fe tournament, analyzing the ecological dynamics of key trading heuristics to address foundational questions from the literature (Rust et al. 1994, Chen & Tai 2010).

**Research Questions:**
1.  Why do simple "sniper" heuristics (Kaplan, Ringuette) outperform more complex strategies?
2.  Can the "Sniper's Dilemma" (market failure in homogeneous sniper populations) be experimentally induced?
3.  Is there a "penalty for adaptivity," where general learners (ZIP) are out-competed by specialized heuristics?
4.  What is the relationship between agent complexity and ecological fitness (e.g., Skeleton vs. GD/EL)?

---

### 2.1 Agent Roster

A representative agent from each major strategic category will be included:
- **Baseline:** `ZIC`
- **Simple/Fixed Heuristic:** `Skeleton`
- **Adaptive Learner:** `ZIP`
- **Parasitic Snipers:** `Kaplan`, `Ringuette`
- **Belief-Based Model:** `GD`
- **Reservation Price Model:** `EL`

*(Note: GD/EL may be excluded from final analysis if computationally prohibitive.)*

---

### 2.2 Experimental Design

Four experimental setups will be used across all 10 Santa Fe environments.

#### 2.2.1 Experiment 1: Invasibility Against Control (1 vs. 7 ZIC)
- **Goal:** Measure raw exploitative power.
- **Setup:** 1 challenger agent vs. 7 `ZIC` agents.
- **Primary Metric:** Profit Ratio (Challenger Profit / Mean ZIC Profit).

#### 2.2.2 Experiment 2: Homogeneous Self-Play (8x Same Agent)
- **Goal:** Test for collective stability and coordination failures.
- **Setup:** 8 identical agents.
- **Primary Metrics:** Allocative Efficiency (%), V-Inefficiency (missed trades), especially in `SHRT` and `RAN` environments.

#### 2.2.3 Experiment 3: Heterogeneous Round-Robin Tournament
- **Goal:** Measure overall ecological fitness and robustness in a static mixed population.
- **Setup:** A mixed market containing a representative sample of all agent types.
- **Primary Metrics:** Overall Profit Ranking, Average Rank per environment, Win Rate (%).

#### 2.2.4 Experiment 4: Evolutionary Tournament
- **Goal:** To identify Evolutionarily Stable Strategies (ESS) by simulating a dynamic population over multiple generations.
- **Setup:**
    - **Initial Population:** Start with an equal number of each agent type.
    - **Generations:** A generation consists of a full round-robin tournament across all 10 environments.
    - **Selection:** After each generation, an agent's share of the total profit determines its population share in the next generation. Poorly performing strategies will be eliminated over time.
    - **Mutation:** A small probability of randomly re-introducing an agent type into the population each generation to ensure diversity.
- **Primary Metrics:** Population Share of each strategy vs. Generation number.

---

### 2.3 Planned Outputs and Narrative Links

| Research Question | Primary Evidence Source (Planned Output) |
| :--- | :--- |
| **1. Why do snipers dominate?** | `table_s6_roundrobin_summary.tex` (shows they win) + `table_s6_invasibility.tex` (shows *how* they win via exploitation). |
| **2. Is the "Sniper's Dilemma" real?**| `table_s6_selfplay_efficiency.tex` (shows efficiency collapse for `Kaplan`/`Ringuette` in `SHRT`). |
| **3. Is adaptivity punished?** | Contrast `ZIP`'s robust self-play from Part 1 with its middling rank in `table_s6_roundrobin_summary.tex`. |
| **4. Is complexity worth it?**| Compare ranks of `Skeleton`, `GD`, and `EL` in `table_s6_roundrobin_summary.tex` to analyze the performance-complexity trade-off. |
| **5. What is the ESS?** | `figure_s6_evolutionary_dynamics.pdf` (shows which strategies survive and dominate over many generations). |

---

## Part 3: RL Agents (PPO)

Reference: Chen et al. (2010)

**Objective:** Evaluate PPO reinforcement learning in the double auction, progressing from simple to complex environments.

### Phase 1: Foundational Capabilities

#### 1.1 Control Experiment (Invasibility)
- **Setup:** 1 PPO vs 7 ZIC
- **Hypothesis H2 (Exploitation):** PPO can exploit naive opponents, capturing majority of surplus
- **Configs:** `p3_ctrl_ppo_*.yaml` (10 environments)
- **Metrics:** Profit ratio (PPO/ZIC), market efficiency, invasibility

#### 1.2 Easy-Play (Buyer Specialist)
- **Setup:** 1 PPO buyer vs TruthTeller sellers
- **Hypothesis H2 variant:** PPO learns surplus extraction without strategic counter-play
- **Configs:** `p3_easy_ppo_*.yaml` (10 environments)
- **Metrics:** Profit/trade, mean trade time, early trade %

#### 1.3 Mixed-ZI Competition (vs Set 1)
- **Setup:** 1 PPO vs ZI/ZIC/ZIP mix
- **Hypothesis H3 (Superior Adaptability):** PPO outperforms heuristic-based ZIP
- **Configs:** `p3_mixed_zi_*.yaml` (10 environments)
- **Metrics:** Profit rank among ZI hierarchy

### Phase 2: The Crucible (Advanced Competition)

#### 2.1 Santa Fe Tournament (vs Set 2)
- **Setup:** 1 PPO vs Kaplan/GD/Skeleton/Ringuette/EL
- **Hypothesis H4 (Emergent Dominance):** PPO beats human-designed heuristics through trial-and-error
- **Configs:** `p3_rr_ppo_*.yaml` (10 environments)
- **Metrics:** Profit rank, beat Kaplan?, behavioral signature (mean trade time, PASS%)

#### 2.2 Self-Play (Stability Test)
- **Setup:** 8 identical PPO agents
- **Hypothesis H1 (Stability):** PPO avoids "Sniper's Dilemma" (market collapse observed in Kaplan self-play)
- **Configs:** `p3_self_ppo_*.yaml` (10 environments)
- **Metrics:** Efficiency, V-inefficiency, trades/period, price convergence

### Phase 3: Synthesis
- [ ] Document quantitative results in results.md
- [ ] Update paper (07_results_rl.tex)
- [ ] Finalize status

### Configuration
- 7,000 trading periods (Chen protocol)
- 25 steps per period
- 4 tokens per trader

### Outputs (In Paper)
- [ ] table_ppo_control.tex: Control/invasibility results
- [ ] table_ppo_easyplay.tex: Easy-play results
- [ ] table_ppo_mixed_zi.tex: vs ZI hierarchy
- [ ] table_ppo_santafe.tex: vs Santa Fe traders
- [ ] table_ppo_selfplay.tex: Self-play stability
- [ ] figure_ppo_learning_curve.pdf: Training dynamics
- [ ] figure_ppo_behavior.pdf: Behavioral signatures (vs Kaplan)

---

## Part 4: LLM Agents

LLM treated as another AI agent type (zero-shot, no training).

### 4.0 Model Baseline Comparison — 35 runs (7 models × 5 seeds)

**Goal**: Establish baseline benchmark across model tiers. Pick 3 models (one per tier) for subsequent experiments.

**Model Tiers**:
- **TOP**: o4-mini-low, o4-mini-high (reasoning models)
- **MID**: GPT-4.1, GPT-4.1-mini, GPT-4o (production-grade)
- **LOW**: GPT-4o-mini, GPT-3.5-turbo (cheap baseline)

**Configuration**:
- Environment: BASE (4B/4S, 4 tokens)
- Market: 1 round × 1 period × 100 steps
- Seeds: 42, 123, 456, 789, 1000
- Prompt: Dense (full market mechanics, no strategy hints)
- Temperature: Default

**Outputs**:
- [x] **Table 4.0**: Model Comparison (7 models × 5 seeds → mean±std ratio) ✅
- [x] **Notes 4.0**: Qualitative "persona" for each model's reasoning style ✅

**Post-Experiment Selection**: Pick 1 model from each tier for Part 4.1-4.3.

---

### Competitor Set 1: Against Control (LLM vs 7 ZIC) — 10 configs

`p4_ctrl_llm_*.yaml`

### Competitor Set 2: Against Self (LLM Self-Play) — 10 configs

`p4_self_llm_*.yaml`

### Competitor Set 3: Round Robin Tournament (LLM in Mixed) — 10 configs

`p4_rr_llm_*.yaml`

### Cost Analysis — 1 config

`p4_model_comparison.yaml` — GPT-4, GPT-3.5, Claude

### Configuration
- Same as PPO (7,000 periods, 25 steps)
- Zero-shot (no training)

### Metrics
- LLM efficiency ratio
- Market efficiency
- LLM profit rank
- API cost per decision
- Latency per decision

### Outputs (In Paper)
- [x] **table_llm_performance.tex:** LLM model comparison ✅

### Planned (Not Yet Completed - High API Cost)
- [ ] Table 4.1: LLM Against Control (LLM vs 7 ZIC × environment)
- [ ] Table 4.2: LLM Self-play results
- [ ] Table 4.3: LLM Round Robin results
- [ ] Figure 4.1: LLM vs legacy trader comparison
- [ ] Figure 4.2: Cost vs performance scatter

---

## Output Artifacts Summary (Aligned with Actual Paper)

### Section 5: Zero-Intelligence (Part 1) - COMPLETE
**Tables:** table_foundational, table_efficiency_full, table_volatility_full, table_vineff_full, table_dispersion_full, table_trades_full ✅
**Figures:** learning_curves.pdf, case_study_zi.pdf ✅

### Section 6: Santa Fe Tournament (Part 2) - COMPLETE
**Tables:** table_control, table_control_volatility, table_invasibility, table_selfplay, table_selfplay_volatility, table_selfplay_vineff, table_pairwise, table_zip_tuning, table_profit_analysis, table_roundrobin, table_roundrobin_summary ✅
**Figures:** kaplan_mixed_vs_pure.pdf, price_autocorrelation.pdf, case_study_mixed.pdf, trading_volume_timing.pdf, trader_hierarchy.pdf ✅

### Section 7: PPO RL (Part 3) - OPEN
**Tables:** table_ppo_control, table_ppo_control_volatility, table_ppo_invasibility, table_ppo_pairwise ✅
**Figures:** ppo_zi_combined.pdf, ppo_tournament_bar.pdf, ppo_learning_curve.pdf ✅

### Section 8: LLM (Part 4) - OPEN
**Tables:** table_llm_performance ✅
**Figures:** None

### What's Left (Optional/Expensive)
- [ ] LLM Control/Self-play/Round-Robin experiments (high API cost)
- [ ] LLM figures (4.1, 4.2)
- [ ] Statistical tests (ANOVA, confidence intervals)
