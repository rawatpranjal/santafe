# Paper Experiment Plan

All experiments use the 10 Santa Fe environments: BASE, BBBS, BSSS, EQL, RAN, PER, SHRT, TOK, SML, LAD

---

## Part 1: Foundational Replication (Smith, Gode-Sunder, Cliff-Bruten)

ZI, ZIC, ZIP self-play across all 10 Santa Fe environments.

References: Smith (1962), Gode & Sunder (1993), Cliff & Bruten (1997)

### Experiments (30 total)

| Strategy | Configs | Count |
|----------|---------|-------|
| ZI | `p1_self_zi_*.yaml` | 10 |
| ZIC | `p1_self_zic_*.yaml` | 10 |
| ZIP | `p1_self_zip_*.yaml` | 10 |

### Configuration
- 4 tokens per trader
- 100 steps per period (except SHRT: 20 steps)
- 10 periods per round
- Multiple rounds for statistical significance

### Metrics
- Allocative efficiency (%)
- Price RMSD from equilibrium
- Price autocorrelation (lag-1)
- Trading volume distribution
- Profit dispersion

### Outputs (In Paper)
- [x] **table_foundational.tex:** Foundational ZI results ✅
- [x] **table_efficiency_full.tex:** Full efficiency matrix ✅
- [x] **table_volatility_full.tex:** Price volatility matrix ✅
- [x] **table_vineff_full.tex:** V-inefficiency matrix ✅
- [x] **table_dispersion_full.tex:** Profit dispersion matrix ✅
- [x] **table_trades_full.tex:** Trades per period matrix ✅
- [x] **learning_curves.pdf:** ZIP learning convergence ✅
- [x] **case_study_zi.pdf:** ZI vs ZIC case study ✅

### Additional Generated (Not in Paper)
- [x] efficiency_by_environment.pdf (grouped bar chart) ✅
- [x] price_convergence.pdf ✅
- [x] efficiency_boxplots.pdf ✅

---

## Part 2: Santa Fe Tournament Replication (Rust et al. 1994)

Strategies: Skeleton, ZIC, ZIP, GD, Kaplan

Reference: Rust et al. (1994)

### Competitor Set 1: Against Control (1 vs 7 ZIC) — 40 configs

| Strategy | Configs |
|----------|---------|
| Skeleton | `p2_ctrl_skel_*.yaml` |
| ZIP | `p2_ctrl_zip_*.yaml` |
| GD | `p2_ctrl_gd_*.yaml` |
| Kaplan | `p2_ctrl_kap_*.yaml` |

### Competitor Set 2: Against Self (Self-Play) — 50 configs

| Strategy | Configs |
|----------|---------|
| Skeleton | `p2_self_skel_*.yaml` |
| ZIC | `p2_self_zic_*.yaml` |
| ZIP | `p2_self_zip_*.yaml` |
| GD | `p2_self_gd_*.yaml` |
| Kaplan | `p2_self_kap_*.yaml` |

### Competitor Set 3: Round Robin Tournament (Mixed) — 10 configs

`p2_rr_mixed_*.yaml` — Randomly sampled heterogeneous markets

### Configuration
- 4 buyers, 4 sellers (except environment-specific)
- 4 tokens per trader
- 100 steps per period
- 10 periods per round

### Metrics
- Allocative efficiency (%)
- Individual trader efficiency ratios
- Price autocorrelation (lag-1)
- Trading volume by period percentage
- Bid-ask spread evolution
- Profit rankings

### Outputs (In Paper)
- [x] **table_control.tex:** Against Control results ✅
- [x] **table_control_volatility.tex:** Control price volatility ✅
- [x] **table_invasibility.tex:** Invasibility ratios ✅
- [x] **table_selfplay.tex:** Self-play efficiency ✅
- [x] **table_selfplay_volatility.tex:** Self-play volatility ✅
- [x] **table_selfplay_vineff.tex:** Self-play V-inefficiency ✅
- [x] **table_pairwise.tex:** Pairwise matchups ✅
- [x] **table_zip_tuning.tex:** ZIP hyperparameter sensitivity ✅
- [x] **table_profit_analysis.tex:** ZIP vs ZIC profit analysis ✅
- [x] **table_roundrobin.tex:** Round Robin full results ✅
- [x] **table_roundrobin_summary.tex:** Round Robin summary ✅
- [x] **kaplan_mixed_vs_pure.pdf:** Kaplan mixed vs pure markets ✅
- [x] **price_autocorrelation.pdf:** Price autocorrelation by trader ✅
- [x] **case_study_mixed.pdf:** Mixed market case study ✅
- [x] **trading_volume_timing.pdf:** Trading volume by period ✅
- [x] **trader_hierarchy.pdf:** Trader strategy hierarchy ✅

---

## Part 3: RL Agents (PPO)

Reference: Chen et al. (2010)

### Training Curriculum — 3 configs

| Training | Config |
|----------|--------|
| vs ZIC | `p3_train_zic.yaml` |
| vs Skeleton | `p3_train_skel.yaml` |
| vs Mixed | `p3_train_mixed.yaml` |

### Competitor Set 1: Against Control (PPO vs 7 ZIC) — 10 configs

`p3_ctrl_ppo_*.yaml`

### Competitor Set 2: Against Self (PPO Self-Play) — 10 configs

`p3_self_ppo_*.yaml`

### Competitor Set 3: Round Robin Tournament (PPO in Mixed) — 10 configs

`p3_rr_ppo_*.yaml`

### Configuration
- 7,000 trading periods (Chen protocol)
- 25 steps per period
- 4 tokens per trader

### Metrics
- PPO efficiency ratio
- Opponent efficiency ratio
- Market efficiency
- PPO profit rank
- Training curves

### Outputs (In Paper)
- [x] **table_ppo_control.tex:** PPO vs 7 ZIC control results ✅
- [x] **table_ppo_control_volatility.tex:** PPO control volatility ✅
- [x] **table_ppo_invasibility.tex:** PPO invasibility ratios ✅
- [x] **table_ppo_pairwise.tex:** PPO pairwise tournament results ✅
- [x] **ppo_zi_combined.pdf:** PPO vs ZI/ZIC/ZIP metrics ✅
- [x] **ppo_tournament_bar.pdf:** PPO tournament ranking bar ✅
- [x] **ppo_learning_curve.pdf:** PPO training learning curve ✅

### Additional Generated (Not in Paper)
- [x] ppo_training_curves.pdf ✅
- [x] ppo_vs_legacy.pdf ✅

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

### Section 7: PPO RL (Part 3) - COMPLETE
**Tables:** table_ppo_control, table_ppo_control_volatility, table_ppo_invasibility, table_ppo_pairwise ✅
**Figures:** ppo_zi_combined.pdf, ppo_tournament_bar.pdf, ppo_learning_curve.pdf ✅

### Section 8: LLM (Part 4) - MINIMAL
**Tables:** table_llm_performance ✅
**Figures:** None

### What's Left (Optional/Expensive)
- [ ] LLM Control/Self-play/Round-Robin experiments (high API cost)
- [ ] LLM figures (4.1, 4.2)
- [ ] Statistical tests (ANOVA, confidence intervals)
