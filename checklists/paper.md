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

### Outputs
- [ ] **Table 1.1:** Efficiency matrix (trader × environment)
- [ ] **Figure 1.1:** Efficiency by environment (grouped bar chart)
- [ ] **Figure 1.2:** Price convergence comparison
- [ ] **Figure 1.3:** Efficiency distribution (box plots by trader type)

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

### Outputs
- [ ] **Table 2.1:** Against Control results (strategy vs 7 ZIC × environment)
- [ ] **Table 2.2:** Self-play efficiency matrix (5 strategies × 10 environments)
- [ ] **Table 2.3:** Round Robin tournament results
- [ ] **Figure 2.1:** Kaplan efficiency: mixed vs pure markets
- [ ] **Figure 2.2:** Price autocorrelation by trader type
- [ ] **Figure 2.3:** Trading volume by period percentage (closing panic)
- [ ] **Figure 2.4:** Trader hierarchy chart

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

### Outputs
- [ ] **Table 3.1:** Against Control results (PPO vs 7 ZIC × environment)
- [ ] **Table 3.2:** Self-play results (PPO × environment)
- [ ] **Table 3.3:** Round Robin results (PPO in Mixed × environment)
- [ ] **Figure 3.1:** PPO training curves
- [ ] **Figure 3.2:** PPO vs legacy trader comparison

---

## Part 4: LLM Agents

LLM treated as another AI agent type (zero-shot, no training).

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

### Outputs
- [ ] **Table 4.1:** Against Control results (LLM vs 7 ZIC × environment)
- [ ] **Table 4.2:** Self-play results (LLM × environment)
- [ ] **Table 4.3:** Round Robin results (LLM in Mixed × environment)
- [ ] **Table 4.4:** Cost-benefit summary
- [ ] **Figure 4.1:** LLM vs legacy trader comparison
- [ ] **Figure 4.2:** Cost vs performance scatter

---

## Output Artifacts Summary

### Tables
- [ ] 1.1: Part 1 efficiency matrix (ZI/ZIC/ZIP × 10 environments)
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

