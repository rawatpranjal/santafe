# LLM Results Section - Guide & Status

**Section:** 7 - Large Language Model Trader
**File:** `paper/arxiv/sections/07_results_llm.tex`
**Status:** Placeholder tables ready, experiments configured
**Last Updated:** November 24, 2025

---

## Section Structure

### 7.1 Zero-Shot Economic Rationality
- **Table 7.1**: LLM Performance Matrix
- **Metrics**: Efficiency, Mean Profit, Profit Ratio vs ZIC, Invalid Actions, Token Cost
- **Agents**: GPT-4o-mini, GPT-3.5 (buyer/seller roles)
- **Experiment**: `gpt4_vs_legacy.yaml`

### 7.2 Invasibility Test: LLM vs ZIC
- **Table 7.2**: LLM Invasibility Analysis
- **Metrics**: Overall Ratio, As Buyer, As Seller
- **Methodology**: 1 LLM + 7 ZIC (matching Section 5 legacy tests)
- **Experiment**: `invasibility_gpt4_mini.yaml`

### 7.3 Grand Melee: LLMs vs Legacy vs RL
- **Table 7.3**: Multi-Agent Tournament Results
- **Agents**: Mix of GPT-4o-mini, GPT-3.5, Kaplan, ZIP, ZIC (PPO if trained)
- **Metrics**: Avg. Rank, Mean Profit, Efficiency, Std. Dev.
- **Experiment**: `grand_melee_llm.yaml`

### 7.4 The Intelligence Premium
- **Table 7.4**: GPT-4o-mini vs GPT-3.5 Head-to-Head
- **Metrics**: Mean Profit, Win Rate, Efficiency, Invalid Actions, Response Time
- **Experiment**: `model_comparison.yaml`

### 7.5 Computational Cost-Performance Trade-offs
- **Table 7.6**: Computational Requirements Comparison
- **Comparison**: LLM API costs vs RL training vs Legacy (zero compute)
- **Analysis**: Cost-performance frontier and scalability

### 7.6 Behavioral Analysis
- Qualitative analysis of chain-of-thought reasoning
- "Fairness bias" from RLHF alignment
- Conservative bidding patterns
- Insensitivity to time pressure

---

## Experiment Configurations

### Priority 1 (Ready to Run - Cost: $3.99)

**gpt4_vs_legacy.yaml**
- 10 rounds × 5 periods
- 1 GPT-4o-mini + 3 legacy (ZIC, Kaplan, ZIP) per side
- Cost: ~$0.63
- Status: Running

**model_comparison.yaml**
- 10 rounds × 5 periods
- 4 GPT-4o-mini vs 4 GPT-3.5 (each side)
- Cost: ~$3.36
- Status: Ready

### Priority 2 (Awaiting Approval - Cost: $8.15)

**invasibility_gpt4_mini.yaml**
- 50 rounds × 10 periods (match legacy methodology)
- 1 GPT-4o-mini + 3 ZIC per side
- Cost: ~$3.15
- Status: Ready

**grand_melee_llm.yaml**
- 10 rounds × 5 periods
- 2 GPT-4o-mini + 2 GPT-3.5 + 2 Kaplan + 2 ZIP per side
- Cost: ~$5.00
- Status: Ready

---

## Analysis Pipeline

### 1. Run Experiments
```bash
# Priority 1
python scripts/run_llm_experiment.py --config gpt4_vs_legacy
python scripts/run_llm_experiment.py --config model_comparison

# Priority 2 (if approved)
python scripts/run_llm_experiment.py --config invasibility_gpt4_mini
python scripts/run_llm_experiment.py --config grand_melee_llm
```

### 2. Generate Tables
```bash
# Generate all tables
python scripts/analyze_llm_results.py --all

# Or specific experiments
python scripts/analyze_llm_results.py --experiment gpt4_vs_legacy
python scripts/analyze_llm_results.py --experiment invasibility_gpt4_mini
```

### 3. Update Paper
Tables are automatically generated in `paper/arxiv/figures/`:
- `table_llm_performance.tex`
- `table_llm_invasibility.tex`
- `table_llm_melee.tex` (manual creation needed)
- `table_llm_intelligence.tex` (manual creation needed)
- `table_llm_cost.tex` (manual creation needed)

---

## Metrics Reference (Santa Fe Template)

### From 1994 Paper (Table 4):
- **Allocative Efficiency**: (Actual profit / Equilibrium profit) × 100%
- **Mean Profit**: Average profit per agent per period
- **Profit Standard Deviation**: Consistency measure
- **Trade Volume**: Number of trades executed
- **Price Volatility**: Standard deviation of transaction prices

### LLM-Specific Additions:
- **Invalid Action Rate**: Constraint violations (%)
- **Token Cost**: API charges per experiment ($)
- **Response Latency**: Decision time per prompt (ms)
- **Cache Hit Rate**: Semantic caching effectiveness (%)
- **Profit Ratio vs ZIC**: LLM profit / ZIC mean profit

---

## File Organization

### Experiment Outputs (gitignored)
```
llm_outputs/
├── experiments/                    # CSV results + config snapshots
│   ├── gpt4_vs_legacy_YYYYMMDD_HHMMSS/
│   │   ├── results.csv
│   │   └── config.yaml
│   ├── model_comparison_YYYYMMDD_HHMMSS/
│   ├── invasibility_gpt4_mini_YYYYMMDD_HHMMSS/
│   └── grand_melee_llm_YYYYMMDD_HHMMSS/
├── analysis/                       # JSON summaries
│   ├── gpt4_vs_legacy_summary.json
│   └── invasibility_gpt4_mini_summary.json
├── logs/                           # Execution logs
├── prompts/                        # Sample prompts
└── responses/                      # Raw LLM responses
```

### Paper Files (committed)
```
paper/arxiv/
├── sections/07_results_llm.tex     # Main results section
└── figures/
    ├── table_llm_performance.tex
    ├── table_llm_invasibility.tex
    └── [other LLM tables]
```

---

## Status Checklist

### Infrastructure ✅
- [x] BaseLLMAgent with AURORA protocol integration
- [x] PromptBuilder with explicit constraint warnings
- [x] ActionParser with Pydantic validation
- [x] GPTAgent with LiteLLM and caching
- [x] CacheManager for cost reduction
- [x] Agent factory integration

### Experiments
- [x] gpt4_vs_legacy config
- [x] model_comparison config
- [x] invasibility_gpt4_mini config (new)
- [x] grand_melee_llm config (new)
- [ ] Run Priority 1 experiments
- [ ] Run Priority 2 experiments (pending approval)

### Analysis
- [x] analyze_llm_results.py script
- [ ] Generate performance table (after exp 1)
- [ ] Generate invasibility table (after exp 3)
- [ ] Generate intelligence premium table (after exp 2)
- [ ] Generate cost analysis table (manual)
- [ ] Generate melee table (after exp 4)

### Paper ✅
- [x] Section 7.1: Zero-Shot Economic Rationality
- [x] Section 7.2: Invasibility Test
- [x] Section 7.3: Grand Melee
- [x] Section 7.4: Intelligence Premium
- [x] Section 7.5: Cost-Performance Trade-offs
- [x] Section 7.6: Behavioral Analysis
- [x] All 5 placeholder tables created

### Documentation
- [x] PHASE5_LLM_IMPLEMENTATION_SUMMARY.md
- [x] This guide (LLM_RESULTS_GUIDE.md)
- [ ] Update tracker.md (pending)

---

## Next Steps

1. **Wait for gpt4_vs_legacy experiment to complete** (~10 min)
2. **Run model_comparison experiment** (~15 min, $3.36)
3. **Generate initial tables** from Priority 1 data
4. **Review results** and request approval for Priority 2
5. **Run invasibility and melee** if approved ($8.15)
6. **Populate all TBD values** in Section 7 tables
7. **Finalize behavioral analysis** with reasoning traces

---

## Cost Summary

| Experiment | Rounds | Periods | Est. Cost | Status |
|-----------|--------|---------|-----------|--------|
| gpt4_vs_legacy | 10 | 5 | $0.63 | Running |
| model_comparison | 10 | 5 | $3.36 | Ready |
| invasibility_gpt4_mini | 50 | 10 | $3.15 | Ready |
| grand_melee_llm | 10 | 5 | $5.00 | Ready |
| **Total Priority 1** | | | **$3.99** | |
| **Total Priority 2** | | | **$8.15** | |
| **Grand Total** | | | **$12.14** | |

All costs use GPT-4o-mini (cheapest option). GPT-3.5 experiments included in model_comparison for intelligence premium analysis.
