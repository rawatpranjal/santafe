# CLAUDE.md - AI Agent Constitution

**Project:** Santa Fe v2.0 - Double Auction Tournament Replication & AI Extensions

---

## Document System

| Doc | Purpose | When to Use |
|-----|---------|-------------|
| [CLAUDE.md](./CLAUDE.md) | Master brain | Read for context, goals, phases, technical specs |
| [CHECKLISTS.md](./CHECKLISTS.md) | Testing/Quality control | Read for key things to test/validate in the whole experimental design |
| [RESULTS.md](./RESULTS.md) | Experimental Design and Results | Read for the main experimental plan and its results |

---

## The Prime Directive

**Qualitative replication is the goal.** Reproduce the market dynamics and stylized facts of the 1993 Santa Fe Tournament, not line-by-line code translation. Success = Table 4 efficiency within 5%, Kaplan paradox observed, negative price autocorrelation confirmed.

**Research Agenda.** Extend with modern AI agents (Deep RL & LLMs).

---

## Repo Structure (Don't Clutter!)

```
/
├── PLAN.md        # This file
├── PLAN.md        # This file
├── CHECKLISTS.md    # How to validate
├── RESULTS.md       # Experiment design and reuslts
├── README.md        # Basic project info
├── pyproject.toml   # Dependencies
│
├── checklists/      # Validation checklists and rules
├── engine/          # Core market logic - DON'T add random scripts
├── traders/         # Agent implementations (organized by type)
│   ├── base.py      # Base trader class
│   ├── legacy/      # Classical traders (ZIC, ZIP, GD, Kaplan, etc.)
│   ├── llm/         # LLM-based traders
│   └── rl/          # RL-based traders (PPO, etc.)
├── envs/            # Gymnasium RL environments
├── scripts/         # ALL runner/utility scripts go here
├── tests/           # ALL tests (organized by phase)
├── conf/            # ALL Hydra configs
├── results/         # Experiment outputs (JSON, CSV)
├── outputs/         # Hydra run outputs (logs, configs)
├── checkpoints/     # Trained RL models
├── models/          # Alternative model storage
├── llm_outputs/     # LLM experiment artifacts
├── logs/            # Training/run logs
├── reference/       # Source materials (read-only)
├── paper/           # Paper drafts and figures
└── archive/         # Old/deprecated code
```

---

## Tinkering Guidelines

| What You're Adding | Where It Goes |
|-------------------|---------------|
| New script | `scripts/` |
| Experiment output | `results/` |
| Config variation | `conf/experiment/` |
| Trained model | `checkpoints/` |
| Log file | `logs/` |
| Temporary analysis | `paper/scratch/` (gitignored) |
| New classical trader | `traders/legacy/` |
| New LLM trader | `traders/llm/` |
| New RL trader | `traders/rl/` |
| RL environment | `envs/` |
| Hydra run output | `outputs/` |
| LLM experiment results | `llm_outputs/` |
| Deprecated code | `archive/` |
| New test | `tests/` |

**Rules:**
- NEVER add loose files to root
- NEVER add scripts to `engine/` or `traders/`
- Results go in `results/`, not scattered around
- One trader = one file in appropriate `traders/` subdirectory

---

## Workflow Loop

```
1. READ CLAUDE.md        → Understand current phase and goals
2. READ checklists/    → Check implementation requirements
3. DO THE WORK         → Follow tinkering guidelines above
4. UPDATE TRACKER.md   → Short entry: Tried/Result
```

---

## Quality Standards

- **Type Safety:** `mypy . --strict`
- **Linting:** `ruff check .`
- **Formatting:** `black .`
- **Tests:** `pytest tests/`
- **Coverage:** 90% minimum for traders

---

## Command Reference

```bash
# Environment
uv sync                    # Install dependencies

# Quality
mypy . --strict           # Type checking
ruff check .              # Linting
black .                   # Formatting

# Testing
pytest tests/             # Run all tests
pytest tests/phase2/      # Run phase-specific tests

# Experiments
python scripts/run_experiment.py experiment=<name>
python scripts/run_ai_experiments.py --suite chen_ppo
```

---

## Definition of Done

**For any task:**
- [ ] Code passes mypy/ruff/black
- [ ] Tests written and passing
- [ ] No loose files added to root
- [ ] TRACKER.md updated with Tried/Result

**For experiments:**
- [ ] Config in `conf/experiment/`
- [ ] Results saved to `results/`
- [ ] TRACKER.md updated

---

## TLDR Requirement

Always end responses with a one-paragraph TL;DR. No bullets, just prose. Focus on big picture, not tiny details. Break the TL'DR into plan (what are you trying to do and why), progress (on current plans), problems (where you are stuck), future plans (what is next). Each gets 1-2 lines. 


# Master Plan: Santa Fe Double Auction Tournament v2.0

**Objective:** Achieve qualitative replication of the 1993 Santa Fe Tournament market dynamics, then extend with modern AI agents (Deep RL & LLMs).
**Replication Strategy:** Focus on behavioral validation and market dynamics over line-by-line code fidelity.

---

## Reference Documentation (ALL VERIFIED ✅)
**Primary Sources (1993 Santa Fe Tournament):**
- **AURORA Protocol:** [DATManual_full_ocr.txt](./reference/oldcode/DATManual_full_ocr.txt) - Complete rules (4,303 lines)
- **Tournament Analysis:** [chartradingdat_full_ocr.txt](./reference/oldcode/chartradingdat_full_ocr.txt) - Results & strategies (1,856 lines)
- **Java Implementation:** `reference/oldcode/extracted/double_auction/java/da2.7.2/` - 49 Java files including:
  - SRobotKaplan.java, SRobotZI1.java, SRobotZI2.java (trader strategies)
  - SGameRunner.java, PeriodHistory.java (market logic)
- **Key Papers:**
  - [1994_santafe_double_auction.txt](./reference/literature/1994_santafe_double_auction.txt) - Rust et al. findings
  - [1992_santafe_report.txt](./reference/literature/1992_santafe_report.txt) - Foundation work
  - [1993_gode_sunder.pdf](./reference/literature/1993_gode_sunder.pdf) - ZIC benchmark
  - [1997_zip.pdf](./reference/literature/1997_zip.pdf) - ZIP algorithm
  - [1998_GD.pdf](./reference/literature/1998_GD.pdf) - GD algorithm

### Historical Context: The 1993 Santa Fe Tournament
**30 programs competed** for a $10,000 prize pool (1990-1991):
- **15 economists, 9 computer scientists, 3 mathematicians**
- **10 environments** testing robustness (competitive, oligopoly, time-pressure, etc.)
- **18,114 games** producing statistically significant rankings

**The Stunning Result: Simple Beat Complex**
- **Winner:** Kaplan (363 lines) - simple "wait-and-steal" heuristic
- **Losers:** Neural networks, genetic algorithms, Bayesian optimizers (all > 1,000 lines)
- **Why complex failed:** Low signal/noise, insufficient training data, implementation bugs, overfitting

**Why This Matters:**
Testing whether simple heuristics outperform sophisticated optimization has direct relevance to modern RL/LLM evaluation. The 1993 findings suggest that in high-noise, data-limited domains, robustness > complexity.

## 1. The Prime Directive
**Qualitative replication is the goal.**
The Python implementation must reproduce the key market dynamics and stylized facts from the 1993 tournament, not necessarily exact code translation.

### Success Criteria: The Six Stylized Facts
Our replication succeeds if we reproduce these empirical patterns from Rust et al. (1994):

1. **Convergence to Competitive Equilibrium**
   - Transaction prices approach CE within first 20% of each period
   - Final trades cluster tightly around equilibrium price

2. **High Ex-Post Efficiency - The Minimal Intelligence Ladder**

   Our replication tests a progression from zero to minimal intelligence:

   - **Pure ZI (unconstrained): 60-70%** (Gode & Sunder 1993 control condition)
     → Proves: Random bidding WITHOUT constraints = poor efficiency
     → Mechanism: Many unprofitable trades (negative surplus)
     → Role: Control group establishing that randomness alone is insufficient
     → Key finding: 28% LOWER than ZIC (proves budget constraint is critical)

   - **Pure ZIC (constrained): 98.7%** (Gode & Sunder 1993 treatment condition)
     → Proves: Budget constraints + AURORA protocol = near-optimal efficiency WITHOUT learning
     → Mechanism: Progressive narrowing of opportunity sets (geometric convergence, not optimization)
     → BUT: Only works in symmetric markets (70-85% in asymmetric, proves necessity of ZIP)
     → Role: Null model establishing institutional power—the FLOOR for market performance
     → Limitation: High price variance (2-3x humans), suboptimal individual profits
     → **The BIG jump:** ZI → ZIC (+28%) proves institution matters more than learning

   - **Pure ZIP: 99.9%** (Cliff & Bruten 1997 benchmark)
     → Proves: One-parameter margin adaptation beats zero intelligence universally
     → TEST: Must work in asymmetric markets where ZIC fails
     → Role: Minimal learning sufficiency test
     → **Small jump:** ZIC → ZIP (+1.2%) proves learning adds marginal refinement

   - **Pure GD: >99.9%** (Gjerstad & Dickhaut 1998 benchmark)
     → Proves: Observable frequencies encode sufficient info for equilibrium
     → TEST: Should converge faster than ZIP (MAD < $0.04 vs $0.08)
     → Role: Informational minimalism validation

   - **Pure Kaplan: <60%** (Rust et al. 1994 market crash)
     → Proves: Individually rational can be collectively unstable
     → TEST: Parasitic strategy requires active bidders for information flow
     → Role: Evolutionary instability demonstration

   - **Mixed markets: 89.7%** (Overall tournament average)
     → Proves: Strategy heterogeneity needed for market ecology
     → TEST: Kaplan achieves 121% efficiency in mixed, crashes in pure
     → Role: Validates Kaplan paradox and trader hierarchy

3. **Negative Price Autocorrelation**
   - Price changes show negative lag-1 correlation (~-25%)
   - Challenges Wilson's (1987) martingale prediction
   - Evidence of mean-reversion in transaction prices

4. **Closing Panic Behavior**
   - Trading volume spikes in final 10-20% of period
   - Bid-ask spread narrows rapidly near deadline
   - Evidence of time-pressure effects on bidding

5. **Trader Hierarchy Preservation**
   - Rankings: Kaplan > ZIP/GD > ZIC
   - Kaplan extracts highest individual profits (121% efficiency)
   - Hierarchy stable across multiple environments

6. **The Kaplan Paradox (Critical Finding)**
   - **Mixed markets:** Kaplan achieves 121% efficiency (parasitic success)
   - **Pure Kaplan markets:** Efficiency collapses to <60% (collective failure)
   - **Cause:** Individually rational strategy is collectively unstable
   - **Mechanism:** Negative information externalities when all traders "wait in background"

### The Kaplan Paradox: Central Theoretical Contribution
**Kaplan's "Wait-and-Steal" Strategy:**
- Wait passively while other traders bid and narrow the spread
- When bid-ask spread < 10%, jump in and bid exactly the previous ask price
- "Steal the deal" that active bidders negotiated
- No prediction, no learning—just opportunistic exploitation

**The Paradox:**
- **Individual success:** Kaplan won by 2.5 standard deviations, earning $408 vs $394 (2nd place)
- **Collective failure:** Pure Kaplan markets show <60% efficiency (market crash)
- **Parasitic dependency:** Strategy only works when others provide liquidity and information flow
- **Evolutionary instability:** If Kaplan proliferates, it destroys conditions for its own success

**Market Ecology Requirements:**
- **Optimal mix:** ~10-20% active bidders (ZIC/Skeleton) + 80-90% opportunistic (Kaplan)
- **Stability threshold:** Markets fail when >90% use wait-and-steal strategies
- **Information externality:** Active bidders create public good (price discovery) that Kaplan free-rides on

**Why This Matters for Replication:**
This is the PRIMARY finding we're testing. If we can't reproduce both Kaplan's dominance in mixed markets AND the crash in pure markets, we haven't validated the core dynamics. The paradox reveals that:
- Individual rationality ≠ collective rationality
- Market efficiency requires strategy heterogeneity
- Simple exploitation can outperform complex optimization—but only parasitically

### Tech Constraints:
- **Laptop-First Engineering:** No HPC clusters required
- **Artifact-Driven:** Every script outputs paper-ready data

## 2. Official Glossary

Strict definitions for variable names, documentation, and technical discussions.

### 2.1 The Hierarchy of Time

The temporal structure of the simulation, derived from the Tournament Manual.

- **Step**: The atomic unit of time. Consists of exactly one Bid-Offer Phase followed by one Buy-Sell Phase.
- **Period**: A "trading day."
  - *State:* Inventory is replenished (e.g., 4 tokens). Valuations are fixed.
  - *Duration:* Typically 25-100 Steps.
  - *End:* Inventory expires (worthless).
- **Round**: A sequence of Periods using the same market configuration (same traders, same token distribution parameters) but different random seeds for valuations.
- **Game**: The full experimental unit. A collection of Rounds.
- **Epoch (RL)**: One full pass through the training data or a set number of episodes used for a PPO update.

### 2.2 AURORA Protocol Mechanics

The physics of the market engine.

- **Bid-Offer Phase (BO)**: The "Shout" phase.
  - *Action:* Agents submit a price or Pass.
  - *Constraint:* New York Rule (Must improve the market). Buyer must bid > Current Bid; Seller must ask < Current Ask.
  - *Crossed Market:* Invalid. A bid ≥ Ask is rejected in this phase (crucial for Kaplan logic).
- **Buy-Sell Phase (BS)**: The "Execution" phase.
  - *Participants:* Only the Holder (current Best Bidder) and Holder (current Best Offerer).
  - *Action:* Binary (Accept/Reject).
  - *Clearing:* If trade occurs, Book resets. If no trade, Book Persists (prices remain standing).
- **The Book**: The state of the CurrentBid and CurrentAsk.
- **Tokens**: The abstract commodity.
  - *Intra-marginal:* Tokens that should trade in equilibrium (Value > Equilibrium Price).
  - *Extra-marginal:* Tokens that should not trade (Value < Equilibrium Price).

### 2.3 Agent Taxonomy

The "Cast of Characters."

- **ZI (Zero Intelligence)**: Unconstrained random bidder. Can buy above value/sell below cost. Used only as a control to prove the value of constraints.
- **ZIC (Zero Intelligence Constrained)**: Random bidder with budget constraints. Cannot lose money. The baseline for "minimal intelligence."
- **Skeleton**: The "Rational Baseline." Uses a simple fixed profit margin. The reference implementation provided to 1993 contestants.
- **Kaplan**: The "Sniper." Waits for the spread to narrow, then jumps in to steal the deal. *Parasitic:* Relies on others to make price discovery.
- **GD (Gjerstad-Dickhaut)**: The "Belief Learner." Calculates probability of acceptance based on history to maximize expected surplus.
- **ZIP (Zero Intelligence Plus)**: The "Adaptive Learner." Adjusts a profit margin up/down based on market feedback (Widrow-Hoff rule).
- **ZI2 (Market-Aware Random)**: Based on Java `SRobotZI2.java`. Generates random prices constrained by current market bid/ask. More informed than ZIC but still non-adaptive.
- **Ringuette**: Sniper strategy based on JEDC paper (Page 74). Uses SPAN calculation to determine when spread is tight enough to strike. Distinct from ZI2.
- **Glass Cannons**: Agents (like Lin or Perry) that achieve high efficiency in self-play but are fragile/exploitable in mixed markets.

### 2.4 Economic Metrics

How we keep score.

- **Allocative Efficiency**: Total Profit Realized / Maximum Theoretical Surplus. The primary metric of market quality.
- **Profit Ratio**: Agent Profit / Baseline ZIC Profit. The primary metric of agent superiority.
- **V-Inefficiency (Valuation Inefficiency)**: Loss caused by failing to execute profitable trades (laziness/waiting too long).
- **EM-Inefficiency (Extra-Marginal Inefficiency)**: Loss caused by executing the wrong trades (buying high/selling low), displacing better trades.
- **Equilibrium Price (CE)**: The theoretical intersection of Supply and Demand curves.

### 2.5 Experimental Protocols

The specific setups for testing.

- **Grand Melee**: A robustness stress-test where agents face 10 different scenarios (Scarcity, Monopoly, Panic Time, etc.).
- **Invasibility Test**: A specific configuration: 1 Subject Agent vs. 7 Clones of Opponent. Tests if a strategy can "invade" a stable population.
- **The Kaplan Paradox**: The phenomenon where a strategy is individually dominant (wins the tournament) but collectively disastrous (crashes the market if everyone uses it).
- **Curriculum**: The training sequence for RL: Empty Market → ZIC Opponents → Skeleton Opponents → Mixed Sharks.

### 2.6 RL & Implementation Terms

Code-level culture.

- **Game 6453**: The canonical seed/configuration for token generation used in the original paper. We reference this to mean "Standard Setup."
- **Strict Integer Math**: All prices are int. No floats allowed in the order book.
- **Observation Space**: The input vector. MUST be anonymous (no opponent IDs).
- **Valid Action Mask**: The filter that prevents the RL agent from choosing illegal moves (selling empty inventory), ensuring fast learning.

## 3. The Tech Stack
- **Environment:** `uv` (Fast Python package manager)
- **Config:** `Hydra` (Compositional YAML configs)
- **Tracking:** `Weights & Biases` (Artifacts, metrics, logs)
- **RL:** `Stable-Baselines3` + `Gymnasium`
- **LLM:** `LiteLLM` + `Pydantic` + `DiskCache`
- **Data:** `Polars` + `Seaborn`
- **Quality:** `pre-commit` (`ruff`, `black`, `mypy`) + `pytest`

## 4. Core Trader Focus (Phase 2)
We implement 6 core traders from the 1993 tournament, selected to test the "simple beats complex" hypothesis and the ZI vs ZIC control experiment:

### Control Condition (Essential for Validation):
0. **ZI (Zero-Intelligence Unconstrained):** Random bids/asks with NO budget constraint

   **Algorithm (Gode & Sunder 1993 Control Condition):**
   - **Bid generation:** Draw uniformly from [price_min, price_max]
   - **Ask generation:** Draw uniformly from [price_min, price_max]
   - **Budget constraint:** NONE - can bid above valuation, ask below cost (trades at loss allowed)
   - **Memory:** NONE - each bid is independent (no learning, no state)
   - **Expected efficiency:** 60-70% (G&S 1993 finding)

   **Significance:** ZI is the CONTROL condition proving budget constraints matter
   - **Poor performance:** ~65% efficiency (vs 98.7% for ZIC)
   - **Role:** Isolates the effect of budget constraint—the difference between ZI and ZIC proves the "C" matters
   - **Mechanism:** Without constraints, random bids don't narrow opportunity sets → many unprofitable trades

   **Why We Need This:**
   If we only test ZIC, we can't prove the budget constraint is necessary. ZI vs ZIC is THE
   control experiment showing that institutions (budget rules) create efficiency, not randomness.

### Essential Traders (Must Work Perfectly):
1. **ZIC (Zero-Intelligence Constrained):** Random bids/asks within profitable range

   **Algorithm (Gode & Sunder 1993 Treatment Condition):**
   - **Bid generation:** Draw uniformly from [price_min, valuation]
   - **Ask generation:** Draw uniformly from [cost, price_max]
   - **Budget constraint:** STRICT inequality (buyer_val > bid, ask > seller_cost)
   - **Memory:** NONE - each bid is independent (no learning, no state)
   - **Mechanism:** Efficiency emerges from progressive narrowing of opportunity sets, not optimization

   **Significance:** 98.7% efficiency despite zero learning proves institution > intelligence (Gode & Sunder 1993)
   - **Revolutionary finding:** Budget constraints alone (without any learning) achieve near-optimal allocation
   - **The floor:** ZIC establishes minimum market performance—any trader with learning should beat this
   - **Limitation:** ONLY works in symmetric markets (fails when supply/demand asymmetric)

   **Role:** The "null model" baseline—validates DA protocol drives efficiency, not trader sophistication

2. **Kaplan (Sniper):** Strategic waiting with timing heuristic
   - **Significance:** Tournament winner—parasitic "wait-and-steal" strategy
   - **Paradox:** 121% efficiency in mixed markets, <60% in pure markets (individually optimal, collectively unstable)
   - **Role:** Tests whether simple opportunistic exploitation beats complex prediction

3. **ZIP (Zero-Intelligence Plus):** Adaptive profit margin adjustment
   - **Significance:** Simple learning (margin adaptation) achieves 85-95% efficiency
   - **Role:** Demonstrates minimal adaptation sufficient for high performance (Cliff 1997)

### Supporting Traders (Basic Implementation):
4. **GD (Gjerstad-Dickhaut):** Belief-based with price/quantity optimization
   - **Significance:** More sophisticated than ZIP but still simpler than neural nets/GA that failed
   - **Role:** Tests whether belief formation + optimization improves over pure adaptation

5. **ZI2 (Ringuette):** Enhanced ZIC with memory
   - **Significance:** 2nd place in tournament—ZIC + token tracking beats complex algorithms
   - **Role:** Demonstrates that minimal memory enhancement over pure randomness is highly effective

### Behavioral Validation Requirements:
Each trader must pass specific behavioral tests BEFORE tournament experiments:
- **ZI:** Uniform random distribution WITHOUT constraints, low efficiency (~65%), high EM-inefficiency
- **ZIC:** Uniform random distribution within constraints, high efficiency (98.7%), ZIC-ZI comparison (+28%)
- **Kaplan:** Correct timing calculation, "wait-war" behavior in homogeneous markets
- **ZIP:** Profit margin adaptation, convergence to equilibrium
- **GD:** Belief function formation, quote optimization
- **ZI2:** Token tracking, improved efficiency over ZIC

## 5. Execution Roadmap (7 Phases)

### Phase 0: Infrastructure ✅ COMPLETE
*Goal: Repository setup with quality gates*
- [x] Task 0.1: Repo initialization with `uv`
- [x] Task 0.2: Quality gates (pre-commit, mypy, pytest)
- [x] Task 0.3: Config schema with Hydra

### Phase 1: Core Engine ✅ COMPLETE
*Goal: AURORA market mechanism implementation*
- [x] Task 1.1: Order book & token generation
- [x] Task 1.2: Market step function (two-stage protocol)
- [x] Task 1.3: Basic validation framework

#### AURORA Strict Mode (Critical for Kaplan Support)

**Price Physics:**
- Force **Integer Math** for all prices, bids, and asks

**Phase 1 Rule (Bid-Offer):**
- **Rejection Logic:** If `Bid >= CurrentAsk` or `Ask <= CurrentBid` during shout phase → **REJECT** the order. Do NOT execute immediately.
- **Why:** This delay is required for Kaplan to see the spread narrow and "steal" the deal in Phase 2

**Phase 2 Rule (Buy-Sell):**
- **Persistence:** If Buyer (Holder_Bid) says "No" AND Seller (Holder_Ask) says "No" → **Keep the Order Book**. Do NOT reset prices to 0.
- **Clearing:** Reset Order Book ONLY upon a successful trade
- **Fairness:** Shuffle agent execution order randomly **every single time step** (prevent ID-based priority)

### Phase 2: Core Traders & Behavioral Validation 🔄 IN PROGRESS
*Goal: Implement and validate 5 core traders*

#### Task 2.1: Fix Critical Bugs (URGENT - 2 hours)
- [ ] **2.1a: Fix efficiency calculation bug** (30 min)
  - Location: `engine/metrics.py`
  - Issue: Negative efficiency values (-116% ZIC, -21% Kaplan)
  - Test: Known scenario with expected efficiency
  - Success: Positive, reasonable efficiency values

- [ ] **2.1b: Fix Kaplan protection clauses** (15 min)
  - Location: `traders/legacy/kaplan.py` lines 186-187, 275-276
  - Action: Ensure protection ONLY on subsequent bids/asks (not first)
  - Test: Kaplan vs Kaplan tournament
  - Success: Efficiency < 60% (market crash)

- [ ] **2.1c: Complete trader interfaces** (45 min)
  - Traders: ZI2, Lin, Jacobson, Perry
  - Add missing methods or refactor to base interface
  - Test: All 7 traders run in mixed tournament
  - Success: No abstract method errors

- [ ] **2.1d: Integrate efficiency metrics** (30 min)
  - Add V-Inefficiency and EM-Inefficiency to tournament logs
  - Update CSV output format
  - Success: Can diagnose missed trades vs bad trades

- [ ] **2.1e: Implement "Skeleton" Trader** (30 min)
  - **Specs:** Constrained ZIC with basic "Profit Margin" heuristic
  - **Role:** Represents the "Rational Baseline" (smarter than ZIC, dumber than Kaplan)
  - **Essential for:** Realistic mixed markets, RL curriculum training Level 2
  - **Reference:** `SRobotExample.java` in Java codebase
  - Test: Skeleton achieves ~99% self-play efficiency
  - Success: Beats ZIC in mixed markets, loses to Kaplan

#### Task 2.2: Behavioral Validation Tests (3 hours)
*Each trader needs specific behavioral tests before tournaments*

**Why ZI vs ZIC Matters: The Control Experiment**

Gode & Sunder (1993) designed THE critical control experiment in market design:

**The Setup:**
- **ZI (unconstrained):** Random bids from [0, 200] - can trade at loss
- **ZIC (constrained):** Random bids from [0, valuation] - no losses allowed
- **Everything else identical:** Same market, same randomness, same AURORA protocol

**The Result:**
- **ZI efficiency:** 60-70% (poor, many unprofitable trades)
- **ZIC efficiency:** 98.7% (near-optimal, competitive equilibrium)
- **Difference:** +28% efficiency from a single constraint

**What This Proves:**
1. **Budget constraint is sufficient** for near-optimal efficiency (no learning needed)
2. **Randomness alone is NOT sufficient** (ZI fails without constraints)
3. **The institution (AURORA + budget rules) creates efficiency**, not trader intelligence
4. **Progressive narrowing** only works when constraints eliminate bad trades

**Why We Test BOTH:**
If we only test ZIC, we can't prove the constraint matters. The ZI vs ZIC comparison
isolates the CAUSAL effect of the budget constraint. This is experimental economics 101.

**Validation Requirements:**
- ZI must achieve ~65% efficiency (proves unconstrained randomness fails)
- ZIC must achieve ~98.7% efficiency (proves constraints create efficiency)
- Difference must be ~30-35% (proves budget constraint is THE critical feature)

**This establishes the foundation:**
- ZI (random, no constraint) = 65% ← Poor baseline
- ZIC (random, with constraint) = 98.7% ← Institution power
- ZIP (learning, with constraint) = 99.9% ← Minimal learning adds marginal gain
- GD (optimization, with constraint) = >99.9% ← Optimization adds tiny gain

The BIG jump is ZI → ZIC (institution). The small jump is ZIC → ZIP → GD (learning).

- [ ] **2.2a-pre: ZI Validation (Control Condition)** (30 min)

  **Purpose:** Validate that unconstrained randomness FAILS to create efficiency.
  This is the control group for the ZI vs ZIC comparison.

  **Distributional Tests (Algorithm Correctness):**
  - **Test 1: Uniform distribution** - Bids uniformly from [0, price_max]
  - **Test 2: No budget constraint** - Verify bids can exceed valuation
  - **Test 3: Independence** - No autocorrelation across periods
  - **Test 4: Memorylessness** - No state persistence

  **Efficiency Tests (Expected Failure):**
  - **Test 5: Low efficiency** - 60-70% efficiency (G&S 1993 Table 1)
    → Market config: Same as ZIC test (symmetric)
    → Expected: ~65% ± 5%
    → Purpose: Proves randomness alone is insufficient

  - **Test 6: High EM-Inefficiency** - Many unprofitable trades
    → Expect: 20-30% of trades have negative surplus (buyer_val < seller_cost)
    → Purpose: Shows why ZI fails—bad trades destroy surplus

  - **Test 7: No convergence** - Price variance should be HIGH
    → No within-period convergence (RMSD flat or increasing)
    → Purpose: Without constraints, no mechanism for price discovery

  - **Test 8: No learning curve** - Efficiency flat across periods
    → Period 1 ≈ Period 50 (both poor)
    → Purpose: Confirms true zero intelligence

  **Success Criteria:**
  - ✅ Tests 1-4 pass (algorithm correctness)
  - ✅ Test 5: Efficiency 60-70% (significantly LOWER than ZIC)
  - ✅ Test 6: High EM-Inefficiency (20-30% bad trades)
  - ✅ Test 7: No price convergence (high variance throughout)
  - ✅ Test 8: Flat learning curve (no adaptation)

  **Reference:** Gode & Sunder (1993) Table 1, Control Condition

- [ ] **2.2a: ZIC Validation** (45 min)

  **Distributional Tests (Algorithm Correctness):**
  - **Test 1: Uniform distribution** - Chi-square test on 1000 bids (p > 0.05)
  - **Test 2: Independence** - Autocorrelation of bids across periods (ρ ≈ 0)
  - **Test 3: Budget constraint** - STRICT inequality (buyer_val > bid, ask > seller_cost)
  - **Test 4: Memorylessness** - No state persistence between periods

  **Efficiency Tests (Institution Power):**
  - **Test 5: Symmetric market** - 98.7% efficiency (Gode & Sunder 1993 benchmark)
    → Market config: 5 buyers [100,90,80,70,60], 5 sellers [40,50,60,70,80]
    → 50 periods × 10 rounds = 500 observations
    → Expected: 98.7% ± 1.5% (matching G&S Table 1)

  - **Test 6: Within-period convergence** - Price RMSD decreasing over time
    → Early trades (t=1-10): high variance
    → Late trades (t=40-50): low variance (convergence to CE)
    → Metric: RMSD should decline monotonically

  - **Test 7: NO learning curve** - Efficiency FLAT across periods
    → Period 1 efficiency ≈ Period 50 efficiency (±2%)
    → Proves: No adaptation, no memory, pure randomness

  **Failure Mode Tests (Asymmetric Markets):**
  - **Test 8: Asymmetric market failure** - Should NOT achieve 98.7%
    → Market config: Flat supply [60,60,60,60,60], downward demand [100,90,80,70,60]
    → Expected: Efficiency 70-85% (ZIC fails, price variance high)
    → Purpose: Validates that ZIC is NOT universal (sets up ZIP necessity)

  **Comparison Tests (Benchmarking):**
  - **Test 9: Profit dispersion** - Higher than ZIP/GD
    → ZIC: 0.35-0.60 (Cliff & Bruten 1997 baseline)
    → Purpose: Shows ZIC is less competitive than learning traders

  - **Test 10: Price variance** - Higher than human subjects
    → G&S finding: ZIC has 2-3x price variance of humans
    → Purpose: Institution creates efficiency but not price stability

  **Comparison Test (THE Critical Experiment):**
  - **Test 11: ZIC vs ZI efficiency difference** - Must be ~30-35%
    → Run identical market with ZI and ZIC
    → Expected: ZIC (98.7%) - ZI (65%) ≈ +33%
    → Purpose: ISOLATES the causal effect of budget constraint
    → **This is the control experiment proving institution > intelligence**

  **Success Criteria:**
  - ✅ Tests 1-4 pass (algorithm correctness)
  - ✅ Test 5: 98.7% ± 1.5% efficiency in symmetric markets
  - ✅ Test 6: Within-period RMSD decreasing
  - ✅ Test 7: NO efficiency improvement across periods (flat learning curve)
  - ✅ Test 8: Efficiency < 90% in asymmetric markets (proves limitation)
  - ✅ Tests 9-10: Higher variance than adaptive traders
  - ✅ Test 11: +28-35% efficiency gain over ZI (proves budget constraint causality)

  **References:**
  - Gode & Sunder (1993) Table 1: 98.7% efficiency across 5 market types
  - Cliff & Bruten (1997): ZIC profit dispersion 0.35-0.60

- [ ] **2.2b: Kaplan Validation** (45 min)
  - Test timing calculation: `(t-lasttime) >= (ntimes-t)/2`
  - Test "rational" bid calculation (best + worst case values)
  - Test wait-war in homogeneous market (<60% efficiency)
  - Test sniping behavior against ZIC
  - Success: Matches Java SRobotKaplan behavior

**Why ZIP Matters: The Minimal Learning Test**
Cliff & Bruten (1997) proved that ZI fails in asymmetric markets but one-parameter margin adaptation succeeds universally. ZIP validation tests whether:
1. Minimal learning (single parameter β) suffices for equilibrium convergence
2. Simple feedback-driven adaptation can handle market shifts that complex prediction cannot
3. Observable acceptance rates provide sufficient signal for coordination

**If ZIP achieves 99.9% efficiency in asymmetric markets, we've proven minimal intelligence works.**

- [ ] **2.2c: ZIP Validation** (45 min)
  - Test profit margin adaptation (increase when losing, decrease when winning)
  - Test convergence to equilibrium price in symmetric markets
  - **Test asymmetric market convergence** (flat supply, downward demand - where ZIC fails)
  - **Test response to market shift** mid-period (should adapt within 1-2 periods)
  - Test learning rate dynamics (β = 0.1-0.5)
  - **Test memory length effect** (L=5 should outperform L=1 or L=∞)
  - **Benchmark:** Profit dispersion 0.01-0.40 (vs ZIC's 0.35-0.60)
  - Success: **99.9% efficiency** (Cliff & Bruten 1997), convergence in all market types

**Why GD Matters: Solving Hayek's Coordination Puzzle**
Gjerstad & Dickhaut (1998) showed that observable transaction frequencies encode sufficient information for decentralized equilibration. GD validation tests whether:
1. Traders with only public market data (no knowledge of others' types) can form beliefs leading to equilibrium
2. Myopic expected surplus maximization achieves globally optimal allocation
3. Belief-based optimization beats simple margin adaptation (GD should outperform ZIP)

**If GD achieves >99.9% efficiency using only observable frequencies, we've validated informational minimalism.**

- [ ] **2.2d: GD Validation** (45 min)
  - Test belief function formation from observable frequencies
  - **Test belief monotonicity** (p(a) decreasing in a, q(b) increasing in b)
  - **Test belief concentration** near equilibrium (should be sharp, not uniform)
  - Test cubic spline interpolation of beliefs between observed prices
  - Test quote optimization via expected surplus maximization (`p*(a)` and `q*(b)`)
  - **Test myopic unit-by-unit optimization** achieves global efficiency
  - Test history seeding (5x weight for initial beliefs)
  - **Test convergence speed** vs ZIP (GD should converge faster)
  - **Benchmark:** Mean Absolute Deviation < $0.04 (vs ZIP's $0.08)
  - Success: **>99.9% efficiency** (Gjerstad & Dickhaut 1998), faster convergence than ZIP

- [ ] **2.2e: ZI2 Validation** (15 min)
  - Test token memory tracking
  - Test improved efficiency over ZIC
  - Success: 5-10% efficiency gain over ZIC

#### Task 2.3: Unit Test Suite (2 hours)
- [ ] **2.3a: Create comprehensive test files**
  - `tests/test_traders/test_zic_behavior.py`
  - `tests/test_traders/test_kaplan_behavior.py`
  - `tests/test_traders/test_zip_behavior.py`
  - `tests/test_traders/test_gd_behavior.py`
  - `tests/test_traders/test_zi2_behavior.py`

- [ ] **2.3b: Test coverage requirements**
  - Each trader: minimum 10 behavioral tests
  - Edge cases: empty market, single trader, extreme prices
  - Integration: trader interaction scenarios
  - Success: >90% code coverage per trader

### Phase 3: Replication Experiments (2 days)
*Goal: Reproduce 1993 tournament results*

**Context: What is Table 4 and Why It Matters**
Table 4 from Rust et al. (1994) presents efficiency levels across market configurations and establishes the core empirical findings:
- **Overall efficiency:** 89.7% across all tournaments (our target: within 5%)
- **Trader hierarchy:** Kaplan ($408) > Ringuette ($394) > Staecker ($387) by statistically significant margins
- **Simple > Complex:** Top performers used simple heuristics; complex strategies (neural nets, GA) underperformed
- **The Kaplan Paradox:** Demonstrated through mixed markets (121% efficiency) vs pure Kaplan markets (<60% crash)

**Why We Replicate Table 4:**
1. Validates our market engine produces realistic dynamics
2. Tests whether simple strategies systematically beat complex ones
3. Proves we can reproduce the Kaplan paradox (individual vs collective rationality)
4. Establishes behavioral fidelity before adding modern AI agents (Phases 4-5)

If we can't match Table 4 within ±5%, we haven't successfully replicated the core tournament dynamics.

#### Task 3.1: Table 4 Replication (4 hours)
- [ ] **3.1a: Pure market experiments (symmetric markets)**
  - Pure ZI (expect **60-70%** efficiency - G&S control condition, proves randomness fails)
  - Pure ZIC (expect **98.7% ± 1.5%** efficiency - G&S treatment, proves constraints work)
  - **ZIC - ZI comparison** (expect +28-35% difference, THE control experiment)
  - Pure Kaplan (expect <60% efficiency - market crash)
  - Pure ZIP (expect **99.9%** efficiency - Cliff & Bruten 1997 benchmark)
  - Pure GD (expect **>99.9%** efficiency - Gjerstad & Dickhaut 1998 benchmark)
  - Pure ZI2 (expect ~99% efficiency)

**Context: The Efficiency Ladder - Decomposing What Matters**

Our experiments test a precise progression to isolate causal factors:

| Trader | Efficiency | Change | What This Proves |
|--------|-----------|---------|------------------|
| ZI (random, no constraint) | 60-70% | Baseline | Randomness alone fails |
| **ZIC (random, with constraint)** | **98.7%** | **+28%** | **Institution > Intelligence** |
| ZIP (learning, with constraint) | 99.9% | +1.2% | Learning adds marginal gain |
| GD (optimization, with constraint) | >99.9% | +0.2% | Optimization adds tiny gain |

**The Key Insight:**
The BIG efficiency jump is ZI → ZIC (+28%) from adding budget constraints.
The small jump is ZIC → ZIP (+1.2%) from adding learning.
The tiny jump is ZIP → GD (+0.2%) from adding optimization.

**Implication:**
Market institutions (rules) matter MORE than trader sophistication. This is why:
- Gode & Sunder titled their paper "Allocative Efficiency of Markets with Zero-Intelligence Traders"
- The focus should be on market design, not just agent design
- Adding complex learning (RL/LLM) may have diminishing returns beyond simple constraints

**For Replication:**
We MUST reproduce this ladder. If we can't show the big ZI→ZIC jump and small ZIC→ZIP jump,
we haven't validated the core finding that institutions dominate intelligence.

- [ ] **3.1a-bis: Asymmetric Market Tests (THE Discriminating Test)**

  **Why This Matters:**
  Cliff & Bruten (1997) proved ZI fails in asymmetric markets (flat supply, excess demand).
  This is THE critical test that discriminates between zero intelligence and minimal learning.

  **Test Configurations:**
  1. **Flat supply + downward demand:**
     - ZIC: Should fail to converge (E[P] ≠ P₀, persistent deviation ~$0.20)
     - ZIP: Should converge (MAD < $0.08 from equilibrium)
     - GD: Should converge faster (MAD < $0.04 from equilibrium)

  2. **Box design (flat/flat, excess demand):**
     - ZIC: Should show systematic deviation from equilibrium
     - ZIP: Should succeed slowly but surely
     - GD: Should converge within 1-2 periods

  3. **Market shift mid-period:**
     - Change equilibrium price from $2.35 to $2.85 after period 3
     - ZIP: Should adapt within 1-2 periods
     - GD: Should adapt within 1 period
     - ZIC: Cannot adapt (no learning)

  **Metrics:**
  - Mean Absolute Deviation (MAD) from equilibrium price
  - Convergence time (periods to reach ±5% of new equilibrium)
  - Price dispersion (variance of transaction prices)

  **Success Criterion:**
  If ZIP succeeds where ZIC fails in asymmetric markets, we've validated that minimal
  learning is necessary and sufficient for universal equilibrium convergence.

- [ ] **3.1b: Mixed market experiments**
  - All 5 traders equal mix
  - Tournament matrix (all pairwise combinations)
  - Background trader tests (90% Kaplan + 10% others)

- [ ] **3.1c: Results validation**
  - Compare to Table 4 from Rust et al. (1994)
  - Document any discrepancies >5%
  - Success: Core results within 5% of paper

**Context: Why Price Dynamics Matter**
Price dynamics reveal whether our implementation captures the microstructure of the original tournament:

**Negative Autocorrelation (-25% in Rust et al.):**
- Indicates price corrections—overshoots followed by reversals
- Wilson's (1987) WGDA theory predicted zero autocorrelation (martingale prices)
- Actual data shows -25%, suggesting mean-reversion
- Tests: Are our traders exhibiting realistic bidding dynamics or just matching efficiency numbers?

**Closing Panic (Kaplan-driven phenomenon):**
- Final 10-20% of period shows volume spike + volatility increase
- Caused by Kaplan's parasitic waiting creating information vacuum
- Demonstrates negative information externalities in action
- Tests: Do we reproduce the time-pressure dynamics that cause market instability?

**Convergence to CE:**
- Adaptive traders (ZIP/GD) should converge faster than random traders (ZIC)
- Measures whether learning actually improves price discovery
- Tests: Does adaptation provide measurable benefits beyond static heuristics?

Without these dynamics, we'd just have a simulation that matches efficiency numbers—not a validated replication of tournament behavior.

#### Task 3.2: Price Dynamics Analysis (3 hours)
- [ ] **3.2a: Autocorrelation analysis**
  - Calculate price change autocorrelation
  - Test for negative autocorrelation (efficiency signal)
  - Compare across trader types
  - Success: Negative autocorrelation observed

- [ ] **3.2b: Convergence analysis**
  - Time to equilibrium metrics
  - Price volatility over time
  - Spread dynamics
  - Success: Convergence patterns match literature

- [ ] **3.2c: Closing panic analysis**
  - Trading volume in final periods
  - Price volatility in final periods
  - Urgency metrics
  - Success: "Closing panic" behavior observed

#### Task 3.3: Trading Pattern Analysis (2 hours)
- [ ] **3.3a: Volume analysis**
  - Trading volume by trader type
  - Extra-marginal vs intra-marginal trades
  - Efficiency decomposition (V-Inefficiency vs EM-Inefficiency)

- [ ] **3.3b: Profit distribution**
  - Profit by trader type
  - Wealth concentration metrics
  - Winner persistence analysis

- [ ] **3.3c: Strategic behavior**
  - Bid timing histograms
  - Markup analysis
  - Quote competitiveness

- [ ] **3.3d: ZIP vs GD Comparison - Learning vs Optimization**

  **The Core Question: Does Belief-Based Optimization Beat Simple Adaptation?**

  Based on theoretical predictions from Cliff & Bruten (1997) vs Gjerstad & Dickhaut (1998):

  **Convergence Speed:**
  - **ZIP:** Should reach equilibrium within 2-3 periods (via margin adaptation)
  - **GD:** Should reach equilibrium within 1-2 periods (via belief formation + optimization)
  - **Metric:** Periods to reach ±5% of equilibrium price

  **Adaptation to Market Shifts:**
  - **ZIP:** Adapts via gradual margin adjustment (1-2 periods)
  - **GD:** Adapts via belief updating + re-optimization (<1 period)
  - **Test:** Change equilibrium mid-session, measure adaptation time

  **Profit Dispersion (Cross-Sectional RMS Difference):**
  - **ZIP:** 0.01-0.40 (Cliff & Bruten benchmark)
  - **GD:** Should be even lower (>99.9% efficiency implies tighter distribution)
  - **Lower = closer to competitive outcome**

  **Computational Cost vs Performance:**
  - **ZIP:** Simple (1 parameter β, memory L=5) → should be robust to misspecification
  - **GD:** Complex (belief functions, cubic splines, optimization) → higher computational cost
  - **Trade-off:** Is GD's marginal efficiency gain worth the added complexity?

  **Expected Results:**
  - Both achieve >99% efficiency (both beat ZIC decisively)
  - GD converges faster than ZIP (fewer periods to equilibrium)
  - ZIP more robust to parameter errors (simpler = fewer failure modes)
  - GD achieves slightly higher efficiency but at computational cost

  **Success Criterion:**
  If GD shows measurably faster convergence and tighter profit distribution than ZIP,
  we've validated that belief-based optimization provides incremental benefits over
  simple learning—but both dominate zero intelligence.

- [ ] **3.3e: Simple vs Complex Validation - The Meta-Test**

  **The Fundamental Question: Does Simple Actually Beat Complex?**

  The 1993 tournament showed Kaplan/ZIP (simple) beat neural nets/GA (complex).
  Our replication should PROVE this systematically as foundation for Phase 4-5 AI extensions.

  **Hypothesis 1: Data Efficiency**
  - **Simple strategies** (ZIP/GD) should achieve >99% efficiency with minimal training
  - **Metric:** Plot efficiency vs number of training periods
  - **Expected:** ZIP/GD reach asymptote within 100 periods
  - **Implication:** Sample complexity of simple heuristics << complex learners

  **Hypothesis 2: Robustness to Parameter Variation**
  - **Simple strategies** should maintain performance across wide parameter ranges
  - **Test:** Vary β (learning rate) from 0.1 to 0.5 for ZIP
  - **Test:** Vary L (memory) from 1 to 10 for ZIP/GD
  - **Expected:** Graceful degradation, no catastrophic failure
  - **Implication:** Simple strategies have fewer "brittleness modes"

  **Hypothesis 3: Generalization Across Environments**
  - **Simple strategies** should work in all 10 tournament environments
  - **Complex strategies** would overfit to specific conditions
  - **Test:** Train in BASE environment, test in SHRT/EQL/BBBS
  - **Expected:** ZIP/GD performance stable across environments

  **Comparative Metrics:**
  | Strategy Type | Data Req. | Param. Sensitivity | Generalization | Impl. Bugs |
  |--------------|-----------|-------------------|----------------|------------|
  | ZIC (zero)   | None      | N/A               | Perfect        | Minimal    |
  | ZIP (simple) | ~50       | Low               | High           | Low        |
  | GD (moderate)| ~100      | Medium            | High           | Medium     |
  | Complex*     | ~1000s    | High              | Low            | High       |

  *Based on 1993 neural net/GA failures

  **Why This Matters:**
  Phase 4-5 will add PPO agents and LLM traders. If we can't first prove that simple
  beats complex in 1993 conditions, we can't claim modern AI overcomes historical
  limitations. This establishes the baseline: does RL/LLM beat simple heuristics where
  1993 neural nets failed?

  **Success Criterion:**
  ZIP/GD achieve >99% efficiency with <100 training periods, maintain performance
  across parameter variations and environments → validates "simple beats complex"
  hypothesis and sets up AI extension experiments.

#### Task 3.4: Replication Report (2 hours)
- [ ] **3.4a: Create comprehensive report**
  - Summary table comparing to 1994 paper
  - Key findings and discrepancies
  - Behavioral validation results
  - Market dynamics analysis

- [ ] **3.4b: Publication-ready artifacts**
  - Efficiency comparison table
  - Price convergence plots
  - Trading pattern visualizations
  - Statistical significance tests

#### Task 3.5: Full Tournament Replication (10 Environments)
*Goal: Recreate the exact 1994 Santa Fe Tournament conditions*

**The Original Tournament Structure:**
- 30 programs competed in 10 different market environments
- Each environment tested different aspects of market robustness
- 18,114 total games provided statistical significance
- Winner: Kaplan ($408), followed by Ringuette/ZI2 ($394), Staecker ($387)

**Environment Configurations to Replicate:**

- [ ] **3.5a: BASE Environment** - Standard competitive market
  - Configuration: 4 buyers, 4 sellers, 4 tokens each
  - Token values: Uniform random [1, 1000]
  - Time steps: 100 per period
  - **Periods:**
    - **Training:** Randomized horizon (9-11 periods) to prevent end-game overfitting
    - **Validation:** Fixed 10 periods for reproducible benchmarking
  - Expected: 89.7% average efficiency
  - Validates: Baseline trader performance

- [ ] **3.5b: BBBS Environment** - Duopoly (market power test)
  - Configuration: 2 buyers, 4 sellers
  - Expected: Lower efficiency due to buyer market power
  - Validates: Performance under asymmetric competition

- [ ] **3.5c: BSSS Environment** - Duopsony
  - Configuration: 4 buyers, 2 sellers
  - Expected: Lower efficiency due to seller market power
  - Validates: Inverse market power effects

- [ ] **3.5d: EQL Environment** - Symmetric values
  - Configuration: All traders get same token values + random shift
  - Expected: Higher efficiency (less heterogeneity)
  - Validates: Performance with homogeneous preferences

- [ ] **3.5e: RAN Environment** - Independent random values
  - Configuration: Pure IID uniform token generation
  - Expected: Similar to BASE
  - Validates: Robustness to value distribution

- [ ] **3.5f: PER Environment** - Single period only
  - Configuration: 1 period (no learning opportunity)
  - Expected: Lower efficiency (no adaptation time)
  - Validates: Zero-shot trading capability

- [ ] **3.5g: SHRT Environment** - High time pressure
  - Configuration: 20 time steps (vs 100 standard)
  - Expected: Lower efficiency, more volatility
  - Validates: Performance under time constraint

- [ ] **3.5h: TOK Environment** - Single token per trader
  - Configuration: 1 token (vs 4 standard)
  - Expected: Similar efficiency, less volume
  - Validates: Minimal market functionality

- [ ] **3.5i: SML Environment** - Small market
  - Configuration: 2 buyers, 2 sellers
  - Expected: Higher variance, similar mean efficiency
  - Validates: Small market dynamics

- [ ] **3.5j: LAD Environment** - Large asymmetric demand
  - Configuration: 6 buyers, 2 sellers or vice versa
  - Expected: Market stress test
  - Validates: Extreme asymmetry handling

**Metrics Collection for Each Environment:**
- Overall allocative efficiency
- Individual trader efficiency ratios (actual/theoretical max)
- Price convergence speed (time to ±5% of CE)
- Trading volume distribution across period
- Price autocorrelation coefficient
- Bid-ask spread evolution
- Profit concentration (Gini coefficient)

**Tournament Scoring:**
- Calculate efficiency ratio for each trader in each environment
- Weight equally across all 10 environments
- Rank by average efficiency ratio
- Compare to 1994 rankings (Kaplan > ZI2 > others)

**Success Criteria:**
- [ ] Overall tournament efficiency: 89.7% ± 5%
- [ ] Kaplan wins in heterogeneous markets (121% efficiency)
- [ ] Kaplan crashes in pure markets (<60% efficiency)
- [ ] Negative price autocorrelation observed (-0.25 ± 0.05)
- [ ] Closing panic in final 10-20% of periods
- [ ] Trader hierarchy preserved: Kaplan > GD/ZIP > ZIC > ZI

#### Task 3.6: Evolutionary Tournament Dynamics
*Goal: Replicate the 28,000-game evolutionary tournament*

- [ ] **3.6a: Population evolution setup**
  - Start with equal proportions of each strategy
  - Evolve based on profit performance
  - Track population shares over time
  - Expected: Kaplan initial dominance → eventual crash

- [ ] **3.6b: Invasibility analysis**
  - Test each strategy vs homogeneous opponent populations
  - Measure profit extraction ability
  - Document evolutionary stable strategies (ESS)
  - Expected: No pure ESS exists

- [ ] **3.6c: Mixed equilibrium search**
  - Find stable population mixtures
  - Test robustness to invasion
  - Document cycling dynamics if present
  - Expected: Complex dynamics, possible cycles

#### Task 3.7: Control Experiments & Invasibility Protocols

**ZI Control Experiment (Essential Validation):**
- [ ] **3.7a: ZI vs ZIC efficiency comparison**
  - Run ZI (Unconstrained) vs ZIC (Constrained) in identical symmetric market
  - Purpose: Isolate efficiency gain from budget constraint alone
  - Expected: +28-35% efficiency difference (ZI ~65%, ZIC ~98.7%)
  - Success: Proves "Institution > Intelligence" - constraints matter more than learning
  - Reference: Gode & Sunder (1993) Table 1

**Invasibility Protocol (1 AI vs 7 Opponents):**
- [ ] **3.7b: PPO vs 7 Kaplans invasibility test**
  - Instead of just "PPO vs Mixed," run **1 PPO vs 7 Kaplans**
  - Purpose: Can PPO exploit the Kaplan "waiting" deadlock?
  - Success Metric: PPO profit > Kaplan profit (exploits information starvation)
  - Expected: PPO should recognize Kaplan's passive strategy and become the active bidder
  - Fail case: If PPO also learns to wait → market crash (<60% efficiency)

- [ ] **3.7c: Strategy-specific invasibility matrix**
  - Test each AI (PPO, LLM) vs 7 clones of each strategy
  - Record: Efficiency ratio, profit extraction, market stability
  - Matrix: AI × {ZIC, ZIP, GD, Kaplan, Skeleton, Mixed}
  - Purpose: Identify which opponents AI can exploit vs which exploit AI

### Phase 4: RL Integration (1 week)
*Goal: Train PPO agents after solid foundation*

#### RL Architecture Specification

**Episode Definition:**
- **1 Episode = 1 Full Round** (e.g., 10 Periods)
- Inventory/Valuations reset every period, but Reward (Profit) accumulates across the round
- This matches how the tournament scores traders

**Observation Space:**
- **MUST Include:** `StepCount` (0-99), `PeriodCount` (0-9), `Inventory`, private valuation/cost
- **MUST Include:** Market info (best bid/ask, spread, last trade price, volume)
- **MUST Exclude:** Opponent Private Info (Valuations/Costs), specific Opponent IDs (Anonymity)
- **Rationale:** Agents should only see public market information

**Action Space:**
- **Masking:** Implement `ValidActionMask` layer before Softmax
- **Logic:** If `Inventory=0` → mask "Sell". If `Cash < Ask` → mask "Buy"
- **Actions:** Pass, Accept (current bid/ask), Improve (better bid/ask), Match

**Training Strategy (Curriculum Learning Pipeline):**
- **Level 1:** Train vs 100% ZIC (High liquidity, easy profits) → Learn basic trading
- **Level 2:** Train vs 100% Skeleton (Rational, predictable) → Learn competitive bidding
- **Level 3:** Train vs Mixed (Kaplan, GD, ZIP, ZIC) → Generalization
- Advance to next level when agent achieves >0.9x efficiency ratio

**Reward Constraint:**
- **Zero Reward Shaping** - Use Pure Profit only
- No intermediate rewards for "good" bids or penalties for "bad" bids
- Profit = Trade Price - Valuation (for buyers), Valuation - Trade Price (for sellers)
- Rationale: Shaped rewards can lead to reward hacking, pure profit forces true learning

#### Task 4.1: Gymnasium Environment (1 day) - Curriculum Learning Pipeline
- [ ] **4.1a: Observation space design**
  - Private info: valuation, tokens, time
  - Market info: best bid/ask, spread, last price
  - History: price trend, volume, imbalance

- [ ] **4.1b: Action space design**
  - Discrete: Pass, Accept, Improve, Match
  - Action masking for invalid actions

- [ ] **4.1c: Reward engineering**
  - Profit-based rewards
  - Penalty for invalid actions
  - Exploration bonuses

#### Task 4.2: PPO Training Pipeline (2 days)
- [ ] **4.2a: Single-agent training**
  - PPO vs ZIC (learn to trade)
  - PPO vs Kaplan (learn to snipe)
  - PPO vs mixed (generalization)

- [ ] **4.2b: Curriculum learning**
  - Stage 1: Easy opponents (ZIC)
  - Stage 2: Strategic opponents (Kaplan)
  - Stage 3: Mixed populations

- [ ] **4.2c: Hyperparameter tuning**
  - Learning rate schedules
  - Network architecture
  - Training stability

#### Task 4.3: Multi-Agent Training (2 days)
- [ ] **4.3a: Self-play setup**
  - Independent PPO (IPPO)
  - Parameter sharing
  - Centralized training

- [ ] **4.3b: Stability analysis**
  - Efficiency over training
  - Convergence metrics
  - Emergent strategies

#### Task 4.4: Evaluation & Analysis (1 day)
- [ ] **4.4a: Performance benchmarks**
  - PPO vs legacy traders
  - Invasibility tests
  - Robustness analysis

- [ ] **4.4b: Strategy analysis**
  - Learned behaviors
  - Timing strategies
  - Adaptation capabilities

### Phase 5: LLM Integration (3 days)
*Goal: Zero-shot LLM evaluation*

#### Task 5.1: LLM Infrastructure (1 day)
- [ ] Prompt engineering
- [ ] Structured output parsing
- [ ] Caching and rate limiting

#### Task 5.2: Experiments (1 day)
- [ ] GPT-4 vs legacy traders
- [ ] GPT-4 vs GPT-3.5
- [ ] Chain-of-thought analysis

#### Task 5.3: Analysis (1 day)
- [ ] Performance comparison
- [ ] Behavioral analysis
- [ ] Cost-benefit analysis

### Phase 6: Synthesis & Paper (1 week)
*Goal: Publication-ready results*

#### Task 6.1: Data Aggregation
- [ ] Merge all experimental results
- [ ] Statistical analysis
- [ ] Significance testing

#### Task 6.2: Visualization
- [ ] Efficiency matrices
- [ ] Learning curves
- [ ] Strategy evolution plots

#### Task 6.3: Paper Writing
- [ ] Introduction (replication importance)
- [ ] Methods (behavioral validation)
- [ ] Results (replication + extensions)
- [ ] Discussion (implications)

## 6. Definition of Done

### Replication Phase Complete When:
1. ✅ All 5 core traders implemented and tested
2. ✅ Behavioral validation tests pass for each trader
3. ✅ Table 4 results reproduced within 5%
4. ✅ Negative price autocorrelation observed
5. ✅ Closing panic behavior confirmed
6. ✅ Kaplan dominance in mixed markets verified
7. ✅ Comprehensive replication report published

### Project Complete When:
1. ✅ Replication phase complete
2. ✅ PPO agents trained and evaluated
3. ✅ LLM agents evaluated
4. ✅ All experiments documented
5. ✅ Paper draft ready for submission
---

## 7. AI Extension Experiments

This section documents two distinct experimental paradigms for testing AI agents in double auction markets.

### Two Experiment Types: Chen vs Rust

| Aspect | Chen Experiment (2010) | Rust Experiment (1992) |
|--------|------------------------|------------------------|
| **Goal** | Can AI beat specific opponent types? | How does AI perform in diverse ecology? |
| **Market Size** | 4v4 (8 traders) | 9v9 or 18v18 (larger markets) |
| **AI Position** | 1 AI vs 7 clones of opponent | AI as one strategy among many |
| **Duration** | 7,000 trading days | 5,000 trading days |
| **Metric Focus** | Individual efficiency ratio | Population dynamics |

### Chen AI Experiments (1 AI vs 7 Clones)

**Research Question:** Can a trained PPO agent or zero-shot LLM beat specific opponent types?

**Config Location:** `conf/experiment/chen_ai/`

**PPO Experiments (require pre-trained model):**
| Config | Opponent | Purpose |
|--------|----------|---------|
| `ppo_vs_kaplan.yaml` | 7 Kaplan | Hardest: Beat strategic snipers |
| `ppo_vs_zic.yaml` | 7 ZIC | Baseline: Beat random traders |
| `ppo_vs_zip.yaml` | 7 ZIP | Intermediate: Beat adaptive learners |
| `ppo_vs_gd.yaml` | 7 GD | Advanced: Beat belief-based traders |
| `ppo_vs_mixed.yaml` | Diverse mix | Generalization: Beat heterogeneous market |

**LLM Experiments (require OPENAI_API_KEY):**
| Config | Opponent | Purpose |
|--------|----------|---------|
| `llm_vs_kaplan.yaml` | 7 Kaplan | Zero-shot vs strategic snipers |
| `llm_vs_zic.yaml` | 7 ZIC | Zero-shot vs random baseline |
| `llm_vs_zip.yaml` | 7 ZIP | Zero-shot vs adaptive traders |
| `llm_vs_gd.yaml` | 7 GD | Zero-shot vs belief-based |
| `llm_vs_mixed.yaml` | Diverse mix | Zero-shot generalization |

**Parameters (matching Chen et al. 2010):**
```yaml
market:
  num_periods: 7000     # Trading days
  num_steps: 25         # Steps per day
  num_tokens: 4         # Tokens per trader
  gametype: 6453        # Chen's random seed
```

### Rust Stew Experiments (Heterogeneous Market)

**Research Question:** How does AI perform as one strategy among many in a diverse competitive ecosystem?

**Config Location:** `conf/experiment/rust_stew/`

**9v9 Market Experiments:**
| Config | AI Agent | Purpose |
|--------|----------|---------|
| `9v9_no_ai.yaml` | None | Baseline heterogeneous market |
| `9v9_with_ppo.yaml` | PPO buyer | PPO in diverse competition |
| `9v9_with_llm.yaml` | LLM buyer | LLM in diverse competition |

**18v18 Market Experiments (Tournament Scale):**
| Config | AI Agent | Purpose |
|--------|----------|---------|
| `18v18_no_ai.yaml` | None | Large market baseline |
| `18v18_with_ppo.yaml` | PPO buyer | PPO at tournament scale |
| `18v18_with_llm.yaml` | LLM buyer | LLM at tournament scale |

**Strategy Mix:**
```yaml
# 9v9 market composition:
buyers: [Kaplan×2, ZIP×2, GD×2, ZIC×3] + optional AI
sellers: [Kaplan×2, ZIP×2, GD×2, ZIC×3]

# 18v18 market composition:
buyers: [Kaplan×4, ZIP×5, GD×4, ZIC×5] + optional AI
sellers: [Kaplan×4, ZIP×5, GD×4, ZIC×5]
```

### Running AI Experiments

**Runner Script:** `scripts/run_ai_experiments.py`

```bash
# List all available experiments
python scripts/run_ai_experiments.py --list

# Run single experiment
python scripts/run_ai_experiments.py --experiment chen_ai/ppo_vs_kaplan

# Run experiment suites
python scripts/run_ai_experiments.py --suite chen_ppo    # All Chen PPO
python scripts/run_ai_experiments.py --suite chen_llm    # All Chen LLM
python scripts/run_ai_experiments.py --suite rust_stew   # All Rust Stew
python scripts/run_ai_experiments.py --suite all         # Everything

# Dry run (show commands without executing)
python scripts/run_ai_experiments.py --suite chen_all --dry-run
```

**Prerequisites:**
- **PPO experiments:** Trained model at `checkpoints/ppo_model.zip`
- **LLM experiments:** `OPENAI_API_KEY` environment variable set

### Expected Results & Success Criteria

**Chen AI Experiments:**
- PPO should achieve >1.0x efficiency ratio vs ZIC (beat baseline)
- PPO should achieve >0.9x vs Kaplan (compete with best)
- LLM zero-shot should beat Kaplan (~0.74x observed, Kaplan ~0.69x)

**Rust Stew Experiments:**
- AI should not crash market efficiency (<60%)
- AI should achieve competitive individual profit
- Market ecology should remain stable with AI present

### File Structure Created

```
conf/experiment/
├── chen_ai/
│   ├── ppo_vs_kaplan.yaml
│   ├── ppo_vs_zic.yaml
│   ├── ppo_vs_zip.yaml
│   ├── ppo_vs_gd.yaml
│   ├── ppo_vs_mixed.yaml
│   ├── llm_vs_kaplan.yaml
│   ├── llm_vs_zic.yaml
│   ├── llm_vs_zip.yaml
│   ├── llm_vs_gd.yaml
│   └── llm_vs_mixed.yaml
├── rust_stew/
│   ├── 9v9_no_ai.yaml
│   ├── 9v9_with_ppo.yaml
│   ├── 9v9_with_llm.yaml
│   ├── 18v18_no_ai.yaml
│   ├── 18v18_with_ppo.yaml
│   └── 18v18_with_llm.yaml
scripts/
└── run_ai_experiments.py    # Suite runner with prereq checks
```

