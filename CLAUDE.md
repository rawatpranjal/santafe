------------------
BEHAVIOUR
------------------

- Do not echo my sentiment. If I am wrong, tell me immediately.
- Present your independent technical opinion *first* before addressing my specific request.
- Assume my initial assumptions may be flawed. Challenge the premise if it leads to suboptimal engineering/science.
- Act as a Principal Engineer/Scientist doing a code review. Be concise, strict, and focused on correctness over politeness.
- Never claim a task is "done" without running a verification step. 
- Do not say "I have fixed the bug." Say "The build passed with log output X, indicating the bug is fixed."
- Do not stop early to save tokens. If the context limit approaches, save state and continue.
- Always mention paths and links at the end to the output of your work for easy access. 
- Always run code in background and check stdout logs for updates (do no wait for it to end). 
------------------
HIGH LEVEL CONTEXT
------------------

- Context: This repo is code base for a research paper that impliments traders in double auction and introduces rl and llm agents

- Always end responses with a one-paragraph TL;DR. No bullets, just prose. Focus on big picture, not tiny details. Break the TL'DR into plan (what are you trying to do and why), progress (on current plans), problems (where you are stuck), future plans (what is next). Each gets 1-2 lines. 

- Documents: claude.md (instructions for the ai), plan.md (the master plan), checklists.md (the checklists to verify that we have faithfully replicated the santafe tournament and its traders and the rl and llm traders)

------------------
CORE DETAILS
------------------

- Metrics (metrics.md) in each market run we want to track: a) market efficiency (as a % of the best possible allocation i.e. central planner market clearing), b) individual performance / profits of each trader (and based on these its rank over the entire tournament), c) variation in the prices where trades happened (volatility and deviation from clearing prices) or how random the prices were (autocorrelation).

- Literature (reference/keypapers): key papers are gode-1993, smith-1962, rust-1994, cliff-1997, gd-1998, chen-2010. The insights from the literature are summarized in (literature.md): a) markets converge to the theoretical competitive equilibrium even when traders have strictly *private* information (knowing only their own costs/values) and possess no knowledge of aggregate supply and demand b) Gode & Sunder famously demonstrated that "Zero-Intelligence" can achieve near-perfect market efficiency. This implies that the allocative efficiency of a market is primarily a property of the market rather than the cognitive sophistication of the participants. c) Rust et al.), the winning strategy (sniper-Kaplan) was one of the simplest codebases entered. The dominant strategy was to to "wait in the background" and snipe a deal only when the bid-ask spread narrows. However, this strategy is parasitic—it relies on *other* (less intelligent) traders to reveal information and provide liquidity. For a market to function continuously, there must be a steady inflow of "noise traders" (impatient or random participants). d)Chen & Tai showed that while human-written strategies (like Kaplan) might dominate initially, autonomous "Genetic Programming" agents eventually learn to defeat them. If these 

- Primary Sources (1993 Santa Fe Tournament):**
  - **AURORA Protocol:** [DATManual_full_ocr.txt](./reference/oldcode/DATManual_full_ocr.txt) - Complete rules (4,303 lines)
  - **Tournament Analysis:** [chartradingdat_full_ocr.txt](./reference/oldcode/chartradingdat_full_ocr.txt) - Results & strategies (1,856 lines)
  - **Java Implementation:** `reference/oldcode/extracted/double_auction/java/da2.7.2/` - 49 Java files including:
    - SRobotKaplan.java, SRobotZI1.java, SRobotZI2.java (trader strategies)
    - SGameRunner.java, PeriodHistory.java (market logic)
  - **Key Papers (reference/keypapers/):**
    - [1962_smith_classroom_experiments.txt](./reference/keypapers/1962_smith_classroom_experiments.txt) - Smith's foundational CDA experiments
    - [1992_santafe_report.txt](./reference/keypapers/1992_santafe_report.txt) - Santa Fe tournament foundation
    - [1993_gode_sunder.txt](./reference/keypapers/1993_gode_sunder.txt) - ZIC benchmark
    - [1997_zip.txt](./reference/keypapers/1997_zip.txt) - ZIP algorithm (Cliff)
    - [1998_GD.txt](./reference/keypapers/1998_GD.txt) - GD algorithm
    - [2015_genetic_programming.txt](./reference/keypapers/2015_genetic_programming.txt) - GP agents (Chen et al.)

- Santa fe rules (rules.md): in this paper we work on the santafe rules and only that. No other rule set will apply. We can try different configs (settings) but the rules are god given and fixed. The tournament features autonomous computer programs acting as either buyers or sellers of tokens, operating within a star network topology where all communication is handled through a central Monitor program (the "AURORA Protocol"). The game proceeds in discrete time steps organized hierarchically into rounds and periods. For each round, players are assigned private token values (redemption values for buyers, costs for sellers) generated by a common formula based on four uniform random variables (`A`, `B`, `C`, `D`), whose ranges are governed by the four-digit `gametype` parameter. Critically, agents must consume their tokens sequentially to maximize profit, and the final ranking is determined by the total accumulated token profit, which is converted to dollar prizes based on the game's theoretical competitive equilibrium surplus. Trading is governed by "Chicago Rules" focusing on a continuous interaction between the current highest bid (`CurrentBid`) and the current lowest offer (`CurrentAsk`). In the alternating Bid-Ask steps, all eligible buyers may submit a bid, and all eligible sellers may submit an offer. To be valid, a new bid must be **strictly higher** than the standing `CurrentBid`, and a new offer must be **strictly lower** than the standing `CurrentAsk`. This rule provides essential **incumbent protection**; matching a standing price is illegal. If multiple agents submit the new best price simultaneously, a random draw breaks the tie to determine the new `CurrentBidder` or `CurrentAsker`. In the subsequent Buy-Sell step, only the holders of the best standing prices (`CurrentBidder` and `CurrentAsker`) are eligible to transact. The Bidder may issue a `BUY` request to accept the `CurrentAsk` price, or the Asker may issue a `SELL` request to accept the `CurrentBid` price. If a trade occurs, the transaction price is determined by the accepted standing price (not an average). Upon the completion of any successful trade, the entire market state is immediately cleared: both the `CurrentBid` and `CurrentAsk` are reset to null, forcing the next Bid-Ask step to start fresh. Periods conclude either when a fixed time limit (`ntimes`) is reached or when a specified number of consecutive steps (`deadsteps`) pass without any mutually profitable trades occurring.

- Tournament details (tournament.md): this contains the environments

- Paper structure (paper.md): this document contains the experimental design of the entire paper and all the simulation/experiments/tournaments we will need to do. This also contains the way we log experiments and also the configs that are needed to run all experiments. 

- Results (results.md): this document stores all the results of the experiments run in paper.md. This stores the main sets of results and from which the paper will be written. 

- Configs and Logs (configs_and_logs.md): This tells us the naming convention for logs and configs such that the repo is well structured. When you are running expeirments, you must follow this guidelines. 

- The paper itself: `paper/arxiv/` contains the LaTeX source. Structure:
  ```
  01_intro.tex           # Introduction
  02_research_motivation.tex  # Research motivation
  03_lit_review.tex      # Literature review
  04_method.tex          # Methodology (AURORA protocol, agents)
  05_results_zero_intel.tex   # Results: ZI/ZIC/ZIP baseline [DONE]
  06_results_santafe.tex      # Results: Santa Fe tournament (GD, Kaplan, etc.)
  07_results_rl.tex           # Results: PPO agents
  08_results_llm.tex          # Results: LLM agents
  08_discussion.tex           # Discussion & conclusion
  appendix_a_ppo_trader.tex   # PPO implementation details
  appendix_b_llm_trader.tex   # LLM prompts
  appendix_c_market.tex       # Market mechanics
  ``` 

------------------
PAPER WRITING STYLE
------------------

When writing or editing the research paper, follow these rules strictly. Do not use bullets or bulleted lists in the paper text. Do not use markdown formatting such as bold or italic in running prose. Do not use m-dashes. Structure the paper using paragraphs, sections, and subsections only. Tables are permitted and should remain anchored to the subsection in which they appear. Write in complete sentences and flowing paragraphs. Also when you make an update to the paper, compile it. Do not wait for me to ask you to compile.

------------------
TECHNICAL DETAILS
------------------

- Repo Structure
/
├── claude.md        # Instructions for AI agents
├── results.md       # Experiment design and reuslts
├── readme.md        # Basic project info
├── pyproject.toml   # Dependencies
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
├── reference/       # Source materials (read-only). This includes the original santa fe papers. 
├── paper/           # The research paper drafts and figures
└── archive/         # Old/deprecated code

-  Tinkering Guidelines:
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

- Tinkering Rules: a) NEVER add loose files to root b) NEVER add scripts to `engine/` or `traders/` c) Results go in `results/`, not scattered around d) One trader = one file in appropriate `traders/` subdirectory

- Workflow Loop:
1. read claude.md
2. read relevant parts in plan.md (never edit this)
3. read relevant bits in checklists/ folder
4. do the work and make changes. 
5. update tracker.md for daily log (reverse choronological, minimal, concise bullets only)
6. update results.md if we have main results from the experiment.

- Command Reference
uv sync                    # Install dependencies
mypy . --strict           # Type checking
ruff check .              # Linting
black .                   # Formatting
pytest tests/             # Run all tests
pytest tests/phase2/      # Run phase-specific tests
python scripts/run_experiment.py experiment=<name>
python scripts/run_ai_experiments.py --suite chen_ppo

