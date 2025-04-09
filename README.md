# Santa Fe Double Auction Experiments

This project simulates the Santa Fe Double Auction (SFI DA) environment, based on the mechanism described by Rust et al. (1994). It allows for the comparison of various trading strategies, including classic heuristics and modern reinforcement learning agents, and facilitates the analysis of market outcomes like efficiency and price convergence.

## Introduction

The simulation replicates the discrete-time, order-driven double auction mechanism used in the original SFI DA tournament. It serves as a testbed for evaluating agent-based trading strategies. Key features include:

*   Implementation of the core SFI DA rules (alternating Bid/Ask and Buy/Sell steps, AURORA mechanism for acceptance).
*   Support for multiple trading periods within a round, and multiple rounds per experiment.
*   A framework where each **round** constitutes a single **episode** for Reinforcement Learning agents.
*   A collection of implemented trading strategies.
*   Configurable market parameters (number of agents, tokens, time steps, value distributions via `gametype`).
*   Automated analysis of results, including market performance, individual agent performance (ranked by profit), and strategy tournament rankings (ranked by average rank).

## Implemented Strategies

*   **Baselines:**
    *   `zi`: Zero-Intelligence (Unconstrained)
    *   `zic`: Zero-Intelligence Constrained (Gode & Sunder, 1993)
    *   `zip`: Zero-Intelligence Plus (Cliff & Bruten, 1997 - adaptive margin)
*   **Heuristic / Belief-Based:**
    *   `kaplan`: Kaplan "Sniper" (Rust et al., 1994)
    *   `gd`: Gjerstad-Dickhaut (Simplified belief-based)
    *   `el`: Easley-Ledyard (Adaptive reservation prices)
*   **Reinforcement Learning:**
    *   `ppo_lstm`: Proximal Policy Optimization with LSTM network (implements PPO-LSTM core logic)

## Project Structure

```
santafe/ # Root directory (adjust name as needed)
│
├── code/
│ ├── auction.py # Core double-auction mechanism logic
│ ├── main.py # Entry point, loads config, runs auction, analysis
│ ├── run_experiments.py # Script to run multiple experiments sequentially
│ ├── utils.py # Utility functions (equilibrium calc, plotting, analysis tables)
│ ├── requirements.txt # Python dependencies
│ │
│ ├── configs/ # Directory for experiment configuration files
│ │ ├── 01a_baseline_ziu_symmetric_1k.py
│ │ ├── 02a_kaplan_mix_baseline_1k.py
│ │ └── ... # Other .py config files
│ │
│ ├── traders/ # Trading strategy implementations
│ │ ├── base.py # BaseTrader class definition
│ │ ├── zi.py
│ │ ├── zic.py
│ │ ├── zip.py
│ │ ├── el.py
│ │ ├── gd.py
│ │ ├── kaplan.py
│ │ ├── ppo.py # PPOTrader interface class (uses ppo_lstm_core)
│ │ ├── ppo_lstm_core.py # PPO+LSTM algorithm implementation (Agent + Logic)
│ │ └── registry.py # Maps strategy names to classes
│ │
│ └── pycache/ # Python cache files
│
├── experiments/ # Stores outputs for each experiment run
│ └── 02a_kaplan_mix_baseline_1k/ # Example experiment output folder
│ ├── config_used.py # Copy of the config file used for this run
│ ├── 02a_kaplan_mix_baseline_1k_run.log # Detailed log file for the run
│ ├── round_log_all.csv # CSV with summary statistics for each round
│ ├── step_log_all.csv # CSV with detailed data for each step (can be large)
│ ├── market_summary_plot.png # Plot of efficiency, price deviation over rounds
│ └── ppo_training_curves.png # Plot of RL training metrics (if RL agents were trained)
│ └── models/ # Optional: Saved RL agent models
│ └── ppo_lstm_agent_B0_agent.pth
│ └── ... # Other experiment output folders
│
├── docs/ # (Optional) Documentation files
├── papers/ # (Optional) Related papers
├── LICENSE
└── README.md # This file
```

## Installation

1.  Ensure you have Python 3.9+ installed.
2.  Navigate to the `code` directory.
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Key dependencies include `numpy`, `pandas`, `matplotlib`, `torch`, `tqdm`, `tabulate`)*

## Running Experiments

Experiments are defined by Python configuration files (`.py`) placed in the `code/configs/` directory. To run one or more experiments:

1.  **Define Experiments:** Create or modify `.py` files in the `code/configs/` directory. Each file should contain a `CONFIG` dictionary specifying the parameters for that experiment (see Configuration section below).
2.  **Select Experiments:** Open `code/run_experiments.py` and edit the `config_files_to_run` list to include the *filenames* of the configuration files you want to execute.
3.  **Run the Runner Script:** Navigate to the `code` directory in your terminal and run:
    ```bash
    python run_experiments.py
    ```
    This script will iterate through the specified config files, execute `main.py` for each one, and save the results in separate folders under `experiments/`. The output folder name will match the config filename (without the `.py` extension).

## Configuration

Experiment parameters are defined within Python dictionaries named `CONFIG` inside files within the `code/configs/` directory. Key parameters include:

*   `experiment_name`: A unique name for the experiment (often overridden by `run_experiments.py`).
*   `experiment_dir`: Base directory for saving results (usually `"experiments"`).
*   `num_rounds`: Total number of simulation rounds (episodes).
*   `num_periods`: Number of trading periods within each round.
*   `num_steps`: Number of discrete time steps within each period.
*   `num_training_rounds`: Number of initial rounds dedicated to training RL agents.
*   `num_buyers`, `num_sellers`: Number of agents of each type.
*   `num_tokens`: Number of tokens each agent can trade per period.
*   `min_price`, `max_price`: Market price bounds.
*   `gametype`: Integer defining the SFI value distribution parameters (e.g., 6453).
*   `buyers`, `sellers`: Lists of dictionaries specifying the `type` (strategy name from `registry.py`) and optional `init_args` for each agent.
*   `rl_params`: Dictionary containing hyperparameters for RL agents (e.g., `lr`, `gamma`, `lstm_hidden_size`).
*   `rng_seed_values`, `rng_seed_auction`, `rng_seed_rl`: Seeds for random number generation.
*   `log_level`, `log_level_rl`: Logging verbosity for console/file outputs.
*   `save_rl_model`, `load_rl_model_path`: Options for saving/loading trained RL models.
*   `generate_per_round_plots`, `generate_eval_behavior_plots`: Flags to enable/disable certain plots.

## Output Files

For each experiment run via `run_experiments.py`, a folder is created under `experiments/` (e.g., `experiments/02a_kaplan_mix_baseline_1k/`). This folder contains:

*   `config_used.py`: A copy of the configuration file used for that specific run, ensuring reproducibility.
*   `<experiment_name>_run.log`: A detailed text log file containing setup information, auction progress, debugging messages (depending on log level), and the final analysis tables.
*   `round_log_all.csv`: A CSV file where each row represents one round, containing aggregate market statistics (efficiency, avg price, deviations, total profit, etc.) and serialized details about agent performance within that round.
*   `step_log_all.csv`: A (potentially very large) CSV file where each row represents a single time step within a period, logging market state (best bid/ask, phibid/phiask), submitted quotes, and trade details if a trade occurred.
*   `market_summary_plot.png`: A plot showing market efficiency and average price deviation over the course of the evaluation rounds.
*   `ppo_training_curves.png`: (If RL agents were trained) Plots showing training metrics like policy loss, value loss, and entropy over training rounds.
*   `models/`: (If `save_rl_model` was `True`) Contains saved model files (`.pth`) for trained RL agents.

## Adding New Traders

1.  Create a new Python file in the `code/traders/` directory (e.g., `my_strategy.py`).
2.  Define your trader class(es) within this file, inheriting from `traders.base.BaseTrader`. Implement the required methods, especially `make_bid_or_ask`, `request_buy`, and `request_sell`.
3.  Open `code/traders/registry.py`.
4.  Import your new trader class(es) (e.g., `from .my_strategy import MyBuyer, MySeller`).
5.  Add a new entry to the `_TRADER_CLASSES` dictionary, mapping a unique lowercase string name for your strategy to a tuple containing your buyer and seller classes (e.g., `"mystrat": (MyBuyer, MySeller),`).
6.  You can now use the string name (e.g., `"mystrat"`) as the `type` in the `buyers` or `sellers` list within a configuration file in the `configs/` directory.

## Analysis Output Example

The script automatically performs analysis on the evaluation rounds and prints summary tables to the console and the main log file (`*_run.log`). Example formats:

**Individual Bot Performance (Ranked)**
```
=== INDIVIDUAL BOT PERFORMANCE (Aggregated over 100 Rounds) ===

--- Buyers (Ranked by Mean Profit) ---
+------+----------+---------+------------+-----------+-----------+--------------+-----------+
| Rank | Strategy | BotName | MeanProfit | StdProfit | MinProfit | MedianProfit | MaxProfit |
+------+----------+---------+------------+-----------+-----------+--------------+-----------+
| 1 | gd | B2 | 548.58 | 359.30 | 14.00 | 464.00 | 1673.00 |
| 2 | el | B3 | 485.52 | 306.63 | 0.00 | 468.50 | 1524.00 |
| 3 | zip | B1 | 484.37 | 346.60 | 0.00 | 462.50 | 1610.00 |
| 4 | kaplan | B0 | 378.83 | 299.63 | 0.00 | 344.50 | 1384.00 |
| 5 | zic | B4 | 327.93 | 294.78 | 0.00 | 254.50 | 1394.00 |
+------+----------+---------+------------+-----------+-----------+--------------+-----------+

--- Sellers (Ranked by Mean Profit) ---
+------+----------+---------+------------+-----------+-----------+--------------+-----------+
| Rank | Strategy | BotName | MeanProfit | StdProfit | MinProfit | MedianProfit | MaxProfit |
+------+----------+---------+------------+-----------+-----------+--------------+-----------+
| 1 | gd | S2 | 557.34 | 295.12 | 6.00 | 541.50 | 1398.00 |
| 2 | zip | S1 | 520.13 | 300.35 | 0.00 | 521.50 | 1547.00 |
| 3 | el | S3 | 514.80 | 309.33 | 20.00 | 522.00 | 1367.00 |
| 4 | kaplan | S0 | 466.22 | 316.73 | 0.00 | 419.50 | 1518.00 |
| 5 | zic | S4 | 203.74 | 180.79 | 0.00 | 162.50 | 716.00 |
+------+----------+---------+------------+-----------+-----------+--------------+-----------+
```
**Market Performance**
```
=== MARKET PERFORMANCE (Mean (StdDev) over 100 Rounds) ===
+-------------+----------------+-----------------+---------------+----------------+-------------+-------------+
| Market Eff | AvgBuyerProfit | AvgSellerProfit | BuyerSurplus% | SellerSurplus% | AvgPriceDev | AvgQuantDev |
+-------------+----------------+-----------------+---------------+----------------+-------------+-------------+
| 1.00 (0.00) | 445 (122) | 452 (122) | 49.5% (9.0) | 50.5% (9.0) | 13.4 (10.6) | 20.0 (3.2) |
+-------------+----------------+-----------------+---------------+----------------+-------------+-------------+
Note: Efficiency clamped to [0, 1] for 100 rounds for averaging.

**Strategy Tournament Ranking**

=== STRATEGY TOURNAMENT RANKING (Aggregated over 100 Rounds) ===
+----------+-------------+-----------+-----------+-----------+--------------+
| Strategy | Mean Profit | (Std Dev) | Mean Rank | (Std Dev) | Agent-Rounds |
+----------+-------------+-----------+-----------+-----------+--------------+
| gd | 552.96 | (328.81) | 4.59 | (2.55) | 200 |
| el | 500.16 | (308.33) | 4.87 | (2.86) | 200 |
| zip | 502.25 | (324.80) | 5.00 | (2.74) | 200 |
| kaplan | 422.52 | (311.38) | 5.76 | (2.82) | 200 |
| zic | 265.83 | (252.28) | 7.28 | (2.54) | 200 |
+----------+-------------+-----------+-----------+-----------+--------------+
```
## References

1.  Smith, V. L. (1962). An Experimental Study of Competitive Market Behavior. *Journal of Political Economy*.
2.  Gode, D. K., & Sunder, S. (1993). Allocative Efficiency of Markets with Zero-Intelligence Traders: Market as a Partial Substitute for Individual Rationality. *Journal of Political Economy, 101*(1), 119–137.
3.  Rust, J., Miller, J. H., & Palmer, R. (1994). Characterizing effective trading strategies: Insights from a computerized Double Auction Tournament. *Journal of Economic Dynamics and Control, 18*(1), 61–96. (Also available as SFI Working Paper 94-03-014).
4.  Cliff, D., & Bruten, J. (1997). Zero is Not Enough: On the lower limit of agent intelligence for Continuous Double Auction Markets. *HP Laboratories Technical Report HPL-97-141*.
5.  Gjerstad, S., & Dickhaut, J. (1998). Price formation in double auctions. *Games and Economic Behavior, 22*(1), 1–29.
6.  Easley, D., & Ledyard, J. (1993). Theories of price formation and exchange in double oral auctions. In D. Friedman and J. Rust (Eds.), *The Double Auction Market: Institutions, Theories, and Evidence* (pp. 63–97). Addison-Wesley.
7.  Chen, S.-H., & Tai, C.-C. (2010). The Agent-Based Double Auction Markets: 15 Years On. In *Simulating Interacting Agents and Social Phenomena* (eds. Takadama, K., Cioffi-Revilla, C., & Deffuant, G.), 119–136. Springer Japan, Tokyo.