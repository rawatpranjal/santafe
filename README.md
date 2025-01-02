# Santa Fe Double Auction

Replicates the Santa Fe Double Auction tournament in Rust et al. (1994), with additional experiments studying market efficiency under various traders. Includes simple ZIC traders and advanced RL agents.

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Modify Config and Run Simulation**:
   ```bash
   cd code
   python main.py
   ```
3. **View Results**:
   - Logs and plots are created in `exp/[EXPERIMENT_NAME]/`.
   - `round_log.csv` (round-level data), `log.csv` (step-level data).
   - `round_X_1x3.png` (per-round price/demand plots) and `game_summary.png` (efficiency/surplus/profit).

## Repository Structure

```
DA/
│
├── code/
│   ├── auction.py         # Core double-auction mechanism
│   ├── config.py          # Experiment config (num_rounds, tokens, strategies, etc.)
│   ├── main.py            # Entry point to run the simulation
│   ├── utils.py           # Utility functions (equilibrium calc, plotting, analysis)
│   ├── traders/           # Trading strategy classes
│   │   ├── base.py
│   │   ├── zic.py         # Zero-Intelligence Constrained
│   │   ├── gd.py          # Gjerstad-Dickhaut
│   │   ├── kaplan.py      # Kaplan
│   │   ├── ppo.py         # Proximal Policy Optimization
│   │   └── registry.py    # Registry for all traders
│   └── ...
├── exp/                   # Stores experiment outputs
│   └── 001_zic_only_gode_sunder/
│       ├── config.py
│       ├── log.csv
│       ├── round_log.csv
│       ├── round_0_1x3.png
│       ├── game_summary.png
│       └── ...
├── docs/
├── papers/
├── LICENSE
├── README.md              
└── requirements.txt
```

## Rules of the Double Auction

1. **Participants**: Buyers and sellers each hold private value/cost draws.  
2. **Limit Orders**: Buyers submit bids, sellers submit asks. The best (highest) bid and best (lowest) ask define the quote.  
3. **Matching**: A trade executes when the best bid ≥ best ask; the transaction price is typically the midpoint.  
4. **Steps**: Multiple steps per round; agents can adjust bids/asks.  
5. **Profit**: Buyer profit = valuation - price; seller profit = price - cost.

## Structure of the Traders

- **BaseTrader** (in `traders/base.py`) defines common logic (tokens, profit).
- Each derived file (`zic.py`, `gd.py`, etc.) implements a custom bidding/asking strategy.
- `registry.py` maintains a dictionary so `auction.py` can load traders by string type.

## Adding Traders

1. Create a new file in `traders/`, defining a class inheriting from `BaseTrader`.
2. Register it in `traders/registry.py` with a unique type name.  
3. Modify `config.py` to include that strategy in `"buyers"` or `"sellers"`.

## Adding Experiments

1. Edit `config.py` with new experiment parameters (number of rounds, tokens, which strategies, etc.).  
2. Run `python main.py` to generate a folder in `exp/` named after the experiment.  
3. Check `log.csv` (step-level) and `round_log.csv` (round-level), plus plots (`round_[X]_1x3.png` and `game_summary.png`).

## Example STDOUT LOG

```
=== CONFIG SETTINGS ===
{'experiment_name': '001_zic_only_gode_sunder',
 'experiment_dir': 'experiments',
 'num_rounds': 4,
 'num_periods': 4,
 'num_steps': 50,
 'num_tokens': 4,
 ...
}

=== STEP LOGS FOR ROUND=0 ===
 round  period  step    bids            asks           cbid  cask  trade  price  ...
    0       0     0  [0.53, None, ...] [None,0.58,...] 0.53  0.58     0    None  ...
    1       0     1  ...
...

=== INDIVIDUAL BOT PERFORMANCE (ACROSS ALL ROUNDS) ===
+--------+---------------+--------+------------+-----------+-----------+--------------+-----------+
|  Role  |   Strategy    | BotName| MeanProfit | StdProfit | MinProfit | MedianProfit | MaxProfit |
+--------+---------------+--------+------------+-----------+-----------+--------------+-----------+
| buyer  | zic           | B0     |   0.6234   |  0.0507   |  0.3456   |    0.5901    |  0.9022   |
| buyer  | zic           | B1     |   0.5775   |  0.0412   |  0.3100   |    0.5601    |  0.8744   |
| ...    | ...           | ...    |    ...     |    ...    |    ...    |      ...     |    ...    |
+--------+---------------+--------+------------+-----------+-----------+--------------+-----------+

=== MARKET PERFORMANCE (AGGREGATE) ===
+------------------+----------------+-----------------+-----------------+-------------+--------------+
| MarketEff(Mean)  | MarketEff(Std) | BuyerSurplus%   | SellerSurplus%  | AvgPriceDiff| AvgQuantDiff |
+------------------+----------------+-----------------+-----------------+-------------+--------------+
|     0.9832       |     0.0121    |    46.20%       |     53.80%      |   0.0423    |    1.5000    |
+------------------+----------------+-----------------+-----------------+-------------+--------------+
```

## References

1. Smith, V. L. (1962). An Experimental Study of Competitive Market Behavior. \textit{Journal of Political Economy}.
2. Gode, D. K., \& Sunder, S. (1993). Allocative Efficiency of Markets with Zero-Intelligence Traders: Market as a Partial Substitute for Individual Rationality. \textit{Journal of Political Economy, 101}, 119–137.
3. Rust, J., Miller, J. H., \& Palmer, R. (1994). Characterizing effective trading strategies in a computerized double auction market. \textit{Santa Fe Institute Working Paper 94-03-014}.
4. Chen, S.-H., \& Tai, C.-C. (2010). The Agent-Based Double Auction Markets: 15 Years On. In \textit{Simulating Interacting Agents and Social Phenomena} (eds.\ Takadama, K., Cioffi-Revilla, C., \& Deffuant, G.), 119–136. Springer Japan, Tokyo.