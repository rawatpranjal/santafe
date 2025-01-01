# Santa Fe Double Auction

This repository replicates the Santa Fe Double Auction tournament in Rust et al., (1994) and conducts a number of experiments to study individual and market efficiency under different types of traders. My contribution is to introduce reinforcement learning agents into this market.

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Change Configuration (Optional):**
   - Edit `config.py` to set the type of auction, number of rounds, agent types, and other parameters.
3. **Run Simulation:**
   ```bash
   python main.py
   ```
4. **Check Results:**
   - Console output shows summary stats (market efficiency, average price vs. equilibrium, etc.).
   - Results are stored in the `exp/` folder.

## Repository Structure

```
DA/
│
├── code/               # Contains all core code for the project
│   ├── __pycache__/    # Compiled Python files
│   ├── data/           # Legacy folder for logs, results, and visualizations
│   ├── experiments/    # Folder for experiment configurations or results
│   ├── traders/        # Contains all trading strategies
│   │   ├── __pycache__/    # Compiled Python files for traders
│   │   ├── base.py         # Abstract BaseTrader class (common logic)
│   │   ├── gd.py           # Gjerstad-Dickhaut strategy implementation
│   │   ├── kaplan.py       # Kaplan (sniper-ish) strategy
│   │   ├── ppo.py          # Proximal Policy Optimization strategy
│   │   ├── registry.py     # Maintains registry of available trader classes
│   │   ├── zic.py          # Zero-Intelligence Constrained strategy
│   │   └── zip.py          # Zero-Intelligence Plus strategy
│   ├── auction.py       # Core double-auction mechanism (order book, matching logic)
│   ├── config.py        # High-level configuration (rounds, agents, price bounds, etc.)
│   ├── main.py          # Entry point to run the simulation
│   └── utils.py         # Utility functions for calculations and data handling
├── exp/                 # Contains all experiment outputs
│   ├── run_YYYY_MM_DD_HHMM/   # Unique folder for each run
│   │   ├── config_dump.json    # Snapshot of run configuration
│   │   ├── round_stats.csv     # Aggregated round-level statistics
│   │   ├── figure_efficiency.png  # Efficiency plot over rounds
│   │   └── ...                # Additional outputs (e.g., logs, custom plots)
├── docs/                # Documentation for the project
├── papers/              # Relevant research papers or related resources
├── LICENSE              # License for the project
├── README.md            # This document
└── requirements.txt     # Python dependencies for the project
```

### Key Features

- **Change Auction Settings:**
  - Before running the simulation, modify `config.py` to set:
    - Auction type
    - Agent mix (e.g., buyers/sellers and their strategies)
    - Price bounds, value distributions, etc.

- **Adding a New Trader:**
  1. Add a new file in the `traders/` folder. The file must define a class derived from `BaseTrader` (defined in `base.py`).
  2. Register the new trader in `traders/registry.py` by adding it to the `TRADER_REGISTRY` dictionary.

- **Saving Experiment Outputs:**
  - Each simulation run creates a unique folder under `exp/`.
  - Includes a `config_dump.json` file with the exact parameters used, `round_stats.csv` for stats, and any generated figures.

## Configuring Agents

- Modify **`config.py`** to specify agents and their parameters:
  ```python
  "buyers": [
      {"type": "zipbuyer"},
      {"type": "random"}
  ],
  "sellers": [
      {"type": "zipseller"},
      {"type": "gdseller"}
  ]
  ```
- Adjustable settings:
  - `num_rounds`, `num_steps`, `num_tokens`
  - Price bounds (`min_price`, `max_price`)
  - Value and cost distributions for agents

## Rules of the Double Auction

1. **Participants**: Buyers and sellers with private values/costs.
2. **Limit Orders**: Agents submit bids (buyers) or asks (sellers). An order book tracks the best bid and ask.
3. **Matching**:
   - Trade occurs when the best bid is >= the best ask.
   - Price is typically set as the midpoint.
4. **Iteration**: Runs over multiple steps. Agents can revise their bids/asks based on strategies.
5. **Profit**:
   - Buyer’s profit = (valuation - transaction price).
   - Seller’s profit = (transaction price - cost).

## Sample Results

**Aggregate over 100 rounds**:
```
  average market efficiency = 0.994
  average |avgPrice - eqPrice| = 0.0180
  average |actualTrades - eqQuantity| = 0.9800
```

**Strategy-Role Breakdown**:
```
  role      strategy  totalProfit  avgProfit_perTraderPerRound
 buyer  random-buyer   860.725163                     0.012296
 buyer     zip-buyer   131.555116                     0.013156
 buyer      gd-buyer   136.636631                     0.013664
 buyer  kaplan-buyer   117.466853                     0.011747
seller random-seller   851.181127                     0.012160
seller    zip-seller   130.033183                     0.013003
seller     gd-seller   132.599890                     0.013260
seller kaplan-seller   118.383562                     0.011838
```

**Round-Level Stats**:
```
 round  eq_q     eq_p  eq_surplus  actual_trades  actual_total_profit  market_efficiency  avg_price  abs_diff_price  abs_diff_quantity
     0    49 0.569304   24.260192             51            23.915447           0.985790   0.526847        0.042457                  2
     1    52 0.483575   25.443188             52            25.443188           1.000000   0.478207        0.005369                  0
     2    54 0.400448   26.396005             52            25.520234           0.966822   0.455795        0.055347                  2
     3    52 0.450308   24.151844             52            24.051290           0.995837   0.476160        0.025853                  0
     4    50 0.531266   24.343040             51            24.231057           0.995400   0.533042        0.001776                  1
```

## References

1. Cason, T. N., & Friedman, D. (1996). Price formation in double auction markets. *Journal of Economic Dynamics and Control, 20*, 1307–1337.

2. Chen, S.-H. (2017). Agent-based computational economics: How the idea originated and where it is going. In K. Schenk (Ed.), *Handbook of Computational Economics* (pp. 1–28). Routledge.

3. Chen, S.-H., & Tai, C.-C. (2010). The Agent-Based Double Auction Markets: 15 Years On. In *Simulating Interacting Agents and Social Phenomena* (eds. Takadama, K., Cioffi-Revilla, C., & Deffuant, G.), 119–136. Springer Japan, Tokyo.

4. Cliff, D., & Bruten, J. (n.d.). Zero is not enough: On the lower limit of agent intelligence for continuous double auction markets.

5. De Luca, M., & Cliff, D. (2011). Agent-human interactions in the continuous double auction. In *Proceedings of the 3rd International Conference on Applied Human Factors and Ergonomics*, 768–775.

6. Easley, D., & Ledyard, J. O. (1993). Theories of price formation and exchange in double auction markets. In D. Friedman & J. Rust (Eds.), *The Double Auction Market: Institutions, Theories, and Evidence* (pp. 63–97). Addison-Wesley.

7. Friedman, D. (1991). A simple testable model of price formation in the CDA market. *Unpublished manuscript, University of California Santa Cruz*.

8. Friedman, D. (2018). The Double Auction Market Institution: A Survey. In *The Double Auction Market Institutions, Theories, and Evidence* (eds. Friedman, D., & Rust, J.), 3–26. Routledge.

9. Gjerstad, S., & Dickhaut, J. (1998). Price Formation in Double Auctions. *Games and Economic Behavior, 22*, 1–29.

10. Gode, D. K., & Sunder, S. (1993). Allocative Efficiency of Markets with Zero-Intelligence Traders: Market as a Partial Substitute for Individual Rationality. *Journal of Political Economy, 101*, 119–137.

11. Rust, J., Miller, J. H., & Palmer, R. (1994). Characterizing effective trading strategies in a computerized double auction market. *Santa Fe Institute Working Paper 94-03-014*.

12. Rust, J., Miller, J. H., & Palmer, R. (2018). Behavior of Trading Automata in a Computerized Double Auction Market. In *The Double Auction Market Institutions, Theories, and Evidence* (eds. Friedman, D., & Rust, J.), 155–198. Routledge.

13. Smith, V. L. (1962). An Experimental Study of Competitive Market Behavior. *Journal of Political Economy*.

14. Tesauro, G., & Das, R. (2001). High-performance bidding agents for the continuous double auction. In *Proceedings of the 3rd ACM conference on Electronic Commerce*, 206–209. ACM, Tampa Florida USA.

15. Wilson, R. B. (1984). On equilibria of bid-ask markets. *Technical Report No. 452, Stanford University*.

16. Zhan, W., & Friedman, D. (2007). Markup strategies and the paradox of high double auction efficiency. *Working paper, University of California, Santa Cruz*.
