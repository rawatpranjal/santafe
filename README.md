# "Santa Fe" Discrete Double Auction

A simple Discrete Double Auction (CDA) simulator with multiple agent strategies: **ZIC**, **Kaplan**, **Gjerstad-Dickhaut (GD)**, and **ZIP**.

## Repository Structure

```
my_project/
│
├── main.py               # Entry point to run the entire simulation
├── config.py             # Configuration parameters (rounds, steps, agent specs, etc.)
├── utils.py              # Utility functions (e.g. compute_equilibrium)
├── auction.py            # Auction logic & market session
├── traders/
│   ├── __init__.py
│   ├── base.py           # BaseTrader parent class
│   ├── zic.py            # ZIC (Zero-Intelligence Constrained) agents
│   ├── kaplan.py         # Kaplan agents
│   ├── gd.py             # Gjerstad-Dickhaut (GD) agents
│   └── zip.py            # ZIP (Zero-Intelligence Plus) agents
└── round_stats.csv       # (created after running, logs summary stats per round)
```

## How to Run

1. **Install dependencies**:
   ```bash
   pip install pandas matplotlib numpy
   ```
2. **Run**:
   ```bash
   python main.py
   ```
3. The simulator prints summary stats and saves `round_stats.csv`.

## Changing Agent Config

- Edit **`config.py`**:
  - Adjust `num_rounds`, `num_steps`, `num_tokens`, etc.
  - Modify `buyers`/`sellers` arrays to choose agent types:
    - `"random"`, `"kaplan"`, `"gdbuyer"/"gdseller"`, `"zipbuyer"/"zipseller"`, etc.
  - For example:
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
4. **Results**:
   - Console shows round-level outcomes (market efficiency, average price vs. equilibrium, etc.).
   - `round_stats.csv` stores final stats for all rounds.

## Sample Results

Here is an example of the console output after running 100 rounds:

```
Aggregate over 100 rounds:
  average market efficiency = 0.994
  average |avgPrice - eqPrice| = 0.0180
  average |actualTrades - eqQuantity| = 0.9800

=== Aggregated Strategy-Role ===
  role      strategy  totalProfit  avgProfit_perTraderPerRound
 buyer  random-buyer   860.725163                     0.012296
 buyer     zip-buyer   131.555116                     0.013156
 buyer      gd-buyer   136.636631                     0.013664
 buyer  kaplan-buyer   117.466853                     0.011747
seller random-seller   851.181127                     0.012160
seller    zip-seller   130.033183                     0.013003
seller     gd-seller   132.599890                     0.013260
seller kaplan-seller   118.383562                     0.011838

Saved round-level stats to 'round_stats.csv'
```

Below is a snippet of the **round-level stats**:

```
=== Round-Level Stats ===
 round  eq_q     eq_p  eq_surplus  actual_trades  actual_total_profit  market_efficiency  avg_price  abs_diff_price  abs_diff_quantity
     0    49 0.569304   24.260192             51            23.915447           0.985790   0.526847        0.042457                  2
     1    52 0.483575   25.443188             52            25.443188           1.000000   0.478207        0.005369                  0
     2    54 0.400448   26.396005             52            25.520234           0.966822   0.455795        0.055347                  2
     3    52 0.450308   24.151844             52            24.051290           0.995837   0.476160        0.025853                  0
     4    50 0.531266   24.343040             51            24.231057           0.995400   0.533042        0.001776                  1
```

## Extending / Customizing

- To create your own trading strategy, add a new file under `traders/`, subclass `BaseTrader`, then update the `create_traders_for_round()` method in `auction.py` to support the new type.