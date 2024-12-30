# "Santa Fe" Discrete Double Auction

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install pandas matplotlib numpy
   ```
2. **Run Simulation:**
   ```bash
   python main.py
   ```
3. **Check Results:**
   - Console output shows summary stats (market efficiency, average price vs. equilibrium, etc.).
   - `round_stats.csv` stores final stats per round.

## Repository Structure

```
my_project/
│
├── main.py
├── config.py
├── utils.py
├── auction.py
├── traders/
│   ├── __init__.py
│   ├── base.py
│   ├── zic.py
│   ├── kaplan.py
│   ├── gd.py
│   └── zip.py
└── round_stats.csv
```

- **`main.py`**: Entry point to run the simulation.  
- **`config.py`**: High-level configuration (number of rounds/steps, agent types, etc.).  
- **`utils.py`**: Utility functions (like equilibrium calculation).  
- **`auction.py`**: Core double-auction mechanism (tracks bids, asks, trading).  
- **`traders/`**: Various trading strategies:
  - `base.py`: Abstract `BaseTrader` class (common logic).
  - `zic.py`: Zero-Intelligence Constrained (random).
  - `kaplan.py`: Kaplan (sniper-ish) approach.
  - `gd.py`: Gjerstad-Dickhaut strategy.
  - `zip.py`: Zero-Intelligence Plus (ZIP) strategy.

## Configuring Agents

- In **`config.py`**, edit `buyers` and `sellers` arrays to specify the agent mix:
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
- You can also tweak:
  - `num_rounds`, `num_steps`, and `num_tokens`.
  - Price bounds (`min_price`, `max_price`).
  - Value/cost distributions for buyers and sellers.

---

## Rules of the Double Auction

1. **Participants**: Buyers and sellers, each with private values/costs.  
2. **Limit Orders**: Each agent can post a bid (if buyer) or ask (if seller). The best (highest) bid and best (lowest) ask are maintained by the order book.  
3. **Matching**:  
   - If the best bid is >= the best ask, a trade occurs at some agreed price (e.g. midpoint).  
   - Each trade removes one unit of the buyer’s and one unit of the seller’s “tokens” (inventory).  
4. **Iteration**: Repeated over multiple steps (discrete time). Agents can continuously update their bid/ask based on market conditions or internal strategies.  
5. **Profit**:  
   - Buyer’s profit = (valuation - transaction price).  
   - Seller’s profit = (transaction price - cost).  

---

## Sample Results

**Aggregate over 100 rounds**:
```
  average market efficiency = 0.994
  average |avgPrice - eqPrice| = 0.0180
  average |actualTrades - eqQuantity| = 0.9800
```

**Aggregated by Strategy-Role**:
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

**Snippet of Round-Level Stats**:
```
 round  eq_q     eq_p  eq_surplus  actual_trades  actual_total_profit  market_efficiency  avg_price  abs_diff_price  abs_diff_quantity
     0    49 0.569304   24.260192             51            23.915447           0.985790   0.526847        0.042457                  2
     1    52 0.483575   25.443188             52            25.443188           1.000000   0.478207        0.005369                  0
     2    54 0.400448   26.396005             52            25.520234           0.966822   0.455795        0.055347                  2
     3    52 0.450308   24.151844             52            24.051290           0.995837   0.476160        0.025853                  0
     4    50 0.531266   24.343040             51            24.231057           0.995400   0.533042        0.001776                  1
```