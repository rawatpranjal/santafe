# Visualization Spec

## Market Heartbeat (Bidding Activity)

The primary visualization for understanding trader behavior and market dynamics.

### Specification

| Element | Description |
|---------|-------------|
| **Type** | Histogram / KDE Plot |
| **X-Axis** | Time Step (0 to NTIMES, typically 0-100) |
| **Y-Axis** | Frequency of actions (bids/asks submitted) |
| **Facet** | By Trader Type (ZI, ZIC, ZIP, GD, Kaplan, PPO, LLM) |

### Data Required

- `step`: Time step when bid/ask was submitted
- `trader_type`: Strategy classification
- `action_type`: Bid or Ask
- `accepted`: Whether the action led to a trade

### Expected Patterns

| Trader | Expected Distribution | Interpretation |
|--------|----------------------|----------------|
| **ZI/ZIC** | Uniform across all steps | Random firing, no timing strategy |
| **Kaplan** | Heavy right skew (late period) | Sniper behavior, waits then strikes |
| **ZIP/GD** | Front-loaded, tapering off | Active early, converges to equilibrium |
| **PPO** | TBD | Did AI learn to wait (right skew) or panic (uniform)? |
| **LLM** | TBD | Zero-shot timing intuition |

### Key Questions Answered

1. **Patience**: Which traders wait vs. act immediately?
2. **Sniping**: Does Kaplan show the expected "wait-and-steal" pattern?
3. **Learning**: Did PPO discover timing as a strategy dimension?
4. **Closing Panic**: Is there a volume spike in final 10-20% of period?

### Implementation Notes

```python
# Pseudocode
df_bids = load_bid_history()  # columns: step, trader_type, action_type
sns.histplot(data=df_bids, x='step', hue='trader_type', bins=20, stat='density')
plt.axvline(x=0.9*NTIMES, linestyle='--', label='Closing panic zone')
```

---

## Convergence Tunnel (Price Discovery)

The essential visualization for understanding "what actually happened" in a single market run.

### Specification

| Element | Description |
|---------|-------------|
| **Type** | Time Series / Scatter Plot |
| **X-Axis** | Time Step (sequence number) |
| **Y-Axis** | Price (absolute, integer) |
| **CE Band** | Shaded horizontal region around equilibrium price |
| **Best Bid** | Blue stepped line showing highest bid trajectory |
| **Best Ask** | Red stepped line showing lowest ask trajectory |
| **All Bids** | Small blue dots (alpha=0.3) showing all bid submissions |
| **All Asks** | Small red dots (alpha=0.3) showing all ask submissions |
| **Trades** | Large green circles marking executed transactions |

### Data Required

- `step`: Time step of action
- `price`: Bid/ask/trade price
- `is_buyer`: True for bids, False for asks
- `status`: "winner", "beaten", "pass" etc.
- `event_type`: "bid_ask", "trade", "period_start"
- `equilibrium_price`: From period_start event (or CLI override)

### What This Reveals

| Pattern | Interpretation |
|---------|----------------|
| **Spread narrowing** | Bid/ask lines converging = market finding equilibrium |
| **Trades in CE band** | High efficiency, prices near theoretical optimum |
| **Scattered trades** | Low efficiency, volatile/random pricing |
| **Late trades** | Closing panic or patient snipers |
| **Wide spread persist** | Market illiquidity or strategic waiting |

### Key Questions Answered

1. **Efficiency**: Do transaction prices cluster around CE?
2. **Convergence**: How quickly does the spread narrow?
3. **Strategic Behavior**: Wide spread = passive sniping (Kaplan) vs tight spread = active discovery (ZIC)
4. **Volatility**: Are prices stable or scattered?

### Usage

```bash
# New logs (auto CE price from period_start event)
python scripts/visualize_tunnel.py logs/exp_events.jsonl -r 1 -p 1

# Old logs (manual CE price override)
python scripts/visualize_tunnel.py logs/old_events.jsonl --ce-price 150

# Save to file
python scripts/visualize_tunnel.py logs/exp_events.jsonl -o figures/tunnel.png

# With summary statistics
python scripts/visualize_tunnel.py logs/exp_events.jsonl --summary
```
