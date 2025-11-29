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
