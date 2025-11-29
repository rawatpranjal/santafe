# GD vs ZIC Market Crash Analysis

## Problem Statement

When mixing 3 GD + 2 ZIC agents on each side:
- **Actual Efficiency**: 38.99%
- **Expected Efficiency**: 85-95% (GD should outperform ZIC)
- **Homogeneous Baselines**:
  - ZIC vs ZIC: 95.88%
  - GD vs GD: 100.00%

**Critical finding**: Mixed market performs WORSE than both homogeneous markets.

## Hypothesis: Belief Pollution

### GD's Belief Formation Mechanism

GD agents form beliefs about price acceptance probabilities:
- `p(a)` = probability ask `a` is accepted
- `q(b)` = probability bid `b` is accepted

These are calculated from historical observations:
```
p(a) = TA(a) / [TA(a) + RB(a)]
q(b) = TB(b) / [TB(b) + RA(b)]

Where:
- TA(a) = accepted asks ≤ a
- RB(a) = rejected bids ≥ a
- TB(b) = accepted bids ≥ b
- RA(b) = rejected asks ≤ b
```

### ZIC's Quote Behavior

ZIC submits **random** quotes uniformly distributed across the profitable range:
- Buyers: `bid ~ U[price_min, valuation]`
- Sellers: `ask ~ U[cost, price_max]`

With `price_min=1` and `price_max=1000`, ZIC generates highly **dispersed** quotes.

### The Pollution Mechanism

When GD observes ZIC's quotes:

1. **Noise Injection**: ZIC's random bids/asks at extreme prices (e.g., buyer bids 5, seller asks 990) create misleading history

2. **Distorted Beliefs**: GD calculates acceptance probabilities based on this noisy data
   - If ZIC submits a bid at 500 that's rejected, GD thinks "bids ≥500 are unlikely to be accepted"
   - But the rejection was due to random chance, not market fundamentals

3. **Suboptimal Decisions**: GD's expected surplus calculations become wrong
   - May be too conservative (wait instead of accept)
   - May be too aggressive (accept bad prices)

4. **Market Breakdown**: If GD agents dominate (3 GD vs 2 ZIC), their collective bad decisions prevent efficient trading

## Evidence to Collect

To confirm this hypothesis, we need to check:

1. **Quote Distribution**: Are ZIC agents submitting extreme quotes that pollute GD's history?
2. **GD Belief Values**: Do GD agents have distorted p(a) and q(b) beliefs?
3. **Trading Pattern**: Are GD agents accepting/rejecting at the wrong times?
4. **History Size**: How many observations do GD agents have? (more noise = worse performance)

## Potential Fixes

### Option 1: Bounded Price Range
Instead of `[1, 1000]`, use a tighter range like `[100, 900]` to reduce noise.

**Pros**: Simple, reduces ZIC's dispersal
**Cons**: Doesn't fix the fundamental issue

### Option 2: GD History Filtering
Filter out extreme outliers from GD's history.

**Pros**: Directly addresses belief pollution
**Cons**: Deviates from original GD algorithm

### Option 3: Different Mixing Ratio
Test with fewer ZIC agents (e.g., 4 GD + 1 ZIC).

**Pros**: Reduces noise proportion
**Cons**: Doesn't match tournament spec if all agents should coexist

### Option 4: Accept This is Real Behavior
Perhaps GD + ZIC DO perform poorly together in the real 1993 tournament?

**Action**: Check literature for GD vs ZIC heterogeneous results.

## ROOT CAUSE IDENTIFIED

### The Initial Quote Bug

**Problem**: GD's _calculate_quote() searches over entire price range with uniform 0.5 beliefs when history is empty.

**For buyers** (valuation V, price_min=1):
```
Expected surplus at price P = 0.5 × (V - P)
Maximum at P = price_min = 1
```

**For sellers** (cost C, price_max=1000):
```
Expected surplus at price P = 0.5 × (P - C)
Maximum at P = price_max = 1000
```

**Result**:
- GD buyers bid 1
- GD sellers ask 1000
- Spread = 999 (never crosses)
- Market stalls until enough history accumulates
- With 3 GD agents dominating, few trades occur
- Low efficiency persists

### Evidence

1. **Test with tight range (100-500)**:
   - Average efficiency: 33.22% (WORSE than 38.99%)
   - 12% periods with 0% efficiency
   - Confirms initial quote calculation is the issue

2. **Homogeneous GD (100% efficiency)**:
   - All GD agents make the same bad initial quotes
   - But they quickly learn from each other's quotes
   - Beliefs converge rapidly in symmetric environment

3. **Mixed GD+ZIC (38% efficiency)**:
   - ZIC's random quotes create noisy history
   - GD's beliefs stay distorted longer
   - Market takes longer to converge (if at all)

### The Real Fix

GD's initial quote when history is empty should NOT maximize expected surplus with uniform 0.5 beliefs across entire range. Options:

1. **Use limit price**: Bid at valuation (buyers) or cost (sellers) initially
2. **Use midpoint**: Start at (price_min + valuation)/2 for buyers
3. **Original GD paper approach**: Check what Gjerstad & Dickhaut actually specify

## Next Steps

1. ✅ Document hypothesis
2. ✅ Test tighter price range - FAILED (made it worse)
3. ✅ Identify root cause - Initial quote calculation bug
4. ⏳ Check original GD paper (1998) for correct initialization
5. ⏳ Implement fix for initial quote calculation
6. ⏳ Re-run GD vs ZIC validation
