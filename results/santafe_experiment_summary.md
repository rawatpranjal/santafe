# Santa Fe Tournament Replication Results

## Experiment Date: 2025-11-22

### Overview
We ran several tournaments to replicate key findings from the 1993 Santa Fe Double Auction experiment using the Java reference implementation traders.

### Implemented Traders (7 from 1993 Tournament)
1. **Kaplan** - Background trader, tournament winner
2. **ZIC (Zero Intelligence Constrained)** - Random within profit bounds
3. **ZI2 (Ringuette)** - Market-aware Zero Intelligence
4. **Lin (Truth-Teller)** - Statistical price prediction
5. **Jacobson** - Equilibrium estimation
6. **Perry** - Adaptive learning
7. **Skeleton (Example)** - Simple template

### Tournament Results

#### 1. Pure Kaplan Market (santafe_pure_kaplan)
- **Configuration**: 5 Kaplan buyers vs 5 Kaplan sellers
- **Expected (from paper)**: <60% efficiency (market crash)
- **Actual Results**: 82-90% efficiency
- **Rounds Run**: 50
- **Key Observations**:
  - Period 1: 82.25% efficiency
  - Period 2: 84.04% efficiency
  - Period 3: 89.33% efficiency
  - Period 4: 90.84% efficiency
  - Period 5: 88.29% efficiency
  - Buyers made consistent profits (200-300 per period)
  - Sellers made consistent losses (-50 to -160 per period)
  - No market crash observed

**Analysis**: The Kaplan implementation includes protection clauses that prevent unprofitable trades. The Java reference has these protections for subsequent bids/asks but not first ones. This prevents the market crash seen in the original tournament.

#### 2. Pure ZIC Market (santafe_pure_zic)
- **Configuration**: 5 ZIC buyers vs 5 ZIC sellers
- **Expected (from paper)**: ~98% efficiency
- **Actual Results**: NEGATIVE efficiency (-116%)
- **Rounds Run**: 50
- **Issue Identified**: Calculation error in efficiency metric

#### 3. Mixed Market (santafe_base)
- **Configuration**: All 7 trader types mixed
- **Status**: Failed to run - new traders missing required abstract methods
- **Issue**: ZI2, Lin, Jacobson, Perry need bid_ask() and buy_sell() method implementations

### Key Findings

1. **Kaplan Protection Clauses**: The current Kaplan implementation has protection mechanisms that prevent the market crashes observed in the 1993 tournament. Lines 186-187 (bid) and 275-276 (ask) add protections for subsequent quotes that match Java but may be too protective.

2. **Efficiency Calculation Issues**: The ZIC tournament shows negative efficiency, suggesting a problem with the efficiency calculation when certain market conditions occur.

3. **Implementation Gaps**: The newly implemented traders (ZI2, Lin, Jacobson, Perry) need additional methods to be compatible with the tournament engine.

### Comparison to 1994 Paper Results

| Market Type | Paper Result | Our Result | Match? |
|-------------|-------------|------------|--------|
| Pure Kaplan | <60% (crash) | 82-90% | ❌ |
| Pure ZIC | ~98% | -116% (error) | ❌ |
| Mixed | ~90% | Not run | - |

### Next Steps

1. **Fix Efficiency Calculation**: Investigate why ZIC shows negative efficiency
2. **Complete Trader Implementations**: Add missing methods to new traders
3. **Adjust Kaplan**: Consider removing some protection clauses to match original behavior
4. **Run Full Tournament**: Complete mixed market experiments

### Notes
- The Java reference implementation appears to have some fixes/improvements over the original 1993 code
- Token generation and market mechanics appear to be working correctly
- Trading is happening but efficiency metrics need review