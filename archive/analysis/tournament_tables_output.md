# Tournament Results Tables

### Table 5: Pure Market Self-Play Efficiency

| Trader   | Efficiency (mean ± std) | Periods | Interpretation                  |
|----------|-------------------------|---------|---------------------------------|
| ZI2      | 98.6% ± 2.4%            | 10      | Excellent self-play             |
| LIN      | 80.4% ± 37.5%           | 10      | Moderate self-play              |
| JACOBSON | 75.3% ± 31.7%           | 10      | Poor self-play (market failure?) |
| GD       | 68.1% ± 44.1%           | 10      | Poor self-play (market failure?) |
| KAPLAN   | 54.1% ± 28.5%           | 10      | Poor self-play (market failure?) |
| ZIP      | 36.8% ± 38.5%           | 10      | Poor self-play (market failure?) |
| SKELETON | 32.9% ± 38.1%           | 10      | Poor self-play (market failure?) |
| ZIC      | 32.9% ± 38.1%           | 10      | Poor self-play (market failure?) |
| ZI       | 31.0% ± 37.6%           | 10      | Poor self-play (market failure?) |

### Table 7: Pairwise Tournament Results

| Matchup              | Efficiency (mean ± std) | Winner Profit Share | Periods | Notes           |
|----------------------|-------------------------|---------------------|---------|-----------------|
| Gd Vs Jacobson       | 83.7% ± 24.7%           | 52.8%               | 10      | Balanced        |
| Gd Vs Lin            | 77.8% ± 14.4%           | 61.4%               | 10      | Moderate dominance |
| Gd Vs Skeleton       | 52.0% ± 38.5%           | 187.3%              | 10      | Strong dominance |
| Gd Vs Zi2            | 71.3% ± 14.9%           | 72.6%               | 10      | Strong dominance |
| Jacobson Vs Perry    | 92.2% ± 12.4%           | 51.9%               | 10      | Balanced        |
| Kaplan Vs Gd         | 21.6% ± 26.5%           | 4951.8%             | 10      | Strong dominance |
| Kaplan Vs Jacobson   | 20.8% ± 25.3%           | 5712.0%             | 10      | Strong dominance |
| Kaplan Vs Lin        | 46.1% ± 38.8%           | 563.5%              | 10      | Strong dominance |

### Table 8: Complete 1v7 Invasibility Matrix

*Test: 1 trader (varied) vs 7 ZIC agents. Measures individual trader's ability to invade/exploit ZIC population.*

| Trader   | Invasibility | As Buyer         | As Seller        | Interpretation                |
|----------|--------------|------------------|------------------|-------------------------------|
| GD       | 18.2%        | 36.3% ± 38.7%    | 0.0% ± 0.0%      | Weak invasibility             |
| ZI2      | 18.2%        | 36.3% ± 38.7%    | 0.0% ± 0.0%      | Weak invasibility             |
| LIN      | 18.1%        | 36.3% ± 38.7%    | 0.0% ± 0.0%      | Weak invasibility             |
| ZIP      | 16.5%        | 33.1% ± 38.2%    | 0.0% ± 0.0%      | Weak invasibility             |
| ZIC      | 16.5%        | 32.9% ± 38.2%    | 0.0% ± 0.0%      | Weak invasibility             |
| KAPLAN   | 16.4%        | 32.7% ± 38.0%    | 0.0% ± 0.0%      | Weak invasibility             |
| SKELETON | 16.3%        | 32.7% ± 38.0%    | 0.0% ± 0.0%      | Weak invasibility             |
| PERRY    | 16.2%        | 32.4% ± 36.7%    | 0.0% ± 0.0%      | Weak invasibility             |
| ZI       | 16.1%        | 32.2% ± 37.8%    | 0.0% ± 0.0%      | Weak invasibility             |
| JACOBSON | 16.1%        | 32.1% ± 36.3%    | 0.0% ± 0.0%      | Weak invasibility             |

### Table 9: Mixed Market Efficiency (Kaplan Background Effect)

*Hypothesis: Kaplan-dominated markets crash (expect <60% efficiency at high %)*

| Kaplan % | Efficiency (mean ± std) | Periods | Interpretation                     |
|----------|-------------------------|---------|------------------------------------|
| 0        | 35.4% ± 30.5%           | 10      | Baseline (no Kaplan)               |
| 10       | 28.9% ± 30.6%           | 10      | Low Kaplan concentration           |
| 25       | 33.5% ± 35.7%           | 10      | Low Kaplan concentration           |
| 50       | 36.9% ± 36.2%           | 10      | Moderate - efficiency declining    |
| 75       | 42.4% ± 36.3%           | 10      | High - market failure observed     |
| 90       | 48.2% ± 32.9%           | 10      | Near-homogeneous - CRASH confirmed |
