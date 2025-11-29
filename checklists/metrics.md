# Outcome Metrics for Continuous Double Auctions

Formal definitions for metrics used in Agent-Based Double Auction Markets.

This document uses notation from the Santa Fe Double Auction Tournament (Rust, Palmer, & Miller 1993), with metric definitions from Gode & Sunder (1993), Cliff & Bruten (1997), Gjerstad & Dickhaut (1998), Chen & Tai (2010), and Smith (1962).

---

## 1. Mathematical Notation and Preliminaries

### 1.1 The Environment

| Symbol | Definition |
|--------|------------|
| $B$ | Set of buyers |
| $S$ | Set of sellers |
| $v_{ik}$ | Redemption value for buyer $i$'s $k$-th unit |
| $c_{jk}$ | Cost for seller $j$'s $k$-th unit |

### 1.2 Demand and Supply Schedules

**Demand Schedule $D(q)$:** Constructed by ordering all $v_{ik}$ in descending order. $D(q)$ is the value of the $q$-th unit on the aggregated demand curve.

**Supply Schedule $S(q)$:** Constructed by ordering all $c_{jk}$ in ascending order. $S(q)$ is the cost of the $q$-th unit on the aggregated supply curve.

### 1.3 Equilibrium Definitions

**Equilibrium Quantity ($Q^*$):** The maximum quantity $q$ where demand exceeds supply:
$$Q^* = \max\{q : D(q) > S(q)\}$$

**Equilibrium Price ($P^*$):** Any price $p$ in the marginal interval:
$$S(Q^*) \leq P^* \leq D(Q^*)$$
Often defined as the midpoint: $P^* = \frac{D(Q^*) + S(Q^*)}{2}$

**Maximum Theoretical Surplus ($TS^*$):** The area between demand and supply curves:
$$TS^* = \sum_{q=1}^{Q^*} \bigl(D(q) - S(q)\bigr)$$

### 1.4 Market Activity Notation

| Symbol | Definition |
|--------|------------|
| $t = 1, \ldots, T$ | Sequence of concluded transactions |
| $p_t$ | Transaction price at trade $t$ |
| $v_t$ | Redemption value of unit exchanged at trade $t$ |
| $c_t$ | Cost of unit exchanged at trade $t$ |

---

## 2. Market Efficiency Metrics

These metrics evaluate the aggregate performance of the market in extracting potential gains from trade.

### 2.1 Allocative Efficiency ($E$)

The primary metric from Smith (1962) and Gode & Sunder (1993). Percentage of maximum possible surplus actually realized.

$$E = \frac{\sum_{t=1}^{T} (v_t - c_t)}{TS^*} \times 100$$

| Trader Type | Expected $E$ | Reference |
|-------------|--------------|-----------|
| ZI (unconstrained) | 60-70% | Gode & Sunder 1993 |
| ZIC (constrained) | 98.7% | Gode & Sunder 1993 |
| ZIP | 99.9% | Cliff & Bruten 1997 |
| GD | >99.9% | Gjerstad & Dickhaut 1998 |
| Mixed tournament | 89.7% | Rust et al. 1994 |

**Note:** If traders exchange units where $c_t > v_t$ (negative surplus trades), the numerator decreases, lowering efficiency.

### 2.2 Efficiency Loss Decomposition

Rust, Palmer, & Miller (1993) decompose total lost surplus $(100\% - E)$ into four components:

Let **intra-marginal units** be those that should trade ($q \leq Q^*$).
Let **extra-marginal units** be those that should not trade ($q > Q^*$).

**Intra-marginal Loss (IM / V-Inefficiency):** Surplus lost from failing to trade profitable units.
$$IM = \sum_{q \in \text{Untraded Intra-marginal}} \bigl(D(q) - S(q)\bigr)$$

**Extra-marginal Loss (EM / EM-Inefficiency):** Negative surplus from trading units that should not have been traded.
$$EM = \sum_{t \in \text{Extra-marginal trades}} (c_t - v_t)$$

**Buyer Displacement (BS):** Surplus lost when an extra-marginal buyer displaces an intra-marginal buyer.

**Seller Displacement (SS):** Surplus lost when an extra-marginal seller displaces an intra-marginal seller.

**Decomposition Identity:**
$$100\% - E = IM + EM + BS + SS$$

---

## 3. Price Convergence Metrics

These metrics measure the tendency of transaction prices $p_t$ to approach the equilibrium price $P^*$.

### 3.1 Root Mean Squared Deviation (RMSD)

From Gode & Sunder (1993). Measures distance of prices from equilibrium.

$$RMSD = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (p_t - P^*)^2}$$

### 3.2 Smith's Alpha ($\alpha$)

Coefficient of convergence from Vernon Smith (1962). Standard deviation of prices around equilibrium, normalized by equilibrium price.

Let $\sigma_0$ be the RMSD of prices around equilibrium:
$$\sigma_0 = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (p_t - P^*)^2}$$

Then:
$$\alpha = \frac{100 \cdot \sigma_0}{P^*}$$

**Interpretation:** Lower $\alpha$ = tighter convergence to equilibrium.

**Note:** Some sources (e.g., Cliff & Bruten 1997) use scaling factor 1000 instead of 100. The interpretation remains the same.

### 3.3 Price Standard Deviation

Raw volatility measure.

$$\sigma_p = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (p_t - \bar{p})^2}$$

where $\bar{p} = \frac{1}{T}\sum_{t=1}^{T} p_t$ is the mean transaction price.

| Trader Type | Expected Pattern |
|-------------|------------------|
| ZIC | High, constant (2-3x humans) |
| ZIP | High early, declining |
| GD | Lower than ZIP, faster decline |

### 3.4 Price Volatility Percentage

Normalized volatility for cross-market comparison.

$$\text{Volatility\%} = \frac{\sigma_p}{\bar{p}} \times 100$$

**Interpretation:** <5% indicates good convergence; >20% indicates unstable market.

### 3.5 Hit Rate

From Santa Fe Tournament. Percentage of trades within band around equilibrium.

$$H = \frac{|\{t : |p_t - P^*| \leq 0.05 \cdot P^*\}|}{T}$$

### 3.6 Mean Absolute Deviation (MAD)

From Gjerstad & Dickhaut (1998).

$$MAD = \frac{1}{T} \sum_{t=1}^{T} |p_t - P^*|$$

| Trader Type | Expected MAD |
|-------------|--------------|
| ZIP | ~\$0.08 |
| GD | ~\$0.04 |

---

## 4. Trader Performance Metrics

These metrics evaluate individual agents rather than the market as a whole.

### 4.1 Individual Profit ($\pi_i$)

Raw earnings for trader $i$.

**Buyer:**
$$\pi_i = \sum_{k \in \text{Items Traded}} (v_{ik} - p_k)$$

**Seller:**
$$\pi_j = \sum_{k \in \text{Items Traded}} (p_k - c_{jk})$$

### 4.2 Equilibrium Profit ($\pi_i^*$)

Theoretical profit at competitive equilibrium. The profit trader $i$ would earn if all trades occurred at $P^*$.

**Buyer:**
$$\pi_i^* = \sum_{k : v_{ik} > P^*} (v_{ik} - P^*)$$

**Seller:**
$$\pi_j^* = \sum_{k : c_{jk} < P^*} (P^* - c_{jk})$$

### 4.3 Profit Deviation

How much more or less than fair share.

$$\Delta\pi_i = \pi_i - \pi_i^*$$

| Value | Interpretation |
|-------|----------------|
| $\Delta\pi_i > 0$ | Trader extracted more than fair share |
| $\Delta\pi_i = 0$ | Trader earned exactly fair share |
| $\Delta\pi_i < 0$ | Trader was exploited/underperformed |

### 4.4 Individual Efficiency Ratio ($E_i$)

From Chen & Tai (2010). Ratio of actual to theoretical profit.

$$E_i = \frac{\pi_i}{\pi_i^*}$$

| Value | Interpretation |
|-------|----------------|
| $E_i > 1$ | Captures more than equilibrium share (exploiter) |
| $E_i = 1$ | Captures exactly equilibrium share |
| $E_i < 1$ | Captures less than equilibrium share (exploited) |

**Expected values:**
- Kaplan (mixed): ~1.14-1.21 (114-121%) — captures more than fair share
- ZIC: ~1.0
- ZIP/GD: ~1.0
- Kaplan (pure): ~0.5-0.6 — captures less due to waiting too long

### 4.5 Profit Dispersion (PD)

**THE KEY METRIC** from Cliff & Bruten (1997) for discriminating intelligent vs zero-intelligence traders.

Cross-sectional RMS difference between actual and equilibrium profits:

$$PD = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\pi_i - \pi_i^*)^2}$$

where $N$ is the total number of traders.

| Trader Type | Expected PD | Interpretation |
|-------------|-------------|----------------|
| ZIC | 0.35-0.60 | Random surplus allocation |
| ZIP | 0.05 (after convergence) | Fair allocation emerges |

**Significance:** ZIP achieves 7-10x lower dispersion than ZIC. Even with similar efficiency, profit dispersion reveals whether the "right" traders are earning profits.

### 4.6 Number of Trades

Activity level for agent $i$.

$$N_i = |\{t : \text{agent } i \text{ participated in trade } t\}|$$

**Expected:** Kaplan has fewer trades than ZIC (waiting strategy).

---

## 5. Dynamic Metrics

### 5.1 Price Autocorrelation

Tests whether price changes predict subsequent changes.

$$\rho = \text{Corr}(\Delta p_t, \Delta p_{t-1})$$

where $\Delta p_t = p_t - p_{t-1}$.

| Value | Interpretation |
|-------|----------------|
| $\rho < 0$ | Mean-reversion (prices overshoot then correct) |
| $\rho = 0$ | Random walk (no predictability) |
| $\rho > 0$ | Momentum/trending |

**Expected:** $\rho \approx -0.25$ (Rust et al. finding). Wilson's (1987) martingale prediction ($\rho = 0$) was empirically rejected.

### 5.2 Gode-Sunder Convergence Coefficient ($\beta$)

From Gode & Sunder (1993) Table 1. Tests whether market "learns" within a period.

Let $y_t$ be the Root Mean Squared Deviation (RMSD) of transaction prices at sequence number $t$, calculated across $N$ experimental runs:

$$y_t = \sqrt{\frac{1}{N} \sum_{n=1}^{N} (p_{t,n} - P^*)^2}$$

Regress $y_t$ against $t$:

$$y_t = \alpha + \beta \cdot t + \epsilon_t$$

| Value | Interpretation |
|-------|----------------|
| $\beta < 0$ | Market is converging (variance shrinking) |
| $\beta \approx 0$ | Market is stagnant (common in ZI unconstrained) |

**Note:** The regression is performed on ensemble RMSD (across multiple runs), not single-run squared error, to reduce noise.

### 5.3 Convergence Time

Periods until prices stabilize within $\pm 5\%$ of equilibrium.

$$T^* = \min\{t : |p_t - P^*| \leq 0.05 \cdot P^*\}$$

| Trader Type | Expected $T^*$ |
|-------------|----------------|
| GD | <1 period |
| ZIP | 1-2 periods |
| ZIC | Never (no learning) |

### 5.4 Time of Last Transaction ($T_{last}$)

From Rust, Palmer, & Miller (1993). Measures liquidity risk and "closing panics."

$$T_{last} = \max_t(\tau_t)$$

where $\tau_t$ is the timestamp of trade $t$ and $T_{max}$ is maximum time allowed.

**Interpretation:** If $T_{last} \approx T_{max}$ consistently, indicates "wait in background" strategies (like Kaplan) causing deadline congestion.

### 5.5 Rank Correlation of Efficient Order ($\rho_s$)

Measures whether "right" trades happened in "right" order. Theory suggests highest-value buyer should trade with lowest-cost seller first.

Let $R_{actual}$ be rank vector of trades by surplus as they occurred.
Let $R_{ideal}$ be rank vector sorted by theoretical surplus.

$$\rho_s = \text{Spearman}(R_{actual}, R_{ideal})$$

**Interpretation:** $\rho_s = 1.0$ means market perfectly executed most profitable trades first.

---

## 6. Evolutionary Metrics

For long-run tournament analysis (Rust, Palmer, & Miller; Chen & Tai).

### 6.1 Capital Stock Evolution ($K_{i,g}$)

Market share of strategy $i$ at game/generation $g$.

$$K_{i,g} = K_{i,g-1} + \pi_{i,g} - S_{i,g}$$

where $S_{i,g}$ is theoretical surplus assigned to trader $i$.

**Interpretation:** Strategies with $K$ trending upward are evolutionarily stable. Those trending to 0 are eliminated.

### 6.2 Generations to Convergence ($Gen^*$)

From Chen & Tai (2010). Learning speed metric.

$$Gen^* = \min\{g : E_{pop,g} \geq E_{target}\}$$

where $E_{pop,g}$ is average efficiency at generation $g$ and $E_{target}$ is threshold (e.g., 99%).

---

## 7. Microstructure Metrics

### 7.1 Initiator Price Bias ($\Delta_{init}$)

From Gjerstad & Dickhaut (1998) Table III. Difference between buyer-initiated and seller-initiated trade prices.

Let $T_{buy}$ = trades where buyer accepted standing ask.
Let $T_{sell}$ = trades where seller accepted standing bid.

$$\bar{p}_{buy} = \frac{1}{|T_{buy}|} \sum_{t \in T_{buy}} p_t$$

$$\bar{p}_{sell} = \frac{1}{|T_{sell}|} \sum_{t \in T_{sell}} p_t$$

$$\Delta_{init} = \bar{p}_{sell} - \bar{p}_{buy}$$

**Interpretation:** In human markets, $\Delta_{init} \neq 0$ indicates asymmetric urgency between buyers and sellers.

### 7.2 ZIP Margin Adjustment

From Cliff & Bruten (1997). The learning dynamics of profit margin $\mu$.

$$\Delta\mu_i(t) = \beta \cdot (Target_i(t) - p_i(t))$$

where $\beta$ is the learning rate parameter.

**As metric:** The optimal $\beta$ that matches human data becomes an outcome when calibrating.

---

## References

- **Gode & Sunder (1993):** "Allocative Efficiency of Markets with Zero-Intelligence Traders"
- **Cliff & Bruten (1997):** "Minimal-Intelligence Agents for Bargaining Behaviors in Market-Based Environments"
- **Gjerstad & Dickhaut (1998):** "Price Formation in Double Auctions"
- **Rust, Palmer, & Miller (1994):** "Behavior of Trading Automata in a Computerized Double Auction Market"
- **Chen & Tai (2010):** "The Agent-Based Double Auction Markets: 15 Years On"
- **Smith (1962):** "An Experimental Study of Competitive Market Behavior"
- **Cason & Friedman (1996):** Efficiency decomposition framework
- **Wilson (1987):** WGDA martingale prediction
