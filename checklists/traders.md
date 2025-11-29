# 2. Trader Algorithms

### Trader: Skeleton
**Classification:** Non-Adaptive, Simple, Stochastic
**Source Reference:** *Participant's Manual* (Chapter 3) and *JEDC Paper* (Figure 4, Page 75).

The "Skeleton" is the baseline code provided to all participants. While primarily a structural template, it contained a default "Zero Intelligence Plus" style strategy that uses a weighted average to creep towards a deal.

#### 1. Bid-Ask Step (Bidding Logic)
*   **[ ] Random Factor ($\alpha$) Generation:**
    *   At the start of the step, generate a random scalar $\alpha$.
    *   Formula: $\alpha = 0.25 + (0.1 \times U[0, 1])$. (Resulting range: $0.25$ to $0.35$).
*   **[ ] Define Limits (`MOST`):**
    *   **Buyer:** Calculate `MOST` (My Optimal Strategic Token limit).
        *   `MOST` = Minimum of (`CurrentAsk`, `TokenRedemptionValue` - 1).
        *   *Note:* If no `CurrentAsk` exists, effectively use `TokenRedemptionValue`.
    *   **Seller:** Calculate `MOST`.
        *   `MOST` = Maximum of (`CurrentBid`, `TokenCost` + 1).
        *   *Note:* If no `CurrentBid` exists, effectively use `TokenCost`.
*   **[ ] Validity Check:**
    *   **Buyer:** Verify `MOST` > `CurrentBid`. If false, do not bid (Return 0).
    *   **Seller:** Verify `MOST` < `CurrentAsk`. If false, do not offer (Return 0).
*   **[ ] Price Calculation (Weighted Average):**
    *   **Buyer Bid:** $Bid = (1 - \alpha) \times (CurrentBid + 1) + (\alpha \times MOST)$.
        *   *Logic:* Bids slightly higher than the current best bid, moving a fraction ($\alpha$) of the distance toward its limit (`MOST`).
    *   **Seller Ask:** $Ask = (1 - \alpha) \times (CurrentAsk - 1) + (\alpha \times MOST)$.
        *   *Logic:* Asks slightly lower than the current best offer, moving a fraction ($\alpha$) of the distance toward its limit (`MOST`).
*   **[ ] Rounding:** Resulting prices must be rounded to the nearest integer.

#### 2. Buy-Sell Step (Take Logic)
*   **[ ] Profitability Check:**
    *   **Buyer:** If `CurrentAsk` < `TokenRedemptionValue`, return `BUY` (1).
    *   **Seller:** If `CurrentBid` > `TokenCost`, return `SELL` (1).
*   **[ ] Default Action:**
    *   If the trade is not profitable (or equals the token value exactly), return `PASS` (0).

### Trader: Kaplan
**Classification:** Simple, Wait-in-Background, Rule-of-Thumb
**Status:** Winner of the 1990 Tournament
**Source Reference:** *JEDC Paper* (Figure 2, Page 73; Table 4, Page 90)

The Kaplan strategy is famous for its "waiting" game. It ignores the bidding war until the spread (difference between Bid and Ask) is small, then it jumps in to "steal the deal" by matching the best standing offer immediately.

#### 1. Memory & Initialization
*   **[ ] History Tracking:**
    *   Track `PMAX` (Highest transaction price in the *previous* period).
    *   Track `PMIN` (Lowest transaction price in the *previous* period).
    *   *Initial State:* If Period = 1, set `PMAX = +Infinity` and `PMIN = -Infinity`.
*   **[ ] Reset:** Update `PMAX` and `PMIN` only at the start of a new Period.

#### 2. Bid-Ask Step (Buyer Logic)
*   **[ ] Check Standing Bid:**
    *   If no `CurrentBid` exists on the board:
        *   **Action:** Submit Bid = `MinPrice + 1`.
*   **[ ] Check Standing Ask:**
    *   If no `CurrentAsk` exists:
        *   **Action:** Do nothing (Wait).
*   **[ ] "Snipe" Condition (Steal the Deal):**
    *   Evaluate three conditions simultaneously:
        1.  Is `CurrentAsk` $\le$ `PMAX`? (Is the price reasonable compared to last period?)
        2.  Is `(Value - CurrentAsk) / CurrentAsk` $> 0.02$? (Is profit > 2%?)
        3.  Is `(CurrentAsk - CurrentBid)` $< 0.10 \times CurrentAsk$? (Is the spread small/tight?)
    *   **Action:** If ALL True $\rightarrow$ Submit Bid = `CurrentAsk` (Accept the offer immediately).
*   **[ ] "Bargain Hunter" Condition:**
    *   Evaluate: Is `CurrentAsk` $\le$ `PMIN`? (Is this price cheaper than anything seen last period?)
    *   **Action:** If True $\rightarrow$ Submit Bid = `CurrentAsk`.
*   **[ ] Time Pressure Condition:**
    *   Evaluate: Is time running out? (e.g., `Time > 0.9 * MaxTime` or similar threshold implied by implementation).
    *   **Action:** If True $\rightarrow$ Submit Bid = Minimum of (`CurrentAsk`, `Value - 1`).
*   **[ ] Default Action:**
    *   If none of the above trigger $\rightarrow$ Do nothing (Return 0).

#### 3. Bid-Ask Step (Seller Logic)

> **⚠️ CRITICAL: Paper vs Java Discrepancy**
>
> The 1994 Rust et al. paper (line 587) states: *"We omitted diagrams of the ask routines
> since they are symmetric to the bid routines."*
>
> However, the da2.7.2 Java implementation uses DIFFERENT denominators:
> - **Buyer (line 66):** `(cask-cbid)/(cask+1) < 0.10` → uses **ASK**
> - **Seller (line 94):** `(cask-cbid)/(cbid+1) < 0.10` → uses **BID**
>
> This asymmetry makes sellers LESS aggressive at jumping in when bid is low.
>
> **Available Variants:**
> - `Kaplan` / `KaplanJava`: Follow Java da2.7.2 (asymmetric, seller uses BID)
> - `KaplanPaper`: Follow paper claim (symmetric, seller uses ASK)

*   **[ ] Check Standing Ask:** If none, Ask = `MaxPrice - 1`.
*   **[ ] "Snipe" Condition:**
    *   1. Is `CurrentBid` $\ge$ `PMIN`?
    *   2. Is `(CurrentBid - Cost) / Cost` $> 0.02$?
    *   3. **Java (KaplanJava):** Is `(CurrentAsk - CurrentBid)` $< 0.10 \times (CurrentBid + 1)$?
    *   3. **Paper (KaplanPaper):** Is `(CurrentAsk - CurrentBid)` $< 0.10 \times (CurrentAsk + 1)$?
    *   **Action:** If ALL True $\rightarrow$ Submit Ask = `CurrentBid`.
*   **[ ] Default:** Wait in the background.

#### 4. Buy-Sell Step (Take Logic)
*   **[ ] Profitability Check:**
    *   **Buyer:** If `CurrentAsk` < `TokenValue`, return `BUY` (1).
    *   **Seller:** If `CurrentBid` > `TokenCost`, return `SELL` (1).
    *   *Note:* Kaplan is opportunistic; if it holds the Current Bid/Ask because it "Sniped" in step 2, the deal closes here.

#### 5. Deep Dive: Why Kaplan Won

**The "Sniper" Mechanic:**
- Most agents actively entered bidding wars, driving the price up/down incrementally.
- Kaplan **did nothing** for most of the period. It watched the spread.
- When the spread became small (<10% of price), Kaplan jumped in.

**Exploiting the Buy-Sell Step:**
- Under tournament rules, only the **Current Bidder** could initiate a trade.
- Kaplan submitted a bid **equal** to the current Ask (not +$1).
- This instantly made Kaplan the "Current Bidder" and locked the trade before others could react.

**Environmental Dominance (Table 3, Page 84):**
- **Rank 1:** BASE, BBBS, PER, SHRT, SML
- **Rank 2:** LAD
- **Achilles Heel:** TOK (Rank 14) — In single-token markets, waiting too long meant missing all trades.

**Kaplan vs Ringuette (2nd Place):**
- Ringuette had higher efficiency per trade (better prices).
- Kaplan had higher volume: **96%** of profitable trades vs Ringuette's **84%**.
- In a high-volume tournament, 96% × good margin > 84% × perfect margin.

**The Priority Rule Exploit:**
- Step 1 (Bid-Ask): Agents yell prices.
- Step 2 (Buy-Sell): Only Current Bidder/Asker can execute. Everyone else frozen.
- By matching the opposing standing order during Bid-Ask, Kaplan guaranteed execution.

**PMAX/PMIN Bounds:**
- Kaplan only sniped if `CurrentAsk ≤ PMAX` (from previous period).
- This prevented bad trades during early chaotic steps.

**Key Quotes from JEDC Paper:**
> "Wait in the background and let others do the negotiating, but when bid and ask get sufficiently close, jump in and 'steal the deal'." *(Page 63)*

> "Kaplan's buyer program places a bid equal to the previous current ask whenever the percentage gap between the current bid and ask is less than 10%." *(Page 73-74)*

---

### Trader: ZI (Zero Intelligence Unconstrained)
**Classification:** Simple, Stochastic, Non-Adaptive, Control Condition
**Source Reference:** Gode & Sunder (1993), Table 1; *JEDC Paper* (Page 69)
**Expected Efficiency:** 60-70%

Pure ZI is the CONTROL condition proving that budget constraints matter. It bids randomly with NO regard for profitability and will accept trades at a loss.

#### 1. Bid-Ask Step (Bidding Logic)
*   **[ ] Random Selection (Buyer):**
    *   Generate random bid from `U[MinPrice, MaxPrice]`.
    *   No budget constraint - can bid above token value.
*   **[ ] Random Selection (Seller):**
    *   Generate random ask from `U[MinPrice, MaxPrice]`.
    *   No budget constraint - can ask below token cost.

#### 2. Buy-Sell Step (Take Logic)
*   **[ ] Accept Any Trade:**
    *   **Buyer:** If we are the high bidder and spread is crossed, ACCEPT (even at loss).
    *   **Seller:** If we are the low asker and spread is crossed, ACCEPT (even at loss).
    *   No profitability check - this is what makes ZI inefficient.

---

### Trader: ZIC (Zero Intelligence Constrained)
**Classification:** Simple, Stochastic, Non-Adaptive, Non-Predictive
**Source Reference:** SRobotZI1.java; Gode & Sunder (1993); *JEDC Paper* (Page 69, Page 82)
**Expected Efficiency:** 97-100%

ZIC ignores market state entirely (no awareness of CurrentBid/CurrentAsk). It generates random prices within budget constraints only. The New York Rule (must improve) is enforced by the MARKET, not the agent.

#### 1. Bid-Ask Step (Bidding Logic)
*   **[ ] Random Selection (Buyer):**
    *   Generate random bid from `U[MinPrice, TokenRedemptionValue]`.
    *   Formula: `bid = valuation - floor(random * (valuation - minprice))`
    *   Budget constraint: Cannot bid above token value.
*   **[ ] Random Selection (Seller):**
    *   Generate random ask from `U[TokenCost, MaxPrice]`.
    *   Formula: `ask = cost + floor(random * (maxprice - cost))`
    *   Budget constraint: Cannot ask below token cost.
*   **[ ] Market Rejection:**
    *   If bid/ask doesn't improve the market (New York Rule), the MARKET rejects it.
    *   ZIC does NOT check CurrentBid/CurrentAsk itself - it's purely random within budget.

#### 2. Buy-Sell Step (Take Logic)
*   **[ ] Profitability Check:**
    *   **Buyer:** Check if `CurrentAsk` < `TokenRedemptionValue` (strict inequality).
    *   **Seller:** Check if `CurrentBid` > `TokenCost` (strict inequality).
*   **[ ] Action:**
    *   If True (Profitable): Return 1 (`BUY`/`SELL`).
    *   If False (Loss/Break-even): Return 0 (`PASS`).

**Key Insight:** ZIC achieves ~98% efficiency despite zero intelligence because:
1. Budget constraints prevent unprofitable trades
2. AURORA protocol's New York Rule progressively narrows the spread
3. Efficiency emerges from institution design, not agent sophistication

### Trader: ZI2 (Market-Aware Random)
**Classification:** Simple, Stochastic, Non-Adaptive, Market-Aware
**Source Reference:** SRobotZI2.java; SFDA Tournament 2nd Place
**Expected Efficiency:** ~95-98%

ZI2 is an enhanced version of ZIC that incorporates current market state. It generates random bids/asks constrained by BOTH budget AND current market prices.

#### 1. Bid-Ask Step (Bidding Logic)
*   **[ ] Market-Aware Random Selection (Buyer):**
    *   Generate random bid from `U[max(MinPrice, CurrentBid+1), min(TokenValue, CurrentAsk-1)]`.
    *   If CurrentAsk exists: cannot bid above CurrentAsk.
    *   If CurrentBid exists: must bid above CurrentBid (improvement).
*   **[ ] Market-Aware Random Selection (Seller):**
    *   Generate random ask from `U[max(TokenCost, CurrentBid+1), min(MaxPrice, CurrentAsk-1)]`.
    *   If CurrentBid exists: cannot ask below CurrentBid.
    *   If CurrentAsk exists: must ask below CurrentAsk (improvement).
*   **[ ] Key Distinction from ZIC:**
    *   ZIC ignores market state entirely (just random within budget).
    *   ZI2 constrains by current market prices (more informed, but still random).
    *   ZI2 does NOT wait - it bids every opportunity, just more narrowly.

#### 2. Buy-Sell Step (Take Logic)
*   **[ ] Profitability Check:** Same as ZIC.
    *   **Buyer:** If `CurrentAsk` < `TokenValue`, return `BUY` (1).
    *   **Seller:** If `CurrentBid` > `TokenCost`, return `SELL` (1).

### Trader: Ringuette
**Classification:** Simple, Wait-in-Background, Rule-of-Thumb
**Status:** 2nd Place in the 1990 Tournament
**Source Reference:** *JEDC Paper* (Figure 3, Page 74; Section 4.1).

Ringuette is structurally very similar to Kaplan (wait in the background, then snipe) but differs in *how* it decides a deal is close enough to strike. Instead of a percentage spread, it uses a fixed "Span" derived from the theoretical price range of the token set. It is slightly more aggressive than Kaplan.

**⚠️ CRITICAL VALIDATION:** Ringuette MUST behave as a sniper:
*   **[ ] Verify:** Ringuette calculates SPAN and waits for `Spread < SPAN/5` before acting.
*   **[ ] Verify:** Ringuette does NOT bid randomly like ZI2 - it WAITS then STRIKES.
*   **[ ] Distinguish from ZI2:** ZI2 bids every step (market-aware random). Ringuette waits passively.

#### 1. Initialization (Span Calculation)
*   **[ ] Calculate Span:**
    *   Compute `SPAN` = `MaxPossiblePrice` - `MinPossiblePrice`.
    *   Usually defined as `HighestTokenRedemption` - `LowestTokenCost` + 10.
    *   *Note:* The +10 is a buffer constant used in the original code.

#### 2. Bid-Ask Step (Bidding Logic)
*   **[ ] Time Check:**
    *   Is `CurrentTime` $\le$ (`MaxTime` / 4)?
    *   **Action:** If True (early game), behave like **Skeleton** (default to Skeleton strategy).
*   **[ ] Market Existence Check:**
    *   Does a `CurrentAsk` exist?
    *   **Action:** If No, return 0 (Do nothing / Wait).
*   **[ ] The "Strike" Condition:**
    *   Calculate the Spread: `Gap` = `CurrentAsk` - `CurrentBid`.
    *   **Condition 1:** `Gap` $\le$ (`SPAN` / 5)? (Is the spread tight enough?)
    *   **Condition 2:** `CurrentAsk` $\le$ `NextTokenValue`? (Is the price profitable?)
    *   **Action:** If BOTH True $\rightarrow$ Submit Bid = `CurrentAsk` (Accept the offer immediately).
*   **[ ] The "Nudge" Condition (Wait logic):**
    *   If the Strike Condition fails:
    *   **Action:** Do nothing (Return 0).

#### 3. Buy-Sell Step (Take Logic)
*   **[ ] Profitability Check:**
    *   **Buyer:** If `CurrentAsk` < `TokenRedemptionValue`, return `BUY` (1).
    *   **Seller:** If `CurrentBid` > `TokenCost`, return `SELL` (1).
*   **[ ] Aggressiveness:**
    *   Like Kaplan, Ringuette aims to be the one holding the Current Bid/Offer when the market crosses, often by submitting the matching price in the Bid-Ask step.

### Bonus: Truth-Teller
**Classification:** Control Strategy / Zero-Profit
**Source Reference:** *JEDC Paper* (Page 69, Footnote 14 - "TruthTeller")

Used as a control variable. It reveals its full private value immediately. It guarantees efficiency (trades happen) but guarantees near-zero profit for itself.

#### 1. Bid-Ask Step
*   **[ ] Strategy:**
    *   **Buyer:** Submit Bid = `TokenRedemptionValue`.
    *   **Seller:** Submit Ask = `TokenCost`.
*   **[ ] Validity Constraints:**
    *   Only submit if the value improves the current market (Bid > CurrentBid or Ask < CurrentAsk).

#### 2. Buy-Sell Step
*   **[ ] Strategy:**
    *   Accept any profitable trade. (Since the bid/ask was already at the limit, this usually only triggers if the market moved in their favor by someone else's error).

---

## Key Literature Traders

### Trader: ZIP (Zero-Intelligence Plus)
**Classification:** Adaptive, Simple, Machine Learning
**Source Reference:** Cliff & Bruten (1997), *Minimal-Intelligence Agents for Bargaining Behaviors*
**File:** `traders/legacy/zip.py`

ZIP uses the Widrow-Hoff delta rule to adapt its profit margin based on observed market activity. It learns whether to raise or lower its margin based on trade outcomes.

#### Core Mechanism
*   **[ ] Price Formula:** $p = \lambda \times (1 + \mu)$ where $\lambda$ is limit price, $\mu$ is profit margin.
*   **[ ] Margin Adaptation:** Uses Widrow-Hoff update: $\Delta = \beta \times (\tau - p)$, $\Gamma = \gamma \Gamma + (1-\gamma)\Delta$
*   **[ ] Parameters:**
    *   $\beta$ (learning rate): typically 0.1-0.5
    *   $\gamma$ (momentum): typically 0.0-0.1
    *   $\mu$ (initial margin): ±0.05-0.35

#### 1. Bid-Ask Step
*   **[ ] Calculate Quote:** $quote = valuation \times (1 + margin)$
*   **[ ] Clamp to Price Range:** Ensure quote in [price_min, price_max]

#### 2. Buy-Sell Step
*   **[ ] ZIP Acceptance Rule:** Accept if counterparty price ≤ my shout price (buyers) or ≥ my shout price (sellers)
*   **[ ] Defensive Check:** Never trade at loss relative to limit price

#### 3. Learning (Post-Trade)
*   **[ ] Raise Margin:** If last shout accepted AND my price was competitive
*   **[ ] Lower Margin:** If not competitive with market price (cross-side signals in AURORA)
*   **[ ] Target Price:** $\tau = R \times q + A$ (random R, A for stochasticity)

---

### Trader: GD (Gjerstad-Dickhaut)
**Classification:** Adaptive, Complex, Belief-Based
**Source Reference:** Gjerstad & Dickhaut (1998), *Price Formation in Double Auctions*
**File:** `traders/legacy/gd.py`

GD forms beliefs about acceptance probabilities from historical bid/ask data and chooses prices to maximize expected surplus.

#### Core Mechanism
*   **[ ] Belief Formation (Seller):** $p(a) = \frac{TA(\ge a) + B(\ge a)}{TA(\ge a) + B(\ge a) + RA(\le a)}$
*   **[ ] Belief Formation (Buyer):** $q(b) = \frac{TB(\le b) + A(\le b)}{TB(\le b) + A(\le b) + RB(> b)}$
*   **[ ] Expected Surplus:** $E[surplus] = prob \times (valuation - price)$

#### 1. Bid-Ask Step
*   **[ ] Build Belief Curve:** From historical bids/asks and their outcomes
*   **[ ] Interpolate:** Use PCHIP (monotone-preserving) interpolation
*   **[ ] Optimize:** Search prices to maximize expected surplus

#### 2. Buy-Sell Step
*   **[ ] Compare Surpluses:** Accept if certain surplus ≥ expected surplus from waiting
*   **[ ] Profitability Check:** Only accept if profitable

#### 3. History Management
*   **[ ] Track All Observations:** Record (price, is_bid, accepted) tuples
*   **[ ] Memory Truncation:** Keep last L trades (configurable)
*   **[ ] Reset Per Round:** Clear history when equilibrium changes

---

### Trader: EL (Easley-Ledyard)
**Classification:** Adaptive, Simple, Reservation Price
**Source Reference:** Easley & Ledyard (1992), *Theories of Price Formation*
**File:** `traders/legacy/el.py`

EL uses reservation prices derived from previous period's price bounds. Traders become more aggressive as time runs out.

#### Core Mechanism
*   **[ ] Inframarginal Classification:** Buyer with $v > \bar{P}$ (max price), Seller with $c < \underline{P}$ (min price)
*   **[ ] Marginal Classification:** Trader at boundary → truth-tell
*   **[ ] Time-Based Interpolation:** Early = conservative, Late = aggressive

#### 1. Bid-Ask Step (Buyer)
*   **[ ] If Inframarginal:** $r(t) = (1-t)\bar{P} + t \times v$ (interpolate from max price to valuation)
*   **[ ] If Marginal:** $r(t) = v$ (truth-tell)
*   **[ ] Improvement Constraint:** Must improve current bid

#### 2. Bid-Ask Step (Seller)
*   **[ ] If Inframarginal:** $s(t) = (1-t)\underline{P} + t \times c$ (interpolate from min price to cost)
*   **[ ] If Marginal:** $s(t) = c$ (truth-tell)

#### 3. Buy-Sell Step
*   **[ ] Accept if:** Current price within reservation price AND profitable

#### 4. Period Boundary
*   **[ ] Update Bounds:** Set $[\underline{P}, \bar{P}]$ from previous period's traded prices

---

## Santa Fe 1993 Tournament Traders

### Trader: Jacobson
**Classification:** Adaptive, Complex, Equilibrium Estimation
**Source Reference:** SRobotJacobson.java (1993 Santa Fe Tournament)
**File:** `traders/legacy/jacobson.py`

Uses weighted equilibrium estimation with exponential confidence function.

#### Core Mechanism
*   **[ ] Equilibrium Estimate:** $\hat{e} = \frac{\sum (price \times weight)}{\sum weight}$
*   **[ ] Confidence:** $conf = 0.01^{1/weight}$ (approaches 1 as weight increases)
*   **[ ] Convex Combination:** New price = $(1-conf) \times old + conf \times \hat{e}$

#### 1. Bid-Ask Step
*   **[ ] Compute Equilibrium Estimate** from weighted price history
*   **[ ] Compute Confidence** from accumulated weight
*   **[ ] Blend:** Combine old bid/ask with equilibrium estimate

#### 2. Buy-Sell Step
*   **[ ] Gap Analysis:** Track spread closing rate
*   **[ ] Time Pressure:** More aggressive as time runs out
*   **[ ] Probabilistic Acceptance:** Based on profit/(profit+gap) ratio

---

### Trader: Perry
**Classification:** Adaptive, Complex, Efficiency-Based Learning
**Source Reference:** SRobotPerryOriginal.java (1993 Santa Fe Tournament)
**File:** `traders/legacy/perry.py`

Uses adaptive parameter tuning based on efficiency evaluation each period.

#### Core Mechanism
*   **[ ] Adaptive Parameter a0:** Adjusts based on period efficiency
*   **[ ] Round Statistics:** Tracks price sum, sum of squares, trade count
*   **[ ] Efficiency Evaluation:** Compares actual profit to potential profit

#### 1. Bid-Ask Step
*   **[ ] Conservative Early:** Random bids in first 3 trades of first period
*   **[ ] Statistical Later:** Based on round average ± a1 × std_dev
*   **[ ] Time Pressure:** Adjust a1 based on remaining time

#### 2. Buy-Sell Step
*   **[ ] Threshold Check:** Compare price to statistical threshold
*   **[ ] Desperate Acceptance:** More lenient near period end

#### 3. Period End Evaluation
*   **[ ] Calculate Efficiency:** actual_profit / potential_profit
*   **[ ] Adjust a0:** Scale up/down based on efficiency vs trades made

---

### Trader: Lin
**Classification:** Statistical, Simple
**Source Reference:** SRobotLin.java (1993 Santa Fe Tournament)
**File:** `traders/legacy/lin.py`

Uses statistical price prediction with Box-Muller sampling.

#### Core Mechanism
*   **[ ] Mean Price Tracking:** Store mean price per period
*   **[ ] Target Price:** Average of current and historical means
*   **[ ] Noise Injection:** Box-Muller transform for normal distribution

#### 1. Bid-Ask Step
*   **[ ] Calculate Target:** $target = norm(mean, stderr)$ using Box-Muller
*   **[ ] Weight Formula:** Based on time, tokens remaining, market composition
*   **[ ] Blend:** Weighted combination of conservative and target price

#### 2. Buy-Sell Step
*   **[ ] Statistical Threshold:** Accept if price better than target ± stderr

---

## Simplified/Baseline Traders

### Trader: Markup
**Classification:** Non-Adaptive, Simple, Deterministic
**Source Reference:** Chen et al. (2010) baseline
**File:** `traders/legacy/markup.py`

Fixed percentage markup - even simpler than ZIC (no randomization).

#### 1. Bid-Ask Step
*   **[ ] Buyer Bid:** $bid = valuation \times (1 - markup)$
*   **[ ] Seller Ask:** $ask = cost \times (1 + markup)$
*   **[ ] Default Markup:** 10%

#### 2. Buy-Sell Step
*   **[ ] Accept if Profitable:** Same as ZIC

---

### Trader: HistogramLearner
**Classification:** Adaptive, Simple, Empirical
**Source Reference:** Simplified empirical Bayesian approach
**File:** `traders/legacy/histogram_learner.py`

Tracks recent transaction prices to inform bidding.

#### 1. Bid-Ask Step
*   **[ ] Track Price Window:** Sliding window of recent prices
*   **[ ] Buyer:** Bid below mean (capture buyer-favorable trades)
*   **[ ] Seller:** Ask above mean (capture seller-favorable trades)
*   **[ ] Fallback:** Use Markup strategy when insufficient data

#### 2. Buy-Sell Step
*   **[ ] Accept if Profitable**

---

### Trader: ReservationPrice
**Classification:** Adaptive, Simple, Time-Decaying
**Source Reference:** Simplified BGAN (Friedman 1991)
**File:** `traders/legacy/reservation_price.py`

Time-decaying reservation price - start conservative, become aggressive.

#### 1. Bid-Ask Step
*   **[ ] Buyer:** Start at 50% of valuation, approach 95% as time runs out
*   **[ ] Seller:** Start at 150% of cost, approach 105% as time runs out
*   **[ ] Urgency Rate:** Configurable decay parameter

#### 2. Buy-Sell Step
*   **[ ] Accept if Profitable**

---

### Trader: Gamer
**Classification:** Simple, Non-Adaptive, Fixed-Rule
**Status:** 24th Place (Poor performer, but a good baseline for fixed-margin behavior)
**Source Reference:** *JEDC Paper* (Page 91, Figure 10)
**File:** `traders/legacy/gamer.py`

"Gamer" does not look at the market state (Current Bid/Ask) to set its price. It simply calculates a fixed profit margin based on its private token value and submits that price, hoping someone walks into it.

#### 1. Initialization
*   **[ ] Define Margin Constant:**
    *   Set `Margin` = 0.10 (10%).

#### 2. Bid-Ask Step (Bidding Logic)
*   **[ ] Calculate Target Price:**
    *   **Buyer:** `Target` = `TokenRedemptionValue` $\times$ $(1 - Margin)$. (Bids 10% below value).
    *   **Seller:** `Target` = `TokenCost` $\times$ $(1 + Margin)$. (Asks 10% above cost).
*   **[ ] Rounding:**
    *   Round `Target` to the nearest integer.
*   **[ ] Validity Check:**
    *   **Buyer:** Is `Target` > `CurrentBid`? **Action:** If True → Submit. Else → PASS.
    *   **Seller:** Is `Target` < `CurrentAsk`? **Action:** If True → Submit. Else → PASS.

#### 3. Buy-Sell Step (Take Logic)
*   **[ ] Profitability Check:**
    *   **Buyer:** If `CurrentAsk` < `TokenRedemptionValue`, return `BUY`.
    *   **Seller:** If `CurrentBid` > `TokenCost`, return `SELL`.

---

### Trader: Breton
**Classification:** Stochastic, Adaptive, Simple
**Source Reference:** *JEDC Paper* (Section 3.4, Page 81)
**File:** `traders/legacy/breton.py`

Breton introduces intentional noise into its decision-making. It calculates a "reasonable" price based on the current market state and its private value, then adds a random "shake" (error term) to prevent deadlock.

#### 1. Bid-Ask Step (Bidding Logic)
*   **[ ] Market State Check:**
    *   If no `CurrentBid`/`CurrentAsk`, default to `MinPrice` (Buyer) or `MaxPrice` (Seller).
*   **[ ] Calculate Target Price (Weighted Average):**
    *   **Buyer:** `Target` = $(\lambda \times CurrentBid) + ((1 - \lambda) \times TokenRedemptionValue)$
    *   **Seller:** `Target` = $(\lambda \times CurrentAsk) + ((1 - \lambda) \times TokenCost)$
    *   Default $\lambda = 0.5$
*   **[ ] Add Noise:**
    *   Generate random integer $E \in [-2, 2]$
    *   `ProposedPrice` = `Target` + $E$
*   **[ ] Validity Check:**
    *   Must improve market AND stay profitable

#### 2. Buy-Sell Step (Take Logic)
*   **[ ] Profitability Check:** Same as other traders

---

## Test Utilities

### Trader: GradualBidder
**Classification:** Test Utility
**File:** `traders/legacy/gradual.py`

Narrows spread through gradual bidding but NEVER accepts trades. Used to demonstrate Kaplan's deal-stealing mechanic.

#### 1. Bid-Ask Step
*   **[ ] Use Skeleton's Weighted Average:** Gradually narrow the spread
*   **[ ] Conservative Start:** First bid/ask at extreme to ensure wide spread

#### 2. Buy-Sell Step
*   **[ ] Always Return False:** Never accept trades
*   **[ ] Purpose:** Forces spread to narrow without trades executing, allowing Kaplan to "steal"
