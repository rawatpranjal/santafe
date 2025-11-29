# Santa Fe Double Auction Tournament Configuration

> The tournament was conducted rigorously using fixed environments and randomized matching to ensure fair testing of all strategies.

---

## 1. Tournament Setup and Execution

### 1.1 Game Parameters
Each environment E(i) is a complete specification of 10 parameters:
- **Players:** `NBUYERS`, `NSELLERS` (up to 20 each)
- **Capacity:** `NTOKENS` (up to 4 per player per period)
- **Duration:** `NROUNDS` (up to 20), `NPERIODS` (up to 5), `NTIMES` (up to 400 steps)
- **Token Ranges:** `RAN1`-`RAN4` (derived from `gametype`)

All programs receive common knowledge of these settings (except `gametype` in some environments).

### 1.2 Token Generation
Token values determined by `gametype` using formula with four uniform random variables:
- Ranges R1-R4 set by base-3 coding scheme (digit d → 3^d - 1)
- Guarantees equal potential surplus endowments across strategies

### 1.3 Player Matching
For each environment E(i), N(i) games played with:
- Programs randomly chosen without replacement to fill buyer/seller roles
- Programs unable to fill assigned role skipped for that game
- No program plays against itself

---

## 2. Ranking and Payoff System

### 2.1 Token Profit Calculation
Programs accumulate total token profits TP(i,j) across all games in environment E(i).

### 2.2 Dollar Conversion
Conversion ratio c(i) = $1,000 / Total Theoretical Surplus TS(i)

### 2.3 Final Payout
- Dollar payment: DP(i,j) = c(i) × TP(i,j)
- Final ranking: sum of dollar payments across all ten environments
- Full $10,000 pool linked to maximum theoretical surplus; difference reflects market efficiency

---

## 3. Historical Tournament Details

### 3.1 Event Overview
| Attribute | Value |
|-----------|-------|
| Event | Double Auction Tournament, Santa Fe Institute |
| Date | March 1990 (cash); 1990-1991 (scientific) |
| Participants | 30 entrants |
| Platform | Sun-4 systems, SunOS 4.0 |
| Total Games | 18,114 games, 30,312 periods |
| Prize Pool | $10,000 allocated; $8,937 paid (89% efficiency) |

### 3.2 The Ten Environments

| Env | Description | Key Variation | gametype |
|-----|-------------|---------------|----------|
| **BASE** | Standard base case | 4B/4S, 4 tokens, 3 periods, 75 steps | 6453 |
| **BBBS** | Buyer-dominated (duopsony) | **6B/2S** | 6453 |
| **BSSS** | Seller-dominated (duopoly) | **2B/6S** | 6453 |
| **EQL** | Equal endowment | Values shifted by common constant | 0 |
| **LAD** | Low adaptivity | Same as BASE | 6453 |
| **PER** | Single period | **1 period** per round | 6453 |
| **RAN** | Independent private values | IID draws (R1=R2=R3=0) | 6453 |
| **SHRT** | High-pressure | **25 steps** per period | 6453 |
| **SML** | Small market | **2B/2S** | 0007 |
| **TOK** | Single token | **1 token** per player | 6453 |

### 3.3 Key Finding: The Kaplan Strategy
- **Winner:** Todd Kaplan (University of Minnesota) — $408 earned
- **Classification:** Simple, Nonadaptive, Nonpredictive, Nonstochastic, Nonoptimizing
- **Rule:** "Wait in the background; when bid-ask spread narrows, jump in and steal the deal"
- **Implication:** Outperformed complex adaptive algorithms using neural networks and statistical predictions
