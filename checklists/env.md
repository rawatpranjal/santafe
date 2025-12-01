# Santa Fe Double Auction Tournament Configuration

> The tournament was conducted rigorously using fixed environments and a controlled randomized matching scheme to ensure fair testing of all strategies.

---

## 1. Tournament Setup and Execution

### 1.1 Game Parameters

Each environment \(E(i)\) is a complete specification of the synchronized double-auction game used in that environment. In the original Santa Fe tournament the key parameters were:

- **Token distribution:**
  - `GAMETYPE` → 4-digit code determining four ranges \(R_1, R_2, R_3, R_4\) for the random variables in the token formula.
  - `MINPRICE`, `MAXPRICE` → lower and upper bounds on admissible bids/asks.

- **Market size:**
  - `NBUYERS`, `NSELLERS` → number of buyers and sellers.
  - `NTOKENS` → number of tokens per trader per period.

- **Temporal structure:**
  - `NROUNDS` → rounds per game.
  - `NPERIODS` → periods per round.
  - `NTIMES` → maximum discrete steps per period.

Internally, for each game, the engine computes and reports `RAN1`–`RAN4` from `GAMETYPE` via the base-3 rule and may also fix auxiliary parameters like `DEADSTEPS` and `TIMEOUT`.

At the start of each game, all programs receive the realized values of these parameters (including `GAMETYPE`) via the monitor header; they are common knowledge within that game.

### 1.2 Token Generation

Token values are determined by `GAMETYPE` using a formula with four independent uniform random components:

- Four ranges \(R_1,\dots,R_4\) are set from the digits of `GAMETYPE` by a base-3 coding scheme: digit \(d\) → \(3^d - 1\).
- These ranges are used to draw \((A, B_1, B_2, C, D)\) and construct token values for each trader and token.

Combined with the tournament’s sampling and matching scheme (blocks of games with the same set of token draws), this design guarantees that all programs in an environment face effectively identical potential surplus endowments.

### 1.3 Player Matching

For each environment \(E(i)\), the organizers ran many games using the following scheme:

- For a given token draw:
  - A block of \(N\) games is played, where \(N\) is the number of entrants (30 in the original cash tournament).
  - In this block:
    - Programs are randomly assigned to specific buyer/seller positions (B1, B2, S1, S2, etc.).
    - No program ever plays against an identical copy of itself.
    - Each program plays each labeled position an equal number of times.

- If a program was only coded to play one side (buyer or seller), a standard “Skeleton” trader was used as a stand-in on the missing side rather than skipping that game.

- After the block of \(N\) games, a new set of token values is drawn and the procedure is repeated.

This block design ensures that differences in profits are due to strategy, not token luck or asymmetric opponent sets.

---

## 2. Ranking and Payoff System

### 2.1 Token Profit Calculation

Within environment \(E(i)\), each program \(j\) accumulates **token profits** \(TP(i,j)\) across all games:

- Buyer trades contribute (token value − transaction price).
- Seller trades contribute (transaction price − token cost).

### 2.2 Dollar Conversion

Each environment is allocated \$1,000 of prize money. For environment \(E(i)\):

- Let \(TS(i)\) be the total theoretical surplus available (given the token distributions and parameters).
- Define the conversion ratio:
  \[
  c(i) = \frac{1000}{TS(i)}.
  \]
- Dollar payment in environment \(i\) for program \(j\) is:
  \[
  DP(i,j) = c(i) \times TP(i,j).
  \]

### 2.3 Final Payout

- Total dollar payment for program \(j\):
  \[
  DP(j) = \sum_i DP(i,j)
  \]
  across all ten environments.
- Final ranking is by \(DP(j)\), highest to lowest.
- In the March 1990 cash tournament, the \$10,000 pool was fully allocated across environments, but only \$8,937 was actually paid out because markets did not achieve 100% efficiency in exploiting the theoretical surplus.

---

## 3. Historical Tournament Details

### 3.1 Event Overview

| Attribute   | Value                                             |
|------------|---------------------------------------------------|
| Event      | Double Auction Tournament, Santa Fe Institute     |
| Date       | March 1990 (cash); 1991 (scientific/evolutionary) |
| Participants | 30 entrants                                    |
| Platform   | Unix workstations (Sun-based environment)         |
| Total Games | 2,233 games across 10 environments              |
| Total Periods | 13,398 periods                                |
| Prize Pool | \$10,000 allocated; \$8,937 paid (~89% of pool)  |

### 3.2 The Ten Environments

Canonical parameter values from the original tournament (Table 3.1, Rust-Palmer-Miller 1992):

| Env  | gametype | maxprice | nbuyers | nsellers | ntokens | nrounds | nperiods | ntimes |
|------|----------|----------|---------|----------|---------|---------|----------|--------|
| BASE | 6453     | 2000     | 4       | 4        | 4       | 2       | 3        | 75     |
| BBBS | 6453     | 2000     | 6       | 2        | 4       | 2       | 3        | 50     |
| BSSS | 6453     | 2000     | 2       | 6        | 4       | 2       | 3        | 50     |
| EQL  | 0        | 2000     | 4       | 4        | 4       | 2       | 3        | 75     |
| LAD  | 0        | 2000     | 4       | 4        | 4       | 2       | 3        | 75     |
| PER  | 6453     | 2000     | 4       | 4        | 4       | 6       | 1        | 75     |
| SHRT | 6453     | 2000     | 4       | 4        | 4       | 2       | 3        | 25     |
| SML  | 6453     | 2000     | 2       | 2        | 4       | 2       | 3        | 50     |
| RAN  | 7        | 3000     | 4       | 4        | 4       | 2       | 3        | 50     |
| TOK  | 6453     | 2000     | 4       | 4        | 1       | 2       | 3        | 25     |

Environment descriptions:

| Env  | Description          | Key Variation                            |
|------|----------------------|------------------------------------------|
| BASE | Standard base case   | Reference configuration                  |
| BBBS | Buyer-dominated      | Duopsony: 6 buyers vs 2 sellers         |
| BSSS | Seller-dominated     | Duopoly: 2 buyers vs 6 sellers          |
| EQL  | Equal endowment      | gametype=0: all traders get same token values shifted by common constant |
| LAD  | Low-adaptivity       | gametype=0: same as EQL, tests learning effects |
| PER  | Single-period        | 6 rounds x 1 period: prevents cross-period learning |
| SHRT | High-pressure        | 25 steps: time-constrained trading      |
| SML  | Small market         | 2x2 traders: minimal market size        |
| RAN  | Independent values   | gametype=7: IID uniform token draws (R1=R2=R3=0, R4=2186) |
| TOK  | Single token         | 1 token per trader: simplest allocation problem |

Notes:
- gametype=0 means "unspecified process" - tokens are generated with equal relative values across all traders
- gametype=7 (0007) produces IID draws: R1=R2=R3=0, R4=3^7-1=2186
- All environments use minprice=1

### 3.3 Key Finding: The Kaplan Strategy

- **Winner:** Todd Kaplan (University of Minnesota) – top overall dollar payoff in the March 1990 cash tournament (about \$408). :contentReference[oaicite:12]{index=12}
- **Classification:** Simple, nonadaptive, nonpredictive, nonstochastic, nonoptimizing.
- **Rule of thumb:** “Wait in the background and let others do the negotiating; when bid and ask get sufficiently close, jump in and steal the deal.”
- **Implication:** This simple background strategy consistently outperformed more complex programs based on neural networks, Bayesian updating, or explicit optimization in most environments.
