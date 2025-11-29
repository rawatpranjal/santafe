# Santa Fe Tournament - Implementation Checklists

> **Purpose:** Consolidated checklists for verifying implementation correctness.
> Tick items off as you validate each component.

## Table of Contents
1. [AURORA Protocol Rules](#1-aurora-protocol-rules)
2. [Trader Algorithms](#2-trader-algorithms)
3. [Evaluation Metrics](#3-evaluation-metrics)
4. [Tournament Environments](#4-tournament-environments)
5. [Kaplan Deep Dive](#5-kaplan-deep-dive)

---

# 1. AURORA Protocol Rules

Granular checklist for verifying the Santa Fe Double Auction Tournament implementation.

### 1.1 Global Simulation Structure
*   **[ ] Topology:** The system is a star network. All agents (Traders) communicate *only* with the central node (Monitor). Agents cannot communicate with each other.
*   **[ ] Discrete Time:** Time is not continuous. It proceeds in discrete steps.
*   **[ ] Hierarchy:** The simulation must follow this nesting order:
    1.  **Tournament:** A collection of Environments.
    2.  **Environment:** A specific configuration (e.g., "BASE", "SHRT") consisting of $N$ Games.
    3.  **Game:** A set of `nrounds` (usually 20).
    4.  **Round:** A set of `nperiods` (usually 3 to 5). **Token values are constant within a round.**
    5.  **Period:** A sequence of alternating steps (Bid-Ask $\leftrightarrow$ Buy-Sell). **Inventory resets every period.**
    6.  **Step:** The atomic unit of time.

---

### 1.2 Token Economy (Initialization)
*   **[ ] Token Assignment:**
    *   Buyers are assigned `ntokens` (usually 4) to buy.
    *   Sellers are assigned `ntokens` (usually 4) to sell.
*   **[ ] Value Generation Formula:** Token value $T_{jk}$ for player $j$ and token $k$ is calculated as (per STokenGeneratorOriginal.java):
    *   IF Buyer: $Val = A + B1 + C_k + D_{jk}$
    *   IF Seller: $Cost = A + B2 + C_{k+N} + D_{jk}$ where N=num_tokens
    *   NOTE: Buyers and sellers use DIFFERENT B values (B1 vs B2) and DIFFERENT C indices
*   **[ ] Random Variables:** $A, B1, B2, C, D$ are drawn from Uniform distributions $U[0, R]$. Note: B1 and B2 are independently drawn from same range.
*   **[ ] Range Definitions (`gametype`):** Ranges $R_1, R_2, R_3, R_4$ correspond to variables $A, B, C, D$.
    *   Input is a 4-digit number (e.g., `6453`).
    *   Convert each digit $d$ using base-3 logic: $Range = 3^d - 1$.
    *   *Example:* Digit 4 $\rightarrow$ $3^4 - 1 = 80$.
*   **[ ] Sorting Rule:**
    *   Buyer tokens must be sorted High $\rightarrow$ Low (Redemption Values).
    *   Seller tokens must be sorted Low $\rightarrow$ High (Costs).
*   **[ ] Forced Consumption Order:** The engine MUST force sequential token consumption (1st token for 1st trade, 2nd for 2nd, etc.). Agents cannot "save" their best tokens for later. *(PDF Page 3, Rules 5 & 6)*
*   **[ ] Private Info:** Traders only know their own values, not the values of others.

---

### 1.3 Step 1: Bid-Ask (BA) Logic
*   **[ ] Entry:** All agents (Buyers and Sellers) may submit a message.
*   **[ ] Buyer Constraints:**
    *   Bid must be integer.
    *   Bid $\le$ `maxprice`.
    *   Bid $>$ `CurrentBid` (if a bid already exists on the board).
*   **[ ] Seller Constraints:**
    *   Ask must be integer.
    *   Ask $\ge$ `minprice`.
    *   Ask $<$ `CurrentAsk` (if an ask already exists on the board).
*   **[ ] State Update:**
    *   Identify Highest Bid $\rightarrow$ becomes `CurrentBid`.
    *   Identify Lowest Ask $\rightarrow$ becomes `CurrentAsk`.
    *   Identify the agents holding these values $\rightarrow$ `CurrentBidder` and `CurrentAsker`.
*   **[ ] Tie-Breaking:** If multiple *new* agents submit the same best price (which must be strictly better than the standing bid/ask), choose one randomly. *(PDF Rule 12)*
*   **[ ] Incumbent Protection:** An agent cannot match the standing `CurrentBid` or `CurrentAsk`—they must strictly improve it. *(PDF Rules 10 & 11)*
    *   *Example:* If Agent A holds `CurrentBid` at 100, Agent B cannot submit 100. B must bid ≥101. If B and C both submit 105, one is chosen randomly to become `CurrentBidder`.

---

### 1.4 Step 2: Buy-Sell (BS) Logic
*   **[ ] Exclusivity Rule:** Only the `CurrentBidder` and `CurrentAsker` (from the previous step) are allowed to act. All other agents are blocked.
*   **[ ] Valid Actions:**
    *   `CurrentBidder`: Can send `BUY` (accept current ask) or `PASS`.
    *   `CurrentAsker`: Can send `SELL` (accept current bid) or `PASS`.
*   **[ ] Transaction Logic:**
    *   **Case 1:** Bidder says `BUY`. Trade executes at `CurrentAsk` price.
    *   **Case 2:** Asker says `SELL`. Trade executes at `CurrentBid` price.
    *   **Case 3:** Both say `BUY`/`SELL`. Trade executes. Monitor randomly selects one winner:
        *   If Buyer selected: Price = `CurrentAsk`.
        *   If Seller selected: Price = `CurrentBid`.
        *   *(PDF Rule 14 & 17: No averaging. Discrete integer prices only.)*
    *   **Case 4:** Neither accepts. No trade.
*   **[ ] Post-Step Cleanup:**
    *   **IF Trade Occurred:** BOTH `CurrentBid` AND `CurrentAsk` are cleared to NULL, even if only one side participated in the trade. Next step starts with clean slate. Inventory of involved traders decreases by 1. *(PDF Page 6, Rule 19)*
    *   **IF NO Trade:** `CurrentBid` and `CurrentAsk` carry over to the next Bid-Ask step.

---

### 1.5 Period Termination Rules
*   **[ ] Time Limit:** The period ends if the step count reaches `ntimes`.
*   **[ ] Deadsteps:** The period ends early if `deadsteps` threshold is reached (consecutive steps with no trades).
*   **[ ] Inventory:** The period acts as if ended for a specific agent if they have traded all their assigned tokens (0 inventory).
*   **[ ] Zero-Token Flag:** Engine must flag agents with 0 tokens (e.g., `nobuysell=1`) and reject any subsequent bid/ask submissions from them. *(PDF Page 34, nobuysell variable)*

---

### 1.6 Configuration Parameters (Table 1 Check)
Verify the simulation supports these variable settings:
*   **[ ] `gametype`:** Controls random generation (e.g., 6453, 0007).
*   **[ ] `minprice`/`maxprice`:** Price floor/ceiling (e.g., 1 to 2000).
*   **[ ] `nbuyers`/`nsellers`:** Number of agents (e.g., 4 vs 4, or 6 vs 2).
*   **[ ] `ntokens`:** Tokens per agent (usually 4).
*   **[ ] `ntimes`:** Max steps per period.

### 1.7 Profit Calculation
*   **[ ] Buyer Profit:** $\sum (\text{Redemption Value} - \text{Transaction Price})$.
*   **[ ] Seller Profit:** $\sum (\text{Transaction Price} - \text{Token Cost})$.
*   **[ ] Ranking:** Agents are ranked by total profit accumulated over the tournament.

---

### 1.8 Engine Mechanics (Nuances)

Critical implementation details from Participant's Manual:

*   **[ ] Discrete Prices:** All transaction prices must be integers. No averaging, no floats.
*   **[ ] Full Board Wipe:** Upon ANY trade, set `CurrentBid = NULL` AND `CurrentAsk = NULL`.
*   **[ ] Strict Improvement:** In Bid-Ask step, valid bid must be STRICTLY > CurrentBid. Cannot match to seize control.
*   **[ ] Conflict Resolution:** If BUY and SELL messages arrive simultaneously in Buy-Sell step:
    *   Randomly select one winner.
    *   If Buyer selected: Price = CurrentAsk.
    *   If Seller selected: Price = CurrentBid.
*   **[ ] Incumbent Protection via Price:** Existing holder cannot be dethroned by matching—only by strictly improving the price. Tie-breaking (Rule 12) only applies among *new* simultaneous submissions at the same improved price level.
