This information is detailed primarily in Chapter 3, Section 3.2 ("Variables in the user routines") of the Participant's Manual.

The data available to agents (players) is divided into **Public Constants** (known and unchanging), **Public Variables** (known and varying, detailing the market state and history), and **Private Variables** (known only to the individual player).

---

## 1. Private Variables (Player's Own State & Secret Information)

These variables contain information specific to the player, including their fundamental strategic values.

| Variable | Description | Strategic Significance |
| :--- | :--- | :--- |
| **`token[4]`** | **Redemption Values (Buyer) or Token Costs (Seller).** Given in decreasing order for buyers (to maximize profit $R-P$) and increasing order for sellers (to maximize profit $P-C$). | **Crucial Private Information.** Rule 7 explicitly states these values are **private information not communicated to other players.** |
| `id` | Your own identification number (unique ID, 1 to nbuyers or 1 to nsellers). | |
| `role` | Your role (1 if buyer, 2 if seller). | |
| `ntokens` | Number of tokens available to buy/sell this round. | |
| `mytrades` | Number of trades you have made so far this period. | Used to determine which `token[i]` to use next. |
| `pprofit` | Your total profit so far in this period. | |
| `rprofit` | Your total profit so far in this round. | |
| `gprofit` | Your total profit so far in this game. | |
| `nobidoff` | Code indicating why you cannot bid/offer (e.g., no tokens left). | |
| `nobuysell` | Code indicating why you cannot make a buy/sell request (e.g., no current bid/offer, not the current holder). | |
| `bo` | Outcome code of your request in the last bid-offer step (e.g., chosen, bettered, lost tie). | |
| `bs` | Outcome code of your request in the last buy-sell step. | |
| `late` | Number of times you were late/non-responsive this period (signals missed info). | |
| `tradelist[5]` | Number of trades you made in each period of this round. | |
| `profitlist[5]` | Profit you made in each period of this round. | |
| `mylasttime` | The time (`t`) of your most recent trade this period. | |
| `timeout` | The wall-clock time allowed for your response per step. | |
| `efficiency` | Measure of overall performance (Profit / Equilibrium Profit). **Only available at the end of the game (`GEND` routine).** | |

---

## 2. Public Variables (Market State and History)

These variables change as the game progresses and are known to all players (usually updated after a bid-offer or buy-sell step).

| Variable | Description | Update Timing |
| :--- | :--- | :--- |
| `r` | Current round number. | Start of Round |
| `p` | Current period number. | Start of Period |
| `t` | Current time (step) within the period. | Every Step |
| `cbid` | **Current Bid value** (Highest outstanding bid). | Cleared on Transaction |
| `coffer` | **Current Offer value** (Lowest outstanding offer). | Cleared on Transaction |
| `bidder` | ID of the buyer holding the current bid (0 if none). | |
| `offerer` | ID of the seller holding the current offer (0 if none). | |
| `nbids` | Number of *new* bids made in the last bid-offer step. | End of Bid-Offer Step |
| `noffers` | Number of *new* offers made in the last bid-offer step. | End of Bid-Offer Step |
| **`bids[20]`** | The **actual bids** made by each buyer in the last bid-offer step. | End of Bid-Offer Step |
| **`offers[20]`** | The **actual offers** made by each seller in the last bid-offer step. | End of Bid-Offer Step |
| `bstype` | Code indicating what happened in the last buy-sell step (0, 1, 2, or -1 for unknown). | End of Buy-Sell Step |
| `price` | Price of the transaction (if one occurred). | End of Buy-Sell Step |
| `buyer` | ID of the buyer involved in the transaction (if one occurred). | End of Buy-Sell Step |
| `seller` | ID of the seller involved in the transaction (if one occurred). | End of Buy-Sell Step |
| `btrades[20]` | Summary: Trades made by each buyer so far in this period. | End of Buy-Sell Step |
| `strades[20]` | Summary: Trades made by each seller so far in this period. | End of Buy-Sell Step |
| `ntrades` | Total number of trades made by all players so far in this period. | End of Buy-Sell Step |
| **`prices[80]`** | **Price of every trade** that has occurred so far in this period. | End of Buy-Sell Step |
| `lasttime` | Time (`t`) of the most recent trade. | End of Buy-Sell Step |

***Note on Information Flow (Rules 13 & 18):***

1.  **Bids and Offers:** All players are informed about all legal bids and offers, and who made them, at the conclusion of each **bid-offer step** (Rule 13). They know the current bid/offer and the identity of the current bidder/offerer.
2.  **Transactions:** All players are informed about all transactions that occur. This includes the transaction price, the identity of the buyer and seller involved, and whether it was a buy or a sell request that was accepted (Rule 18).
3.  **Failed Requests:** Players are **not** informed about buy or sell requests that were *not* accepted (Rule 18).

---

## 3. Public Constants (Game Parameters)

These are set at the start of the game and remain fixed.

*   `nplayers`: Total number of players.
*   `nbuyers`: Number of buyers (max 20).
*   `nsellers`: Number of sellers (max 20).
*   `nrounds`: Maximum rounds (max 20).
*   `nperiods`: Maximum periods per round (max 5).
*   `ntimes`: Number of time steps per period (max 400).
*   `minprice`: Minimum price (1).
*   `maxprice`: Maximum price (8000).
*   `gameid`: Unique game identifier.
*   `gametype`: 4-digit number that conveys the parameters (RAN1...RAN4) used for random token generation.

---

## 4. What is NOT in the Information Set

Crucially, no agent ever observes:

| Non-Observable | Explanation |
|----------------|-------------|
| Other traders' token values/costs | Rule 7: Token values are private information not communicated to other players |
| Other traders' profits | Only your own `pprofit`, `rprofit`, `gprofit` are visible |
| Other traders' strategies or code | The mechanism only broadcasts actions, not decision rules |
| Failed buy/sell requests | Rule 18: Players are not informed about requests that were not accepted |
| Future token draws | Only current round tokens are revealed; future draws unknown beyond distribution F |
| Hidden order-book state | All order-book info is publicly announced; no "dark pool" or hidden liquidity |

This asymmetric information structure is fundamental to the DA mechanism: traders have common knowledge of the game parameters and public history, but private knowledge of their own valuations.

---

## 5. Summary: Complete Information Set at Any Decision Point

At any decision point, an agent's information set consists of:

**Static Public Knowledge:**
- Environment specification: `nbuyers`, `nsellers`, `nrounds`, `nperiods`, `ntimes`
- Price bounds: `minprice`, `maxprice`
- Token distribution parameters: `gametype` (encodes the A,B,C,D random ranges)

**Private Knowledge:**
- Own token valuations/costs: `token[4]`
- Own trading state: `mytrades`, `pprofit`, `rprofit`, `gprofit`

**Dynamic Public History:**
- All bids ever posted: reconstructable from `bids[20]` broadcasts each BA step
- All asks ever posted: reconstructable from `offers[20]` broadcasts each BA step
- Current best bid/ask: `cbid`, `coffer` and holders `bidder`, `offerer`
- All transaction prices: `prices[80]`
- All transaction counterparties: `buyer`, `seller` for each trade
- Current time: `r`, `p`, `t`

**Derived State (agent may compute internally):**
- Price statistics from past periods (min, max, average)
- Bid-ask spread history
- Trading patterns of specific counterparties
- Own remaining profitable tokens

This information structure enables sophisticated strategies like Kaplan (which uses price history and timing) and Ledyard (which uses bid/ask history for reservation price estimation).
