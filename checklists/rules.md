High-level: your checklist already captures most of what you need to implement a Santa Fe–style double auction. Below is a compact “how the engine actually works” view plus a few implementation clarifications/gaps to be aware of.

---

## 1. Global structure (what resets where)

You need a nested simulation loop:

1. **Tournament**

   * A set of Environments. Profits are aggregated across all Environments/Games/Rounds/Periods.

2. **Environment**

   * A configuration (e.g. `BASE`, `SHRT`) with:

     * `gametype` (ranges for A,B,C,D)
     * `minprice`, `maxprice`
     * `nbuyers`, `nsellers`
     * `ntokens`, `ntimes`
     * `nrounds`
     * `nperiods`
     * `deadsteps`
   * Contains **N Games** with identical settings.

3. **Game**

   * Contains `nrounds` (≈ 20).

4. **Round**

   * Contains `nperiods` (3–5).
   * **Token values are drawn once per Round** for each agent and token and are **constant across all periods in that Round.**

5. **Period**

   * Inventory resets: each agent starts with `ntokens` units.
   * Uses the same token values as the other periods of that Round.
   * Runs a sequence of discrete **Steps** that alternate:

     * Step 1: Bid–Ask (BA)
     * Step 2: Buy–Sell (BS)
     * Step 3: BA
     * Step 4: BS
     * … until period termination.

6. **Step**

   * Atomic unit of time: a BA step or a BS step.
   * Period ends when:

     * Step index reaches `ntimes`, or
     * Deadsteps condition is hit (no trades), or
     * All agents have 0 inventory (effectively nothing more can happen).

**Game/Round Initialization:**

* At the start of each **Game**, the Monitor broadcasts to all traders:
  * `nbuyers`, `nsellers`, `nrounds`, `nperiods`, `ntimes`, `ntokens`, `gametype`, `minprice`, `maxprice`, `deadsteps`.
* For each **Round**, token values are generated and then **privately** sent to each trader.
* This makes explicit what is **public** (market config) vs **private** (own token values).

---

## 2. Token economy and value generation

For each Round and each agent (j):

* Agent type:

  * Buyer: has `ntokens` units to buy.
  * Seller: has `ntokens` units to sell.

* Values per token (k \in {1, …, ntokens}):

  * Buyer token value:
    [
    Val_{jk} = A + B1 + C_k + D_{jk}
    ]
  * Seller token cost:
    [
    Cost_{jk} = A + B2 + C_{k+N} + D_{jk}
    ]
    where (N = ntokens).

* Random draws:

  * (A \sim U[0, R_1])
  * (B1, B2 \sim U[0, R_2]) independently
  * (C_\ell \sim U[0, R_3]) for (\ell = 1, …, 2N)
  * (D_{jk} \sim U[0, R_4]) for every trader-token pair
  * All independent across agents and tokens.

* **When random draws happen:**

  * All A, B1, B2, C, D draws are done **once per Round** (per environment + game), not per Period or per Step.
  * Token values then stay fixed for all Periods of that Round.

* `gametype` → ranges:

  * 4-digit code, e.g. `6453`.

    * 1st digit → range for A (`R1`)
    * 2nd digit → range for B (`R2`)
    * 3rd digit → range for C (`R3`)
    * 4th digit → range for D (`R4`)
  * Digit (d) gives:
    [
    R = 3^d - 1
    ]
    Example: digit 4 → (3^4 - 1 = 80).

* Sorting and forced consumption:

  * For each agent in a Round:

    * Buyers: sort tokens by value **descending**.
    * Sellers: sort tokens by cost **ascending**.
  * In the Period, trades must use tokens sequentially:

    * 1st trade uses token #1
    * 2nd trade uses token #2
    * etc.
  * Agents cannot select which token to use; they always use “next token index” for the next executed trade.

* Information:

  * Each trader only sees their own ordered list of values/costs.
  * No information about others' tokens or payoffs.

* **Public information broadcast:**

  * After each BA step (and after trades), the engine broadcasts to all traders:
    * `CurrentBid`, `CurrentAsk`
    * identities of `CurrentBidder` and `CurrentAsker`
    * (optionally) the list of executed trades so far in the Period.

---

## 3. Market state variables

At any time within a Period, the market board has:

* `CurrentBid` (integer or null)
* `CurrentBidder` (agent id or null)
* `CurrentAsk` (integer or null)
* `CurrentAsker` (agent id or null)

After any **successful trade**, **both sides of the board are cleared**:

* `CurrentBid = null`, `CurrentAsk = null`
* `CurrentBidder = null`, `CurrentAsker = null`

If **no trade** happens in the BS step, the board is preserved into the next BA step.

**Trade log structure:**

* Each trade is recorded as a tuple:
  * `(buyer_id, seller_id, price, round, period, step, buyer_token_index, seller_token_index)`
* This is needed to compute profits and to debug agent strategies.

---

## 4. Bid–Ask (BA) step rules

In a BA step:

1. **Who can act:**

   * All agents with at least 1 remaining token (inventory > 0).
   * An agent may:

     * Submit a **Bid** (if buyer),
     * Submit an **Ask** (if seller),
     * Or be silent (no message).

2. **Bid constraints (buyers):**

   * Bid is an integer.
   * `minprice <= Bid <= maxprice`.
   * If `CurrentBid` is not null:

     * New Bid must satisfy:
       [
       Bid > CurrentBid
       ]
     * Matching (`Bid == CurrentBid`) is not allowed.

3. **Ask constraints (sellers):**

   * Ask is an integer.
   * `minprice <= Ask <= maxprice`.
   * If `CurrentAsk` is not null:

     * New Ask must satisfy:
       [
       Ask < CurrentAsk
       ]
     * Matching (`Ask == CurrentAsk`) is not allowed.

4. **Illegal / out-of-range messages:**

   * If a bid/ask is not an integer, or lies outside `[minprice, maxprice]`, or fails the strict-improvement condition, it is treated as **illegal** and ignored (does not change the board).
   * Optionally log it for debugging, but the agent is not penalized beyond losing that submission.

5. **Updating the board:**

   * Among all valid new Bids:

     * Take the **highest** bid.
     * If strictly higher than the existing `CurrentBid`, it becomes the new `CurrentBid`, and its agent becomes `CurrentBidder`.
   * Among all valid new Asks:

     * Take the **lowest** ask.
     * If strictly lower than the existing `CurrentAsk`, it becomes the new `CurrentAsk`, and its agent becomes `CurrentAsker`.

6. **Tie-breaking among new challengers:**

   * If several agents submit the same top bid (all > old `CurrentBid`):

     * Randomly select one to become `CurrentBidder`.
   * Similarly for asks: if several submit the same best ask (< old `CurrentAsk`), randomly choose the new `CurrentAsker`.
   * The incumbent cannot be displaced by a matching price; only strictly better prices and then tie-breaking among the new offers.

7. **Explicit first-bid/first-ask case:**

   * If there was **no** `CurrentBid` before the step, select the highest submitted bid; if several tie at that highest price, choose one uniformly at random.
   * Symmetric for `CurrentAsk`: if none existed, select the lowest ask with random tie-breaking.

8. **Incumbent behavior:**

   * The incumbent (current best bid/ask holder) is allowed to improve their own price further (e.g. raise their own bid).
   * They are protected from being matched; others must strictly beat the current price.

---

## 5. Buy–Sell (BS) step rules

In the BS step:

1. **Who can act (revised):**

   * If **both** `CurrentBid` and `CurrentAsk` exist:
     * Only `CurrentBidder` and `CurrentAsker` may act.
   * If **only `CurrentBid` exists** (no ask on board):
     * All sellers with inventory > 0 may act (each chooses `SELL` or `PASS`).
   * If **only `CurrentAsk` exists** (no bid on board):
     * All buyers with inventory > 0 may act (each chooses `BUY` or `PASS`).
   * If **neither** exists:
     * No trade can occur in this BS step.

2. **Actions:**

   * `CurrentBidder` chooses between:

     * `BUY` (accept current ask)
     * `PASS`
   * `CurrentAsker` chooses between:

     * `SELL` (accept current bid)
     * `PASS`

3. **Transaction logic:**

   * Case A: Only buyer accepts

     * Buyer sends `BUY`, seller sends `PASS`.
     * Trade executes at price = `CurrentAsk`.
   * Case B: Only seller accepts

     * Seller sends `SELL`, buyer sends `PASS`.
     * Trade executes at price = `CurrentBid`.
   * Case C: Both accept

     * Buyer sends `BUY` and seller sends `SELL`.
     * Trade executes; monitor randomly selects which side's acceptance is honored:

       * If buyer's side is chosen: price = `CurrentAsk`.
       * If seller's side is chosen: price = `CurrentBid`.
   * Case D: Neither accepts

     * No trade.

4. **Multiple acceptors when only one side is on the board:**

   * If only a bid exists and **multiple sellers** send `SELL`:
     * Randomly select one seller to trade at `CurrentBid`; others are treated as if they had passed.
   * If only an ask exists and **multiple buyers** send `BUY`:
     * Randomly select one buyer to trade at `CurrentAsk`; others are treated as if they had passed.

5. **After a trade:**

   * Decrement inventory of buyer and seller by 1.
   * Advance each to their next token index (forced sequential use).
   * Clear both sides of the board:

     * `CurrentBid`, `CurrentAsk`, `CurrentBidder`, `CurrentAsker` all set to null.
   * Record realized profits for both sides (see below).

6. **After no trade:**

   * Board is unchanged and carried to next BA step.
   * Deadstep counter increases.

---

## 6. Period termination

The Period ends when any of these is true:

1. **Max steps reached:**

   * Step counter `>= ntimes`.

2. **Deadsteps threshold:**

   * A number of consecutive BS steps without any trade reaches `deadsteps`.

**Deadstep definition:**

* `deadsteps` counts consecutive **BS steps** without a trade.
* Reset to 0 whenever a trade occurs.

3. **Inventory exhausted (per agent):**

   * For an individual agent, as soon as their traded units reach `ntokens`, set a flag (`nobuysell = 1`).
   * That agent may not submit further bids or asks in the current Period.
   * Period itself may continue for others as long as at least one buyer and one seller still have tokens.

**Global exhaustion condition:**

* If **no buyer and no seller** has remaining tokens (all have `nobuysell = 1`), the Period ends immediately, even if `ntimes` or `deadsteps` are not reached.

At the end of the Period:

* No token carry-over to the next Period. Everyone’s inventory is reset to `ntokens`.
* Token values are the same as earlier Periods within that Round.
* The board should be cleared.

---

## 7. Profit calculation and ranking

For each executed trade of price (P):

* **Buyer**:

  * Uses their current token’s value (Val_{jk}).
  * Profit contribution:
    [
    \pi^{buyer} = Val_{jk} - P
    ]

* **Seller**:

  * Uses their current token’s cost (Cost_{jk}).
  * Profit contribution:
    [
    \pi^{seller} = P - Cost_{jk}
    ]

Each agent’s total profit is the sum over all trades across:

* All Periods,
* All Rounds,
* All Games,
* All Environments in the Tournament.

Tournament ranking:

* Order agents by total profit (highest → lowest).

**Conversion factor (optional):**

* For tournament realism: `money_profit = conversion_factor[env] * token_profit`
* Only needed if reproducing exact dollar payouts; not required for strategy comparison.

---

## 8. Implementation-oriented skeleton

You can think of the core loop as:

```pseudo
for env in environments:
  config = env.config
  for game in 1..N_games:
    for round in 1..nrounds:
      generate_tokens_for_all_agents(round, config)
      for period in 1..nperiods:
        reset_inventory_and_token_index()
        clear_board()
        deadstep_count = 0

        for step in 1..ntimes:
          if step % 2 == 1:  # BA step
            collect_BA_messages_from_active_traders()
            filter_illegal_messages()  # reject non-integer, out-of-range, non-improving
            update_board_with_best_bid_and_ask()
            broadcast_board_to_traders()  # explicit broadcast
          else:  # BS step
            trade_happened = maybe_execute_trade_using_BS_rules()

            if trade_happened:
              apply_forced_token_consumption()
              update_profits()
              clear_board()
              broadcast_trade_and_cleared_board()  # explicit broadcast
              deadstep_count = 0
            else:
              deadstep_count += 1

            if deadstep_count >= deadsteps:
              break  # end period early

          if all_traders_out_of_tokens():
            break  # global exhaustion
```

This structure plus your checklist is enough to build a faithful simulator. If you want, I can next help you formalize the exact message format and state transitions (e.g., how to represent simultaneous BA messages and apply the tie-breaking).

---

## 9. Implementation notes (deviations from Java baseline)

This implementation intentionally deviates from the original Java codebase in two ways:

1. **nobuysell flag logic (improvement):**
   * Java: Uses OR logic—if either `CurrentBidder` or `CurrentAsker` is missing, ALL agents receive the `+2` flag.
   * Python: Uses precise per-side logic—buyers only blocked (`+2`) if no ask exists; sellers only blocked if no bid exists.
   * Rationale: More logically correct per AURORA semantics. Allows the non-blocked side to still participate when appropriate.

2. **Market crossing allowed:**
   * Java: Unclear if bids exceeding asks were rejected.
   * Python: Bids can exceed asks; trades happen in BS stage, not automatically on crossing.
   * Rationale: Prevents deadlock when spread=1 (e.g., bid=69, ask=70).

These deviations are documented in the codebase (`market.py` lines 434–437, 467–468 and `orderbook.py` lines 176–179).
