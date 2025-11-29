# Official Glossary

Strict definitions for variable names, documentation, and technical discussions.

---

## 1. The Hierarchy of Time

The temporal structure of the simulation, derived from the Tournament Manual.

- **Step**: The atomic unit of time. Consists of exactly one Bid-Offer Phase followed by one Buy-Sell Phase.
- **Period**: A "trading day."
  - *State:* Inventory is replenished (e.g., 4 tokens). Valuations are fixed.
  - *Duration:* Typically 25-100 Steps.
  - *End:* Inventory expires (worthless).
- **Round**: A sequence of Periods using the same market configuration (same traders, same token distribution parameters) but different random seeds for valuations.
- **Game**: The full experimental unit. A collection of Rounds.
- **Epoch (RL)**: One full pass through the training data or a set number of episodes used for a PPO update.

---

## 2. AURORA Protocol Mechanics

The physics of the market engine.

- **Bid-Offer Phase (BO)**: The "Shout" phase.
  - *Action:* Agents submit a price or Pass.
  - *Constraint:* New York Rule (Must improve the market). Buyer must bid > Current Bid; Seller must ask < Current Ask.
  - *Crossed Market:* Invalid. A bid ≥ Ask is rejected in this phase (crucial for Kaplan logic).
- **Buy-Sell Phase (BS)**: The "Execution" phase.
  - *Participants:* Only the Holder (current Best Bidder) and Holder (current Best Offerer).
  - *Action:* Binary (Accept/Reject).
  - *Clearing:* If trade occurs, Book resets. If no trade, Book Persists (prices remain standing).
- **The Book**: The state of the CurrentBid and CurrentAsk.
- **Tokens**: The abstract commodity.
  - *Intra-marginal:* Tokens that should trade in equilibrium (Value > Equilibrium Price).
  - *Extra-marginal:* Tokens that should not trade (Value < Equilibrium Price).

---

## 3. Agent Taxonomy

The "Cast of Characters."

- **ZI (Zero Intelligence)**: Unconstrained random bidder. Can buy above value/sell below cost. Used only as a control to prove the value of constraints.
- **ZIC (Zero Intelligence Constrained)**: Random bidder with budget constraints. Cannot lose money. The baseline for "minimal intelligence."
- **Skeleton**: The "Rational Baseline." Uses a simple fixed profit margin. The reference implementation provided to 1993 contestants.
- **Kaplan**: The "Sniper." Waits for the spread to narrow, then jumps in to steal the deal. *Parasitic:* Relies on others to make price discovery.
- **GD (Gjerstad-Dickhaut)**: The "Belief Learner." Calculates probability of acceptance based on history to maximize expected surplus.
- **ZIP (Zero Intelligence Plus)**: The "Adaptive Learner." Adjusts a profit margin up/down based on market feedback (Widrow-Hoff rule).

---

## 5. Experimental Protocols

The specific setups for testing.

- **Grand Melee**: A robustness stress-test where agents face 10 different scenarios (Scarcity, Monopoly, Panic Time, etc.).
- **Invasibility Test**: A specific configuration: 1 Subject Agent vs. 7 Clones of Opponent. Tests if a strategy can "invade" a stable population.
- **The Kaplan Paradox**: The phenomenon where a strategy is individually dominant (wins the tournament) but collectively disastrous (crashes the market if everyone uses it).
- **Curriculum**: The training sequence for RL: Empty Market → ZIC Opponents → Skeleton Opponents → Mixed Sharks.

---

## 6. RL & Implementation Terms

Code-level culture.

- **Game 6453**: The canonical seed/configuration for token generation used in the original paper. We reference this to mean "Standard Setup."
- **Strict Integer Math**: All prices are int. No floats allowed in the order book.
- **Observation Space**: The input vector. MUST be anonymous (no opponent IDs).
- **Valid Action Mask**: The filter that prevents the RL agent from choosing illegal moves (selling empty inventory), ensuring fast learning.
