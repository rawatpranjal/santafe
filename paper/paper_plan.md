# Research Paper Plan: Santa Fe Redux (v2.0)

**Title:** *The Invisible Hand vs. The Gradient: Revisiting the Santa Fe Double Auction Tournament with Deep RL and LLMs*

**Target Venue:** arXiv (Quantitative Finance / AI), NeurIPS AI for Agent-Based Models Workshop.

---

## 1. Abstract
*   **Hook:** In 1993, the Santa Fe Institute hosted a tournament where simple "sniping" heuristics beat complex AI.
*   **Gap:** 30 years later, can modern Deep RL (PPO) and Generative AI (LLMs) solve the "Hayek Problem" of information aggregation without the need for hard-coded rules?
*   **Method:** We faithfully replicate the 1993 Synchronized Double Auction environment (Rust et al., 1994) and introduce PPO and GPT-4o agents.
*   **Results (Targeted):** 
    1.  **H1 (The Sniper):** PPO agents successfully learn to exploit legacy heuristics, rediscovering the "Kaplan" strategy from scratch.
    2.  **H2 (The Neural Market):** Multi-agent PPO markets maintain high efficiency, avoiding the "market crash" observed in heuristic self-play.
    3.  **H3 (The Semantic Trader):** Zero-shot LLMs exhibit high allocative efficiency but display distinct behavioral biases (fairness over profit).
    4.  **H4 (The Intelligence Gap):** A measurable wealth transfer exists from GPT-3.5 to GPT-4o.

## 2. Introduction
*   **The Problem:** How do decentralized markets reach equilibrium? (Hayek, Smith).
*   **The History:**
    *   Smith (1962): Humans do it efficiently.
    *   Gode & Sunder (1993): Random agents do it (structure matters).
    *   Rust et al. (1993): Simple rules beat complex rules (The Tournament).
*   **The New Question:** Do gradient-based learners (RL) and semantic learners (LLMs) fundamentally change these conclusions?

## 3. Literature Review (A Guided Tour)
*   **Experimental Economics:** Smith (1962), Plott (1982).
*   **The Santa Fe Tournament:** Rust, Miller, Palmer (1993, 1994). The dominance of "Kaplan".
*   **Algorithmic Trading:** Cliff (1997) ZIP, Gjerstad & Dickhaut (1998) GD.
*   **Modern MARL:** Vinyals et al. (2019) AlphaStar (for complexity context), Balduzzi et al. (2019) Open-Ended Learning.

## 4. Methodology (The Time Capsule)
*   **The Mechanism:** Exact specification of the Synchronized DA (AURORA rules).
    *   *Visual:* Timeline of a Trading Period (Bid/Ask Step -> Buy/Sell Step).
*   **The Agents:**
    *   **Legacy Zoo:** ZI-C (Constraint), Kaplan (Sniper).
    *   **Deep RL:** PPO with LSTM, discrete relative action space, manual feature engineering.
    *   **LLM:** GPT-4o with order book context and trade history.

## 5. Results: The Rise of Reinforcement Learning
*   **Single-Agent Invasion:** PPO vs. Kaplan/ZI-C.
    *   *Fig 3:* Profit curve over training episodes.
    *   *Fig 4:* "Bidding Heatmap" (Time vs. Bid Distance). Proof of Sniping.
*   **The Arms Race (Self-Play):** 8 PPO agents.
    *   *Table 1:* The Efficiency Matrix (3x3).

## 6. Results: The Semantic Trader (LLMs)
*   **Zero-Shot Performance:** GPT-4o vs. Legacy Agents.
*   **Behavioral Analysis:** Does GPT-4o "panic" or "hold"?
*   **Intelligence Gap:** GPT-4o vs. GPT-3.5.

## 7. Discussion & Conclusion
*   **Structure vs. Intelligence:** Re-evaluating Gode & Sunder.
*   **Algorithmic Collusion:** Implications for AI finance.

---

## Visuals Checklist
1.  **Fig 1:** Timeline of the Synchronized Double Auction Step.
2.  **Fig 2:** The "Money Table" (Efficiency Matrix).
3.  **Fig 3:** Training Curves (PPO Profit vs. Legacy Agents).
4.  **Fig 4:** "Bidding Heatmap" (Time in Period vs. Bid Distance from Equilibrium).