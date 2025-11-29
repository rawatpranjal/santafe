"""
Prompt builder for LLM agents.

Converts numerical market state into natural language prompts
for LLM-based trading agents.
"""

from typing import Optional


class PromptBuilder:
    """
    Builds natural language prompts for LLM agents.

    Converts numerical state observations into concise text descriptions
    suitable for LLM consumption while keeping token count low.
    """

    @staticmethod
    def build_system_prompt(is_buyer: bool, use_cot: bool = True, stage: str = "bid_ask") -> str:
        """
        Build the system prompt defining agent role and constraints.

        IMPORTANT: Prompts are COMPLETELY SEPARATE for each stage to avoid confusion.
        - bid_ask stage: NEVER mentions "accept"
        - buy_sell stage: NEVER mentions "bid" or "ask"

        Args:
            is_buyer: True for buyers, False for sellers
            use_cot: Enable chain-of-thought reasoning (default True)
            stage: "bid_ask" or "buy_sell"

        Returns:
            System prompt string
        """
        role = "BUYER" if is_buyer else "SELLER"
        action_type = "bid" if is_buyer else "ask"

        if stage == "bid_ask":
            # BID/ASK STAGE - Focus ONLY on price submission
            # NEVER mention "accept", "trade", or anything about accepting
            if is_buyer:
                prompt = f"""You are a {role} in an auction. Your task: submit a BID price.

GOAL: Submit a competitive bid to become the highest bidder.

RULES (you MUST follow these):
1. Your bid must be LESS than your valuation (otherwise you lose money)
2. Your bid must be HIGHER than the current best bid (to beat the competition)
3. If no valid bid is possible, you must PASS

THINK STEP BY STEP before deciding:
1. What is my valuation? (maximum I can bid)
2. What is the current best bid? (I must beat this)
3. Valid range = (best_bid + 1) to (valuation - 1)
4. If this range is empty or invalid, I MUST pass.
5. Otherwise, pick a price in that range.

Example: valuation=100, best bid=80 → valid range is 81-99 → bid something like 85.
Example: valuation=50, best bid=60 → NO valid range → MUST pass.

Reason it out fully, then act.

OUTPUT FORMAT:
{{"reasoning": "your step-by-step analysis", "action": "bid", "price": <integer>}}
OR
{{"reasoning": "why passing is necessary", "action": "pass"}}

CRITICAL: Only "bid" or "pass" are valid. Nothing else."""
            else:
                prompt = f"""You are a {role} in an auction. Your task: submit an ASK price.

GOAL: Submit a competitive ask to become the lowest seller.

RULES (you MUST follow these):
1. Your ask must be GREATER than your cost (otherwise you lose money)
2. Your ask must be LOWER than the current best ask (to beat the competition)
3. If no valid ask is possible, you must PASS

THINK STEP BY STEP before deciding:
1. What is my cost? (minimum I can ask)
2. What is the current best ask? (I must beat this)
3. Valid range = (cost + 1) to (best_ask - 1)
4. If this range is empty or invalid, I MUST pass.
5. Otherwise, pick a price in that range.

Example: cost=50, best ask=80 → valid range is 51-79 → ask something like 70.
Example: cost=70, best ask=60 → NO valid range → MUST pass.

Reason it out fully, then act.

OUTPUT FORMAT:
{{"reasoning": "your step-by-step analysis", "action": "ask", "price": <integer>}}
OR
{{"reasoning": "why passing is necessary", "action": "pass"}}

CRITICAL: Only "ask" or "pass" are valid. Nothing else."""

        else:  # buy_sell stage
            # BUY/SELL STAGE - Focus ONLY on accepting/passing
            # NEVER mention "bid", "ask", or price submission
            if is_buyer:
                prompt = f"""You are a {role}. A purchase opportunity is available.

SITUATION: You can buy now at a specific price.
DECISION: Accept the purchase or pass.

If you ACCEPT: You complete the purchase at the offered price.
If you PASS: You wait for a better opportunity (but time is limited).

Profit calculation: Your valuation minus the purchase price.
Only accept if profit > 0.

OUTPUT FORMAT:
{{"reasoning": "1-2 sentences", "action": "accept"}}
OR
{{"reasoning": "1-2 sentences", "action": "pass"}}

CRITICAL: Only use "accept" or "pass". No other actions exist."""
            else:
                prompt = f"""You are a {role}. A sale opportunity is available.

SITUATION: You can sell now at a specific price.
DECISION: Accept the sale or pass.

If you ACCEPT: You complete the sale at the offered price.
If you PASS: You wait for a better opportunity (but time is limited).

Profit calculation: The sale price minus your cost.
Only accept if profit > 0.

OUTPUT FORMAT:
{{"reasoning": "1-2 sentences", "action": "accept"}}
OR
{{"reasoning": "1-2 sentences", "action": "pass"}}

CRITICAL: Only use "accept" or "pass". No other actions exist."""

        return prompt

    @staticmethod
    def build_bid_ask_prompt(
        is_buyer: bool,
        valuation: int,
        tokens_remaining: int,
        tokens_total: int,
        time_step: int,
        max_time: int,
        best_bid: int,
        best_ask: int,
        spread: int,
        price_min: int,
        price_max: int,
        recent_trades: Optional[list[int]] = None
    ) -> str:
        """
        Build prompt for bid/ask stage decision.

        Args:
            is_buyer: Agent role
            valuation: Private valuation for next token
            tokens_remaining: Tokens left to trade
            tokens_total: Total tokens allocated
            time_step: Current time step
            max_time: Maximum time steps
            best_bid: Current best bid (0 if none)
            best_ask: Current best ask (0 if none)
            spread: Bid-ask spread
            price_min: Market minimum price
            price_max: Market maximum price
            recent_trades: Recent transaction prices

        Returns:
            Bid/ask stage prompt
        """
        role = "BUYER" if is_buyer else "SELLER"
        action_type = "BID" if is_buyer else "ASK"
        valuation_label = "Valuation" if is_buyer else "Cost"

        # Time urgency
        time_pct = (time_step / max_time) * 100
        urgency = "LOW" if time_pct < 33 else "MEDIUM" if time_pct < 66 else "HIGH"

        # Market state
        bid_str = f"{best_bid}" if best_bid > 0 else "None"
        ask_str = f"{best_ask}" if best_ask > 0 else "None"
        spread_str = f"{spread}" if spread > 0 else "No spread (no bid/ask)"

        # Compute valid price range and show explicitly
        if is_buyer:
            min_valid = best_bid + 1 if best_bid > 0 else price_min
            max_valid = valuation - 1  # Must be below valuation to profit
            if min_valid <= max_valid:
                valid_range = f"\n\n*** VALID BID RANGE: {min_valid} to {max_valid} ***"
                constraint_reminder = f"\nYour bid must be in range [{min_valid}, {max_valid}] to be valid."
            else:
                valid_range = f"\n\n*** NO VALID BID POSSIBLE - you must PASS ***"
                constraint_reminder = f"\nCannot bid: need > {best_bid} but < {valuation}. You must pass."
        else:  # seller
            max_valid = best_ask - 1 if best_ask > 0 else price_max
            min_valid = valuation + 1  # Must be above cost to profit
            if min_valid <= max_valid:
                valid_range = f"\n\n*** VALID ASK RANGE: {min_valid} to {max_valid} ***"
                constraint_reminder = f"\nYour ask must be in range [{min_valid}, {max_valid}] to be valid."
            else:
                valid_range = f"\n\n*** NO VALID ASK POSSIBLE - you must PASS ***"
                constraint_reminder = f"\nCannot ask: need < {best_ask} but > {valuation}. You must pass."

        # Recent activity
        trades_str = ""
        if recent_trades and len(recent_trades) > 0:
            trades_str = f"\nRecent Trades (last 5): {', '.join(str(p) for p in recent_trades[-5:])}"

        prompt = f"""
SITUATION:
You are a {role}. Time to submit a {action_type}.

PRIVATE INFO:
- {valuation_label} for next unit: {valuation}
- Tokens remaining: {tokens_remaining}/{tokens_total}

MARKET STATE:
- Time: Step {time_step}/{max_time} (Urgency: {urgency})
- Best Bid: {bid_str}
- Best Ask: {ask_str}
- Spread: {spread_str}
- Price Range: [{price_min}, {price_max}]{trades_str}
{valid_range}{constraint_reminder}

DECISION:
What {action_type.lower()} will you submit (or pass)?"""

        return prompt

    @staticmethod
    def build_buy_sell_prompt(
        is_buyer: bool,
        valuation: int,
        tokens_remaining: int,
        time_step: int,
        max_time: int,
        high_bid: int,
        low_ask: int,
        is_high_bidder: bool = False,
        is_low_asker: bool = False
    ) -> str:
        """
        Build prompt for buy/sell stage decision.

        Args:
            is_buyer: Agent role
            valuation: Private valuation for next token
            tokens_remaining: Tokens left to trade
            time_step: Current time step
            max_time: Maximum time steps
            high_bid: Current high bid
            low_ask: Current low ask
            is_high_bidder: Whether this agent is high bidder
            is_low_asker: Whether this agent is low asker

        Returns:
            Buy/sell stage prompt
        """
        role = "BUYER" if is_buyer else "SELLER"

        # Can agent accept?
        can_accept = is_high_bidder if is_buyer else is_low_asker

        if not can_accept:
            return f"""
SITUATION:
You are a {role}. Trade opportunity exists but you cannot accept.

REASON:
{"You are not the high bidder" if is_buyer else "You are not the low asker"}

DECISION:
You must PASS (you cannot accept this trade)."""

        # Calculate potential profit
        if is_buyer:
            trade_price = low_ask
            profit = valuation - trade_price
        else:
            trade_price = high_bid
            profit = trade_price - valuation

        # Time pressure
        time_pct = (time_step / max_time) * 100
        urgency = "LOW" if time_pct < 33 else "MEDIUM" if time_pct < 66 else "HIGH"

        action = "BUY" if is_buyer else "SELL"
        price_label = f"ask of {low_ask}" if is_buyer else f"bid of {high_bid}"

        prompt = f"""
SITUATION:
You are a {role} and can accept the current {price_label}.

TRADE ANALYSIS:
- Your {"valuation" if is_buyer else "cost"}: {valuation}
- Trade price: {trade_price}
- Profit if accepted: {profit}
- Tokens remaining after: {tokens_remaining - 1}
- Time pressure: {urgency} (Step {time_step}/{max_time})

DECISION:
Do you {action} at {trade_price}?
- ACCEPT: Lock in profit of {profit}
- PASS: Wait for better price (risk: time running out)"""

        return prompt

    # =========================================================================
    # MINIMAL PROMPTS - Pure facts, no hand-holding, trust LLM intuition
    # =========================================================================

    @staticmethod
    def build_minimal_system_prompt(is_buyer: bool) -> str:
        """
        Build pure minimal system prompt (~60 words).

        No rules, no constraints, no formulas. Just context and goal.
        Let the LLM figure out how to profit through intuition.

        Args:
            is_buyer: True for buyers, False for sellers

        Returns:
            Minimal system prompt string
        """
        if is_buyer:
            return """You are a trader buying goods in an auction.

You have a private value for each item - the maximum you'd pay without losing money.
Your goal is to profit. Profit = what it's worth to you minus what you pay.

How it works: You bid to become the "high bidder". A seller can accept your bid at any time.
Bidding doesn't commit you to buy - it just establishes your position.

Respond with JSON only:
- To place a bid: {"reasoning": "brief thought", "action": "bid", "price": <int>}
- To accept a trade: {"reasoning": "brief thought", "action": "accept"}
- To pass: {"reasoning": "brief thought", "action": "pass"}"""
        else:
            return """You are a trader selling goods in an auction.

You have a private cost for each item - the minimum you need to break even.
Your goal is to profit. Profit = what you receive minus your cost.

How it works: You ask to become the "low asker". A buyer can accept your ask at any time.
Asking doesn't commit you to sell - it just establishes your position.

Respond with JSON only:
- To place an ask: {"reasoning": "brief thought", "action": "ask", "price": <int>}
- To accept a trade: {"reasoning": "brief thought", "action": "accept"}
- To pass: {"reasoning": "brief thought", "action": "pass"}"""

    @staticmethod
    def build_minimal_bid_ask_prompt(
        is_buyer: bool,
        valuation: int,
        tokens_remaining: int,
        tokens_total: int,
        time_step: int,
        max_time: int,
        best_bid: int,
        best_ask: int
    ) -> str:
        """
        Build pure facts bid/ask prompt (~40 words).

        No hints, no valid ranges, no warnings. Just facts.
        The LLM must figure out what's profitable on its own.

        Args:
            is_buyer: Agent role (used for value/cost label)
            valuation: Private valuation for next token
            tokens_remaining: Tokens left to trade
            tokens_total: Total tokens allocated
            time_step: Current time step
            max_time: Maximum time steps
            best_bid: Current best bid (0 if none)
            best_ask: Current best ask (0 if none)

        Returns:
            Minimal bid/ask prompt
        """
        value_label = "Your value" if is_buyer else "Your cost"
        bid_str = str(best_bid) if best_bid > 0 else "none"
        ask_str = str(best_ask) if best_ask > 0 else "none"

        return f"""{value_label}: {valuation}
Best bid: {bid_str}
Best ask: {ask_str}
Time: {time_step}/{max_time}
Tokens left: {tokens_remaining}/{tokens_total}

What do you do?"""

    @staticmethod
    def build_minimal_buy_sell_prompt(
        is_buyer: bool,
        valuation: int,
        trade_price: int,
        can_accept: bool
    ) -> str:
        """
        Build minimal buy/sell stage prompt.

        Args:
            is_buyer: Agent role
            valuation: Private valuation
            trade_price: Price if trade accepted
            can_accept: Whether agent can accept this trade

        Returns:
            Minimal buy/sell prompt
        """
        if not can_accept:
            return """You cannot accept this trade (not your turn).
{"reasoning": "cannot accept", "action": "pass"}"""

        value_label = "Your value" if is_buyer else "Your cost"
        if is_buyer:
            profit = valuation - trade_price
        else:
            profit = trade_price - valuation

        return f"""{value_label}: {valuation}
Trade price: {trade_price}
Profit if you accept: {profit}

Accept or pass?"""

    @staticmethod
    def format_minimal_error_feedback(error_msg: str) -> str:
        """
        Format minimal error feedback - direct and simple.

        Args:
            error_msg: The validation error message

        Returns:
            Simple error feedback
        """
        return f"That didn't work: {error_msg}\nTry again."

    @staticmethod
    def format_error_feedback(error_msg: str, attempt: int) -> str:
        """
        Format validation error feedback for retry.

        Args:
            error_msg: The validation error message
            attempt: Current retry attempt number

        Returns:
            Error feedback string
        """
        return f"""
ERROR (Attempt {attempt}/3):
{error_msg}

Please try again with a valid action that respects all constraints."""
