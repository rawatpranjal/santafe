"""
Prompt builder for LLM agents.

Converts numerical market state into natural language prompts
for LLM-based trading agents.
"""


class PromptBuilder:
    """
    Builds natural language prompts for LLM agents.

    Converts numerical state observations into concise text descriptions
    suitable for LLM consumption while keeping token count low.
    """

    @staticmethod
    def build_system_prompt(is_buyer: bool, use_cot: bool = True, stage: str = "bid_ask") -> str:
        """
        Build ultra-simple system prompt for GPT-3.5 compatibility.

        Args:
            is_buyer: True for buyers, False for sellers
            use_cot: Ignored (kept for API compatibility)
            stage: "bid_ask" or "buy_sell"

        Returns:
            Simple system prompt string
        """
        if stage == "bid_ask":
            if is_buyer:
                prompt = """Pick a number. Reply JSON only.

{"action": "bid", "price": N}
or
{"action": "pass"}

Constraints:
- N > best_bid
- N < your_value
- If no valid N exists, pass"""
            else:
                prompt = """Pick a number. Reply JSON only.

{"action": "ask", "price": N}
or
{"action": "pass"}

Constraints:
- N < best_ask
- N > your_cost
- If no valid N exists, pass"""
        else:  # buy_sell stage
            prompt = """Decide yes or no. Reply JSON only.

{"action": "accept"}
or
{"action": "pass"}

Say accept only if profit > 0."""

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
        recent_trades: list[int] | None = None,
    ) -> str:
        """
        Build ultra-simple bid/ask prompt - just the numbers.

        Args:
            is_buyer: Agent role
            valuation: Private valuation for next token
            All other args kept for API compatibility but mostly ignored.

        Returns:
            Simple bid/ask prompt with just essential numbers
        """
        # Only show what's relevant to avoid confusion:
        # Buyers need to beat best_bid, sellers need to beat best_ask
        if is_buyer:
            bid_str = str(best_bid) if best_bid > 0 else "0"
            return f"your_value={valuation}, best_bid={bid_str}"
        else:
            ask_str = str(best_ask) if best_ask > 0 else "999"
            return f"your_cost={valuation}, best_ask={ask_str}"

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
        is_low_asker: bool = False,
    ) -> str:
        """
        Build ultra-simple buy/sell prompt - just the profit calculation.

        Args:
            is_buyer: Agent role
            valuation: Private valuation for next token
            high_bid: Current high bid
            low_ask: Current low ask
            is_high_bidder: Whether this agent is high bidder
            is_low_asker: Whether this agent is low asker
            Other args kept for API compatibility.

        Returns:
            Simple buy/sell prompt
        """
        # Can agent accept?
        can_accept = is_high_bidder if is_buyer else is_low_asker

        if not can_accept:
            return "You cannot accept. Pass."

        # Calculate potential profit
        if is_buyer:
            trade_price = low_ask
            profit = valuation - trade_price
            value_label = "your_value"
        else:
            trade_price = high_bid
            profit = trade_price - valuation
            value_label = "your_cost"

        return f"{value_label}={valuation}, trade_price={trade_price}, profit={profit}"

    # =========================================================================
    # MINIMAL PROMPTS - Pure facts, no hand-holding, trust LLM intuition
    # =========================================================================

    @staticmethod
    def build_minimal_system_prompt(is_buyer: bool, stage: str = "bid_ask") -> str:
        """
        Build rich system prompt with rules and context.

        Args:
            is_buyer: True for buyers, False for sellers
            stage: "bid_ask" or "buy_sell" - determines available actions

        Returns:
            System prompt string with rules
        """
        if stage == "bid_ask":
            # BID/ASK stage with explicit rules
            if is_buyer:
                return """You are a BUYER in a Santa Fe double auction.

GOAL: Buy tokens below your private value to profit.

CONSTRAINTS (both must be satisfied):
- bid > best_bid (must beat current best)
- bid <= your_value (cannot exceed your value)

If best_bid >= your_value, no valid bid exists. PASS.

VALID EXAMPLES:
- value=70, best_bid=40 -> valid bids: 41-70. Bid 45.
- value=70, best_bid=70 -> no valid bid (must be >70 but <=70). Pass.
- value=50, best_bid=60 -> no valid bid (must be >60 but <=50). Pass.

RESPOND JSON:
{"reasoning": "...", "action": "bid", "price": N}
{"reasoning": "...", "action": "pass"}"""
            else:
                return """You are a SELLER in a Santa Fe double auction.

GOAL: Sell tokens above your private cost to profit.

CONSTRAINTS (both must be satisfied):
- ask < best_ask (must beat current best)
- ask >= your_cost (cannot go below your cost)

If best_ask <= your_cost, no valid ask exists. PASS.

VALID EXAMPLES:
- cost=50, best_ask=60 -> valid asks: 50-59. Ask 55.
- cost=50, best_ask=50 -> no valid ask (must be <50 but >=50). Pass.
- cost=60, best_ask=55 -> no valid ask (must be <55 but >=60). Pass.

RESPOND JSON:
{"reasoning": "...", "action": "ask", "price": N}
{"reasoning": "...", "action": "pass"}"""
        else:
            # BUY/SELL stage
            if is_buyer:
                return """You are a buyer in a Santa Fe double auction.

You are the high bidder. You can now accept the seller's asking price.

DECISION: Accept if the trade is profitable (price < your_value).

RESPONSE FORMAT (JSON only):
{"reasoning": "...", "action": "accept"}
{"reasoning": "...", "action": "pass"}"""
            else:
                return """You are a seller in a Santa Fe double auction.

You are the low asker. You can now accept the buyer's bid price.

DECISION: Accept if the trade is profitable (price > your_cost).

RESPONSE FORMAT (JSON only):
{"reasoning": "...", "action": "accept"}
{"reasoning": "...", "action": "pass"}"""

    @staticmethod
    def build_minimal_bid_ask_prompt(
        is_buyer: bool,
        valuation: int,
        tokens_remaining: int,
        tokens_total: int,
        time_step: int,
        max_time: int,
        best_bid: int,
        best_ask: int,
        recent_trades: list[int] | None = None,
        period_profit: int = 0,
    ) -> str:
        """
        Build rich context bid/ask prompt.

        Includes trade history, position, and current market state.
        The LLM applies rules from system prompt to decide action.

        Args:
            is_buyer: Agent role (used for value/cost label)
            valuation: Private valuation for next token
            tokens_remaining: Tokens left to trade
            tokens_total: Total tokens allocated
            time_step: Current time step
            max_time: Maximum time steps
            best_bid: Current best bid (0 if none)
            best_ask: Current best ask (0 if none)
            recent_trades: List of recent trade prices (optional)
            period_profit: Accumulated profit this period (optional)

        Returns:
            Rich context bid/ask prompt
        """
        value_label = "Your private value" if is_buyer else "Your private cost"
        bid_str = str(best_bid) if best_bid > 0 else "none"
        ask_str = str(best_ask) if best_ask > 0 else "none"

        # Build trade history section
        if recent_trades:
            trade_history = ", ".join(str(p) for p in recent_trades[-5:])
            history_section = f"Recent trade prices: {trade_history}"
        else:
            history_section = "No trades yet this period"

        # Build position section
        tokens_traded = tokens_total - tokens_remaining
        position_section = (
            f"Tokens traded: {tokens_traded}/{tokens_total}, Profit so far: {period_profit}"
        )

        return f"""=== CURRENT SITUATION ===
{value_label}: {valuation}
Current best bid: {bid_str}
Current best ask: {ask_str}
Time step: {time_step}/{max_time}

=== TRADE HISTORY ===
{history_section}

=== YOUR POSITION ===
{position_section}

What do you do?"""

    @staticmethod
    def build_minimal_buy_sell_prompt(
        is_buyer: bool, valuation: int, trade_price: int, can_accept: bool
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

    # =========================================================================
    # DEEP CONTEXT PROMPTS - Full history and market structure
    # =========================================================================

    @staticmethod
    def build_deep_context_system_prompt(
        is_buyer: bool,
        stage: str = "bid_ask",
        num_buyers: int = 1,
        num_sellers: int = 1,
        num_tokens: int = 3,
        max_time: int = 50,
    ) -> str:
        """
        Build deep context system prompt with full rules and market structure.

        Args:
            is_buyer: True for buyer, False for seller
            stage: "bid_ask" or "buy_sell"
            num_buyers: Number of buyers in market
            num_sellers: Number of sellers in market
            num_tokens: Tokens per trader
            max_time: Max time steps per period

        Returns:
            Deep context system prompt
        """
        role = "BUYER" if is_buyer else "SELLER"
        value_label = "value" if is_buyer else "cost"
        action_label = "bid" if is_buyer else "ask"
        constraint_label = "best_bid" if is_buyer else "best_ask"

        if stage == "bid_ask":
            if is_buyer:
                return f"""You are a {role} in a Santa Fe double auction.

=== MARKET STRUCTURE ===
{num_buyers} buyers vs {num_sellers} sellers
{max_time} steps per period, {num_tokens} tokens each

=== BIDDING RULES ===
To place a valid bid:
1. bid > best_bid (must beat the standing bid)
2. bid <= your_value (cannot exceed your value)

If BOTH constraints cannot be satisfied, you MUST pass.

EXAMPLES:
- value=70, best_bid=40 -> valid: 41-70. Bid 45.
- value=70, best_bid=70 -> IMPOSSIBLE. Pass.
- value=50, best_bid=60 -> IMPOSSIBLE. Pass.

=== RESPONSE FORMAT (JSON only) ===
{{"reasoning": "brief", "action": "bid", "price": N}}
{{"reasoning": "brief", "action": "pass"}}"""
            else:
                return f"""You are a {role} in a Santa Fe double auction.

=== MARKET STRUCTURE ===
{num_buyers} buyers vs {num_sellers} sellers
{max_time} steps per period, {num_tokens} tokens each

=== ASKING RULES ===
To place a valid ask:
1. ask < best_ask (must beat the standing ask)
2. ask >= your_cost (cannot go below your cost)

If BOTH constraints cannot be satisfied, you MUST pass.

EXAMPLES:
- cost=50, best_ask=60 -> valid: 50-59. Ask 55.
- cost=50, best_ask=50 -> IMPOSSIBLE. Pass.
- cost=60, best_ask=55 -> IMPOSSIBLE. Pass.

=== RESPONSE FORMAT (JSON only) ===
{{"reasoning": "brief", "action": "ask", "price": N}}
{{"reasoning": "brief", "action": "pass"}}"""
        else:
            # buy_sell stage
            if is_buyer:
                return f"""You are a {role} in a Santa Fe double auction.

You are the high bidder. You can accept the seller's asking price.

RULE: Accept if trade_price < your_value (profitable).

=== RESPONSE FORMAT (JSON only) ===
{{"reasoning": "brief", "action": "accept"}}
{{"reasoning": "brief", "action": "pass"}}"""
            else:
                return f"""You are a {role} in a Santa Fe double auction.

You are the low asker. You can accept the buyer's bid.

RULE: Accept if trade_price > your_cost (profitable).

=== RESPONSE FORMAT (JSON only) ===
{{"reasoning": "brief", "action": "accept"}}
{{"reasoning": "brief", "action": "pass"}}"""

    @staticmethod
    def build_deep_context_bid_ask_prompt(
        is_buyer: bool,
        valuation: int,
        tokens_remaining: int,
        tokens_total: int,
        time_step: int,
        max_time: int,
        best_bid: int,
        best_ask: int,
        order_book_history: list[tuple[int, int, int]],
        trade_history: list[tuple[int, int]],
        period_profit: int,
    ) -> str:
        """
        Build deep context user prompt with full history.

        Args:
            is_buyer: Agent role
            valuation: Private valuation for next token
            tokens_remaining: Tokens left to trade
            tokens_total: Total tokens
            time_step: Current time step
            max_time: Max time steps
            best_bid: Current best bid
            best_ask: Current best ask
            order_book_history: List of (time, bid, ask) tuples
            trade_history: List of (time, price) tuples
            period_profit: Accumulated profit

        Returns:
            Deep context user prompt
        """
        value_label = "Your value" if is_buyer else "Your cost"
        bid_str = str(best_bid) if best_bid > 0 else "none"
        ask_str = str(best_ask) if best_ask > 0 else "none"

        # Build order book history section
        if order_book_history:
            history_lines = []
            for t, b, a in order_book_history[-5:]:
                b_str = str(b) if b > 0 else "-"
                a_str = str(a) if a > 0 else "-"
                history_lines.append(f"  Step {t}: bid={b_str}, ask={a_str}")
            ob_section = "\n".join(history_lines)
        else:
            ob_section = "  (no history yet)"

        # Build trade history section
        if trade_history:
            trade_lines = []
            for i, (t, p) in enumerate(trade_history, 1):
                trade_lines.append(f"  Trade {i} (step {t}): price {p}")
            trade_section = "\n".join(trade_lines)
        else:
            trade_section = "  (no trades yet)"

        tokens_traded = tokens_total - tokens_remaining

        return f"""=== CURRENT STATE ===
Time: step {time_step}/{max_time}
{value_label}: {valuation}
Best bid: {bid_str}
Best ask: {ask_str}

=== ORDER BOOK HISTORY ===
{ob_section}

=== TRADE HISTORY ===
{trade_section}

=== YOUR POSITION ===
Tokens: {tokens_traded}/{tokens_total}
Profit: {period_profit}

What do you do?"""

    @staticmethod
    def build_deep_context_buy_sell_prompt(
        is_buyer: bool, valuation: int, trade_price: int, can_accept: bool, period_profit: int = 0
    ) -> str:
        """
        Build deep context buy/sell prompt.

        Args:
            is_buyer: Agent role
            valuation: Private valuation
            trade_price: Price if trade accepted
            can_accept: Whether agent can accept
            period_profit: Accumulated profit

        Returns:
            Deep context buy/sell prompt
        """
        if not can_accept:
            return """You cannot accept (not your turn).
{"reasoning": "cannot accept", "action": "pass"}"""

        value_label = "Your value" if is_buyer else "Your cost"
        if is_buyer:
            profit = valuation - trade_price
        else:
            profit = trade_price - valuation

        return f"""{value_label}: {valuation}
Trade price: {trade_price}
Profit if accept: {profit}
Current profit: {period_profit}

Accept or pass?"""
