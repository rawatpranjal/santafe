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

    # =========================================================================
    # DASHBOARD PROMPTS - Rich visual display with all market information
    # =========================================================================

    @staticmethod
    def build_dashboard_system_prompt(
        is_buyer: bool,
        stage: str = "bid_ask",
        num_buyers: int = 4,
        num_sellers: int = 4,
        num_tokens: int = 4,
        max_time: int = 100,
    ) -> str:
        """
        Build dashboard-style system prompt with reasoning instructions.

        Args:
            is_buyer: True for buyer, False for seller
            stage: "bid_ask" or "buy_sell"
            num_buyers: Number of buyers
            num_sellers: Number of sellers
            num_tokens: Tokens per trader
            max_time: Max time steps

        Returns:
            Dashboard system prompt
        """
        role = "BUYER" if is_buyer else "SELLER"

        if stage == "bid_ask":
            if is_buyer:
                return f"""You are a {role} in a Santa Fe double auction market.

MARKET RULES:
- {num_buyers} buyers compete against {num_sellers} sellers
- Each trader has {num_tokens} tokens to trade over {max_time} steps
- Your goal: maximize profit by buying tokens BELOW your private value

HOW BIDDING WORKS:
- This is a two-stage process: (1) BID/ASK stage, then (2) BUY/SELL stage
- In THIS stage, you submit a BID to become the HIGH BIDDER
- Your bid competes against OTHER BUYERS, not against sellers
- To become high bidder, you must bid HIGHER than the current best bid

CRITICAL CONSTRAINT:
>>> Your bid MUST be > best_bid (to beat the current high bidder) <<<
>>> Your bid MUST be <= your_value (to ensure profit) <<<

COMMON MISTAKE: Do NOT bid at or near the ask price! The ask is what sellers want.
You must beat the CURRENT BID from other buyers, not match the ask.

If best_bid >= your_value, you cannot profitably outbid. You MUST pass.

STRATEGY:
- Bid just above best_bid to become high bidder cheaply
- Later (in buy/sell stage), you can accept the ask if profitable

RESPONSE FORMAT (JSON only):
{{"reasoning": "your analysis here", "action": "bid", "price": N}}
{{"reasoning": "your analysis here", "action": "pass"}}"""
            else:
                return f"""You are a {role} in a Santa Fe double auction market.

MARKET RULES:
- {num_buyers} buyers compete against {num_sellers} sellers
- Each trader has {num_tokens} tokens to trade over {max_time} steps
- Your goal: maximize profit by selling tokens ABOVE your private cost

HOW ASKING WORKS:
- This is a two-stage process: (1) BID/ASK stage, then (2) BUY/SELL stage
- In THIS stage, you submit an ASK to become the LOW ASKER
- Your ask competes against OTHER SELLERS, not against buyers
- To become low asker, you must ask LOWER than the current best ask

CRITICAL CONSTRAINT:
>>> Your ask MUST be < best_ask (to beat the current low asker) <<<
>>> Your ask MUST be >= your_cost (to ensure profit) <<<

COMMON MISTAKE: Do NOT ask at or near the bid price! The bid is what buyers offer.
You must beat the CURRENT ASK from other sellers, not match the bid.

If best_ask <= your_cost, you cannot profitably undercut. You MUST pass.

STRATEGY:
- Ask just below best_ask to become low asker profitably
- Later (in buy/sell stage), you can accept the bid if profitable

RESPONSE FORMAT (JSON only):
{{"reasoning": "your analysis here", "action": "ask", "price": N}}
{{"reasoning": "your analysis here", "action": "pass"}}"""
        else:  # buy_sell stage
            if is_buyer:
                return """You are a BUYER who placed the highest bid.

You can now ACCEPT the seller's asking price to complete a trade.

RULE: Accept ONLY if the trade is profitable (price < your_value).

RESPONSE FORMAT (JSON only):
{"reasoning": "your analysis", "action": "accept"}
{"reasoning": "your analysis", "action": "pass"}"""
            else:
                return """You are a SELLER who placed the lowest ask.

You can now ACCEPT the buyer's bid price to complete a trade.

RULE: Accept ONLY if the trade is profitable (price > your_cost).

RESPONSE FORMAT (JSON only):
{"reasoning": "your analysis", "action": "accept"}
{"reasoning": "your analysis", "action": "pass"}"""

    @staticmethod
    def build_dashboard_bid_ask_prompt(
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
        remaining_values: list[int] | None = None,
        period: int = 1,
        num_periods: int = 10,
        num_buyers: int = 4,
        num_sellers: int = 4,
    ) -> str:
        """
        Build rich dashboard-style prompt with all market information.

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
            remaining_values: List of remaining token values
            period: Current period number
            num_periods: Total periods
            num_buyers: Number of buyers
            num_sellers: Number of sellers

        Returns:
            Dashboard-style prompt
        """
        role = "BUYER" if is_buyer else "SELLER"
        value_label = "value" if is_buyer else "cost"
        action_label = "bid" if is_buyer else "ask"

        tokens_traded = tokens_total - tokens_remaining
        time_remaining = max_time - time_step
        time_pct = int(time_remaining / max_time * 100)

        # Format best bid/ask
        bid_str = f"${best_bid}" if best_bid > 0 else "none"
        ask_str = f"${best_ask}" if best_ask > 0 else "none"

        # Calculate spread
        if best_bid > 0 and best_ask > 0:
            spread = best_ask - best_bid
            spread_pct = int(spread / ((best_bid + best_ask) / 2) * 100)
            spread_str = f"${spread} ({spread_pct}%)"
        else:
            spread_str = "N/A"

        # Calculate valid range
        if is_buyer:
            if best_bid > 0:
                valid_min = best_bid + 1
            else:
                valid_min = 1
            valid_max = valuation
            if valid_min <= valid_max:
                valid_range = f"${valid_min} - ${valid_max}"
            else:
                valid_range = "NONE (must pass)"
        else:
            if best_ask > 0:
                valid_max = best_ask - 1
            else:
                valid_max = 999
            valid_min = valuation
            if valid_min <= valid_max:
                valid_range = f"${valid_min} - ${valid_max}"
            else:
                valid_range = "NONE (must pass)"

        # Format remaining values
        if remaining_values:
            values_str = ", ".join(f"${v}" for v in remaining_values[:4])
        else:
            values_str = f"${valuation}"

        # Build order book history section
        ob_lines = []
        for t, b, a in order_book_history[-8:]:
            b_str = f"${b}" if b > 0 else "-"
            a_str = f"${a}" if a > 0 else "-"
            # Check if trade happened
            trade_at_t = [p for (tt, p) in trade_history if tt == t]
            if trade_at_t:
                ob_lines.append(
                    f"  Step {t:3d}: bid={b_str:>4}, ask={a_str:>4} -> TRADE @ ${trade_at_t[0]}"
                )
            else:
                ob_lines.append(f"  Step {t:3d}: bid={b_str:>4}, ask={a_str:>4}")
        ob_section = "\n".join(ob_lines) if ob_lines else "  (no history yet)"

        # Build trade history section
        trade_lines = []
        for i, (t, p) in enumerate(trade_history[-5:], 1):
            trade_lines.append(f"  Trade {i}: step {t}, price ${p}")
        trade_section = "\n".join(trade_lines) if trade_lines else "  (no trades yet)"

        # Calculate statistics
        num_trades = len(trade_history)
        if trade_history:
            prices = [p for _, p in trade_history]
            avg_price = sum(prices) / len(prices)
            if len(prices) > 1:
                import statistics

                volatility = statistics.stdev(prices) / avg_price * 100
                # Determine trend
                recent = prices[-3:] if len(prices) >= 3 else prices
                if len(recent) >= 2:
                    if recent[-1] > recent[0]:
                        trend = "rising"
                    elif recent[-1] < recent[0]:
                        trend = "falling"
                    else:
                        trend = "stable"
                else:
                    trend = "N/A"
            else:
                volatility = 0
                trend = "N/A"
            stats_section = f"""  Trades this period: {num_trades}
  Average price: ${avg_price:.0f}
  Price trend: {trend}
  Volatility: {volatility:.1f}%"""
        else:
            stats_section = "  Trades this period: 0\n  (no price data yet)"

        # Build the dashboard
        dashboard = f"""
================== TRADING DASHBOARD ==================
                  You are {role} #{1}
=======================================================

MARKET STRUCTURE
  {num_buyers} buyers vs {num_sellers} sellers
  {tokens_total} tokens each
  Period {period}/{num_periods}, Step {time_step}/{max_time}

YOUR POSITION
  Tokens traded: {tokens_traded}/{tokens_total}
  Current profit: ${period_profit}
  Your {value_label} for next token: ${valuation}
  Remaining token {value_label}s: [{values_str}]

CURRENT MARKET STATE
  Best bid: {bid_str}
  Best ask: {ask_str}
  Spread: {spread_str}
  Valid {action_label} range: {valid_range}

ORDER BOOK HISTORY (recent)
{ob_section}

TRADE HISTORY
{trade_section}

MARKET STATISTICS
{stats_section}
  Time remaining: {time_remaining} steps ({time_pct}%)

=======================================================
"""

        # Add constraint reminder
        if is_buyer:
            dashboard += f"""
CONSTRAINTS (you must satisfy BOTH):
1. Your bid MUST be > {bid_str} (beat current highest bid from other buyers)
2. Your bid MUST be <= ${valuation} (stay at or below your value)

>>> DO NOT bid at the ask price ({ask_str})! That's a seller's price. <<<
>>> You must outbid the current BUYER at {bid_str}. <<<

Valid bid range: {valid_range}
What is your decision?"""
        else:
            dashboard += f"""
CONSTRAINTS (you must satisfy BOTH):
1. Your ask MUST be < {ask_str} (beat current lowest ask from other sellers)
2. Your ask MUST be >= ${valuation} (stay at or above your cost)

>>> DO NOT ask at the bid price ({bid_str})! That's a buyer's price. <<<
>>> You must undercut the current SELLER at {ask_str}. <<<

Valid ask range: {valid_range}
What is your decision?"""

        return dashboard

    @staticmethod
    def build_dashboard_buy_sell_prompt(
        is_buyer: bool,
        valuation: int,
        trade_price: int,
        can_accept: bool,
        period_profit: int = 0,
    ) -> str:
        """
        Build dashboard-style buy/sell prompt.

        Args:
            is_buyer: Agent role
            valuation: Private valuation
            trade_price: Price if trade accepted
            can_accept: Whether agent can accept
            period_profit: Accumulated profit

        Returns:
            Dashboard buy/sell prompt
        """
        if not can_accept:
            return """You cannot accept (not your turn).
{"reasoning": "cannot accept", "action": "pass"}"""

        value_label = "value" if is_buyer else "cost"
        if is_buyer:
            profit = valuation - trade_price
            profitable = profit > 0
        else:
            profit = trade_price - valuation
            profitable = profit > 0

        return f"""
================== TRADE OPPORTUNITY ==================

Your {value_label}: ${valuation}
Trade price: ${trade_price}
Profit if you accept: ${profit}
Current period profit: ${period_profit}

{"PROFITABLE TRADE!" if profitable else "UNPROFITABLE - Consider passing"}

Accept or pass?
======================================================="""

    # =========================================================================
    # CONSTRAINTS-ONLY PROMPTS - No heuristics, only objective + hard rules
    # =========================================================================

    @staticmethod
    def build_constraints_system_prompt(is_buyer: bool, stage: str = "bid_ask") -> str:
        """
        Build constraints-only system prompt. NO strategy hints.

        Only: objective + hard constraints + output format.
        """
        role = "BUYER" if is_buyer else "SELLER"

        if stage == "bid_ask":
            if is_buyer:
                return """You are an automated BUYER in a Santa Fe double auction market.

ROLE
- You act as a BUYER.
- In each decision, you choose either:
  - to SUBMIT a BID, or
  - to PASS (submit no bid in this step).

HIGH-LEVEL OBJECTIVE
- Maximize your total profit over time, subject to the hard constraints below.
- Profit is generated only by buying tokens at prices strictly below your private value.

HARD CONSTRAINTS (MUST BE SATISFIED)
1. Bid feasibility:
   - If you choose to BID, the bid price must satisfy:
     - price > best_bid (if best_bid is a number)
       NOTE: ">" means HIGHER. To beat best_bid of 100, bid 101 or more.
     - price <= your_value

2. Infeasible / forced-pass conditions:
   - If must_pass: true, or valid_bid_range: NONE, then:
     - You MUST choose "action": "pass"
     - You MUST NOT output any "price" field

3. Consistency:
   - Treat all variables in STATE SUMMARY as authoritative
   - If must_pass: true, respect it over numeric values

OUTPUT FORMAT (STRICT)
Output exactly ONE JSON object:

If you BID:
{"reasoning": "<brief>", "action": "bid", "price": <integer>}

If you PASS:
{"reasoning": "<brief>", "action": "pass"}

No extra text before or after the JSON."""
            else:
                return """You are an automated SELLER in a Santa Fe double auction market.

ROLE
- You act as a SELLER.
- In each decision, you choose either:
  - to SUBMIT an ASK, or
  - to PASS (submit no ask in this step).

HIGH-LEVEL OBJECTIVE
- Maximize your total profit over time, subject to the hard constraints below.
- Profit is generated only by selling tokens at prices strictly above your private cost.

HARD CONSTRAINTS (MUST BE SATISFIED)
1. Ask feasibility:
   - If you choose to ASK, the ask price must satisfy:
     - price < best_ask (if best_ask is a number)
       NOTE: "<" means LOWER. To beat best_ask of 100, ask 99 or less.
     - price >= your_cost

2. Infeasible / forced-pass conditions:
   - If must_pass: true, or valid_ask_range: NONE, then:
     - You MUST choose "action": "pass"
     - You MUST NOT output any "price" field

3. Consistency:
   - Treat all variables in STATE SUMMARY as authoritative
   - If must_pass: true, respect it over numeric values

OUTPUT FORMAT (STRICT)
Output exactly ONE JSON object:

If you ASK:
{"reasoning": "<brief>", "action": "ask", "price": <integer>}

If you PASS:
{"reasoning": "<brief>", "action": "pass"}

No extra text before or after the JSON."""
        else:  # buy_sell stage
            if is_buyer:
                return """You are a BUYER in a Santa Fe double auction.

You can ACCEPT the current ask price to complete a trade.

OBJECTIVE: Maximize profit.
CONSTRAINT: Accept only if trade_price < your_value (profitable).

OUTPUT FORMAT (JSON only):
{"reasoning": "<brief>", "action": "accept"}
{"reasoning": "<brief>", "action": "pass"}"""
            else:
                return """You are a SELLER in a Santa Fe double auction.

You can ACCEPT the current bid price to complete a trade.

OBJECTIVE: Maximize profit.
CONSTRAINT: Accept only if trade_price > your_cost (profitable).

OUTPUT FORMAT (JSON only):
{"reasoning": "<brief>", "action": "accept"}
{"reasoning": "<brief>", "action": "pass"}"""

    @staticmethod
    def build_constraints_bid_ask_prompt(
        is_buyer: bool,
        valuation: int,
        tokens_remaining: int,
        tokens_total: int,
        time_step: int,
        max_time: int,
        best_bid: int,
        best_ask: int,
        period_profit: int = 0,
        period: int = 1,
    ) -> str:
        """
        Build constraints-only STATE SUMMARY prompt.

        Clean, machine-readable format with must_pass flag.
        """
        role = "buyer" if is_buyer else "seller"
        tokens_traded = tokens_total - tokens_remaining

        # Format values
        bid_val = best_bid if best_bid > 0 else "null"
        ask_val = best_ask if best_ask > 0 else "null"

        # Calculate valid range and must_pass
        if is_buyer:
            if best_bid > 0:
                min_bid = best_bid + 1
            else:
                min_bid = 1
            max_bid = valuation

            if min_bid <= max_bid:
                valid_range = f"[{min_bid}, {max_bid}]"
                must_pass = "false"
            else:
                valid_range = "NONE"
                must_pass = "true"
                min_bid = "null"
                max_bid = "null"
        else:
            if best_ask > 0:
                max_ask = best_ask - 1
            else:
                max_ask = 999
            min_ask = valuation

            if min_ask <= max_ask:
                valid_range = f"[{min_ask}, {max_ask}]"
                must_pass = "false"
            else:
                valid_range = "NONE"
                must_pass = "true"
                min_ask = "null"
                max_ask = "null"

        value_label = "your_value" if is_buyer else "your_cost"
        range_label = "valid_bid_range" if is_buyer else "valid_ask_range"

        if is_buyer:
            return f"""=== STATE SUMMARY ===
role: {role}
period: {period}
step: {time_step}

tokens_traded: {tokens_traded}
current_profit: {period_profit}

{value_label}: {valuation}
best_bid: {bid_val}
best_ask: {ask_val}

min_bid: {min_bid if must_pass == "false" else "null"}
max_bid: {max_bid if must_pass == "false" else "null"}
{range_label}: {valid_range}
must_pass: {must_pass}
====================="""
        else:
            return f"""=== STATE SUMMARY ===
role: {role}
period: {period}
step: {time_step}

tokens_traded: {tokens_traded}
current_profit: {period_profit}

{value_label}: {valuation}
best_bid: {bid_val}
best_ask: {ask_val}

min_ask: {min_ask if must_pass == "false" else "null"}
max_ask: {max_ask if must_pass == "false" else "null"}
{range_label}: {valid_range}
must_pass: {must_pass}
====================="""

    @staticmethod
    def build_constraints_buy_sell_prompt(
        is_buyer: bool,
        valuation: int,
        trade_price: int,
        can_accept: bool,
        period_profit: int = 0,
    ) -> str:
        """
        Build constraints-only buy/sell prompt.
        """
        if not can_accept:
            return """=== STATE SUMMARY ===
can_accept: false
must_pass: true
====================="""

        value_label = "your_value" if is_buyer else "your_cost"
        if is_buyer:
            profit = valuation - trade_price
            profitable = profit > 0
        else:
            profit = trade_price - valuation
            profitable = profit > 0

        return f"""=== STATE SUMMARY ===
{value_label}: {valuation}
trade_price: {trade_price}
profit_if_accept: {profit}
current_profit: {period_profit}
can_accept: true
profitable: {str(profitable).lower()}
====================="""

    # =========================================================================
    # DENSE PROMPT STYLE - Full context, no strategy hints
    # =========================================================================

    @staticmethod
    def build_dense_system_prompt(is_buyer: bool, stage: str = "bid_ask") -> str:
        """
        Build dense system prompt with full market context but NO strategy hints.
        Explains rules, price formation, and options - lets model reason.
        """
        role = "BUYER" if is_buyer else "SELLER"

        base = f"""You are a {role} in a Santa Fe double auction. Goal: maximize total profit over the round.

=== MARKET STRUCTURE ===
Hierarchy: Round → Periods → Steps
- Token values fixed for the entire round
- Inventory resets each period (fresh tokens)
- You only know YOUR values; others' values are hidden

=== TWO-STAGE TRADING ===

STAGE 1: BID/ASK
All traders submit prices simultaneously.
- Buyers submit BIDs; Sellers submit ASKs
- To become CurrentBidder: bid must be STRICTLY > CurrentBid
- To become CurrentAsker: ask must be STRICTLY < CurrentAsk
- Multiple agents at same improved price → random tie-break

STAGE 2: BUY/SELL
Only CurrentBidder and CurrentAsker can act. Others blocked.

=== PRICE FORMATION ===
| Who Accepts | Transaction Price |
|-------------|-------------------|
| Buyer (BUY) | CurrentAsk |
| Seller (SELL) | CurrentBid |
| Both | Random pick |
| Neither | No trade |

=== BOOK STATE ===
- After trade: BOTH bid and ask clear to NULL
- After no trade: Book PERSISTS (you keep your position)

=== PROFIT ==="""

        if is_buyer:
            base += """
Buyer profit = Token Value - Transaction Price
Tokens trade in order: 1st first, then 2nd, etc.

RISK: Your bid is a PRICE COMMITMENT. If you bid above your value and seller accepts, you LOSE money."""
        else:
            base += """
Seller profit = Transaction Price - Token Cost
Tokens trade in order: 1st first, then 2nd, etc.

RISK: Your ask is a PRICE COMMITMENT. If you ask below your cost and buyer accepts, you LOSE money."""

        base += """

=== OUTPUT FORMAT ===
Respond with exactly one JSON object. No other text."""

        return base

    @staticmethod
    def build_dense_bid_ask_prompt(
        is_buyer: bool,
        valuation: int,
        tokens_remaining: int,
        tokens_total: int,
        time_step: int,
        max_time: int,
        best_bid: int | None,
        best_ask: int | None,
        period: int = 1,
        period_profit: int = 0,
        trade_history: list[tuple[int, int]] | None = None,
    ) -> str:
        """
        Build dense bid/ask prompt with state and explicit options.
        """
        role = "BUYER" if is_buyer else "SELLER"
        value_label = "Your value" if is_buyer else "Your cost"

        # Format prices
        bid_str = f"${best_bid}" if best_bid and best_bid > 0 else "none"
        ask_str = f"${best_ask}" if best_ask and best_ask > 0 else "none"

        # Calculate valid range
        if is_buyer:
            min_valid = (best_bid + 1) if best_bid and best_bid > 0 else 1
            max_valid = valuation
            can_act = min_valid <= max_valid
            range_str = f"[${min_valid}, ${max_valid}]" if can_act else "NONE"
        else:
            max_valid = (best_ask - 1) if best_ask and best_ask > 0 else 999
            min_valid = valuation
            can_act = min_valid <= max_valid
            range_str = f"[${min_valid}, ${max_valid}]" if can_act else "NONE"

        # Trade history
        history_str = ""
        if trade_history and len(trade_history) > 0:
            recent = trade_history[-5:]  # Last 5 trades
            history_str = "Recent trades: " + ", ".join([f"${p}" for _, p in recent])
        else:
            history_str = "No trades yet this period"

        prompt = f"""=== CURRENT STATE ===
Stage: BID/ASK
Period: {period}, Step: {time_step}/{max_time}
{value_label} for next token: ${valuation}
Tokens: {tokens_remaining}/{tokens_total} remaining
Profit so far: ${period_profit}

CurrentBid: {bid_str}
CurrentAsk: {ask_str}
{history_str}

=== YOUR OPTIONS ==="""

        if is_buyer:
            if can_act:
                prompt += f"""
1. BID: Submit price in {range_str}
   - Must be > CurrentBid to become CurrentBidder
   - If seller later accepts your bid, you pay YOUR BID

2. PASS: Do nothing this step

Output format:
{{"reasoning": "...", "action": "bid", "price": <integer>}}
{{"reasoning": "...", "action": "pass"}}"""
            else:
                prompt += """
Only option: PASS (CurrentBid >= your value, cannot profitably outbid)

Output: {"reasoning": "...", "action": "pass"}"""
        else:  # seller
            if can_act:
                prompt += f"""
1. ASK: Submit price in {range_str}
   - Must be < CurrentAsk to become CurrentAsker
   - If buyer later accepts your ask, you receive YOUR ASK

2. PASS: Do nothing this step

Output format:
{{"reasoning": "...", "action": "ask", "price": <integer>}}
{{"reasoning": "...", "action": "pass"}}"""
            else:
                prompt += """
Only option: PASS (CurrentAsk <= your cost, cannot profitably undercut)

Output: {"reasoning": "...", "action": "pass"}"""

        return prompt

    @staticmethod
    def build_dense_buy_sell_prompt(
        is_buyer: bool,
        valuation: int,
        trade_price: int,
        my_standing_price: int | None,
        can_accept: bool,
        is_current_holder: bool = True,
        period_profit: int = 0,
    ) -> str:
        """
        Build dense buy/sell prompt with explicit options and price formation.

        Args:
            my_standing_price: The bid (for buyer) or ask (for seller) that you hold
        """
        role = "BUYER" if is_buyer else "SELLER"
        value_label = "Your value" if is_buyer else "Your cost"
        my_price_label = "Your standing bid" if is_buyer else "Your standing ask"
        opponent_price_label = "CurrentAsk" if is_buyer else "CurrentBid"

        if is_buyer:
            profit_if_accept = valuation - trade_price
            profit_if_opponent_accepts = valuation - (my_standing_price or 0)
        else:
            profit_if_accept = trade_price - valuation
            profit_if_opponent_accepts = (my_standing_price or 0) - valuation

        holder_role = "CurrentBidder" if is_buyer else "CurrentAsker"
        prompt = f"""=== CURRENT STATE ===
Stage: BUY/SELL
You are the {holder_role}
{value_label}: ${valuation}
{my_price_label}: ${my_standing_price}
{opponent_price_label}: ${trade_price}
Profit so far: ${period_profit}

=== YOUR OPTIONS ==="""

        if not is_current_holder:
            prompt += """
You are NOT the current holder. Only option: PASS

Output: {"reasoning": "...", "action": "pass"}"""
        elif can_accept:
            if is_buyer:
                prompt += f"""
1. BUY: Accept the CurrentAsk
   - You pay ${trade_price}
   - Your profit: ${profit_if_accept}
   - Book clears (both bid and ask reset to null)

2. PASS: Keep your bid standing
   What happens next:
   - If seller SELLs → Trade at YOUR BID (${my_standing_price}), profit = ${profit_if_opponent_accepts}
   - If seller PASSes → No trade, book persists, you remain CurrentBidder
   - Next BID/ASK step: someone may outbid you

Output format:
{{"reasoning": "...", "action": "accept"}}
{{"reasoning": "...", "action": "pass"}}"""
            else:
                prompt += f"""
1. SELL: Accept the CurrentBid
   - You receive ${trade_price}
   - Your profit: ${profit_if_accept}
   - Book clears (both bid and ask reset to null)

2. PASS: Keep your ask standing
   What happens next:
   - If buyer BUYs → Trade at YOUR ASK (${my_standing_price}), profit = ${profit_if_opponent_accepts}
   - If buyer PASSes → No trade, book persists, you remain CurrentAsker
   - Next BID/ASK step: someone may undercut you

Output format:
{{"reasoning": "...", "action": "accept"}}
{{"reasoning": "...", "action": "pass"}}"""
        else:
            prompt += f"""
Trade would be unprofitable (profit = ${profit_if_accept})
Only option: PASS

Output: {{"reasoning": "...", "action": "pass"}}"""

        return prompt
