"""
Kaplan Agent.

Ports the logic from SRobotKaplan.java (1993 Santa Fe Tournament).
This agent uses a "sniper" strategy and price history to decide bids/asks.
It is deterministic.

Key behavioral characteristics:
1. Uses worst-case token for first bid/ask (most conservative)
2. "Jump-in" logic: submits competitive bids when spread is small, price is favorable, or time is short
3. Protection clauses: all jump-in bids/asks are clamped to profitability bounds
4. Sniper behavior: accepts any profitable trade in last 2 timesteps
5. Learns from history: tracks min/max/avg prices across periods

Expected performance:
- All-Kaplan markets: Should achieve moderate efficiency (~60-80%)
- Mixed markets: Effective against passive strategies like ZIC
- The jump-in logic with protection prevents extreme waiting games
"""

from traders.base import Agent


class Kaplan(Agent):
    """
    Kaplan trading agent (Sniper).
    
    Strategy:
    - Wait for favorable prices.
    - "Jump in" if spread is small, price is better than last period, or time is running out.
    - Accept any profitable trade in the last few seconds (Sniper).
    - Tracks min/max/avg prices from previous periods.
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        price_min: int = 0,
        price_max: int = 100,
        num_times: int = 100, # Default, should be set correctly
        seed: int | None = None,
    ) -> None:
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min_limit = price_min
        self.price_max_limit = price_max
        self.num_times = num_times

        # Kaplan State
        self.minpr: int = 0
        self.maxpr: int = 0
        self.avepr: int = 0
        self.prices: list[int] = [0] * (num_times + 100)  # 1-indexed list of prices (with buffer)
        self.trade_count: int = 0
        self.last_time: int = 0
        self.my_last_time: int = 0
        self.period: int = 1  # Start at period 1

        # Current Step State
        self.current_time = 0
        self.current_bid = 0
        self.current_ask = 0
        self.current_bidder = 0
        self.current_asker = 0
        self.nobidask = 0
        self.nobuysell = 0

    def start_period(self, period_number: int) -> None:
        super().start_period(period_number)
        self.period = period_number
        # Reset period stats
        self.prices = [0] * (self.num_times + 100) # Buffer
        self.trade_count = 0
        self.last_time = 0
        self.my_last_time = 0

    def end_period(self) -> None:
        # Logic from SRobotKaplan.java playerPeriodEnd
        p1 = self.prices[1] if self.trade_count > 0 else 0

        self.avepr = 0
        self.minpr = abs(p1)
        self.maxpr = abs(p1)

        if self.is_buyer:
            self.maxpr += -100
        else:
            self.minpr += 100

        for t1 in range(1, self.trade_count + 1):
            p_val = self.prices[t1]
            abs_p = abs(p_val)

            if not self.is_buyer: # Seller
                if self.maxpr < abs_p:
                    self.maxpr = abs_p
                if self.minpr > p_val and p_val > 0:
                    self.minpr = abs_p
            else: # Buyer
                if self.maxpr < p_val: # Note: Java uses p_val here, not abs_p?
                    # Java: if (maxpr<prices[t1]) maxpr=abs(prices[t1]);
                    # If prices[t1] is -1, maxpr < -1 is false (if maxpr >= 0).
                    # So it ignores own trades for maxpr update?
                    self.maxpr = abs_p
                if self.minpr > abs_p:
                    self.minpr = abs_p

            self.avepr += abs_p

        if self.trade_count > 0:
            self.avepr = self.avepr // self.trade_count

        # Call parent to accumulate period profit into total profit
        super().end_period()

    def bid_ask(self, time: int, nobidask: int) -> None:
        self.current_time = time
        self.nobidask = nobidask
        self.has_responded = False

    def bid_ask_response(self) -> int:
        self.has_responded = True

        if self.is_buyer:
            return self._player_request_bid()
        else:
            return self._player_request_ask()

    def _player_request_bid(self) -> int:
        if self.nobidask > 0: return 0

        # Get current valuation (token[mytrades+1])
        if self.num_trades >= self.num_tokens: return 0
        token_val = self.valuations[self.num_trades]

        newbid = 0
        most = 0

        # CRITICAL FIX: Use worst-case token for FIRST bid only
        # Java:
        #   if (cbid == 0): most = token[ntokens] - 1  (worst-case)
        #   else: most = token[mytrades+1] - 1  (current token)
        worst_case_val = self.valuations[self.num_tokens - 1]

        if self.current_bid == 0:
            most = worst_case_val - 1
            if self.current_ask > 0 and self.current_ask < most:
                most = self.current_ask
            newbid = self.price_min_limit + 1
        else:
            most = token_val - 1  # Use CURRENT token for subsequent bids
            newbid = self.price_min_limit + 1
            if self.current_ask > 0 and self.current_ask < most:
                most = self.current_ask

            if most <= self.current_bid:
                return 0

            # Jump in logic
            # 1. Spread < 10%
            spread_cond = False
            if self.current_ask > 0:
                ratio = (self.current_ask - self.current_bid) / (self.current_ask + 1)
                if ratio < 0.10:
                    # Check profitability: (token+1)*0.98 - cask > 0
                    profit_cond = ((token_val + 1) * 0.98 - self.current_ask) > 0
                    if profit_cond and (self.period == 1 or self.current_ask <= self.maxpr):
                        newbid = self.current_ask

            # 2. Price better than last period
            if self.period > 1 and self.current_ask > 0 and self.current_ask <= self.minpr:
                newbid = self.current_ask

            # 3. Time running out or been a while
            time_cond = False
            t = self.current_time
            rem = self.num_times - t
            if (t - self.last_time) >= rem / 2:
                time_cond = True
            if (t - self.my_last_time) >= 2 * rem / 3 and (t - self.last_time) >= 5:
                time_cond = True

            if time_cond and self.current_ask > 0:
                newbid = self.current_ask

            # Java line 74: Protection for subsequent bids only (NOT first bid)
            if newbid > most:
                newbid = most

        # Clamp to minimum
        if newbid < self.price_min_limit:
            newbid = self.price_min_limit

        return newbid

    def _player_request_ask(self) -> int:
        import logging
        logger = logging.getLogger("kaplan.ask")

        if self.nobidask > 0: return 0

        if self.num_trades >= self.num_tokens: return 0
        token_val = self.valuations[self.num_trades]

        newoffer = 0
        least = 0

        # CRITICAL FIX: Use worst-case token for FIRST ask only
        # Java:
        #   if (cask == 0): least = token[ntokens] + 1  (worst-case)
        #   else: least = token[mytrades+1] + 1  (current token)
        worst_case_val = self.valuations[self.num_tokens - 1]

        logger.debug(
            f"SELLER P{self.player_id} _request_ask t={self.current_time} "
            f"token[{self.num_trades}]={token_val} worst={worst_case_val} "
            f"cask={self.current_ask} cbid={self.current_bid}"
        )

        if self.current_ask == 0:
            least = worst_case_val + 1
            if self.current_bid > least:
                least = self.current_bid
            newoffer = self.price_max_limit - 1
        else:
            newoffer = self.price_max_limit - 1
            least = token_val + 1  # Use CURRENT token for subsequent asks
            if self.current_bid > least:
                least = self.current_bid

            logger.debug(
                f"SELLER P{self.player_id} SUBSEQUENT ASK initial least={least} newoffer={newoffer}"
            )

            if least >= self.current_ask:
                logger.debug(f"SELLER P{self.player_id} RETURN 0 (least >= cask)")
                return 0

            # Jump in logic
            # 1. Spread < 10%
            if self.current_bid > 0:
                ratio = (self.current_ask - self.current_bid) / (self.current_bid + 1)
                logger.debug(f"SELLER P{self.player_id} spread ratio={ratio:.3f}")
                if ratio < 0.10:
                    profit_cond = ((token_val + 1) * 1.02 - self.current_bid) < 0
                    logger.debug(
                        f"SELLER P{self.player_id} small spread! profit_cond={profit_cond} "
                        f"period={self.period} cbid>= minpr={self.current_bid >= self.minpr}"
                    )
                    if profit_cond and (self.period == 1 or self.current_bid >= self.minpr):
                        logger.debug(f"SELLER P{self.player_id} JUMP IN (spread) newoffer={self.current_bid}")
                        newoffer = self.current_bid

            # 2. Price better than last period
            if self.period > 1 and self.current_bid >= self.maxpr:
                logger.debug(f"SELLER P{self.player_id} JUMP IN (better price) newoffer={self.current_bid}")
                newoffer = self.current_bid

            # 3. Time running out
            t = self.current_time
            rem = self.num_times - t
            time_cond = False
            if (t - self.last_time) >= rem / 2:
                time_cond = True
            if (t - self.my_last_time) >= 2 * rem / 3 and (t - self.last_time) >= 5:
                time_cond = True

            if time_cond:
                logger.debug(f"SELLER P{self.player_id} JUMP IN (time) newoffer={self.current_bid}")
                newoffer = self.current_bid

            # Java line 99: Protection for subsequent asks only (NOT first ask)
            if newoffer < least:
                newoffer = least

        # CRITICAL FIX: Also enforce least for the first ask case (if current_ask == 0)
        # The original Java logic might have relied on max_price being high enough,
        # but we must ensure we never ask below cost + 1.
        if newoffer < least:
            newoffer = least

        # Clamp to maximum
        if newoffer > self.price_max_limit:
            newoffer = self.price_max_limit

        logger.debug(f"SELLER P{self.player_id} FINAL ASK={newoffer}")
        return newoffer

    def buy_sell(
        self,
        time: int,
        nobuysell: int,
        high_bid: int,
        low_ask: int,
        high_bidder: int,
        low_asker: int,
    ) -> None:
        self.current_time = time
        self.nobuysell = nobuysell
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker
        self.has_responded = False

    def buy_sell_response(self) -> bool:
        import logging
        logger = logging.getLogger("kaplan.decision")

        self.has_responded = True
        if self.nobuysell > 0: return False
        if self.num_trades >= self.num_tokens: return False

        token_val = self.valuations[self.num_trades]

        if self.is_buyer:
            # CRITICAL GUARD CLAUSE: Never buy at a loss
            # Matches Java: if (token[mytrades+1] <= cask) return 0;
            logger.debug(
                f"BUYER P{self.player_id} t={self.current_time} "
                f"token[{self.num_trades}]={token_val} ask={self.current_ask} "
                f"guard_check: {token_val}<={self.current_ask}={token_val <= self.current_ask}"
            )

            if token_val <= self.current_ask:
                logger.debug(f"BUYER P{self.player_id} GUARD FIRED - REJECT")
                return False

            # Safe to accept: all paths below are profitable
            # (token_val > current_ask guaranteed by guard above)

            # Accept if winner and spread crossed
            if self.player_id == self.current_bidder and self.current_bid >= self.current_ask:
                logger.debug(
                    f"BUYER P{self.player_id} ACCEPT (crossed spread) "
                    f"bid={self.current_bid} ask={self.current_ask}"
                )
                return True

            # Sniper: Accept if time is short
            # Safe because guard clause ensures profit
            if (self.num_times - self.current_time) <= 2:
                logger.debug(
                    f"BUYER P{self.player_id} ACCEPT (sniper t={self.current_time}/{self.num_times}) "
                    f"token={token_val} ask={self.current_ask} profit={token_val - self.current_ask}"
                )
                return True

            logger.debug(f"BUYER P{self.player_id} REJECT (no condition met)")

        else:
            # CRITICAL GUARD CLAUSE: Never sell at a loss
            # Matches Java: if (cbid <= token[mytrades+1]) return 0;
            logger.debug(
                f"SELLER P{self.player_id} t={self.current_time} "
                f"token[{self.num_trades}]={token_val} bid={self.current_bid} "
                f"guard_check: {self.current_bid}<={token_val}={self.current_bid <= token_val}"
            )

            if self.current_bid <= token_val:
                logger.debug(f"SELLER P{self.player_id} GUARD FIRED - REJECT")
                return False

            # Safe to accept: all paths below are profitable
            # (current_bid > token_val guaranteed by guard above)

            if self.player_id == self.current_asker and self.current_ask <= self.current_bid:
                logger.debug(
                    f"SELLER P{self.player_id} ACCEPT (crossed spread) "
                    f"bid={self.current_bid} ask={self.current_ask}"
                )
                return True

            if (self.num_times - self.current_time) <= 2:
                logger.debug(
                    f"SELLER P{self.player_id} ACCEPT (sniper t={self.current_time}/{self.num_times}) "
                    f"token={token_val} bid={self.current_bid} profit={self.current_bid - token_val}"
                )
                return True

            logger.debug(f"SELLER P{self.player_id} REJECT (no condition met)")

        return False

    def buy_sell_result(
        self,
        status: int,
        trade_price: int,
        trade_type: int,
        high_bid: int,
        high_bidder: int,
        low_ask: int,
        low_asker: int,
    ) -> None:
        import logging
        logger = logging.getLogger("kaplan.trade")

        # Log before super() call to see token value BEFORE num_trades increments
        if status == 1:  # I traded
            token_val = self.valuations[self.num_trades]  # Current token (before increment)
            profit = (token_val - trade_price) if self.is_buyer else (trade_price - token_val)
            logger.debug(
                f"{'BUY' if self.is_buyer else 'SELL'} P{self.player_id} "
                f"EXECUTED price={trade_price} token[{self.num_trades}]={token_val} "
                f"profit={profit} period_profit_before={self.period_profit}"
            )

        super().buy_sell_result(status, trade_price, trade_type, high_bid, high_bidder, low_ask, low_asker)

        if status == 1:
            logger.debug(
                f"{'BUY' if self.is_buyer else 'SELL'} P{self.player_id} "
                f"period_profit_after={self.period_profit} "
                f"num_trades_after={self.num_trades}"
            )

        # Update history
        if trade_type != 0: # Trade occurred
            self.trade_count += 1
            self.last_time = self.current_time

            # Did I trade?
            # status 1 means I traded.
            if status == 1:
                # num_trades updated by super
                self.my_last_time = self.current_time
                # Java: prices[ntrades] = -1
                self.prices[self.trade_count] = -1
            else:
                self.prices[self.trade_count] = trade_price

        # Update current state (not strictly needed as bid_ask/buy_sell update it, but good for consistency)
        self.current_bid = high_bid
        self.current_ask = low_ask
        self.current_bidder = high_bidder
        self.current_asker = low_asker
