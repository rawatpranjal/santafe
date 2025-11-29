"""
Zero-Intelligence Plus (ZIP) trading agent.

Based on:
Cliff, D., & Bruten, J. (1997). "Minimal-Intelligence Agents for Bargaining 
Behaviors in Market-Based Environments"

The ZIP agent uses simple machine learning (Widrow-Hoff delta rule) to adapt
its profit margin based on observed market activity.
"""

from typing import Any, Optional
import numpy as np
import logging
from traders.base import Agent

logger = logging.getLogger(__name__)


class ZIP(Agent):
    """
    Zero-Intelligence Plus trader with adaptive profit margins.
    
    Core mechanism:
    - Maintains profit margin μ that adapts via Widrow-Hoff rule
    - Shout price p = λ × (1 + μ) where λ is limit price
    - Raises margin when could have gotten better price
    - Lowers margin when not competitive
    """
    
    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        price_min: int = 0,
        price_max: int = 100,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize ZIP agent.
        
        Args:
            player_id: Agent ID
            is_buyer: True for buyer, False for seller
            num_tokens: Number of tokens
            valuations: Private valuations/costs
            price_min: Minimum allowed price
            price_max: Maximum allowed price
            seed: Random seed for reproducibility
            **kwargs: Hyperparameter overrides (beta, gamma, margin_init, etc.)
        """
        super().__init__(player_id, is_buyer, num_tokens, valuations)
        self.price_min = price_min
        self.price_max = price_max
        self.rng = np.random.default_rng(seed)
        
        # ZIP parameters (from paper) - allow overrides via kwargs
        # Calibrated for AURORA protocol via hyperparameter search (see calibrate_zip.py)

        # Learning rate: β
        # Paper: U[0.1, 0.5]
        # Calibrated default: 0.01 (v3 calibration - optimal for selfplay efficiency)
        # Lower beta = more conservative learning = better convergence
        if 'beta' in kwargs:
            # Fixed value passed explicitly
            self.beta = kwargs['beta']
        elif 'beta_min' in kwargs and 'beta_max' in kwargs:
            # Range passed, draw random
            self.beta = self.rng.uniform(kwargs['beta_min'], kwargs['beta_max'])
        else:
            # Use calibrated fixed default (achieves ~98% selfplay efficiency)
            self.beta = 0.01

        # Momentum coefficient: γ
        # Paper: U[0.0, 0.1]
        # Calibrated default: 0.008 (v4 calibration - optimal for Santa Fe gametype)
        # Low momentum = responsive to market signals
        if 'gamma' in kwargs:
            # Fixed value passed explicitly
            self.gamma = kwargs['gamma']
        elif 'gamma_min' in kwargs and 'gamma_max' in kwargs:
            # Range passed, draw random
            self.gamma = self.rng.uniform(kwargs['gamma_min'], kwargs['gamma_max'])
        else:
            # Use calibrated fixed default (achieves ~98% selfplay efficiency)
            self.gamma = 0.008

        # Profit margin: μ
        # Sellers: positive margin (markup), Buyers: negative margin (markdown)
        # Paper: U[0.05, 0.35] for sellers, U[-0.35, -0.05] for buyers
        # Calibrated default: 0.02 (v3 calibration - efficiency optimized)
        # Lower margin = faster convergence to equilibrium = higher efficiency
        if 'margin_init' in kwargs or 'margin' in kwargs:
            # Fixed value passed explicitly (assume positive, negate for buyers)
            margin_val = kwargs.get('margin_init', kwargs.get('margin'))
            self.margin = margin_val if not is_buyer else -abs(margin_val)
        elif 'margin_min' in kwargs and 'margin_max' in kwargs:
            # Range passed, draw random
            self.margin = self.rng.uniform(kwargs['margin_min'], kwargs['margin_max'])
        else:
            # Use calibrated optimal (v3 calibration - ~98% selfplay efficiency)
            self.margin = 0.02 if not is_buyer else -0.02

        # Target price calculation parameters (R and A from paper)
        # These control convergence speed - CRITICAL for self-play performance
        # Paper defaults: R ∈ [1.0, 1.05] or [0.95, 1.0], A ∈ [0, 0.05] or [-0.05, 0]
        self.R_increase_min = kwargs.get('R_increase_min', 1.0)
        self.R_increase_max = kwargs.get('R_increase_max', 1.05)
        self.R_decrease_min = kwargs.get('R_decrease_min', 0.95)
        self.R_decrease_max = kwargs.get('R_decrease_max', 1.0)
        self.A_increase_min = kwargs.get('A_increase_min', 0.0)
        self.A_increase_max = kwargs.get('A_increase_max', 0.05)
        self.A_decrease_min = kwargs.get('A_decrease_min', -0.05)
        self.A_decrease_max = kwargs.get('A_decrease_max', 0.0)

        # Momentum term Γ (capital gamma)
        self.momentum_delta = 0.0
        
        # Last shout observed
        self.last_shout_price = 0
        self.last_shout_was_bid = False
        self.last_shout_accepted = False

        # Current quote and market state
        self.current_quote = 0

        self.current_high_bid = 0
        self.current_low_ask = 0
        self.current_high_bidder = 0
        self.current_low_asker = 0

    def start_round(self, valuations: list[int]) -> None:
        """
        Reset ZIP learning state for new round with new valuations.

        CRITICAL: When valuations change between rounds, we must reset
        learned parameters (margin, momentum) because they were adapted
        to the OLD valuation distribution.

        Note: start_period() does NOT reset these - that's intentional!
        ZIP should learn ACROSS periods within a round (Cliff 1997).
        """
        # Call base class to update valuations and reset trade counters
        super().start_round(valuations)

        # Randomized margins per Cliff & Bruten (1997) Section 4.2
        # Ensures diversity: some agents conservative (can trade immediately),
        # others aggressive (must learn). Prevents systematic deadlock.
        #
        # v4 CALIBRATION: Near-zero margins for Santa Fe gametype efficiency
        # Original paper: U[0.05, 0.35] but this causes slow convergence
        # Santa Fe gametype creates tight supply/demand - need small margins
        # Optimized: U[0.0, 0.003] for ~98% selfplay efficiency
        if self.is_buyer:
            self.margin = self.rng.uniform(-0.003, 0.0)
        else:
            self.margin = self.rng.uniform(0.0, 0.003)

        # Reset momentum (Widrow-Hoff delta accumulator)
        self.momentum_delta = 0.0

        # Clear stale market observations from previous round
        self.last_shout_price = 0
        self.last_shout_was_bid = False
        self.last_shout_accepted = False

        # Clear stale market state
        self.current_quote = 0
        self.current_high_bid = 0
        self.current_low_ask = 0
        self.current_high_bidder = 0
        self.current_low_asker = 0

    def _calculate_quote(self) -> int:
        """
        Calculate shout price from limit price and profit margin.
        
        p = λ × (1 + μ)
        
        For sellers: higher μ = higher price
        For buyers: lower μ = lower price
        """
        if self.num_trades >= self.num_tokens:
            return 0
            
        limit_price = self.valuations[self.num_trades]
        quote = limit_price * (1.0 + self.margin)
        
        # Clamp to valid range
        quote = max(self.price_min, min(self.price_max, int(round(quote))))
        
        logger.debug(f"ZIP P{self.player_id} QUOTE: limit={limit_price} margin={self.margin:.4f} quote={quote}")
        return quote
        
    def _calculate_target_price(self, last_price: int, raise_margin: bool) -> float:
        """
        Calculate target price τ for Widrow-Hoff update.

        τ = R × q + A

        Where:
        - R ~ U[R_increase_min, R_increase_max] for increases
        - R ~ U[R_decrease_min, R_decrease_max] for decreases
        - A ~ U[A_increase_min, A_increase_max] for increases
        - A ~ U[A_decrease_min, A_decrease_max] for decreases

        These parameters control convergence speed - critical for self-play!
        """
        if raise_margin:
            R = self.rng.uniform(self.R_increase_min, self.R_increase_max)
            A = self.rng.uniform(self.A_increase_min, self.A_increase_max)
        else:
            R = self.rng.uniform(self.R_decrease_min, self.R_decrease_max)
            A = self.rng.uniform(self.A_decrease_min, self.A_decrease_max)

        target = R * last_price + A

        # Clamp target to valid price range
        target = max(self.price_min, min(self.price_max, target))

        return target
        
    def _update_margin(self, target_price: float) -> None:
        """
        Update profit margin using Widrow-Hoff delta rule with momentum.
        
        Δ(t) = β × (τ(t) - p(t))
        Γ(t+1) = γ × Γ(t) + (1-γ) × Δ(t)
        μ(t+1) = (p(t) + Γ(t)) / λ - 1
        """
        if self.num_trades >= self.num_tokens:
            # No limit price for next trade
            return
            
        current_price = self._calculate_quote()
        limit_price = self.valuations[self.num_trades]
        
        # Widrow-Hoff delta
        delta = self.beta * (target_price - current_price)
        
        # Momentum update
        self.momentum_delta = self.gamma * self.momentum_delta + (1.0 - self.gamma) * delta
        
        # Update margin
        new_price = current_price + self.momentum_delta
        old_margin = self.margin
        # Guard against zero limit price (can happen with gametype=0)
        if limit_price > 0:
            self.margin = (new_price / limit_price) - 1.0
        else:
            self.margin = 0.0  # Default margin when limit is zero
        
        logger.debug(
            f"ZIP P{self.player_id} UPDATE: target={target_price:.2f} curr={current_price} "
            f"delta={delta:.4f} mom_delta={self.momentum_delta:.4f} "
            f"margin: {old_margin:.4f} -> {self.margin:.4f}"
        )
        
        # Clamp margin to valid ranges
        # Tight bounds prevent runaway divergence (10.0 was causing market failures)
        if self.is_buyer:
            self.margin = max(-0.50, min(0.0, self.margin))
            # Verify buyer margin is non-positive
            assert self.margin <= 0.0, f"Buyer margin must be ≤ 0, got {self.margin}"
        else:
            self.margin = max(0.0, min(0.50, self.margin))
            # Verify seller margin is non-negative
            assert self.margin >= 0.0, f"Seller margin must be ≥ 0, got {self.margin}"
            
    def _should_raise_margin(self, my_price: int, trade_price: int) -> bool:
        """
        Check if agent should raise its profit margin.
        
        Raise if: last shout accepted AND my price ≤ transaction price
        """
        if not self.last_shout_accepted:
            return False
            
        if self.is_buyer:
            # Buyer: raise margin (lower price) if my bid was too low
            return my_price >= trade_price
        else:
            # Seller: raise margin (higher price) if my ask was too low
            return my_price <= trade_price
            
    def _should_lower_margin(self, my_price: int, last_price: int, high_bidder: int = 0, low_asker: int = 0) -> bool:
        """
        Check if agent should lower its profit margin.

        Per Cliff (1997) Figure 27 WITH AURORA PROTOCOL ADAPTATION:
        
        ACCEPTED case (matches paper exactly):
        - Lower if active AND counterparty shout AND my price worse than trade
        
        REJECTED case (AURORA adaptation - differs from paper):
        - Lower if active AND my price not competitive with ANY standing order
        - Respond to CROSS-SIDE signals (not just same-side as in paper)
        
        CRITICAL AURORA ADAPTATION:
        The paper's CDA has:
        - Continuous asynchronous trading (shouts at any time)
        - Book CLEARS after each trade
        - "Rejected" = shout not YET accepted (transient state)
        
        AURORA has:
        - Two-stage synchronous protocol (bid/ask, then buy/sell)
        - Book PERSISTS (only clears if trade executes)
        - "Rejected" = end of period, NO trade (terminal state)
        
        In the paper's CDA, Figure 27's same-side-only logic works because:
        - Constant new bids/offers arrive asynchronously
        - Book clears frequently
        - Agents don't get stuck with persistent standing orders
        
        In AURORA, strict Figure 27 causes deadlock:
        - bid=$191, ask=$202 persists for many steps
        - Buyers wait for rejected bids (but only see standing ask)
        - Sellers wait for rejected offers (but only see standing bid)
        - Spread never closes, efficiency drops to 70%
        
        Cross-side response in REJECTED case is NECESSARY for AURORA:
        - Both sides see persistent standing orders
        - To close spread, both must move toward each other
        - Buyers seeing high asks must raise bids
        - Sellers seeing low bids must lower asks
        - Self-shout prevention stops divergence
        
        This is a justified protocol adaptation, not a bug.
        """
        if self.num_trades >= self.num_tokens:
            # Inactive
            return False

        if self.last_shout_accepted:
            # Trade occurred - respond to COUNTERPARTY shouts (matches paper)
            if self.is_buyer:
                # I'm buyer, last shout was ask (counterparty)
                if not self.last_shout_was_bid:
                    return my_price <= last_price
            else:
                # I'm seller, last shout was bid (counterparty)
                if self.last_shout_was_bid:
                    return my_price >= last_price
        else:
            # No trade - Simplified AURORA logic
            # Respond to market pressure but avoid own shouts
            if self.is_buyer:
                # Don't respond to own shout (prevents divergence)
                if self.player_id == high_bidder:
                    return False
                # Lower margin (raise bid) if my quote is below market price
                return my_price <= last_price
            else:
                # Don't respond to own shout (prevents divergence)
                if self.player_id == low_asker:
                    return False
                # Lower margin (lower ask) if my quote is above market price
                return my_price >= last_price

        return False
        
    def bid_ask(self, time: int, nobidask: int) -> None:
        """Receive notification to submit bid or ask."""
        self.current_time = time
        self.current_quote = self._calculate_quote()
        
    def bid_ask_response(self) -> int:
        """Return bid or ask."""
        return self.current_quote
        
    def bid_ask_result(
        self,
        status: int,
        num_trades: int,
        new_bids: list[int],
        new_asks: list[int],
        high_bid: int,
        high_bidder: int,
        low_ask: int,
        low_asker: int,
    ) -> None:
        """
        Process bid/ask stage results and potentially update margin.
        
        Note: In AURORA, we observe the bid/ask phase results but don't know
        yet if a trade will occur. We'll update based on this information.
        """
        super().bid_ask_result(status, num_trades, new_bids, new_asks, 
                               high_bid, high_bidder, low_ask, low_asker)
        
        # Store market state for buy/sell stage
        self.current_high_bid = high_bid
        self.current_low_ask = low_ask
        
    def buy_sell(
        self,
        time: int,
        nobuysell: int,
        high_bid: int,
        low_ask: int,
        high_bidder: int,
        low_asker: int,
    ) -> None:
        """Receive notification for buy/sell decision."""
        self.current_high_bid = high_bid
        self.current_high_bidder = high_bidder
        self.current_low_ask = low_ask
        self.current_low_asker = low_asker

    def buy_sell_response(self) -> bool:
        """
        Decide whether to accept current market price.

        Per Cliff & Bruten 1997 Section 4.1 (page 21):
        "A ZIP buyer will, in principle, buy from any seller that makes an
        offer less than the buyer's current bid shout-price; similarly, a ZIP
        seller sells to any buyer making a bid greater than the seller's
        current offer shout-price."

        Key difference from ZIC: ZIP compares against its SHOUT PRICE (current_quote),
        not its limit price (valuation).

        DEFENSIVE IMPLEMENTATION:
        - Extra validation to prevent irrational trades
        - Guards against state corruption across rounds
        - Ensures we never trade at a loss relative to our limit price
        """
        # Defensive: Check we have tokens left
        if self.num_trades >= self.num_tokens:
            return False

        # Defensive: Validate num_trades is within bounds
        if self.num_trades < 0 or self.num_trades >= len(self.valuations):
            return False

        # Check if spread is crossed
        if self.current_high_bid <= 0 or self.current_low_ask <= 0:
            return False

        if self.current_high_bid < self.current_low_ask:
            return False  # Spread not crossed

        # Get our limit price for defensive checks
        limit_price = self.valuations[self.num_trades]

        # Defensive: Validate market prices are reasonable
        if self.current_low_ask < 0 or self.current_high_bid < 0:
            return False

        if self.is_buyer:
            # Accept if I'm high bidder AND offer ≤ my bid shout price
            if self.player_id == self.current_high_bidder:
                # Primary check: ZIP's shout price rule
                if self.current_low_ask <= self.current_quote:
                    # Defensive: NEVER buy above our limit price (would lose money)
                    if self.current_low_ask <= limit_price:
                        return True
        else:
            # Accept if I'm low asker AND bid ≥ my ask shout price
            if self.player_id == self.current_low_asker:
                # Primary check: ZIP's shout price rule
                if self.current_high_bid >= self.current_quote:
                    # Defensive: NEVER sell below our limit price (would lose money)
                    if self.current_high_bid >= limit_price:
                        return True

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
        """
        Process trade results and update profit margin.
        
        This is where the ZIP learning occurs.
        """
        super().buy_sell_result(status, trade_price, trade_type, high_bid, 
                                high_bidder, low_ask, low_asker)
        
        # Update last shout information
        if trade_type != 0:  # Trade occurred
            self.last_shout_accepted = True
            self.last_shout_price = trade_price
            
            # Determine if last shout was bid or ask based on trade_type
            # Type 1: buyer accepted (price = ask)
            # Type 2: seller accepted (price = bid)
            # Type 3: both accepted (random)
            if trade_type == 1:
                self.last_shout_was_bid = False  # Ask was accepted
            elif trade_type == 2:
                self.last_shout_was_bid = True   # Bid was accepted
            else:
                # Both accepted, use price to infer
                self.last_shout_was_bid = (trade_price == high_bid)
        else:
            self.last_shout_accepted = False
            # Last shout was the current outstanding bid or ask
            if high_bid > 0:
                self.last_shout_was_bid = True
                self.last_shout_price = high_bid
            elif low_ask > 0:
                self.last_shout_was_bid = False
                self.last_shout_price = low_ask
                
        # Now apply ZIP learning rules
        my_quote = self._calculate_quote()

        if my_quote > 0:  # Have a valid quote
            # Check if should raise margin
            if self._should_raise_margin(my_quote, trade_price if trade_type != 0 else self.last_shout_price):
                target = self._calculate_target_price(trade_price if trade_type != 0 else self.last_shout_price, raise_margin=True)
                self._update_margin(target)

            # Check if should lower margin (only if active)
            elif self._should_lower_margin(my_quote, trade_price if trade_type != 0 else self.last_shout_price, high_bidder, low_asker):
                target = self._calculate_target_price(trade_price if trade_type != 0 else self.last_shout_price, raise_margin=False)
                self._update_margin(target)
