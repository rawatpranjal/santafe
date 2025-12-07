"""
Zero-Intelligence Plus 2 (ZIP2) - Patient ZIP with ZIC2 Market Constraints.

ZIP2 is a direct evolutionary step from ZIC2:
- ZIC2: Random prices within budget + market constraints
- ZIP2: Learned prices within budget + market constraints

When ZIP2's learned target price falls outside the valid ZIC2 bounds,
it PASSES (returns 0) instead of bidding - exhibiting patience.

Hierarchy: ZI → ZIC1 (budget) → ZIC2 (budget + market) → ZIP1 → ZIP2
"""

import logging
from typing import Any

from traders.legacy.zip import ZIP1

logger = logging.getLogger(__name__)


class ZIP2(ZIP1):
    """
    Patient ZIP with ZIC2-style market constraints.

    Key difference from ZIP1:
    - ZIP1: Quote = limit * (1 + margin), clamped to [price_min, price_max]
    - ZIP2: Quote = limit * (1 + margin), but must respect ZIC2 bounds:
        - Buyers: quote must be in [current_bid + 1, limit_price]
        - Sellers: quote must be in [limit_price, current_ask - 1]
      If target is outside bounds, ZIP2 PASSES (returns 0).

    Hierarchy: ZI → ZIC1 (budget) → ZIC2 (budget + market) → ZIP1 → ZIP2
    """

    def __init__(
        self,
        player_id: int,
        is_buyer: bool,
        num_tokens: int,
        valuations: list[int],
        price_min: int = 0,
        price_max: int = 100,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ZIP2 agent (same parameters as ZIP)."""
        super().__init__(
            player_id=player_id,
            is_buyer=is_buyer,
            num_tokens=num_tokens,
            valuations=valuations,
            price_min=price_min,
            price_max=price_max,
            seed=seed,
            **kwargs,
        )

    def _calculate_quote(self) -> int:
        """
        Calculate shout price from limit price and profit margin,
        clipped to ZIC2-style market bounds.

        ZIP2 = ZIC2 with learned prices instead of random ones.

        If target is outside valid ZIC2 bounds, return 0 (PASS).
        This makes ZIP2 patient - it waits for favorable market conditions.
        """
        if self.num_trades >= self.num_tokens:
            return 0

        limit_price = self.valuations[self.num_trades]

        # ZIP's learned target price
        target = limit_price * (1.0 + self.margin)

        if self.is_buyer:
            # ZIC2 bounds for buyers: [current_bid + 1, limit_price]
            # Must bid higher than current bid to be competitive
            if self.current_high_bid > 0:
                min_valid = self.current_high_bid + 1
            else:
                min_valid = self.price_min

            max_valid = limit_price  # Budget constraint

            if target < min_valid:
                # Target is below market - PASS (patient behavior)
                logger.debug(
                    f"ZIP2 P{self.player_id} PASS: target={target:.2f} < min_valid={min_valid}"
                )
                return 0
            elif target > max_valid:
                # Cap at valuation (budget constraint)
                return int(max_valid)
            else:
                return int(round(target))
        else:
            # ZIC2 bounds for sellers: [limit_price, current_ask - 1]
            # Must ask lower than current ask to be competitive
            min_valid = limit_price  # Budget constraint

            if self.current_low_ask > 0:
                max_valid = self.current_low_ask - 1
            else:
                max_valid = self.price_max

            if target > max_valid:
                # Target is above market - PASS (patient behavior)
                logger.debug(
                    f"ZIP2 P{self.player_id} PASS: target={target:.2f} > max_valid={max_valid}"
                )
                return 0
            elif target < min_valid:
                # Floor at cost (budget constraint)
                return int(min_valid)
            else:
                return int(round(target))
