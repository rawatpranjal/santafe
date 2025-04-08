# traders/registry.py
import logging
from .base import BaseTrader
from .zic import ZICBuyer, ZICSeller
from .kaplan import KaplanBuyer, KaplanSeller # <<< ADD THIS IMPORT
from .ql import QLTrader

logger = logging.getLogger('trader.registry')

# Registry mapping strategy names to (BuyerClass, SellerClass) tuples
_trader_registry = {
    "zic": (ZICBuyer, ZICSeller),
    "kaplan": (KaplanBuyer, KaplanSeller), # <<< ADD THIS LINE
    # "dqn": (QLTrader, QLTrader), # Optional: Keep old name mapping?
    "dqn_pricegrid": (QLTrader, QLTrader),
    # Add 'gd', 'el', 'skeleton', etc. when implemented
}

def get_trader_class(strategy_type: str, is_buyer: bool) -> type[BaseTrader]:
    """
    Factory function to retrieve the appropriate trader class based on strategy type and role.
    """
    strategy_type_lower = strategy_type.lower() # Case-insensitive lookup
    if strategy_type_lower not in _trader_registry:
        available = available_strategies()
        logger.error(f"Unknown trader strategy type: '{strategy_type}'. Available: {available}")
        raise ValueError(f"Unknown trader strategy type: '{strategy_type}'. Available: {available}")

    buyer_class, seller_class = _trader_registry[strategy_type_lower]
    TraderClass = buyer_class if is_buyer else seller_class

    # Check if the selected class expects rl_config (useful for auction setup)
    import inspect
    try:
        sig = inspect.signature(TraderClass.__init__)
        expects_rl_config = 'rl_config' in sig.parameters
    except ValueError: # Handles built-in types or classes without standard __init__
        expects_rl_config = False

    logger.debug(f"Registry returning class: {TraderClass.__name__} for type '{strategy_type}' (Buyer={is_buyer}), expects_rl_config={expects_rl_config}")
    return TraderClass

def available_strategies():
    """Returns a list of registered strategy type names."""
    return list(_trader_registry.keys())

# Create an empty file named __init__.py in the traders directory if it doesn't exist