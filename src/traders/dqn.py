# traders/registry.py
import logging
from .base import BaseTrader
from .zic import ZICBuyer, ZICSeller
from .kp import KaplanBuyer, KaplanSeller
from .ql import QLTrader # Import the new QL trader

logger = logging.getLogger('trader.registry')

_trader_registry = {
    "zic": (ZICBuyer, ZICSeller),
    "kaplan": (KaplanBuyer, KaplanSeller),
    "dqn": (QLTrader, QLTrader), # Register DQN trader
    # Add 'gd' and 'el' when implemented
}

def get_trader_class(strategy_type: str, is_buyer: bool) -> type[BaseTrader]:
    """Factory function to retrieve the appropriate trader class."""
    strategy_type = strategy_type.lower()
    if strategy_type not in _trader_registry:
        logger.error(f"Unknown trader strategy type: '{strategy_type}'. Available: {available_strategies()}")
        raise ValueError(f"Unknown trader strategy type: '{strategy_type}'")
    buyer_class, seller_class = _trader_registry[strategy_type]
    cls = buyer_class if is_buyer else seller_class
    # Check if the class expects rl_config (heuristic check)
    import inspect
    sig = inspect.signature(cls.__init__)
    expects_rl_config = 'rl_config' in sig.parameters
    logger.debug(f"Registry returning class: {cls.__name__} for type '{strategy_type}' (Buyer={is_buyer}), expects_rl_config={expects_rl_config}")
    return cls

def available_strategies():
    """Returns a list of registered strategy type names."""
    return list(_trader_registry.keys())