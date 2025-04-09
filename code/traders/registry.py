# traders/registry.py
import logging
from .zic import ZICBuyer, ZICSeller
from .zi import ZIUBuyer, ZIUSeller
from .zip import ZIPBuyer, ZIPSeller # Import the updated ZIP
# Import other strategies like Kaplan, GD, EL, PPO here
# from .kaplan import KaplanBuyer, KPSeller # Example
# ... etc

logger = logging.getLogger('registry')

# Dictionary mapping strategy names (lowercase strings) to (BuyerClass, SellerClass) tuples
_TRADER_CLASSES = {
    "zic": (ZICBuyer, ZICSeller),
    "zi": (ZIUBuyer, ZIUSeller),
    "zip": (ZIPBuyer, ZIPSeller), # Added ZIP
    # "kaplan": (KaplanBuyer, KPSeller), # Example
    # "gd": (GDBuyer, GDSeller), # Example
    # "el": (ELBuyer, ELSeller), # Example
    # "ppo_lstm": (PPOBuyer, PPOSeller), # Example
    # Add other strategies here
}

def get_trader_class(strategy_name, is_buyer):
    """ Gets the appropriate trader class based on strategy name and role. """
    strategy_name = strategy_name.lower()
    if strategy_name in _TRADER_CLASSES:
        buyer_cls, seller_cls = _TRADER_CLASSES[strategy_name]
        if is_buyer:
            return buyer_cls
        else:
            return seller_cls
    else:
        logger.error(f"Unknown strategy name requested: '{strategy_name}'")
        raise ValueError(f"Unknown strategy: {strategy_name}")

def available_strategies():
    """ Returns a list of available strategy names. """
    return list(_TRADER_CLASSES.keys())