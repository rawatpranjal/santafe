# traders/registry.py
import logging

# --- Import Base Strategies ---
from .zic import ZICBuyer, ZICSeller
from .kaplan import KaplanBuyer, KaplanSeller
from .zi import ZIBuyer, ZISeller
from .zip import ZipBuyer, ZipSeller

# --- Import Heuristic/Belief Strategies ---
from .gd import GDBuyer, GDSeller
from .el import ELBuyer, ELSeller
# --- ADD IMPORTS FOR NEW STRATEGIES ---
from .tt import TTBuyer, TTSeller          # Truth Teller
from .mu import MUBuyer, MUSeller          # Markup
from .sk import SKBuyer, SKSeller          # Skeleton (Interpreted)
from .rg import RGBuyer, RGSeller          # Ringuette
# --- END ADDED IMPORTS ---

# --- Import RL Strategies ---
from .ppo import PPOTrader

# --- Trader Registration Dictionary ---
# Maps lowercase strategy names to (BuyerClass, SellerClass) tuples
_TRADER_CLASSES = {
    # Baselines
    "zic": (ZICBuyer, ZICSeller),
    "kaplan": (KaplanBuyer, KaplanSeller),
    "zi": (ZIBuyer, ZISeller),
    "zip": (ZipBuyer, ZipSeller),

    # Heuristic/Belief-based
    "gd": (GDBuyer, GDSeller),
    "el": (ELBuyer, ELSeller),
    # --- ADD NEW STRATEGIES ---
    "tt": (TTBuyer, TTSeller),
    "mu": (MUBuyer, MUSeller),
    "sk": (SKBuyer, SKSeller),
    "rg": (RGBuyer, RGSeller),
    # --- END ADDED STRATEGIES ---

    # RL Agents
    "ppo_lstm": (PPOTrader, PPOTrader),
}

# ... (get_trader_class and available_strategies functions remain the same) ...
def get_trader_class(strategy_name, is_buyer):
    """Gets the appropriate trader class based on strategy name and role."""
    logger = logging.getLogger('registry')
    strategy_name = strategy_name.lower() # Ensure consistent lookup

    if strategy_name not in _TRADER_CLASSES:
        logger.error(f"Unknown trader strategy name: '{strategy_name}'")
        available = available_strategies()
        logger.error(f"Available strategies are: {available}")
        # Provide a more informative error message
        raise ValueError(f"Unknown trader strategy name: '{strategy_name}'. Available: {available}")

    buyer_class, seller_class = _TRADER_CLASSES[strategy_name]

    if is_buyer:
        # logger.debug(f"Registry providing Buyer class for '{strategy_name}': {buyer_class.__name__}")
        return buyer_class
    else:
        # logger.debug(f"Registry providing Seller class for '{strategy_name}': {seller_class.__name__}")
        return seller_class

def available_strategies():
    """Returns a sorted list of available strategy names."""
    return sorted(list(_TRADER_CLASSES.keys())) # Sort for consistent output