# traders/registry.py
import logging
from .zic import ZICBuyer, ZICSeller
from .zi import ZIUBuyer, ZIUSeller
from .zip import ZIPBuyer, ZIPSeller
from .mgd import MGDBuyer, MGDSeller
from .gd import GDBuyer, GDSeller
from .el import ELBuyer, ELSeller
from .kp import KaplanBuyer, KPSeller
from .rg import RGBuyer, RGSeller
from .sk import SKBuyer, SKSeller
from .tt import TTBuyer, TTSeller
from .mu import MUBuyer, MUSeller
from .pt import PTBuyer, PTSeller
from .ppo import PPOTrader
from .ppo_handcrafted import PPOHandcraftedTrader

logger = logging.getLogger('registry')

# Dictionary mapping strategy names (lowercase strings) to (BuyerClass, SellerClass) tuples
_TRADER_CLASSES = {
    "zic": (ZICBuyer, ZICSeller),
    "zi": (ZIUBuyer, ZIUSeller),
    "zip": (ZIPBuyer, ZIPSeller),
    "mgd": (MGDBuyer, MGDSeller),
    "gd": (GDBuyer, GDSeller),
    "el": (ELBuyer, ELSeller),
    "kaplan": (KaplanBuyer, KPSeller),
    "rg": (RGBuyer, RGSeller),
    "sk": (SKBuyer, SKSeller),
    "tt": (TTBuyer, TTSeller),
    "mu": (MUBuyer, MUSeller),
    "pt": (PTBuyer, PTSeller),
    "ppo_lstm": (PPOTrader, PPOTrader),  # PPOTrader handles both buyer/seller roles
    "ppo_handcrafted": (PPOHandcraftedTrader, PPOHandcraftedTrader),  # Feature-engineered PPO agent
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