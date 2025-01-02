# traders/registry.py

from traders.zi import ZIBuyer, ZISeller
from traders.zic import RandomBuyer, RandomSeller
from traders.gd import GDBuyer, GDSeller
from traders.zip import ZipBuyer, ZipSeller
from traders.kaplan import KaplanBuyer, KaplanSeller
from traders.ppo import PPOBuyer#, PPOSeller  # Example if you have PPO

# A dictionary keyed by (type_name, is_buyer_bool)
TRADER_REGISTRY = {
    ("zi", True):  ZIBuyer,
    ("zi", False): ZISeller,
    ("random",  True):  RandomBuyer,
    ("random",  False): RandomSeller,
    ("kaplan",  True):  KaplanBuyer,
    ("kaplan",  False): KaplanSeller,
    ("gdbuyer", True):  GDBuyer,
    ("gdseller",False): GDSeller,
    ("zic",True):  RandomBuyer,    
    ("zic",False):  RandomSeller,
    ("zip",True):  ZipBuyer,
    ("zip",False): ZipSeller,
    ("ppobuyer",True):  PPOBuyer,
    #("pposeller",False): PPOSeller,
    # Add new lines for new Traders...
}

def get_trader_class(type_str, is_buyer):
    """
    Return the Trader class from TRADER_REGISTRY.
    Raise an error if not found.
    """
    key = (type_str, is_buyer)
    if key not in TRADER_REGISTRY:
        raise ValueError(f"Trader type '{type_str}' with is_buyer={is_buyer} not in registry.")
    return TRADER_REGISTRY[key]
