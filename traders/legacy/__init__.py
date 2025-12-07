"""Legacy trading agents from the 1993 Santa Fe Tournament."""

from traders.legacy.bgan import BGAN
from traders.legacy.gd import GD
from traders.legacy.kaplan import Kaplan
from traders.legacy.ledyard import Ledyard
from traders.legacy.ringuette import Ringuette
from traders.legacy.skeleton import Skeleton
from traders.legacy.staecker import Staecker
from traders.legacy.zi2 import ZI2, ZIC2
from traders.legacy.zic import ZIC, ZIC1
from traders.legacy.zip import ZIP, ZIP1
from traders.legacy.zip2 import ZIP2

__all__ = [
    "BGAN",
    "GD",
    "Kaplan",
    "Ledyard",
    "Ringuette",
    "Skeleton",
    "Staecker",
    "ZIC",
    "ZIC1",
    "ZI2",
    "ZIC2",
    "ZIP",
    "ZIP1",
    "ZIP2",
]
