"""Legacy trading agents from the 1993 Santa Fe Tournament."""

from traders.legacy.zic import ZIC
from traders.legacy.kaplan import Kaplan
from traders.legacy.zip import ZIP
from traders.legacy.gd import GD
from traders.legacy.el import EL

__all__ = ["ZIC", "Kaplan", "ZIP", "GD", "EL"]
