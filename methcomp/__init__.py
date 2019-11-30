# Capture the original matplotlib rcParams
import matplotlib as mpl

_orig_rc_params = mpl.rcParams.copy()

# Import functions from all modules
from .blandaltman import *
from .regression import *

# Define version
__version__ = "0.2.0"
