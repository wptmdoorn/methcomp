# Capture the original matplotlib rcParams
import matplotlib as mpl

_orig_rc_params = mpl.rcParams.copy()

# Import functions from all modules
from .blandaltman import *

# Define version
__version__ = "0.0.1.dev0"
