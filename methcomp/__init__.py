# Capture the original matplotlib rcParams
import warnings
import matplotlib as mpl

_orig_rc_params = mpl.rcParams.copy()

# Import functions from all modules
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    from .blandaltman import *
    from .regression import *
    from .glucose import *

# Define version
__version__ = "1.0.0"
