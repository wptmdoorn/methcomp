# Capture the original matplotlib rcParams
import matplotlib as mpl

_orig_rc_params = mpl.rcParams.copy()

# Import functions from all modules
from .blandaltman import *
from .regression import *
from .regressor import *
from .glucose import *

# Define version
__version__ = "1.0.0"
