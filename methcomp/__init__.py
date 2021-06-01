# -*- coding: utf-8 -*-
import matplotlib as mpl

# Import functions from all modules
from .blandaltman import blandaltman, BlandAltman
from .glucose import clarke, clarkezones, parkes, parkeszones, seg, segscores
from .regression import deming, linear, passingbablok
from .regressor import Deming, Linear, PassingBablok

# Capture the original matplotlib rcParams
_orig_rc_params = mpl.rcParams.copy()

# Define version
__version__ = "1.0.0"
