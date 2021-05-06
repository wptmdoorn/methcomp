# -*- coding: utf-8 -*-
import math

import numpy as np
import scipy.stats as st


def calc_passing_bablok(y1, y2, CI):
    """Compute passing bablok slope and intercept

    Args:
        y1 (np.ndarray): Method 1 values
        y2 (np.ndarray): Method 2 values (same length as y1)
        CI (float): Confidence level, i.e 95%, 99%...

    Returns:
        Tuple[Tuple[float]]: slope, slope lower ci, slope upper ci, interval, interval lower ci, interval upper ci
    """
    m = len(y2)
    # Define pair indices
    idx = np.array(np.triu_indices(m, 1))
    # Find pairwise differences for y1 and y2
    d1 = np.diff(y1[idx], axis=0)
    d2 = np.diff(y2[idx], axis=0)
    # Avoid 0 division (nan if difference both 0, inf with sign of d2 if d1 is 0
    d2 = np.where(
        (d1 == d2) & (d1 == 0),
        np.nan,
        np.where(d1 == 0, np.where(d2 < 0, -np.inf, np.inf), d2),
    )
    # Ensure sign of 0s are correct
    S = np.where(d2 == 0, np.where(d1 < 0, np.NZERO, np.PZERO), d2 / d1)
    # Sort and drop nan
    S = np.sort(S[~np.isnan(S)])
    n = len(S)
    # Find half index of first element larger than 0
    k = np.argmax(S > 0) // 2
    if n % 2 == 1:
        # Use central element
        slope = S[(n + 1) // 2 + k]
    else:
        # Use geometric mean of central 2 elements
        slope = math.sqrt(S[n // 2 + k] * S[n // 2 + k + 1])

    ci = st.norm.ppf((CI + 1) * 0.5) * math.sqrt(m * (m - 1) * (2 * m + 5) / 18)
    m1 = int((n - ci) // 2)
    m2 = n - m1 + 1

    slope = (slope, S[k + m1], S[k + m2])
    intercept = (
        np.median(y2 - slope[0] * y1),
        np.median(y2 - slope[2] * y1),
        np.median(y2 - slope[1] * y1),
    )

    return slope, intercept
