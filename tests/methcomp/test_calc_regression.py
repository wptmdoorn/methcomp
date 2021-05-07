# -*- coding: utf-8 -*-
import warnings

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from methcomp.calc_regression import calc_passing_bablok


@pytest.mark.parametrize(
    "y1, y2, ci, slope_expected, intercept_expected",
    [
        [
            # Check with R package mcr mcreg(method.reg="PaBa")
            # PB.reg <- mcreg(x1, x2, method.reg = "PaBa", alpha=0.05, method.ci="analytical")
            # Note: implementation differenecs doesn't allow for high accuracy here
            np.array(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            ),
            np.array(
                [
                    1.03,
                    2.05,
                    2.79,
                    3.67,
                    5.00,
                    5.82,
                    7.16,
                    7.69,
                    8.53,
                    10.38,
                    11.11,
                    12.17,
                    13.47,
                    13.83,
                    15.15,
                    16.12,
                    16.94,
                    18.09,
                    19.13,
                    19.54,
                ]
            ),
            0.95,
            (1.0050, 0.9848077, 1.0265805),
            (0.0125, -0.2975146, 0.1393271),
        ],
    ],
)
def test_calc_passing_bablok(y1, y2, ci, slope_expected, intercept_expected):
    slope, intercept = calc_passing_bablok(y1, y2, ci)
    np.testing.assert_allclose(slope, slope_expected, rtol=1e-2)
    np.testing.assert_allclose(intercept, intercept_expected, rtol=1e-1, atol=1e-2)
