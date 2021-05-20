# -*- coding: utf-8 -*-

"""Tests for regressors."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from methcomp.regressor import Linear, PassingBablok, Deming


@pytest.fixture
def method1():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


@pytest.fixture
def method2():
    return [
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


def test_check_params(method1, method2):
    # Check for exception when length missmatch
    with pytest.raises(ValueError):
        Linear(method1, method2[:-1])
    # Check for exception when CI is out of bounds
    with pytest.raises(ValueError):
        Linear(method1, method2, -100)


@pytest.mark.parametrize("model", (Deming, Linear, PassingBablok))
def test_calc_hi_lo(method1, method2, model):
    # Ensure result is ci_low < value < ci_high
    result = model(method1, method2).calculate()
    assert result["slope"][0] > result["slope"][1]
    assert result["slope"][0] < result["slope"][2]
    assert result["intercept"][0] > result["intercept"][1]
    assert result["intercept"][0] < result["intercept"][2]


@pytest.mark.parametrize(
    "model, CI, s, slo, shi, i, ilo, ihi",
    [      
        # Computed with R mcreg from mcr package
        # Note: some internal difference makes exact comparison impossible
        # mcreg(method1, method2, method.reg="LinReg", method.ci="analytical")
        (Linear,  0.95, 1.00570677, 0.9845263, 1.0268873, -0.07642105, -0.3301455, 0.1773034),
        # mcreg(method1, method2, method.reg="Deming", method.ci="bootstrap")
        (Deming,  0.95, 1.00662190, 0.988821, 1.0259598, -0.08602996, -0.2976630, 0.1048322),
        # mcreg(method1, method2, method.reg="PaBa", method.ci="analytical")
        (PassingBablok, 0.95, 1.0050,  0.9848077, 1.0265805, 0.0125, -0.2975146, 0.1393271),
    ],
)
def test_models(method1, method2, model, CI, s, slo, shi, i, ilo, ihi):
    result = model(method1, method2, CI=CI).calculate()
    # Expected
    np.testing.assert_allclose(result["slope"][:3], (s, slo, shi), rtol=1e-2)
    np.testing.assert_allclose(result["intercept"][:3], (i, ilo, ihi), atol=1e-1)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_plot_linear(method1, method2):
    fig, ax = plt.subplots(1, 1)
    ax = Linear(method1, method2).plot(ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_plot_no_line_references(method1, method2):
    fig, ax = plt.subplots(1, 1)
    ax = Linear(method1, method2).plot(ax=ax, line_reference=False)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_plot_no_line_CI(method1, method2):
    fig, ax = plt.subplots(1, 1)
    ax = Linear(method1, method2).plot(ax=ax, line_CI=False)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_plot_no_legend(method1, method2):
    fig, ax = plt.subplots(1, 1)
    ax = Linear(method1, method2).plot(ax=ax, legend=False)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_plot_square(method1, method2):
    fig, ax = plt.subplots(1, 1)
    ax = Linear(method1, method2).plot(ax=ax, square=True)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_plot_noaxis(method1, method2):
    # Cover case where ax must be created
    ax = Linear(method1, method2).plot()
    return plt.gcf()
