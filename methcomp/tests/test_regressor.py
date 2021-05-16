# -*- coding: utf-8 -*-

"""Tests for regressors."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from methcomp.regressor import Linear

@pytest.fixture
def method1():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

@pytest.fixture
def method2():
    return [1.03, 2.05, 2.79, 3.67, 5.00, 5.82, 7.16, 7.69, 8.53, 10.38, 11.11, 12.17, 13.47, 13.83, 15.15, 16.12, 16.94, 18.09, 19.13, 19.54]

def test_check_params(method1, method2):
    # Check for exception when length missmatch
    with pytest.raises(ValueError):
        Linear(method1, method2[:-1])
    # Check for exception when CI is out of bounds
    with pytest.raises(ValueError):
        Linear(method1, method2, -100)

@pytest.mark.parametrize("CI, s, slo, shi, i, ilo, ihi",[
    (0.90, 0.99253036, 0.99126234, 0.99379838, 0.09480798, 0.07961312, 0.11000284),
    (0.95, 0.99253036, 0.99189771, 0.99316301, 0.09480798, 0.08722681, 0.10238915),
    (0.99, 0.99253036, 0.99240391, 0.99265680, 0.09480798, 0.09329278, 0.09632318),
])
def test_calc_linear(method1, method2, CI, s, slo, shi, i, ilo, ihi):
    slope, intercept = Linear(method1, method2, CI=CI).calculate()
    # Expected
    np.testing.assert_allclose(slope, (s, slo, shi))
    np.testing.assert_allclose(intercept, (i, ilo, ihi))

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
