
from methcomp import regression, passingbablok, deming, linear
import numpy as np
import matplotlib.pyplot as plt
import pytest

method1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
method2 = [1.03, 2.05, 2.79, 3.67,
           5.00, 5.82, 7.16, 7.69,
           8.53, 10.38, 11.11, 12.17,
           13.47, 13.83, 15.15, 16.12,
           16.94, 18.09, 19.13, 19.54]


@pytest.mark.parametrize(
    "y1, y2, ci, slope_expected, intercept_expected",
    [
        [
            # Check with R package mcr mcreg(method.reg="PaBa")
            # PB.reg <- mcreg(x1, x2, method.reg = "PaBa", alpha=0.05, method.ci="analytical")
            # Note: implementation differenecs doesn't allow for high accuracy here
            np.array(method1),
            np.array(method2),
            0.95,
            (1.0050, 0.9848077, 1.0265805),
            (0.0125, -0.2975146, 0.1393271),
        ],
    ],
)
def test_calc_passing_bablok(y1, y2, ci, slope_expected, intercept_expected):
    slope, intercept = regression._PassingBablok(method1=y1, method2=y2,
                  x_label='Method 1', y_label='Method 2', title=None,
                  CI=ci, line_reference=True, line_CI=False, legend=True,
                  color_points='#000000', color_paba='#008bff',
                  point_kws=None)._derive_params()
    np.testing.assert_allclose(slope, slope_expected, rtol=1e-2)
    np.testing.assert_allclose(intercept, intercept_expected, rtol=1e-1, atol=1e-2)

@pytest.mark.mpl_image_compare(tolerance=10)
def test_passing_bablok_basic():
    fig, ax = plt.subplots(1, 1)
    passingbablok(method1, method2, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_passing_bablok_basic_title():
    fig, ax = plt.subplots(1, 1)
    passingbablok(method1, method2, title='Test', ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_passing_bablok_basic_with_ci():
    fig, ax = plt.subplots(1, 1)
    passingbablok(method1, method2, line_CI=True, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_passing_bablok_basic_squared():
    fig, ax = plt.subplots(1, 1)
    passingbablok(method1, method2, square=True, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_deming_basic():
    fig, ax = plt.subplots(1, 1)
    deming(method1, method2, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_deming_basic_title():
    fig, ax = plt.subplots(1, 1)
    deming(method1, method2, title='Test', ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_deming_basic_with_ci():
    fig, ax = plt.subplots(1, 1)
    deming(method1, method2, line_CI=True, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_deming_no_bootstrap():
    fig, ax = plt.subplots(1, 1)
    deming(method1, method2, bootstrap=None, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_deming_squared():
    fig, ax = plt.subplots(1, 1)
    deming(method1, method2, square=True, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_linear_basic():
    fig, ax = plt.subplots(1, 1)
    linear(method1, method2, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_linear_basic_title():
    fig, ax = plt.subplots(1, 1)
    linear(method1, method2, title='Test', ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_linear_with_ci():
    fig, ax = plt.subplots(1, 1)
    linear(method1, method2, line_CI=True, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_linear_squared():
    fig, ax = plt.subplots(1, 1)
    linear(method1, method2, square=True, ax=ax)
    return fig
