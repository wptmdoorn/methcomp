from methcomp import blandaltman
import matplotlib.pyplot as plt
import pytest
import numpy as np
from typing import Union


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


@pytest.mark.parametrize(
    "CI, m, mlo, mhi, l, llo, lhi, h, hlo, hhi",
    [
        (
            0.95,
            0.016499999999,
            -0.092549364,
            0.1255493648768,
            -0.471192547000,
            -0.6703265295,
            -0.2720585644,
            0.50419254700,
            0.3050585644,
            0.7033265295,
        ),
    ],
)
def test_blandaltman_compute(
    method1, method2, CI, m, mlo, mhi, l, llo, lhi, h, hlo, hhi
):
    results = blandaltman(method1, method2, CI=CI).compute()
    # Expected
    np.testing.assert_allclose(results["mean"], m, rtol=1e-5)
    np.testing.assert_allclose(results["mean_CI"], (mlo, mhi), rtol=1e-5)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_blandaltman_basic(method1, method2):
    fig, ax = plt.subplots(1, 1)
    blandaltman(method1, method2).plot(ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_blandaltman_basic_title(method1, method2):
    fig, ax = plt.subplots(1, 1)
    blandaltman(method1, method2).plot(graph_title="Test", ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_blandaltman_basic_without_CI(method1, method2):
    fig, ax = plt.subplots(1, 1)
    blandaltman(method1, method2, CI=None).plot(ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_blandaltman_basic_percentage(method1, method2):
    fig, ax = plt.subplots(1, 1)
    blandaltman(method1, method2, diff="percentage").plot(ax=ax)
    return fig
