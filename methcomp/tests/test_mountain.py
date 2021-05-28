# -*- coding: utf-8 -*-

"""Tests for mountain plot."""

import matplotlib.pyplot as plt
import pytest

from methcomp.mountain import Mountain, mountain


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


@pytest.mark.mpl_image_compare(tolerance=10)
def test_mountain_plot(method1, method2):
    fig, ax = plt.subplots(1, 1)
    ax = Mountain(method1, method2).plot(ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_mountain_plot_unit(method1, method2):
    fig, ax = plt.subplots(1, 1)
    ax = Mountain(method1, method2).plot(ax=ax, unit="X")
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_mountain_plot_no_iqr(method1, method2):
    fig, ax = plt.subplots(1, 1)
    ax = Mountain(method1, method2, iqr=0).plot(ax=ax)
    return fig


def test_mountain_median(method1, method2):
    median = -0.04
    m = Mountain(method1, method2)
    assert m.result["median"] == pytest.approx(median, rel=1e-2, abs=1e-2)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_mountain_deprecated(method1, method2):
    fig, ax = plt.subplots(1, 1)
    with pytest.deprecated_call():
        mountain(method1, method2, ax=ax)
    return fig


def test_mountain_auc(method1, method2):
    auc = 18.5
    m = Mountain(method1, method2)
    assert m.result["auc"] == pytest.approx(auc, rel=1e-2, abs=1e-2)


def test_mountain_iqr(method1, method2):
    iqr = 0.47
    m = Mountain(method1, method2)
    assert m.result["iqr"][1] - m.result["iqr"][0] == pytest.approx(
        iqr, rel=1e-2, abs=1e-2
    )


@pytest.mark.parametrize("n_percentiles", (50, 100, 1000))
def test_mountain_n_percentiles(method1, method2, n_percentiles):
    m = Mountain(method1, method2, n_percentiles=n_percentiles)
    assert len(m.result["mountain"]) == n_percentiles
    assert len(m.result["quantile"]) == n_percentiles


def test_mountain_bad_n(method1, method2):
    with pytest.raises(ValueError):
        Mountain(method1, method2, n_percentiles=-1)


def test_mountain_bad_iqr(method1, method2):
    with pytest.raises(ValueError):
        Mountain(method1, method2, iqr=-1)
