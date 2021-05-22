# -*- coding: utf-8 -*-

"""Tests for comparer."""

import numpy as np
import pytest

from methcomp.comparer import Comparer


@pytest.fixture
def method1():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


@pytest.fixture
def dummy(method1):
    class DummyComparer(Comparer):
        def _calculate_impl(self):
            self.result["dummy"] = "dummy"

        def plot(self):
            return super().plot()

    return DummyComparer(method1, method1)


def test_comparer(dummy, method1):
    assert not dummy.calculated
    assert len(dummy.result) == 0
    assert dummy.n == len(method1)
    assert isinstance(dummy.method1, np.ndarray)


def test_calculate(dummy):
    ax = dummy.plot()
    assert dummy.calculate()
    assert dummy.result["dummy"] == "dummy"


def test_plot(dummy):
    ax = dummy.plot()
    assert dummy.calculated
    assert ax is not None
