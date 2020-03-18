from methcomp import clarke, parkes
import matplotlib.pyplot as plt
import pytest

reference = [4.6, 13.73, 16.09, 17.16, 18.69, 19.48, 19.56, 20.76, 26.82, 27.95]
test = [1.11, 2.04, 7.5, 7.87, 14.85, 15.76, 17.63, 21.08, 21.29, 29.6]


@pytest.mark.mpl_image_compare(tolerance=10)
def test_clarke_basic():
    fig, ax = plt.subplots(1, 1)
    clarke(reference, test, units='mmol', ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_clarke_basic_title():
    fig, ax = plt.subplots(1, 1)
    clarke(reference, test, units='mmol', title='Test', ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_clarke_basic_without_grid():
    fig, ax = plt.subplots(1, 1)
    clarke(reference, test, units='mmol', grid=False, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_clarke_no_percentages():
    fig, ax = plt.subplots(1, 1)
    clarke(reference, test, units='mmol', percentage=False, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_clarke_no_mgdl():
    fig, ax = plt.subplots(1, 1)
    _ref = [x*18 for x in reference]
    _test = [x*18 for x in test]
    clarke(_ref, _test, units='mgdl', percentage=False, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_clarke_squared():
    fig, ax = plt.subplots(1, 1)
    clarke(reference, test, units='mmol', square=True, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_parkes_basic_t1d():
    fig, ax = plt.subplots(1, 1)
    parkes(1, reference, test, units='mmol', ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_parkes_basic_t2d():
    fig, ax = plt.subplots(1, 1)
    parkes(2, reference, test, units='mmol', ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_parkes_basic_title():
    fig, ax = plt.subplots(1, 1)
    parkes(1, reference, test, units='mmol', title='Test', ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_parkes_basic_without_grid():
    fig, ax = plt.subplots(1, 1)
    parkes(1, reference, test, units='mmol', grid=False, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_parkes_no_percentages():
    fig, ax = plt.subplots(1, 1)
    parkes(1, reference, test, units='mmol', percentage=False, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_parkes_no_mgdl():
    fig, ax = plt.subplots(1, 1)
    _ref = [x*18 for x in reference]
    _test = [x*18 for x in test]
    parkes(1, _ref, _test, units='mgdl', percentage=False, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_parkes_squared():
    fig, ax = plt.subplots(1, 1)
    parkes(1, reference, test, units='mmol', square=True, ax=ax)
    return fig
