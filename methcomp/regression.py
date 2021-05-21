# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, Tuple
import matplotlib
import numpy as np

__all__ = ["deming", "passingbablok", "linear"]
from .regressor import Deming, PassingBablok, Linear


def deming(
    method1: np.ndarray,
    method2: np.ndarray,
    CI: float = 0.95,
    vr: float = None,
    sdr: float = None,
    bootstrap: int = 1000,
    x_label: str = "Method 1",
    y_label: str = "Method 2",
    title: str = None,
    line_reference: bool = True,
    line_CI: bool = True,
    legend: bool = True,
    square: bool = False,
    ax: matplotlib.axes.Axes = None,
    point_kws: Optional[Dict] = None,
    color_regr: Optional[str] = None,
    alpha_regr: Optional[float] = None,
):
    """Provide a method comparison using Deming regression.

    This is an Axis-level function which will draw the Deming plot
    onto the current active Axis object unless ``ax`` is provided.

    Parameters
    ----------
    method1 : np.ndarray
        Values for method 1
    method2 : np.ndarray
        Values for method 2
    CI : float, optional
        The confidence interval employed in regression line (default=0.95)
    vr : float, optional
        The assumed known ratio of the (residual) variance of the ys
        relative to that of the xs.
        [default=1]
    sdr : float, optional
        The assumed known standard deviations. Parameter vr takes
        precedence if both are given.
        [default=1]
    bootstrap : int, optional
        Amount of bootstrap estimates that should be performed to acquire
        standard errors (and confidence intervals). If None, no bootstrap
        is performed.
        [default=1000]
    x_label : str, optional
        The label which is added to the X-axis. (default: "Method 1")
    y_label : str, optional
        The label which is added to the Y-axis. (default: "Method 2")
    title : str, optional
        Title of the regression plot. If None is provided, no title will be plotted.
    line_reference : bool, optional
        If True, a grey reference line at y=x will be plotted in the plot
        (default: True)
    line_CI : bool, optional
        If True, dashed lines will be plotted at the boundaries of the confidence intervals.
        (default: False)
    legend : bool, optional
        If True, will provide a legend containing the computed regression equation.
        (default: True)
    square : bool, optional
        If True, set the Axes aspect to "equal" so each cell will be
        square-shaped. (default: True)
    ax : matplotlib.axes.Axes, optional
        matplotlib axis object, if not passed, uses gca()
    point_kws : Optional[Dict], optional
        Additional keywords to plt
    color_regr : Optional[str], optional
        color for regression line and CI area
    alpha_regr : Optional[float], optional
        alpha for regression CI area

    Returns
    -------
    matplotlib.axes.Axes
        axes object with the Deming regression plot

    See Also
    -------
    regressor.Deming - class that implements Deming regression and plot


    References
    ----------
    .. [Koopmans_1937] Koopmans, T. C. (1937).
                       "Linear regression analysis of economic time series."
                       DeErven F. Bohn, Haarlem, Netherlands.
    .. [Deming_1943] Deming, W. E. (1943).
                     "Statistical adjustment of data."
                     Wiley, NY (Dover Publications edition, 1985).
    """

    return Deming(
        method1=method1, method2=method2, CI=CI, vr=vr, sdr=sdr, bootstrap=bootstrap
    ).plot(
        x_label=x_label,
        y_label=y_label,
        title=title,
        line_reference=line_reference,
        line_CI=line_CI,
        legend=legend,
        square=square,
        ax=ax,
        point_kws=point_kws,
        color_regr=color_regr,
        alpha_regr=alpha_regr,
    )


def passingbablok(
    method1: np.ndarray,
    method2: np.ndarray,
    CI: float = 0.95,
    x_label: str = "Method 1",
    y_label: str = "Method 2",
    title: str = None,
    line_reference: bool = True,
    line_CI: bool = True,
    legend: bool = True,
    square: bool = False,
    ax: matplotlib.axes.Axes = None,
    point_kws: Optional[Dict] = None,
    color_regr: Optional[str] = None,
    alpha_regr: Optional[float] = None,
):
    """Provide a method comparison using Passing-Bablok regression.

    This is an Axis-level function which will draw the Passing-Bablok plot
    onto the current active Axis object unless ``ax`` is provided.

    Parameters
    ----------
    method1 : np.ndarray
        Values for method 1
    method2 : np.ndarray
        Values for method 2
    CI : float, optional
        The confidence interval employed in regression line (default=0.95)
    x_label : str, optional
        The label which is added to the X-axis. (default: "Method 1")
    y_label : str, optional
        The label which is added to the Y-axis. (default: "Method 2")
    title : str, optional
        Title of the regression plot. If None is provided, no title will be plotted.
    line_reference : bool, optional
        If True, a grey reference line at y=x will be plotted in the plot
        (default: True)
    line_CI : bool, optional
        If True, dashed lines will be plotted at the boundaries of the confidence intervals.
        (default: False)
    legend : bool, optional
        If True, will provide a legend containing the computed regression equation.
        (default: True)
    square : bool, optional
        If True, set the Axes aspect to "equal" so each cell will be
        square-shaped. (default: True)
    ax : matplotlib.axes.Axes, optional
        matplotlib axis object, if not passed, uses gca()
    point_kws : Optional[Dict], optional
        Additional keywords to plt
    color_regr : Optional[str], optional
        color for regression line and CI area
    alpha_regr : Optional[float], optional
        alpha for regression CI area

    Returns
    -------
    matplotlib.axes.Axes
        axes object with the Passing-Bablok regression plot

    References
    ----------
    .. [passing_1983] Passing, H. and W. Bablok W.
                      "A New Biometrical Procedure for Testing the Equality of
                      Measurements from Two Different Analytical Methods."
                      J. Clin. Chem. Clin. Biochem 21 (1983): 709-720.
    .. [passing_1988] Bablok, W., et al. "A General Regression Procedure for
                      Method Transformation. Application of Linear Regression
                      Procedures for Method Comparison Studies in Clinical
                      Chemistry, Part III." Journal of clinical chemistry
                      and clinical biochemistry. Zeitschrift fur klinische
                      Chemie und klinische Biochemie 26.11 (1988): 783-790.

    See Also
    -------
    regressor.PassingBablok - class that implements Passing-Bablok regression and plot


    References
    ----------
    .. [passing_1983] Passing, H. and W. Bablok W.
                      "A New Biometrical Procedure for Testing the Equality of
                      Measurements from Two Different Analytical Methods."
                      J. Clin. Chem. Clin. Biochem 21 (1983): 709-720.
    .. [passing_1988] Bablok, W., et al. "A General Regression Procedure for
                      Method Transformation. Application of Linear Regression
                      Procedures for Method Comparison Studies in Clinical
                      Chemistry, Part III." Journal of clinical chemistry
                      and clinical biochemistry. Zeitschrift fur klinische
                      Chemie und klinische Biochemie 26.11 (1988): 783-790.
    """
    return PassingBablok(method1=method1, method2=method2, CI=CI).plot(
        x_label=x_label,
        y_label=y_label,
        title=title,
        line_reference=line_reference,
        line_CI=line_CI,
        legend=legend,
        square=square,
        ax=ax,
        point_kws=point_kws,
        color_regr=color_regr,
        alpha_regr=alpha_regr,
    )


def linear(
    method1: np.ndarray,
    method2: np.ndarray,
    CI: float = 0.95,
    x_label: str = "Method 1",
    y_label: str = "Method 2",
    title: str = None,
    line_reference: bool = True,
    line_CI: bool = True,
    legend: bool = True,
    square: bool = False,
    ax: matplotlib.axes.Axes = None,
    point_kws: Optional[Dict] = None,
    color_regr: Optional[str] = None,
    alpha_regr: Optional[float] = None,
):
    """Provide a method comparison using simple, linear regression.

    This is an Axis-level function which will draw the linear regression plot
    onto the current active Axis object unless ``ax`` is provided.

    Parameters
    ----------
    method1 : np.ndarray
        Values for method 1
    method2 : np.ndarray
        Values for method 2
    CI : float, optional
        The confidence interval employed in regression line (default=0.95)
    x_label : str, optional
        The label which is added to the X-axis. (default: "Method 1")
    y_label : str, optional
        The label which is added to the Y-axis. (default: "Method 2")
    title : str, optional
        Title of the regression plot. If None is provided, no title will be plotted.
    line_reference : bool, optional
        If True, a grey reference line at y=x will be plotted in the plot
        (default: True)
    line_CI : bool, optional
        If True, dashed lines will be plotted at the boundaries of the confidence intervals.
        (default: False)
    legend : bool, optional
        If True, will provide a legend containing the computed regression equation.
        (default: True)
    square : bool, optional
        If True, set the Axes aspect to "equal" so each cell will be
        square-shaped. (default: True)
    ax : matplotlib.axes.Axes, optional
        matplotlib axis object, if not passed, uses gca()
    point_kws : Optional[Dict], optional
        Additional keywords to plt
    color_regr : Optional[str], optional
        color for regression line and CI area
    alpha_regr : Optional[float], optional
        alpha for regression CI area

    Returns
    -------
    matplotlib.axes.Axes
        axes object with the linear regression plot

    See Also
    -------
    regressor.Linear - class that implements linear regression and plot
    """
    return Linear(method1=method1, method2=method2, CI=CI).plot(
        x_label=x_label,
        y_label=y_label,
        title=title,
        line_reference=line_reference,
        line_CI=line_CI,
        legend=legend,
        square=square,
        ax=ax,
        point_kws=point_kws,
        color_regr=color_regr,
        alpha_regr=alpha_regr,
    )
