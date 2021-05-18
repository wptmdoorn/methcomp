# -*- coding: utf-8 -*-

"""Implementation of regressors.
"""
from abc import ABC, abstractmethod
import math
from typing import Any, Dict, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import statsmodels.api as sm

from .comparer import Comparer


class Regressor(Comparer):
    """Method comparison regression baseclass.

    Attributes
    ----------
    CI : float
        The confidence interval employed in regression line
    DEFAULT_POINT_KWS : Dict[str, Any]
        Default keywords for method value scatter plot
    DEFAULT_REGRESSION_KWS : Dict[str, Any]
        Default keywords for regression line
    """

    DEFAULT_POINT_KWS = {"s": 20, "alpha": 0.6, "color": "#000000"}
    DEFAULT_REGRESSION_KWS = {"color": "#008bff", "alpha": 0.2}

    def __init__(self, method1: np.ndarray, method2: np.ndarray, CI: float = 0.95):
        """Construct a regressor.

        Parameters
        ----------
        method1 : np.ndarray
            Values for method 1
        method2 : np.ndarray
            Values for method 2
        CI : float, optional
            The confidence interval employed in regression line (default=0.95)
        """
        # Process args
        self.CI = CI
        super().__init__(method1, method2)

    def _check_params(self):
        """Check validity of parameters

        Raises
        ------
        ValueError
            If method values are of different shape or CI outside of range 0,1
        """
        super()._check_params()
        if self.CI is not None and (self.CI > 1 or self.CI < 0):
            raise ValueError("Confidence interval must be between 0 and 1.")

    @abstractmethod
    def calculate(self) -> Tuple[np.array]:
        """Calculate regression parameters.

        Fill in contents to self.result, should include
        - "slope": (value, ci_lower, ci_upper)
        - "intercept": (value, ci_lower, ci_upper)

        Note: Must set self.slope and self.intercept
        """
        super().calculate()

    def plot(
        self,
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
    ) -> matplotlib.axes.Axes:
        """Plot regression result

        Parameters
        ----------
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
        """
        ax = super().plot(ax=ax)

        # Set scatter plot keywords to defaults and apply override
        pkws = self.DEFAULT_POINT_KWS.copy()
        pkws.update(point_kws or {})

        # Get regression parameters
        slope = self.result["slope"]
        intercept = self.result["intercept"]

        # plot individual points
        ax.scatter(self.method1, self.method2, **pkws)

        # plot reference line
        if line_reference:
            ax.plot(
                [0, 1],
                [0, 1],
                label="Reference",
                color="grey",
                linestyle="--",
                transform=ax.transAxes,
            )

        # Compute x and y values
        xvals = np.array(ax.get_xlim())
        yvals = xvals[:, None] * slope + intercept

        # Plot regression line 0
        ax.plot(
            xvals,
            yvals[:, 0],
            label=f"{y_label} = {intercept[0]:.2f} + {slope[0]:.2f} * {x_label}",
            color=color_regr,
            linestyle="-",
        )

        # Plot confidence region
        if yvals.shape[1] > 2:
            ax.fill_between(
                xvals,
                yvals[:, 1],
                yvals[:, 2],
                color=color_regr or self.DEFAULT_REGRESSION_KWS["color"],
                alpha=alpha_regr or self.DEFAULT_REGRESSION_KWS["alpha"],
            )
            if line_CI:
                ax.plot(xvals, yvals[:, 1], linestyle="--")
                ax.plot(xvals, yvals[:, 2], linestyle="--")

        # Set axes labels
        ax.set(
            xlabel=x_label or "",
            ylabel=y_label or "",
            title=title or "",
        )

        if legend:
            ax.legend(loc="upper left", frameon=False)

        if square:
            ax.set_aspect("equal")
        return ax


class PassingBablok(Regressor):

    """Passing-Bablok Regressor

    Attributes
    ----------
    result : Dict[str, Any]
        regression result
    """

    def __init__(self, method1: np.ndarray, method2: np.ndarray, CI: float = 0.95):
        """Construct a Passing-Bablok Regressor

        Parameters
        ----------
        method1 : np.ndarray
            Values for method 1
        method2 : np.ndarray
            Values for method 2
        CI : float, optional
            The confidence interval employed in regression line (default=0.95)
        """
        super().__init__(method1, method2, CI)

    def calculate(self) -> Tuple[np.array]:
        """Calculate regression parameters.

        Returns
        -------
        Tuple[np.array]
            Description
        """
        # Define pair indices
        super().calculate()

        idx = np.array(np.triu_indices(self.n, 1))
        # Find pairwise differences for y1 and y2
        d1 = np.diff(self.method1[idx], axis=0)
        d2 = np.diff(self.method2[idx], axis=0)
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

        ci = st.norm.ppf((self.CI + 1) * 0.5) * math.sqrt(
            self.n * (self.n - 1) * (2 * self.n + 5) / 18
        )
        m1 = int((n - ci) // 2)
        m2 = n - m1 + 1

        slope = np.array((slope, S[k + m1], S[k + m2]))
        intercept = np.median(self.method2 - slope[:, None] * self.method1, axis=1)[
            [0, 2, 1]
        ]

        self.result = {"slope": slope, "intercept": intercept}

        return slope, intercept


class Deming(Regressor):

    """Deming Regressor

    Attributes
    ----------
    bootstrap : int
        Amount of bootstrap estimates that should be performed to acquire standard errors (and confidence
        intervals). If None, no bootstrapping is performed.
    sdr : float
        The assumed known standard deviations. Parameter vr takes precedence if both are given.
    vr : float
        The assumed known ratio of the (residual) variance of the ys relative to that of the xs.
    """

    def __init__(
        self,
        method1: np.ndarray,
        method2: np.ndarray,
        CI: float = 0.95,
        vr: float = None,
        sdr: float = None,
        bootstrap: int = 1000,
    ):
        """Construct a Deming Regressor

        Parameters
        ----------
        method1 : np.ndarray
            Values for method 1
        method2 : np.ndarray
            Values for method 2
        CI : float, optional
            The confidence interval employed in regression line
            [default=0.95]
        vr : float, optional
            The assumed known ratio of the (residual) variance of the ys relative to that of the xs.
            [default=1]
        sdr : float, optional
            The assumed known standard deviations. Parameter vr takes precedence if both are given.
            [default=1]
        bootstrap : int
            Amount of bootstrap estimates that should be performed to acquire standard errors (and confidence
            intervals). If None, no bootstrapping is performed.
            [default=1000]
        """
        self.vr = vr
        self.sdr = sdr
        self.bootstrap = bootstrap
        super().__init__(method1, method2, CI)

    def _check_params(self):
        """Check validity of parameters

        Raises
        ------
        ValueError
            If method values are of different shape or CI outside of range 0,1
        """
        if self.vr is not None < 0:
            raise ValueError("vr parameter must be positive or None")
        if self.sdr is not None < 0:
            raise ValueError("sdr parameter must be positive or None")
        if self.bootstrap is not None and self.bootstrap <= 0:
            raise ValueError("bootstrap parameter must be postivie or None")

    def calculate(self) -> Tuple[np.array]:
        """Calculate regression parameters."""

        def deming(n, x, y, lamb):
            axis = 1 if len(x.shape) > 1 else None
            mx = x.mean(axis=axis, keepdims=True)
            my = y.mean(axis=axis, keepdims=True)
            dx = x - mx
            dy = y - my
            sxx = np.sum(dx * dx, axis=axis, keepdims=True)
            syy = np.sum(dy * dy, axis=axis, keepdims=True)
            sxy = np.sum(dx * dy, axis=axis, keepdims=True)
            dxy = syy - lamb * sxx
            beta = (dxy + np.sqrt(dxy * dxy + 4 * lamb * sxy * sxy)) / (2 * sxy)
            alpha = my - beta * mx
            xi = (lamb * x + beta * (y - alpha)) / (lamb + beta * beta)
            dxxi = x - xi
            dyxi = y - alpha - beta * xi
            sigmasq = (
                lamb * np.sum(dxxi * dxxi, axis=axis, keepdims=True)
                + np.sum(dyxi * dyxi, axis=axis, keepdims=True)
            ) / (2 * lamb * (n - 2))
            sigmay = np.sqrt(lamb * sigmasq)
            sigmax = np.sqrt(sigmasq)
            return np.hstack((alpha, beta, sigmax, sigmay))

        super().calculate()

        _lambda = self.vr or self.sdr or 1

        if self.bootstrap is None:
            alpha, beta, sigmax, sigmay = deming(self.n, self.method1, self.method2, _lambda)
        else:
            # Perform bootstrap evaluation
            idx = np.random.choice(
                self.n, (self.bootstrap, self.n), replace=True
            )
            params = deming(
                self.n, np.take(self.method1, idx), np.take(self.method2, idx), _lambda
            )
            se = np.sqrt(np.var(np.cov(params.T), axis=1, ddof=1))
            t = np.transpose(
                np.apply_along_axis(
                    np.quantile, 0, params, [0.5, (1 - self.CI) / 2, 1 - (1 - self.CI) / 2]
                )
            )
            alpha = np.hstack((t[0, :], se[0]))
            beta = np.hstack((t[1, :], se[1]))
            sigmax = np.hstack((t[2, :], se[2]))
            sigmay = np.hstack((t[3, :], se[3]))

        self.result = {
            "intercept": alpha,
            "slope": beta,
            "sx": sigmax,
            "sy": sigmay,
        }
        return beta, alpha


class Linear(Regressor):

    """Linear Regressor"""

    def __init__(self, method1: np.ndarray, method2: np.ndarray, CI: float = 0.95):
        """Construct a Linear regressor

        Parameters
        ----------
        method1 : np.ndarray
            Values for method 1
        method2 : np.ndarray
            Values for method 2
        CI : float, optional
            The confidence interval employed in regression line (default=0.95)
        """
        super().__init__(method1, method2, CI)

    def calculate(self) -> Tuple[np.array]:
        """Calculate regression parameters."""
        super().calculate()

        _model = sm.OLS(self.method1, sm.add_constant(self.method2)).fit()
        _params = _model.params
        _confint = _model.conf_int(alpha=self.CI)
        intercept = np.array((_params[0], _confint[0][0], _confint[0][1]))
        slope = np.array((_params[1], _confint[1][0], _confint[1][1]))

        self.result = {"slope": slope, "intercept": intercept}

        return slope, intercept
