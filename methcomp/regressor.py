# -*- coding: utf-8 -*-

"""Implementation of regressors.
"""
import math
from typing import Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from .comparer import Comparer

__all__ = ["Deming", "PassingBablok", "Linear"]


class Regressor(Comparer):
    """Method comparison regression base class.

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

    def __init__(
        self,
        method1: Union[List[float], np.ndarray],
        method2: Union[List[float], np.ndarray],
        CI: float = 0.95,
    ):
        """Construct a regressor.

        Parameters
        ----------
        method1 : Union[List[float], np.ndarray]
            Values for method 1
        method2 : Union[List[float], np.ndarray]
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
            If True, dashed lines will be plotted at the boundaries of the confidence
            intervals.
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
        ------------------
        matplotlib.axes.Axes
            axes object with the plot
        """
        ax = ax or plt.gca()

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
        Regression result with `slope` and `intercept`.
        Each of these contain
            `value`, `ci_low`, `ci_high`, and `SE`

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

    def __init__(
        self,
        method1: Union[List[float], np.ndarray],
        method2: Union[List[float], np.ndarray],
        CI: float = 0.95,
    ):
        """Construct a Passing-Bablok Regressor

        Parameters
        ----------
        method1 : Union[List[float], np.ndarray]
            Values for method 1
        method2 : Union[List[float], np.ndarray]
            Values for method 2
        CI : float, optional
            The confidence interval employed in regression line (default=0.95)
        """
        super().__init__(method1, method2, CI)

    def _calculate_impl(self):
        """Calculate regression parameters."""
        # Define pair indices
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
        # Compute CI
        ci = st.norm.ppf((self.CI + 1) * 0.5) * math.sqrt(
            self.n * (self.n - 1) * (2 * self.n + 5) / 18
        )
        m1 = int((n - ci) // 2)
        m2 = n - m1 + 1
        # Make slope and intercept as value, lower ci, upper ci
        slope = np.array((slope, S[k + m1], S[k + m2]))
        intercept = np.median(self.method2 - slope[:, None] * self.method1, axis=1)[
            [0, 2, 1]
        ]
        self._result = {"slope": slope, "intercept": intercept}


class Deming(Regressor):

    """Deming Regressor

    Attributes
    ----------
    bootstrap : int
        Amount of bootstrap estimates that should be performed to acquire
        standard errors (and confidence intervals).
    result : Dict[str, Any]
        Regression result with
            `slope`, `intercept`, `sx`, and `sy`
        if bootstrap > 0 each of these contain
            `value`, `ci_low`, `ci_high`, and `SE`
    sdr : float
        The assumed known standard deviations.
    vr : float
        The assumed known ratio of the (residual) variance of the ys relative
        to that of the xs.


    References
    ----------
    .. [Koopmans_1937] Koopmans, T. C. (1937).
                       "Linear regression analysis of economic time series."
                       DeErven F. Bohn, Haarlem, Netherlands.
    .. [Deming_1943] Deming, W. E. (1943).
                     "Statistical adjustment of data."
                     Wiley, NY (Dover Publications edition, 1985).
    """

    def __init__(
        self,
        method1: Union[List[float], np.ndarray],
        method2: Union[List[float], np.ndarray],
        CI: float = 0.95,
        vr: float = None,
        sdr: float = None,
        bootstrap: int = 1000,
    ):
        """Construct a Deming Regressor

        Parameters
        ----------
        method1 : Union[List[float], np.ndarray]
            Values for method 1
        method2 : Union[List[float], np.ndarray]
            Values for method 2
        CI : float, optional
            The confidence interval employed in regression line
            [default=0.95]
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
        super()._check_params()
        if self.vr is not None < 0:
            raise ValueError("vr parameter must be positive or None")
        if self.sdr is not None < 0:
            raise ValueError("sdr parameter must be positive or None")
        if self.bootstrap is not None and self.bootstrap <= 0:
            raise ValueError("bootstrap parameter must be postivie or None")

    def _calculate_impl(self):
        """Calculate regression parameters."""

        def _calc_deming(
            n: int, x: np.ndarray, y: np.ndarray, lamb: float
        ) -> np.ndarray:
            """Calculate deming regresison parameters

            Parameters
            ----------
            n : int
                Length of data
            x : np.ndarray
                method 1 data
            y : np.ndarray
                method 2 data
            lamb : float
                assummed variation

            Returns
            ------------------
            np.ndarray
                alpha, beta, sigmax, sigmay as columns in array
            """
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

        _lambda = self.vr or self.sdr or 1

        if self.bootstrap is None:
            # Non bootstrap evaluation - no CI computation
            result = _calc_deming(self.n, self.method1, self.method2, _lambda)[:, None]
        else:
            # Perform bootstrap evaluation
            idx = np.random.choice(self.n, (self.bootstrap, self.n), replace=True)
            params = _calc_deming(
                self.n, np.take(self.method1, idx), np.take(self.method2, idx), _lambda
            )
            # Compute standard errors of each column in params
            se = np.sqrt(np.var(np.cov(params.T), axis=1, ddof=1))

            # Calculate median, lower and upper CI
            t = np.quantile(
                params, axis=0, q=[0.5, (1 - self.CI) / 2, 1 - (1 - self.CI) / 2]
            ).T
            # Add SE column to median, low ci, high ci
            result = np.hstack((t, se[:, None]))

        # Form result
        self._result = {
            "intercept": result[0, :],
            "slope": result[1, :],
            "sx": result[2, :],
            "sy": result[3, :],
        }


class Linear(Regressor):

    """Linear Regressor

    Attributes
    ----------
    result : Dict[str, Any]
        Regression result with `slope` and `intercept` each consisting of
            `value`, `lower ci`, and `upper ci`.
        In addition `scipy.stats.linregress` outputs:
            `pvalue`, `rvalue`, `std_err`, `intercept_stderr`
        and `t-score` for confidence interval estimation

    """

    def __init__(
        self,
        method1: Union[List[float], np.ndarray],
        method2: Union[List[float], np.ndarray],
        CI: float = 0.95,
    ):
        """Construct a Linear regressor

        Uses `scipy.stats.linregress` internally


        Parameters
        ----------
        method1 : Union[List[float], np.ndarray]
            Values for method 1
        method2 : Union[List[float], np.ndarray]
            Values for method 2
        CI : float, optional
            The confidence interval employed in regression line (default=0.95)
        """
        super().__init__(method1, method2, CI)

    def _calculate_impl(self):
        """Calculate regression parameters."""

        # Use scipy.stats.linregress
        result = st.linregress(self.method1, self.method2)._asdict()

        # Hack to support scipy < 1.60
        if "intercept_stderr" not in result:
            result["intercept_stderr"] = result["stderr"] * np.sqrt(
                np.var(self.method1) + self.method1.mean() ** 2
            )

        ts = abs(st.t.ppf((1 - self.CI) / 2, df=self.n - 2))
        result.update(
            {
                "t-score": ts,
            }
        )

        # Calculate ci width from centre
        slope_ciw = ts * result["stderr"]
        intercept_ciw = ts * result["intercept_stderr"]

        # Put CI in slope and intercept
        result.update(
            {
                "slope": np.array(
                    [
                        result["slope"],
                        result["slope"] - slope_ciw,
                        result["slope"] + slope_ciw,
                    ]
                ),
                "intercept": np.array(
                    [
                        result["intercept"],
                        result["intercept"] - intercept_ciw,
                        result["intercept"] + intercept_ciw,
                    ]
                ),
            }
        )
        self._result = result
