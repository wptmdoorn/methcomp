import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import math
import numpy as np
from collections import Iterable

__all__ = ["deming", "passingbablok"]


class _Deming(object):
    """Internal class for drawing a Deming regression plot"""

    def __init__(self, method1, method2,
                 vr, sdr, bootstrap,
                 x_label, y_label, title,
                 CI, line_reference, line_CI, legend,
                 color_points, color_deming):
        self.method1: np.array = np.asarray(method1)
        self.method2: np.array = np.asarray(method2)
        self.vr = vr
        self.sdr = sdr
        self.bootstrap = bootstrap
        self.x_title = x_label
        self.y_title = y_label
        self.graph_title = title
        self.color_points = color_points
        self.color_deming = color_deming
        self.CI = CI
        self.line_reference = line_reference
        self.line_CI = line_CI
        self.legend = legend

        self._check_params()
        self._derive_params()

    def _check_params(self):
        if len(self.method1) != len(self.method2):
            raise ValueError('Length of method 1 and method 2 are not equal.')
        if self.bootstrap is not None and not isinstance(self.bootstrap, int):
            raise ValueError('Bootstrap argument should either be None or an integer.')
        if self.CI is not None and (self.CI > 1 or self.CI < 0):
            raise ValueError('Confidence interval must be between 0 and 1.')
        if any([not isinstance(x, str) for x in [self.x_title, self.y_title]]):
            raise ValueError('Axes labels arguments should be provided as a str.')

    def _derive_params(self):
        def _deming(x, y, lamb):
            ssdx = np.var(x, ddof=1) * (self.n - 1)
            ssdy = np.var(y, ddof=1) * (self.n - 1)
            spdxy = np.cov(x, y)[1][1] * (self.n - 1)

            beta = (ssdy - lamb * ssdx + math.sqrt((ssdy - lamb * ssdx) ** 2 + 4 * lamb * (ssdy ** 2))) / (
                    2 * spdxy)
            alpha = y.mean() - beta * x.mean()

            ksi = (lamb * x + beta * (y - alpha)) / (lamb + beta ** 2)
            sigmax = lamb * sum((x - ksi) ** 2) + sum((y - alpha - beta * ksi) ** 2) / (
                    (self.n - 2) * beta)
            sigmay = math.sqrt(lamb * sigmax)
            sigmax = math.sqrt(sigmax)

            return alpha, beta, sigmax, sigmay

        self.n = len(self.method1)
        if self.vr is not None:
            _lambda = self.vr
        elif self.sdr is not None:
            _lambda = self.sdr
        else:
            _lambda = 1

        params = _deming(self.method1, self.method2, _lambda)

        if self.bootstrap is None:
            self.alpha = params[0]
            self.beta = params[1]
            self.sigmax = params[2]
            self.sigmay = params[3]
        else:
            _params = np.zeros([self.bootstrap, 4])

            for i in range(self.bootstrap):
                idx = np.random.choice(range(self.n), self.n, replace=True)
                _params[i] = _deming(np.take(self.method1, idx), np.take(self.method2, idx), _lambda)

            _paramsdf = pd.DataFrame(_params, columns=['alpha', 'beta', 'sigmax', 'sigmay'])
            se = np.sqrt(np.diag(np.cov(_paramsdf.cov())))
            t = np.transpose(
                np.apply_along_axis(np.quantile, 0, _params, [0.5, (1 - self.CI) / 2, 1 - (1 - self.CI) / 2]))

            self.alpha = [t[0][0], se[0], t[0][1], t[0][2]]
            self.beta = [t[1][0], se[1], t[0][1], t[0][2]]
            self.sigmax = [t[2][0], se[2], t[0][1], t[0][2]]
            self.sigmay = [t[3][0], se[3], t[0][1], t[0][2]]

    def plot(self, ax):
        # plot individual points
        ax.scatter(self.method1, self.method2, s=20, alpha=0.6, color=self.color_points)

        # plot reference line
        if self.line_reference:
            ax.plot([0, 1], [0, 1], label='Reference',
                    color='grey', linestyle='--', transform=ax.transAxes)

        # plot Deming-line
        _xvals = np.array(ax.get_xlim())
        if self.bootstrap is None:
            _yvals = self.alpha + self.beta * _xvals
            ax.plot(_xvals, _yvals, label=f'{self.alpha:.2f} + {self.beta:.2f} * Method 1',
                    color=self.color_deming, linestyle='-')
        else:
            _yvals = [self.alpha[s] + self.beta[0] * _xvals for s in range(0, 4)]
            ax.plot(_xvals, _yvals[0], label=f'{self.alpha[0]:.2f} + {self.beta[0]:.2f} * Method 1',
                    color=self.color_deming, linestyle='-')
            ax.fill_between(_xvals, _yvals[2], _yvals[3], color=self.color_deming, alpha=0.2)
            if self.line_CI:
                ax.plot(_xvals, _yvals[2], linestyle='--')
                ax.plot(_xvals, _yvals[3], linestyle='--')

        if self.legend:
            ax.legend(loc='upper left', frameon=False)

        ax.set_ylabel(self.y_title)
        ax.set_xlabel(self.x_title)
        if self.graph_title is not None:
            ax.set_title(self.graph_title)


def deming(method1, method2,
           vr=None, sdr=None, bootstrap=1000,
           x_label='Method 1', y_label='Method 2', title=None,
           CI=0.95, line_reference=True, line_CI=False, legend=True,
           color_points='#000000', color_deming='#008bff',
           square=False, ax=None):
    """Provide a method comparison using Deming regression.

    This is an Axis-level function which will draw the Passing-Bablok plot
    onto the current active Axis object unless ``ax`` is provided.


    Parameters
    ----------
    method1, method2 : array, or list
        Values obtained from both methods, preferably provided in a np.array.
    vr : float
        The assumed known ratio of the (residual) variance of the ys relative to that of the xs.
        Defaults to 1.
    sdr : float
        The assumed known standard deviations. Parameter vr takes precedence if both are given.
        Defaults to 1.
    bootstrap : int
        Amount of bootstrap estimates that should be performed to acquire standard errors (and confidence
        intervals). If None, no bootstrapping is performed. Defaults to 1000.
    x_label : str, optional
        The label which is added to the X-axis. If None is provided, a standard
        label will be added.
    y_label : str, optional
        The label which is added to the Y-axis. If None is provided, a standard
        label will be added.
    title : str, optional
        Title of the Passing-Bablok plot. If None is provided, no title will be plotted.
    CI : float, optional
        The confidence interval employed in the mean difference and limit of agreement
        lines. Defaults to 0.95.
    line_reference : bool, optional
        If True, a grey reference line at y=x will be plotted in the plot.
        Defaults to true.
    line_CI : bool, optional
        If True, dashed lines will be plotted at the boundaries of the confidence intervals.
        Defaults to false.
    legend : bool, optional
        If True, will provide a legend containing the computed Passing-Bablok equation.
        Defaults to true.
    color_points : str, optional
        Color of the individual differences that will be plotted.
        Color should be provided in format compatible with matplotlib.
    color_deming : str, optional
        Color of the mean difference line that will be plotted.
        Color should be provided in format compatible with matplotlib.
    square : bool, optional
        If True, set the Axes aspect to "equal" so each cell will be
        square-shaped.
    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the Bland-Altman plot.

    See Also
    -------
    Koopmans, T. C. (1937). Linear regression analysis of economic time series. DeErven F. Bohn, Haarlem, Netherlands.
    Deming, W. E. (1943). Statistical adjustment of data. Wiley, NY (Dover Publications edition, 1985).
    """

    plotter: _Deming = _Deming(method1, method2,
                               vr, sdr, bootstrap,
                               x_label, y_label, title,
                               CI, line_reference, line_CI, legend,
                               color_points, color_deming)

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()

    if square:
        ax.set_aspect('equal')

    plotter.plot(ax)

    return ax


class _PassingBablok(object):
    """Internal class for drawing a Passing-Bablok regression plot"""

    def __init__(self, method1, method2,
                 x_label, y_label, title,
                 CI, line_reference, line_CI, legend,
                 color_points, color_paba):
        self.method1: np.array = np.asarray(method1)
        self.method2: np.array = np.asarray(method2)
        self.x_title = x_label
        self.y_title = y_label
        self.graph_title = title
        self.CI = CI
        self.color_points = color_points
        self.color_paba = color_paba
        self.line_reference = line_reference
        self.line_CI = line_CI
        self.legend = legend

        self._check_params()
        self._derive_params()

    def _check_params(self):
        if len(self.method1) != len(self.method2):
            raise ValueError('Length of method 1 and method 2 are not equal.')
        if self.CI is not None and (self.CI > 1 or self.CI < 0):
            raise ValueError('Confidence interval must be between 0 and 1.')
        if any([not isinstance(x, str) for x in [self.x_title, self.y_title]]):
            raise ValueError('Axes labels arguments should be provided as a str.')

    def _derive_params(self):
        self.n = len(self.method1)
        self.sv = []

        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                self.sv.append((self.method2[i] - self.method2[j]) /
                               (self.method1[i] - self.method1[j]))

        self.sv.sort()
        n = len(self.sv)
        k = math.floor(len([a for a in self.sv if a < 0]) / 2)

        if n % 2 == 1:
            self.slope = self.sv[int((n + 1) / k + 2)]
        else:
            self.slope = math.sqrt(self.sv[int(n / 2 + k)] * self.sv[int(n / 2 + k + 1)])

        _ci = st.norm.ppf(1 - (1 - self.CI) / 2) * math.sqrt((self.n * (self.n - 1) * (2 * self.n + 5)) / 18)
        _m1 = int(round((n - _ci) / 2))
        _m2 = n - _m1 - 1

        self.slope = [self.slope, self.sv[k + _m1], self.sv[k + _m2]]
        self.intercept = [np.median(self.method2 - self.slope[0] * self.method1),
                          np.median(self.method2 - self.slope[1] * self.method1),
                          np.median(self.method2 - self.slope[2] * self.method1)]

    def plot(self, ax):
        # plot individual points
        ax.scatter(self.method1, self.method2, s=20, alpha=0.6, color=self.color_points)

        # plot reference line
        if self.line_reference:
            ax.plot([0, 1], [0, 1], label='Reference',
                    color='grey', linestyle='--', transform=ax.transAxes)

        # plot PaBa-line
        _xvals = np.array(ax.get_xlim())
        _yvals = [self.intercept[s] + self.slope[s] * _xvals for s in range(0, 3)]
        ax.plot(_xvals, _yvals[0], label=f'{self.intercept[0]:.2f} + {self.slope[0]:.2f} * Method 1',
                color=self.color_paba, linestyle='-')
        ax.fill_between(_xvals, _yvals[1], _yvals[2], color=self.color_paba, alpha=0.2)
        if self.line_CI:
            ax.plot(_xvals, _yvals[1], linestyle='--')
            ax.plot(_xvals, _yvals[2], linestyle='--')

        if self.legend:
            ax.legend(loc='upper left', frameon=False)

        ax.set_ylabel(self.y_title)
        ax.set_xlabel(self.x_title)
        if self.graph_title is not None:
            ax.set_title(self.graph_title)


def passingbablok(method1, method2,
                  x_label='Method 1', y_label='Method 2', title=None,
                  CI=0.95, line_reference=True, line_CI=False, legend=True,
                  color_points='#000000', color_paba='#008bff',
                  square=False, ax=None):
    """Provide a method comparison using Passing-Bablok regression.

    This is an Axis-level function which will draw the Passing-Bablok plot
    onto the current active Axis object unless ``ax`` is provided.


    Parameters
    ----------
    method1, method2 : array, or list
        Values obtained from both methods, preferably provided in a np.array.
    x_label : str, optional
        The label which is added to the X-axis. If None is provided, a standard
        label will be added.
    y_label : str, optional
        The label which is added to the Y-axis. If None is provided, a standard
        label will be added.
    title : str, optional
        Title of the Passing-Bablok plot. If None is provided, no title will be plotted.
    CI : float, optional
        The confidence interval employed in the mean difference and limit of agreement
        lines. Defaults to 0.95.
    line_reference : bool, optional
        If True, a grey reference line at y=x will be plotted in the plot.
        Defaults to true.
    line_CI : bool, optional
        If True, dashed lines will be plotted at the boundaries of the confidence intervals.
        Defaults to false.
    legend : bool, optional
        If True, will provide a legend containing the computed Passing-Bablok equation.
        Defaults to true.
    color_points : str, optional
        Color of the individual differences that will be plotted.
        Color should be provided in format compatible with matplotlib.
    color_paba : str, optional
        Color of the mean difference line that will be plotted.
        Color should be provided in format compatible with matplotlib.
    square : bool, optional
        If True, set the Axes aspect to "equal" so each cell will be
        square-shaped.
    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the Bland-Altman plot.

    See Also
    -------
    Passing H and Bablok W. J Clin Chem Clin Biochem, vol. 21, no. 11, 1983, pp. 709 - 720
    """

    plotter: _PassingBablok = _PassingBablok(method1, method2,
                                             x_label, y_label, title,
                                             CI, line_reference, line_CI, legend,
                                             color_points, color_paba)

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()

    if square:
        ax.set_aspect('equal')

    plotter.plot(ax)

    return ax
