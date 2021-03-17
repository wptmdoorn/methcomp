import matplotlib.pyplot as plt
import matplotlib
import matplotlib.transforms as transforms
import numpy as np
from scipy import stats

__all__ = ["blandaltman"]


class _BlandAltman(object):
    """Internal class for drawing a Bland-Altman plot"""

    def __init__(self, method1, method2,
                 x_title, y_title, graph_title,
                 diff, limit_of_agreement, reference, CI,
                 xlim, ylim,
                 color_mean, color_loa, color_points,
                 point_kws):
        # variables assignment
        self.method1: np.array = np.asarray(method1)
        self.method2: np.array = np.asarray(method2)
        self.diff_method: str = diff
        self.graph_title: str = graph_title
        self.x_title: str = x_title
        self.y_title: str = y_title
        self.loa: float = limit_of_agreement
        self.reference: bool = reference
        self.CI: float = CI
        self.xlim: list = xlim
        self.ylim: list = ylim
        self.color_mean: str = color_mean
        self.color_loa: str = color_loa
        self.color_points: str = color_points
        self.point_kws: dict = {} if point_kws is None else point_kws.copy()

        # check provided parameters
        self._check_params()
        self._derive_params()

    def _derive_params(self):
        # perform necessary calculations and processing
        self.n: float = len(self.method1)
        self.mean: np.array = np.mean([self.method1, self.method2], axis=0)

        if self.diff_method == 'absolute':
            self.diff = self.method1 - self.method2
        elif self.diff_method == 'percentage':
            self.diff = ((self.method1 - self.method2) / self.mean) * 100
        else:
            self.diff = self.method1 - self.method2

        self.mean_diff = np.mean(self.diff)
        self.sd_diff = np.std(self.diff, axis=0)
        self.loa_sd = self.loa * self.sd_diff

        if self.CI is not None:
            self.CI_mean = stats.norm.interval(alpha=self.CI, loc=self.mean_diff,
                                               scale=self.sd_diff / np.sqrt(self.n))
            se_loa = (1.71 ** 2) * ((self.sd_diff**2) / self.n)
            conf_loa = np.sqrt(se_loa) * stats.t.ppf(q=(1 - self.CI) / 2., df=self.n - 1)
            self.CI_upper = [self.mean_diff + self.loa_sd + conf_loa,
                             self.mean_diff + self.loa_sd - conf_loa]
            self.CI_lower = [self.mean_diff - self.loa_sd + conf_loa,
                             self.mean_diff - self.loa_sd - conf_loa]

    def _check_params(self):
        if len(self.method1) != len(self.method2):
            raise ValueError('Length of method 1 and method 2 are not equal.')

        if self.CI is not None and (self.CI > 1 or self.CI < 0):
            raise ValueError('Confidence interval must be between 0 and 1.')

        if self.diff_method not in ['absolute', 'percentage']:
            raise ValueError('The provided difference method must be either absolute or percentage.')

        if any([not isinstance(x, str) for x in [self.x_title, self.y_title]]):
            raise ValueError('Axes labels arguments should be provided as a str.')

    def plot(self, ax: matplotlib.axes.Axes):
        # individual points
        ax.scatter(self.mean, self.diff, s=20, alpha=0.6, color=self.color_points,
                   **self.point_kws)

        # mean difference and SD lines
        ax.axhline(self.mean_diff, color=self.color_mean, linestyle='-')
        ax.axhline(self.mean_diff + self.loa_sd, color=self.color_loa, linestyle='--')
        ax.axhline(self.mean_diff - self.loa_sd, color=self.color_loa, linestyle='--')

        if self.reference:
            ax.axhline(0, color='grey', linestyle='-', alpha=0.4)

        # confidence intervals (if requested)
        if self.CI is not None:
            ax.axhspan(self.CI_mean[0],  self.CI_mean[1], color=self.color_mean, alpha=0.2)
            ax.axhspan(self.CI_upper[0], self.CI_upper[1], color=self.color_loa, alpha=0.2)
            ax.axhspan(self.CI_lower[0], self.CI_lower[1], color=self.color_loa, alpha=0.2)

        # text in graph
        trans: matplotlib.transform = transforms.blended_transform_factory(
            ax.transAxes, ax.transData)
        offset: float = (((self.loa * self.sd_diff) * 2) / 100) * 1.2
        ax.text(0.98, self.mean_diff + offset, 'Mean', ha="right", va="bottom", transform=trans)
        ax.text(0.98, self.mean_diff - offset, f'{self.mean_diff:.2f}', ha="right", va="top", transform=trans)
        ax.text(0.98, self.mean_diff + self.loa_sd + offset,
                f'+{self.loa:.2f} SD', ha="right", va="bottom", transform=trans)
        ax.text(0.98, self.mean_diff + self.loa_sd - offset,
                f'{self.mean_diff + self.loa_sd:.2f}', ha="right", va="top", transform=trans)
        ax.text(0.98, self.mean_diff - self.loa_sd - offset,
                f'-{self.loa:.2f} SD', ha="right", va="top", transform=trans)
        ax.text(0.98, self.mean_diff - self.loa_sd + offset,
                f'{self.mean_diff - self.loa_sd:.2f}', ha="right", va="bottom", transform=trans)

        # transform graphs
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # set X and Y limits
        if self.xlim is not None:
            ax.set_xlim(self.xlim[0], self.xlim[1])
        if self.ylim is not None:
            ax.set_ylim(self.ylim[0], self.ylim[1])

        # graph labels
        ax.set_ylabel(self.y_title)
        ax.set_xlabel(self.x_title)
        if self.graph_title is not None:
            ax.set_title(self.graph_title)


def blandaltman(method1, method2,
                x_label='Mean of methods', y_label='Difference between methods', title=None,
                diff='absolute', limit_of_agreement=1.96, reference=False, CI=0.95,
                xlim=None, ylim=None,
                color_mean='#008bff', color_loa='#FF7000', color_points='#000000',
                point_kws=None,
                ax=None):
    """Provide a method comparison using Bland-Altman plotting.

    This is an Axis-level function which will draw the Bland-Altman plot
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
        Title of the Bland-Altman plot. If None is provided, no title will be plotted.
    diff : "absolute"  or "percentage"
        The difference to display, whether it is an absolute one or a percentual one.
        If None is provided, it defaults to absolute.
    limit_of_agreement : float, optional
        Multiples of the standard deviation to plot the limit of afgreement bounds at.
        This defaults to 1.96.
    reference : bool, optional
        If True, a grey reference line at y=0 will be plotted in the Bland-Altman.
    CI : float, optional
        The confidence interval employed in the mean difference and limit of agreement
        lines. Defaults to 0.95.
    xlim : list, optional
        Minimum and maximum limits for X-axis. Should be provided as list or tuple.
        If not set, matplotlib will decide its own bounds.
    ylim : list, optional
        Minimum and maximum limits for Y-axis. Should be provided as list or tuple.
        If not set, matplotlib will decide its own bounds.
    color_mean : str, optional
        Color of the mean difference line that will be plotted.
    color_loa : str, optional
        Color of the limit of agreement lines that will be plotted.
    color_points : str, optional
        Color of the individual differences that will be plotted.
    point_kws : dict of key, value mappings, optional
        Additional keyword arguments for `plt.scatter`.
    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the Bland-Altman plot.

    See Also
    -------
    pyCompare package on github
    Altman, D. G., and Bland, J. M. Series D (The Statistician), vol. 32, no. 3, 1983, pp. 307–317.
    Altman, D. G., and Bland, J. M. Statistical Methods in Medical Research, vol. 8, no. 2, 1999, pp. 135–160.

    """

    plotter: _BlandAltman = _BlandAltman(method1, method2, x_label, y_label, title,
                                         diff, limit_of_agreement, reference, CI,
                                         xlim, ylim,
                                         color_mean, color_loa, color_points,
                                         point_kws)

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()

    plotter.plot(ax)

    return ax