import matplotlib.pyplot as plt
import matplotlib
import matplotlib.transforms as transforms
import numpy as np
from scipy import stats
from typing import Tuple

__all__ = ["blandaltman"]


class _BlandAltman(object):
    """Internal class for drawing a Bland-Altman plot"""

    def __init__(self, method1, method2,
                 diff, limit_of_agreement, CI):
        # variables assignment
        self.method1: np.array = np.asarray(method1)
        self.method2: np.array = np.asarray(method2) 
        self.diff_method: str = diff
        self.CI: float = CI
        self.loa: float = limit_of_agreement

        # check provided parameters
        self._check_params()

    def compute(self) -> dict:
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

        self._output = {'mean': self.mean_diff,
                       'mean_CI': self.CI_mean if self.CI else None, 
                       'loa_lower': self.mean_diff - self.loa_sd,
                       'loa_lower_CI': self.CI_lower if self.CI else None,
                       'loa_upper': self.mean_diff + self.loa_sd,
                       'loa_upper_CI': self.CI_upper if self.CI else None,
                       }
        
        return self._output

    def _check_params(self):
        if len(self.method1) != len(self.method2):
            raise ValueError('Length of method 1 and method 2 are not equal.')

        if self.CI is not None and (self.CI > 1 or self.CI < 0):
            raise ValueError('Confidence interval must be between 0 and 1.')

        if self.diff_method not in ['absolute', 'percentage']:
            raise ValueError('The provided difference method must be either absolute or percentage.')

        #if any([not isinstance(x, str) for x in [self.x_title, self.y_title]]):
        #    raise ValueError('Axes labels arguments should be provided as a str.')
        
        
    def plot(self,
             x_title='Mean of methods', y_title='Difference between methods', graph_title=None,
             reference=False, xlim=None, ylim=None,
             color_mean='#008bff', color_loa='#FF7000', color_points='#000000', point_kws=None,
             ax: matplotlib.axes.Axes=None):
        
        if not(hasattr(self, '_output')):
            self.compute()
        
        point_kws: dict = {} if point_kws is None else point_kws.copy()
        ax = plt.gca()
        
        # individual points
        ax.scatter(self.mean, self.diff, s=20, alpha=0.6, color=color_points,
                   **point_kws)

        # mean difference and SD lines
        ax.axhline(self.mean_diff, color=color_mean, linestyle='-')
        ax.axhline(self.mean_diff + self.loa_sd, color=color_loa, linestyle='--')
        ax.axhline(self.mean_diff - self.loa_sd, color=color_loa, linestyle='--')

        if reference:
            ax.axhline(0, color='grey', linestyle='-', alpha=0.4)

        # confidence intervals (if requested)
        if self.CI is not None:
            ax.axhspan(self.CI_mean[0],  self.CI_mean[1], color=color_mean, alpha=0.2)
            ax.axhspan(self.CI_upper[0], self.CI_upper[1], color=color_loa, alpha=0.2)
            ax.axhspan(self.CI_lower[0], self.CI_lower[1], color=color_loa, alpha=0.2)

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
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

        # graph labels
        ax.set_ylabel(y_title)
        ax.set_xlabel(x_title)
        if graph_title is not None:
            ax.set_title(graph_title)


def blandaltman(method1, method2,
                diff='absolute', limit_of_agreement=1.96, CI=0.95) -> _BlandAltman:
    """Provide a method comparison using Bland-Altman. 

    This functions creates a BlandAltman class which can be used to
    access the statistics (using .statistics()) or to generate a plot
    with additional arguments (using .plots()).


    Parameters
    ----------
    method1, method2 : array, or list
        Values obtained from both methods, preferably provided in a np.array.
    diff : "absolute"  or "percentage"
        The difference to display, whether it is an absolute one or a percentual one.
        If None is provided, it defaults to absolute.
    limit_of_agreement : float, optional
        Multiples of the standard deviation to plot the limit of afgreement bounds at.
        This defaults to 1.96.
    CI : float, optional
        The confidence interval employed in the mean difference and limit of agreement
        lines. Defaults to 0.95.

    Returns
    -------
    _BlandAltman : class object containing the statistics and plot functionality

    See Also
    -------
    pyCompare package on github
    Altman, D. G., and Bland, J. M. Series D (The Statistician), vol. 32, no. 3, 1983, pp. 307–317.
    Altman, D. G., and Bland, J. M. Statistical Methods in Medical Research, vol. 8, no. 2, 1999, pp. 135–160.

    """

    ba: _BlandAltman = _BlandAltman(method1, method2, diff, limit_of_agreement, CI)

    # Return BlandAltman
    return ba