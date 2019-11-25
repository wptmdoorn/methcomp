import matplotlib.pyplot as plt
import matplotlib
import matplotlib.transforms as transforms
import numpy as np
from scipy import stats

__all__ = ["blandaltman"]


class _BlandAltman(object):
    """Internal class for drawing a Bland-Altman plot"""

    def __init__(self, x, y,
                 x_title, y_title, graph_title,
                 diff, limit_of_agreement, reference, CI,
                 color_mean, color_loa, color_points):
        # variables assignment
        self.x: np.array = np.asarray(x)
        self.y: np.array = np.asarray(y)
        self.n: float = len(x)
        self.diff_method: str = diff
        self.graph_title: str = graph_title
        self.x_title: str = x_title
        self.y_title: str = y_title
        self.loa: float = limit_of_agreement
        self.reference: bool = reference
        self.CI: float = CI
        self.color_mean: str = color_mean
        self.color_loa: str = color_loa
        self.color_points: str = color_points

        # check provided parameters
        self._check_params()

        # perform necessary calculations and processing
        self.mean: np.array = np.mean([self.x, self.y], axis=0)

        if diff == 'absolute':
            self.diff = self.x - self.y
        elif diff == 'percentage':
            self.diff = ((self.x - self.y) / self.mean) * 100
        else:
            self.diff = self.x - self.y

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
        if len(self.x) != len(self.y):
            raise ValueError('Length of X and Y are not equal.')

        if self.CI is not None and (self.CI > 1 or self.CI < 0):
            raise ValueError('Confidence interval must be between 0 and 1.')

        if self.diff_method not in ['absolute', 'percentage']:
            raise ValueError('The provided difference method must be either absolute or percentage.')

        if any([not isinstance(x, str) for x in [self.x_title, self.y_title]]):
            raise ValueError('Axes labels arguments should be provided as a str.')

    def plot(self, ax: matplotlib.axes.Axes):
        # individual points
        ax.scatter(self.mean, self.diff, s=20, alpha=0.6, color=self.color_points)

        # mean difference and SD lines
        ax.axhline(self.mean_diff, color=self.color_mean, linestyle='-')
        ax.axhline(self.mean_diff + self.loa_sd, color=self.color_loa, linestyle='--')
        ax.axhline(self.mean_diff - self.loa_sd, color=self.color_loa, linestyle='--')

        if self.reference:
            ax.axhline(0, color='grey', linestyle='-')

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

        # graph labels
        ax.set_ylabel(self.y_title)
        ax.set_xlabel(self.x_title)
        if self.graph_title is not None:
            ax.set_title(self.graph_title)


def blandaltman(x, y,
                x_label='Mean of methods', y_label='Difference between methods', title=None,
                diff='absolute', limit_of_agreement=1.96, reference=False, CI=0.95,
                color_mean='#008bff', color_loa='#FF7000', color_points='#000000',
                ax=None):
    plotter: _BlandAltman = _BlandAltman(x, y, x_label, y_label, title,
                                         diff, limit_of_agreement, reference, CI,
                                         color_mean, color_loa, color_points)

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()

    plotter.plot(ax)

    return ax