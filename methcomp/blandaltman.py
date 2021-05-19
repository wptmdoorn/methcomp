import matplotlib.pyplot as plt
import matplotlib
import matplotlib.transforms as transforms
import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any, Union, List

__all__ = ["blandaltman"]


class BlandAltman:
    """Class for drawing a Bland-Altman plot"""

    DEFAULT_POINTS_KWS = {"s": 20, "alpha": 0.6, "color": "#000000"}

    def __init__(
        self,
        method1: Union[List[float], np.ndarray],
        method2: Union[List[float], np.ndarray],
        diff: str,
        limit_of_agreement: float,
        CI: float,
    ):
        """
        Initialize a Bland-Altman class object. This class is center to the calculate
        and compute functionality of a Bland-Altman comparison.

        Parameters
        ----------
        method1, method2: Union[List[float], np.ndarray]
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
        """

        # variables assignment
        self.computed = False
        self.method1: np.array = np.asarray(method1)
        self.method2: np.array = np.asarray(method2)
        self.diff_method: str = diff
        self.CI: float = CI
        self.loa: float = limit_of_agreement

        # check provided parameters
        self._check_params()

    def compute(self) -> Dict[str, Any]:
        """Calculates the statistics for method comparison using
        Bland-Altman plotting. Returns a dictionary with the results.

        Parameters
        ----------
        None

        Returns
        ----------
        Dict[str, Any] : a dictionary containing the mean and limit of agreement values and
                         their confidence intervals
        """

        self.n: float = len(self.method1)
        self.mean: np.array = np.mean([self.method1, self.method2], axis=0)

        self.diff = self.method1 - self.method2
        if self.diff_method == "percentage":
            self.diff = self.diff * 100 / self.mean

        self.mean_diff = np.mean(self.diff)
        self.sd_diff = np.std(self.diff, axis=0)
        self.loa_sd = self.loa * self.sd_diff

        if self.CI is not None:
            self.CI_mean = stats.norm.interval(
                alpha=self.CI, loc=self.mean_diff, scale=self.sd_diff / np.sqrt(self.n)
            )
            se_loa = (1.71 ** 2) * ((self.sd_diff ** 2) / self.n)
            conf_loa = np.sqrt(se_loa) * stats.t.ppf(
                q=(1 - self.CI) / 2.0, df=self.n - 1
            )
            self.CI_upper = [
                self.mean_diff + self.loa_sd + conf_loa,
                self.mean_diff + self.loa_sd - conf_loa,
            ]
            self.CI_lower = [
                self.mean_diff - self.loa_sd + conf_loa,
                self.mean_diff - self.loa_sd - conf_loa,
            ]

        self._output = {
            "mean": self.mean_diff,
            "mean_CI": self.CI_mean if self.CI else None,
            "loa_lower": self.mean_diff - self.loa_sd,
            "loa_lower_CI": self.CI_lower if self.CI else None,
            "loa_upper": self.mean_diff + self.loa_sd,
            "loa_upper_CI": self.CI_upper if self.CI else None,
        }

        self.computed = True

        return self._output

    def _check_params(self):
        if len(self.method1) != len(self.method2):
            raise ValueError("Length of method 1 and method 2 are not equal.")

        if self.CI is not None and (self.CI > 1 or self.CI < 0):
            raise ValueError("Confidence interval must be between 0 and 1.")

        if self.diff_method not in ["absolute", "percentage"]:
            raise ValueError(
                "The provided difference method must be either absolute or percentage."
            )

        # if any([not isinstance(x, str) for x in [self.x_title, self.y_title]]):
        #    raise ValueError('Axes labels arguments should be provided as a str.')

    def plot(
        self,
        x_label: str = "Mean of methods",
        y_label: str = "Difference between methods",
        graph_title: str = None,
        reference: bool = False,
        xlim: Tuple = None,
        ylim: Tuple = None,
        color_mean: int = "#008bff",
        color_loa: int = "#FF7000",
        color_points: int = "#000000",
        point_kws: Dict = None,
        ax: matplotlib.axes.Axes = None,
    ):
        """Provide a method comparison using Bland-Altman plotting.
        This is an Axis-level function which will draw the Bland-Altman plot
        onto the current active Axis object unless ``ax`` is provided.
        Parameters
        ----------
        x_label : str, optional
            The label which is added to the X-axis. If None is provided, a standard
            label will be added.
        y_label : str, optional
            The label which is added to the Y-axis. If None is provided, a standard
            label will be added.
        graph_title : str, optional
            Title of the Bland-Altman plot. If None is provided, no title will be plotted.
        reference : bool, optional
            If True, a grey reference line at y=0 will be plotted in the Bland-Altman.
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
        """

        if not (self.computed):
            self.compute()

        pkws = self.DEFAULT_POINTS_KWS.copy()
        pkws.update(point_kws or {})
        ax = ax or plt.gca()

        # individual points
        ax.scatter(self.mean, self.diff, **pkws)

        # mean difference and SD lines
        ax.axhline(self.mean_diff, color=color_mean, linestyle="-")
        ax.axhline(self.mean_diff + self.loa_sd, color=color_loa, linestyle="--")
        ax.axhline(self.mean_diff - self.loa_sd, color=color_loa, linestyle="--")

        if reference:
            ax.axhline(0, color="grey", linestyle="-", alpha=0.4)

        # confidence intervals (if requested)
        if self.CI is not None:
            ax.axhspan(self.CI_mean[0], self.CI_mean[1], color=color_mean, alpha=0.2)
            ax.axhspan(self.CI_upper[0], self.CI_upper[1], color=color_loa, alpha=0.2)
            ax.axhspan(self.CI_lower[0], self.CI_lower[1], color=color_loa, alpha=0.2)

        # text in graph
        trans: matplotlib.transform = transforms.blended_transform_factory(
            ax.transAxes, ax.transData
        )
        offset: float = (((self.loa * self.sd_diff) * 2) / 100) * 1.2
        ax.text(
            0.98,
            self.mean_diff + offset,
            "Mean",
            ha="right",
            va="bottom",
            transform=trans,
        )
        ax.text(
            0.98,
            self.mean_diff - offset,
            f"{self.mean_diff:.2f}",
            ha="right",
            va="top",
            transform=trans,
        )
        ax.text(
            0.98,
            self.mean_diff + self.loa_sd + offset,
            f"+{self.loa:.2f} SD",
            ha="right",
            va="bottom",
            transform=trans,
        )
        ax.text(
            0.98,
            self.mean_diff + self.loa_sd - offset,
            f"{self.mean_diff + self.loa_sd:.2f}",
            ha="right",
            va="top",
            transform=trans,
        )
        ax.text(
            0.98,
            self.mean_diff - self.loa_sd - offset,
            f"-{self.loa:.2f} SD",
            ha="right",
            va="top",
            transform=trans,
        )
        ax.text(
            0.98,
            self.mean_diff - self.loa_sd + offset,
            f"{self.mean_diff - self.loa_sd:.2f}",
            ha="right",
            va="bottom",
            transform=trans,
        )

        # transform graphs
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # set X and Y limits
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

        # graph labels
        ax.set(xlabel=x_label, ylabel=y_label, title=graph_title)

        return ax


def blandaltman(
    method1, method2, diff="absolute", limit_of_agreement=1.96, CI=0.95
) -> BlandAltman:
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
    BlandAltman : class object containing the statistics and plot functionality

    See Also
    -------
    pyCompare package on github
    Altman, D. G., and Bland, J. M. Series D (The Statistician), vol. 32, no. 3, 1983, pp. 307–317.
    Altman, D. G., and Bland, J. M. Statistical Methods in Medical Research, vol. 8, no. 2, 1999, pp. 135–160.

    """

    return BlandAltman(method1, method2, diff, limit_of_agreement, CI)
