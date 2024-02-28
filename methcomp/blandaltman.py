# -*- coding: utf-8 -*-
import warnings
from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

from .comparer import Comparer

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from scipy.stats import norm, t

__all__ = ["blandaltman", "BlandAltman"]


class BlandAltman(Comparer):
    """Class for drawing a Bland-Altman plot"""

    DEFAULT_POINTS_KWS = {"s": 20, "alpha": 0.6, "color": "#000000"}

    def __init__(
        self,
        method1: Union[Sequence[float], np.ndarray],
        method2: Union[Sequence[float], np.ndarray],
        diff: str = "absolute",
        limit_of_agreement: float = 1.96,
        CI: float = 0.95,
    ):
        """
        Initialize a Bland-Altman class object. This class is center to the calculate
        and compute functionality of a Bland-Altman comparison.

        Parameters
        ----------
        method1, method2: Union[List[float], np.ndarray]
            Values obtained from both methods, preferably provided in a np.array.
        diff : "absolute"  or "percentage"
            The difference to display, whether it is an absolute one or a percentual
            one. If None is provided, it defaults to absolute.
        limit_of_agreement : float, optional
            Multiples of the standard deviation to plot the limit of afgreement bounds
            at. This defaults to 1.96.
        CI : float, optional
            The confidence interval employed in the mean difference and limit of
            agreement lines. Defaults to 0.95.
        """

        # variables assignment
        self.computed = False
        self.diff_method: str = diff
        self.CI: float = CI
        self.loa: float = limit_of_agreement
        super().__init__(method1, method2)

    def _check_params(self):
        super()._check_params()

        if len(self.method1) != len(self.method2):
            raise ValueError("Length of method 1 and method 2 are not equal.")

        if self.CI is not None and (self.CI > 1 or self.CI < 0):
            raise ValueError("Confidence interval must be between 0 and 1.")

        if self.diff_method not in ["absolute", "percentage"]:
            raise ValueError(
                "The provided difference method must be either absolute or percentage."
            )

    def _calculate_impl(self):
        """Calculates the statistics for method comparison using
        Bland-Altman plotting. Returns a dictionary with the results.

        Parameters
        ----------
        None

        Returns
        ----------
        Dict[str, Any] :
            Dictionary containing the mean and limit of agreement
            values and their confidence intervals
        """

        self.mean: np.array = np.mean(
            [self.method1, self.method2], axis=0
        )  # type: ignore

        self.diff = self.method1 - self.method2
        if self.diff_method == "percentage":
            self.diff = self.diff * 100 / self.mean

        self.mean_diff = np.mean(self.diff)
        self.sd_diff = np.std(self.diff, axis=0)
        self.loa_sd = self.loa * self.sd_diff

        if self.CI is not None:
            self.CI_mean = norm.interval(
                confidence=self.CI,
                loc=self.mean_diff,
                scale=self.sd_diff / np.sqrt(self.n),
            )
            se_loa = (1.71**2) * ((self.sd_diff**2) / self.n)
            conf_loa = np.sqrt(se_loa) * t.ppf(q=(1 - self.CI) / 2.0, df=self.n - 1)
            self.CI_upper = [
                self.mean_diff + self.loa_sd + conf_loa,
                self.mean_diff + self.loa_sd - conf_loa,
            ]
            self.CI_lower = [
                self.mean_diff - self.loa_sd + conf_loa,
                self.mean_diff - self.loa_sd - conf_loa,
            ]

        self._result = {
            "mean": self.mean_diff,
            "mean_CI": self.CI_mean if self.CI else None,
            "loa_lower": self.mean_diff - self.loa_sd,
            "loa_lower_CI": self.CI_lower if self.CI else None,
            "loa_upper": self.mean_diff + self.loa_sd,
            "loa_upper_CI": self.CI_upper if self.CI else None,
            "sd_diff": self.sd_diff,
        }

    def plot(
        self,
        x_label: str = "Mean of methods",
        y_label: str = "Difference between methods",
        graph_title: Optional[str] = None,
        reference: bool = False,
        xlim: Optional[Tuple] = None,
        ylim: Optional[Tuple] = None,
        color_mean: str = "#008bff",
        color_loa: str = "#FF7000",
        color_points: str = "#000000",
        point_kws: Optional[Dict] = None,
        ci_alpha: float = 0.2,
        loa_linestyle: str = "--",
        ax: Optional[matplotlib.axes.Axes] = None,
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
            Title of the Bland-Altman plot.
            If None is provided, no title will be plotted.
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
        ci_alpha: float, optional
            Alpha value of the confidence interval.
        loa_linestyle: str, optional
            Linestyle of the limit of agreement lines.
        ax : matplotlib Axes, optional
            Axes in which to draw the plot, otherwise use the currently-active
            Axes.

        Returns
        -------
        ax : matplotlib Axes
            Axes object with the Bland-Altman plot.
        """

        ax = ax or plt.gca()

        pkws = self.DEFAULT_POINTS_KWS.copy()
        pkws.update(point_kws or {})

        # Get parameters
        mean, mean_CI = self.result["mean"], self.result["mean_CI"]
        loa_upper, loa_upper_CI = self.result["loa_upper"], self.result["loa_upper_CI"]
        loa_lower, loa_lower_CI = self.result["loa_lower"], self.result["loa_lower_CI"]
        sd_diff = self.result["sd_diff"]

        # individual points
        ax.scatter(self.mean, self.diff, **pkws)  # type: ignore

        # mean difference and SD lines
        ax.axhline(mean, color=color_mean, linestyle=loa_linestyle)
        ax.axhline(loa_upper, color=color_loa, linestyle=loa_linestyle)
        ax.axhline(loa_lower, color=color_loa, linestyle=loa_linestyle)

        if reference:
            ax.axhline(0, color="grey", linestyle="-", alpha=0.4)

        # confidence intervals (if requested)
        if self.CI is not None:
            ax.axhspan(*mean_CI, color=color_mean, alpha=ci_alpha)
            ax.axhspan(*loa_upper_CI, color=color_loa, alpha=ci_alpha)
            ax.axhspan(*loa_lower_CI, color=color_loa, alpha=ci_alpha)

        # text in graph
        trans: matplotlib.transforms.Transform = transforms.blended_transform_factory(
            ax.transAxes, ax.transData
        )
        offset: float = (((self.loa * sd_diff) * 2) / 100) * 1.2
        ax.text(
            0.98,
            mean + offset,
            "Mean",
            ha="right",
            va="bottom",
            transform=trans,
        )
        ax.text(
            0.98,
            mean - offset,
            f"{mean:.2f}",
            ha="right",
            va="top",
            transform=trans,
        )
        ax.text(
            0.98,
            loa_upper + offset,
            f"+{self.loa:.2f} SD",
            ha="right",
            va="bottom",
            transform=trans,
        )
        ax.text(
            0.98,
            loa_upper - offset,
            f"{loa_upper:.2f}",
            ha="right",
            va="top",
            transform=trans,
        )
        ax.text(
            0.98,
            loa_lower - offset,
            f"-{self.loa:.2f} SD",
            ha="right",
            va="top",
            transform=trans,
        )
        ax.text(
            0.98,
            loa_lower + offset,
            f"{loa_lower:.2f}",
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
    method1,
    method2,
    diff="absolute",
    limit_of_agreement=1.96,
    CI=0.95,
    x_label: str = "Mean of methods",
    y_label: str = "Difference between methods",
    graph_title: Optional[str] = None,
    reference: bool = False,
    xlim: Optional[Tuple] = None,
    ylim: Optional[Tuple] = None,
    color_mean: str = "#008bff",
    color_loa: str = "#FF7000",
    color_points: str = "#000000",
    point_kws: Optional[Dict] = None,
    ci_alpha: float = 0.2,
    loa_linestyle: str = "--",
    ax: Optional[matplotlib.axes.Axes] = None,
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
    [altman_1983] Altman, D. G., and Bland, J. M.
                  Series D (The Statistician), vol. 32, no. 3, 1983, pp. 307–317.
    [altman_1999] Altman, D. G., and Bland, J. M.
                  Statistical Methods in Medical Research,
                  vol. 8, no. 2, 1999, pp. 135–160.

    """

    return BlandAltman(method1, method2, diff, limit_of_agreement, CI).plot(
        x_label=x_label,
        y_label=y_label,
        graph_title=graph_title,
        reference=reference,
        xlim=xlim,
        ylim=ylim,
        color_mean=color_mean,
        color_loa=color_loa,
        color_points=color_points,
        point_kws=point_kws,
        ci_alpha=ci_alpha,
        loa_linestyle=loa_linestyle,
        ax=ax,
    )
