# -*- coding: utf-8 -*-

"""Mountain plot.
"""
import warnings
from typing import List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .comparer import Comparer


class Mountain(Comparer):

    """Mountain plot

    A mountain plot is useful to compare how two methods perform on paired
    measurements [krouwer_1995]. It forms the empirical cumulative distirbution
    of the difference between pairs, folded at the median or any other quantile p
    [monti_1995]. This makes it easy to see bias and spread. Furthermore, it has
    been shown that the area under the curve AUC is the  mean absolute devation
    from p [xue_2011].

    Attributes
    ----------
    iqr : float
        Interquartile range
    n_percentiles : int
        Number of percentiles to use - more gives a smoother mountain plot
    result : Dict[str, Any]
        Mountain calculation result with keys
        * mountain - folded CDF in percent
        * quantile - value along difference as function of quantile
        * auc - area under curve
        * median - quantile at median
        * mountain_median - mountain at median
        * iqr - interquantile range: index at low=50-iqr/2, high=50+iqr/2
        * mountain_iqr - mountain low and high iqr

    Examples
    --------
    Compare method 1 to method 2 and 3:

    >>> import matplotlib.pyplot as plt
    >>> from methcomp import mountain
    >>> method1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    >>>    18, 19, 20]
    >>> method2 = [1.03, 2.05, 2.79, 3.67, 5.00, 5.82, 7.16, 7.69, 8.53, 10.38,
    >>>    11.11, 12.17, 13.47, 13.83, 15.15, 16.12, 16.94, 18.09, 19.13, 19.54]
    >>> method3 = [0.82, 2.09, 4.31, 4.18, 4.85, 6.01, 7.73, 7.95, 9.78, 10.07,
    >>>    11.64, 11.88, 12.90, 13.55, 14.57, 16.82, 16.95, 17.62, 18.44, 20.14]
    >>> mountain.Mountain(method1, method2, n_percentiles=500).plot(
    >>>    color="blue", label="$M_1$ - $M_2$", unit="ng/ml")
    >>> mountain.Mountain(method1, method3, n_percentiles=500).plot(
    >>>    color="green", label="$M_1$ - $M_3$", unit="ng/ml")
    >>> plt.show()

    This will superimpose the comparisons by default, as plot creates it's own
    axis if necessary, and reuses the current one if in place.

    References
    ----------
    .. [krouwer_1995] Krouwer, Jan S., and Katherine L. Monti.
                   "A simple, graphical method to evaluate laboratory assays."
                   European journal of clinical chemistry and clinical biochemistry
                   33.8 (1995): 525-528.
    .. [monti_1995] Monti, Katherine L.
                 "Folded empirical distribution function curves—mountain plots."
                 The American Statistician 49.4 (1995): 342-345.
    .. [xue_2011] Xue, Jing-Hao, and D. Michael Titterington.
               "The p-folded cumulative distribution function and the mean absolute
               deviation from the p-quantile." Statistics & probability letters
               81.8 (2011): 1179-1182.
    """

    def __init__(
        self,
        method1: Union[List[float], np.ndarray],
        method2: Union[List[float], np.ndarray],
        n_percentiles: int = 100,
        iqr: float = 68.27,
    ):
        """Construct a regressor.

        Parameters
        ----------
        method1 : Union[List[float], np.ndarray]
            Values for method 1
        method2 : Union[List[float], np.ndarray]
            Values for method 2
        n_percentiles : int, optional
            Number of percentile streps - more gives a smoother mountain plot
            (default n=100)
        iqr : float, optional
            Interquartile range for to show in plot. If <0 iqr will be skipped.
            (default: 90)
        """
        self.n_percentiles = n_percentiles
        self.iqr = iqr
        # Process args
        super().__init__(method1, method2)

    def _check_params(self):
        """Check validity of parameters

        Raises
        ------
        ValueError
            If method values are of different shape or CI outside of range 0,1
        """
        super()._check_params()
        if self.n_percentiles <= 0:
            raise ValueError(
                "n_percentiles: Number of percentile steps should be positive"
            )
        if not 0 <= self.iqr <= 100:
            raise ValueError("iqr: Interquartile range must be in [0-100]")

    def _calculate_impl(self):
        """Calculate mountain parameters."""

        # quantile values to evaluate
        qrange = np.linspace(0, 1, self.n_percentiles)
        diff = self.method1 - self.method2
        quantile = np.quantile(diff, qrange)
        iqr = np.quantile(diff, [0.5 - self.iqr / 200, 0.5 + self.iqr / 200])
        # Find id corresponding to iqr and median
        median_idx = self.n_percentiles // 2
        iqr_idx = np.abs(quantile - iqr[0]).argmin(), np.abs(quantile - iqr[1]).argmin()
        # Split qrange in the middle and convert to percentile
        mountain = np.where(qrange < 0.5, qrange, 1.0 - qrange) * 100
        # Calcualte area under curve
        auc = (np.diff(quantile) * (mountain[:-1] + mountain[1:]) / 2).sum()

        # Build result
        self._result = {
            "mountain": mountain,
            "quantile": quantile,
            "auc": auc,
            "iqr": iqr,
            "mountain_iqr": mountain[iqr_idx, None],
            "median": quantile[median_idx],
            "mountain_median": mountain[median_idx],
        }

    def plot(
        self,
        xlabel: str = "Method difference",
        ylabel: str = "Folded CDF (%)",
        label: str = "$M_1$ - $M_2$",
        unit: str = None,
        title: str = None,
        color: str = "blue",
        legend: bool = True,
        square: bool = False,
        show_hline: bool = True,
        show_vlines: bool = True,
        show_markers: bool = True,
        ax: matplotlib.axes.Axes = None,
    ) -> matplotlib.axes.Axes:
        """Plot mountain plot

        Parameters
        ----------
        xlabel : str, optional
            The label which is added to the X-axis. (default: "Method difference")
        ylabel : str, optional
            The label which is added to the Y-axis. (default: "Folded CDF (%)")
        label : str, optional
            mountaint line legend label (default:"Method 1 - Method 2" )
        unit : str, optional
            unit to disply for x-axis
        title : str, optional
            figure title, if none there will be no title
            (default: None)
        legend : bool, optional
            If True, will provide a legend containing the computed regression equation.
            (default: True)
        square : bool, optional
            If True, set the Axes aspect to "equal" so each cell will be
            square-shaped. (default: True)
        color : str, optional
            Color for mountain plot elements
        show_hline: bool, optional
            If set show horizontal lines for iqr
        show_vlines: bool, optional
            If set show vertical lines at iqr and median
        show_markers: bool, optional
            If set show markers at iqr and median
        ax : matplotlib.axes.Axes, optional
            matplotlib axis object, if not passed, uses gca()

        Returns
        -------
        matplotlib.axes.Axes
            axes object with the plot
        """
        ax = ax or plt.gca()
        ax.step(
            y=self.result["mountain"],
            x=self.result["quantile"],
            where="mid",
            label=f"{label} AUC={self.result['auc']:.2f}",
            color=color,
        )
        if show_hline:
            ax.hlines(
                self.result["mountain_iqr"],
                xmin=self.result["iqr"][0],
                xmax=self.result["iqr"][1],
                color=color,
            )
        if show_vlines:
            ax.vlines(
                self.result["median"],
                ymin=0,
                ymax=50,
                label=f"median={self.result['median']:.2f} {unit or ''}",
                linestyle="--",
                color=color,
            )
            if self.iqr > 0:
                ax.vlines(
                    self.result["iqr"],
                    ymin=0,
                    ymax=50 - self.iqr / 2,
                    label=(
                        f"{self.iqr:.2f}% IQR ="
                        f"{self.result['iqr'][1] - self.result['iqr'][0]:.2f}"
                        "{unit or ''}"
                    ),
                    linestyle=":",
                    color=color,
                )
        if show_markers:
            ax.plot(
                self.result["median"],
                self.result["mountain_median"],
                "o",
                color=color,
            )
            if self.iqr > 0:
                ax.plot(
                    self.result["iqr"],
                    self.result["mountain_iqr"],
                    "o",
                    color=color,
                )
        u = f"({unit})" if unit else ""
        ax.set(xlabel=f"{xlabel} {u}", ylabel=ylabel or None, title=title or None)
        ax.legend(loc="upper left", fontsize="medium")

        if legend:
            ax.legend(loc="upper left", frameon=False)

        if square:
            ax.set_aspect("equal")

        return ax


def mountain(
    method1: Union[List[float], np.ndarray],
    method2: Union[List[float], np.ndarray],
    n_percentiles: int = 100,
    iqr: float = 68.27,
    xlabel: str = "Method difference",
    ylabel: str = "Folded CDF (%)",
    label: str = "$M_1$ - $M_2$",
    unit: str = None,
    title: str = None,
    color: str = "blue",
    legend: bool = True,
    square: bool = False,
    show_hline: bool = True,
    show_vlines: bool = True,
    show_markers: bool = True,
    ax: matplotlib.axes.Axes = None,
):
    """Provide a method comparison using Mountain plot.

    This is an Axis-level function which will draw the Deming plot
    onto the current active Axis object unless ``ax`` is provided.

    NOTE: This function is deprecated and remain for API backwards
    compatibility reasons. Prefer to use: mountain.Mountain

    Parameters
    ----------
    method1 : Union[List[float], np.ndarray]
        Values for method 1
    method2 : Union[List[float], np.ndarray]
        Values for method 2
    n_percentiles : int, optional
        Number of percentile streps - more gives a smoother mountain plot
        (default n=100)
    iqr : float, optional
        Interquartile range for to show in plot. If <0 iqr will be skipped.
        (default: 90)
    xlabel : str, optional
        The label which is added to the X-axis. (default: "Method difference")
    ylabel : str, optional
        The label which is added to the Y-axis. (default: "Folded CDF (%)")
    label : str, optional
        mountaint line legend label (default:"Method 1 - Method 2" )
    unit : str, optional
        unit to disply for x-axis
    title : str, optional
        figure title, if none there will be no title
        (default: None)
    legend : bool, optional
        If True, will provide a legend containing the computed regression equation.
        (default: True)
    square : bool, optional
        If True, set the Axes aspect to "equal" so each cell will be
        square-shaped. (default: True)
    color : str, optional
        Color for mountain plot elements
    show_hline: bool, optional
        If set show horizontal lines for iqr
    show_vlines: bool, optional
        If set show vertical lines at iqr and median
    show_markers: bool, optional
        If set show markers at iqr and median
    ax : matplotlib.axes.Axes, optional
        matplotlib axis object, if not passed, uses gca()

    Returns
    -------
    matplotlib.axes.Axes
        axes object with the Mountain regression plot

    See Also
    -------
    mountain.Mountain - class that implements Mountain plot

    References
    ----------
    .. [krouwer_1995] Krouwer, Jan S., and Katherine L. Monti.
                   "A simple, graphical method to evaluate laboratory assays."
                   European journal of clinical chemistry and clinical biochemistry
                   33.8 (1995): 525-528.
    .. [monti_1995] Monti, Katherine L.
                 "Folded empirical distribution function curves—mountain plots."
                 The American Statistician 49.4 (1995): 342-345.
    .. [xue_2011] Xue, Jing-Hao, and D. Michael Titterington.
               "The p-folded cumulative distribution function and the mean absolute
               deviation from the p-quantile." Statistics & probability letters
               81.8 (2011): 1179-1182.
    """
    warnings.warn(
        "methcomp.mountain.mountain is deprecated,"
        " use methcomp.mountain.Mountain instead",
        DeprecationWarning,
    )

    return Mountain(
        method1=method1,
        method2=method2,
        n_percentiles=n_percentiles,
        iqr=iqr,
    ).plot(
        xlabel=xlabel,
        ylabel=ylabel,
        label=label,
        unit=unit,
        title=title,
        color=color,
        legend=legend,
        square=square,
        show_hline=show_hline,
        show_vlines=show_vlines,
        show_markers=show_markers,
        ax=ax,
    )
