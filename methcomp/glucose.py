import matplotlib.pyplot as plt
import numpy as np

__all__ = ["clarke", "parkes"]


class _Clarke(object):
    """Internal class for drawing a Clarke-Error grid plotting"""

    def __init__(self, reference, test, units,
                 x_title, y_title, graph_title,
                 xlim, ylim,
                 color_grid, color_points):
        # variables assignment
        self.reference: np.array = np.asarray(reference)
        self.test: np.array = np.asarray(test)
        self.units = units
        self.graph_title: str = graph_title
        self.x_title: str = x_title
        self.y_title: str = y_title
        self.xlim: list = xlim
        self.ylim: list = ylim
        self.color_grid: str = color_grid
        self.color_points: str = color_points

        self._check_params()
        self._derive_params()

    def _check_params(self):
        if len(self.reference) != len(self.test):
            raise ValueError('Length of reference and test values are not equal')

        if self.units not in ['mmol', 'mg/dl', 'mgdl']:
            raise ValueError('The provided units should be one of the following: mmol, mgdl or mg/dl.')

        if any([x is not None and not isinstance(x, str) for x in [self.x_title, self.y_title]]):
            raise ValueError('Axes labels arguments should be provided as a str.')


    def _derive_params(self):
        if self.x_title is None:
            _unit = 'mmol/L' if 'mmol' else 'mg/dL'
            self.x_title = 'Reference glucose concentration ({})'.format(_unit)

        if self.y_title is None:
            _unit = 'mmol/L' if 'mmol' else 'mg/dL'
            self.y_title = 'Predicted glucose concentration ({})'.format(_unit)

        self.xlim = self.xlim or [0, 400]
        self.ylim = self.ylim or [0, 400]

    def _calc_error_zone(self):
        # ref, pred
        ref = self.reference
        pred = self.test

        # calculate conversion factor if needed
        n = 18 if self.units == 'mmol' else 1

        # we initialize an array with ones
        # this in fact very smart because all the non-matching values will automatically
        # end up in zone B (which is 1)!
        _zones = np.ones(len(ref))

        # absolute relative error = abs(bias)/reference*100
        bias = pred - ref
        are = abs(bias) / ref * 100
        eq1 = (7 / 5) * (ref - 130 / n)
        eq2 = ref + 110 / n

        # zone E: (ref <= 70 and test >= 180) or (ref >=180 and test <=70)
        zone_e = ((ref<= 70 / n) & (pred >= 180 / n)) | ((ref >= 180 / n) & (pred <= 70 / n))
        _zones[zone_e] = 4

        # zone D: ref < 70 and (test > 70 and test < 180) or
        #   ref > 240 and (test > 70 and test < 180)
        test_d = (pred >= 70 / n) & (pred < 180 / n)  # error corrected >=70 instead of >70
        zone_d = ((ref < 70 / n) & test_d) | ((ref > 240 / n) & test_d)
        _zones[zone_d] = 3

        # zone C: (ref >= 130 and ref <= 180 and test < eq1) or
        #   (ref > 70 and ref > 180 and ref > eq2)
        zone_c = ((ref >= 130 / n) & (ref <= 180 / n) & (pred < eq1)) | ((ref > 70 / n) & (pred > 180 / n) & (pred > eq2))
        _zones[zone_c] = 2

        # zone A: are <= 20  or (ref < 58.3 and test < 70)
        zone_a = (are <= 20) | ((ref < 70 / n) & (pred < 70 / n))
        _zones[zone_a] = 0

        return _zones

    def plot(self, ax):
        _gridlines = [
            ([0, 400], [0, 400], ':'),
            ([0, 175 / 3], [70, 70], '-'),
            ([175 / 3, 400 / 1.2], [70, 400], '-'),
            ([70, 70], [84, 400], '-'),
            ([0, 70], [180, 180], '-'),
            ([70, 290], [180, 400], '-'),
            ([70, 70], [0, 56], '-'),
            ([70, 400], [56, 320], '-'),
            ([180, 180], [0, 70], '-'),
            ([180, 400], [70, 70], '-'),
            ([240, 240], [70, 180], '-'),
            ([240, 400], [180, 180], '-'),
            ([130, 180], [0, 70], '-')
        ]

        _gridlabels = [
            (30, 15, "A"),
            (370, 260, "B"),
            (280, 370, "B"),
            (160, 370, "C"),
            (160, 15, "C"),
            (30, 140, "D"),
            (370, 120, "D"),
            (30, 370, "E"),
            (370, 15, "E"),
        ]

        # calculate conversion factor if needed
        n = 18 if self.units == 'mmol' else 1

        # plot individual points
        ax.scatter(self.reference,
                   self.test, marker='o', color=self.color_points, s=8)

        # plot grid lines
        for g in _gridlines:
            ax.plot(np.array(g[0])/n,
                    np.array(g[1])/n,
                    g[2], color=self.color_grid)

        for l in _gridlabels:
            ax.text(l[0]/n, l[1]/n, l[2], fontsize=15)

        # limits and ticks
        ax.set_xlim(self.xlim[0]/n, self.xlim[1]/n)
        ax.set_ylim(self.ylim[0]/n, self.ylim[1]/n)

        # graph labels
        ax.set_ylabel(self.y_title)
        ax.set_xlabel(self.x_title)
        if self.graph_title is not None:
            ax.set_title(self.graph_title)

def clarke(reference, test, units='mmol',
           x_label=None, y_label=None, title=None,
           xlim=None, ylim=None,
           color_grid='#000000', color_points='#FF0000',
           square=False, ax=None):
    """Provide a glucose error grid analyses as designed by Clarke.

    This is an Axis-level function which will draw the Clarke-error grid plot.
    onto the current active Axis object unless ``ax`` is provided.


    Parameters
    ----------
    reference, test : array, or list
        Glucose values obtained from the reference and predicted methods, preferably provided in a np.array.
    units : str
        The SI units which the glucose values are provided in. Options: 'mmol', 'mgdl' or 'mg/dl'.
    x_label : str, optional
        The label which is added to the X-axis. If None is provided, a standard
        label will be added.
    y_label : str, optional
        The label which is added to the Y-axis. If None is provided, a standard
        label will be added.
    title : str, optional
        Title of the Bland-Altman plot. If None is provided, no title will be plotted.
    xlim : list, optional
        Minimum and maximum limits for X-axis. Should be provided as list or tuple.
        If not set, matplotlib will decide its own bounds.
    ylim : list, optional
        Minimum and maximum limits for Y-axis. Should be provided as list or tuple.
        If not set, matplotlib will decide its own bounds.
    color_grid : str, optional
        Color of the Clarke error grid lines.
    color_points : str, optional
        Color of the individual differences that will be plotted.
    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the Bland-Altman plot.

    See Also
    -------
    Clarke, W. L., Cox, D., et al. Diabetes Care, vol. 10, no. 5, 1987, pp. 622–628.
    """

    plotter: _Clarke = _Clarke(reference, test, units,
                               x_label, y_label, title,
                               xlim, ylim,
                               color_grid, color_points)

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()

    if square:
        ax.set_aspect('equal')

    plotter.plot(ax)

    return ax


class _Parkes(object):
    """Internal class for drawing a Deming regression plot"""

    def __init__(self):
        self._check_params()
        self._derive_params()

    def _check_params(self):
        pass

    def _derive_params(self):
        pass

    def plot(self, ax):
        pass

def parkes(
           square=False, ax=None):

    plotter: _Parkes = _Parkes()

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()

    if square:
        ax.set_aspect('equal')

    plotter.plot(ax)

    return ax