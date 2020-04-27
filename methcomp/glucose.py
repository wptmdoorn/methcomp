import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from shapely.geometry import Polygon, Point

__all__ = ["clarke", "parkes", "seg",
           "clarkezones", "parkeszones", "segscores"]


class _Clarke(object):
    """Internal class for drawing a Clarke-Error grid plotting"""

    def __init__(self, reference, test, units,
                 x_title, y_title, graph_title,
                 xlim, ylim,
                 color_grid, color_gridlabels, color_points,
                 grid, percentage,
                 point_kws, grid_kws):
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
        self.color_gridlabels: str = color_gridlabels
        self.color_points: str = color_points
        self.grid: bool = grid
        self.percentage: bool = percentage
        self.point_kws = {} if point_kws is None else point_kws.copy()
        self.grid_kws = {} if grid_kws is None else grid_kws.copy()

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

        return [int(i) for i in _zones]

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

        colors = ['#196600', '#7FFF00', '#FF7B00', '#FF5700', '#FF0000']

        _gridlabels = [
            (30, 15, "A", colors[0]),
            (370, 260, "B", colors[1]),
            (280, 370, "B", colors[1]),
            (160, 370, "C", colors[2]),
            (160, 15, "C", colors[2]),
            (30, 140, "D", colors[3]),
            (370, 120, "D", colors[3]),
            (30, 370, "E", colors[4]),
            (370, 15, "E", colors[4]),
        ]

        # calculate conversion factor if needed
        n = 18 if self.units == 'mmol' else 1

        # plot individual points
        if self.color_points == 'auto':
            ax.scatter(self.reference,
                       self.test,
                       marker='o',
                       alpha=0.6,
                       c=[colors[i] for i in self._calc_error_zone()], s=8,
                       **self.point_kws)
        else:
            ax.scatter(self.reference,
                       self.test, marker='o',
                       color=self.color_points,
                       alpha=0.6,
                       s=8,
                       **self.point_kws)

        # plot grid lines
        if self.grid:
            for g in _gridlines:
                ax.plot(np.array(g[0])/n,
                        np.array(g[1])/n,
                        g[2], color=self.color_grid,
                        **self.grid_kws)

            if self.percentage:
                zones = [['A', 'B', 'C', 'D', 'E'][i] for i in self._calc_error_zone()]

                for l in _gridlabels:
                    ax.text(l[0] / n, l[1] / n, l[2],
                            fontsize=12,
                            fontweight='bold',
                            color=l[3] if self.color_gridlabels == 'auto' else self.color_gridlabels)
                    ax.text(l[0] / n + (8 / n),
                            l[1] / n + (8 / n),
                            "{:.1f}".format((zones.count(l[2]) / len(zones)) * 100),
                            fontsize=9,
                            fontweight='bold',
                            color=l[3] if self.color_gridlabels == 'auto' else self.color_gridlabels)

            else:
                for l in _gridlabels:
                    ax.text(l[0] / n, l[1] / n, l[2],
                            fontsize=12,
                            fontweight='bold',
                            color=l[3] if self.color_gridlabels == 'auto' else self.color_gridlabels)

        # limits and ticks
        ax.set_xlim(self.xlim[0]/n, self.xlim[1]/n)
        ax.set_ylim(self.ylim[0]/n, self.ylim[1]/n)

        # graph labels
        ax.set_ylabel(self.y_title)
        ax.set_xlabel(self.x_title)
        if self.graph_title is not None:
            ax.set_title(self.graph_title)

def clarke(reference, test, units,
           x_label=None, y_label=None, title=None,
           xlim=None, ylim=None,
           color_grid='#000000', color_gridlabels='auto', color_points='auto',
           grid=True, percentage=False,
           point_kws=None, grid_kws=None,
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
        Title of the Clarke error grid plot. If None is provided, no title will be plotted.
    xlim : list, optional
        Minimum and maximum limits for X-axis. Should be provided as list or tuple.
        If not set, matplotlib will decide its own bounds.
    ylim : list, optional
        Minimum and maximum limits for Y-axis. Should be provided as list or tuple.
        If not set, matplotlib will decide its own bounds.
    color_grid : str, optional
        Color of the Clarke error grid lines.
    color_gridlabels : str, optional
        Color of the gridlabels (A, B, C, ..) that will be plotted. If set to 'auto',
        it will plot the points according to their zones.
    color_points : str, optional
        Color of the individual differences that will be plotted. If set to 'auto',
        it will plot the points according to their zones.
    grid : bool, optional
        Enable the grid lines of the Parkes error. Defaults to True.
    percentage : bool, optional
        If True, percentage of the zones will be depicted in the plot.
    point_kws : dict of key, value mappings, optional
        Additional keyword arguments for `plt.scatter`.
    grid_kws : dict of key, value mappings, optional
        Additional keyword arguments for the grid with `plt.plot`.
    square : bool, optional
        If True, set the Axes aspect to "equal" so each cell will be square-shaped.
    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the Clarke-error grid plot.

    See Also
    -------
    Clarke, W. L., Cox, D., et al. Diabetes Care, vol. 10, no. 5, 1987, pp. 622–628.
    """

    plotter: _Clarke = _Clarke(reference, test, units,
                               x_label, y_label, title,
                               xlim, ylim,
                               color_grid, color_gridlabels, color_points,
                               grid, percentage,
                               point_kws, grid_kws)

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()

    if square:
        ax.set_aspect('equal')

    plotter.plot(ax)

    return ax

def clarkezones(reference, test, units,
                numeric=False):
    """Provides the error zones as depicted by the
    Clarke error grid analysis for each point in the reference and test datasets.


    Parameters
    ----------
    reference, test : array, or list
        Glucose values obtained from the reference and predicted methods, preferably provided in a np.array.
    units : str
        The SI units which the glucose values are provided in. Options: 'mmol', 'mgdl' or 'mg/dl'.
    numeric : bool, optional
        If this is set to true, returns integers (0 to 4) instead of characters for each of the
        zones.

    Returns
    -------
    clarkezones : list
        Returns a list depecting the zones for each of the reference and test values.

    """

    # obtain zones from a Clarke reference object
    _zones = _Clarke(reference, test, units,
                     None, None, None,
                     None, None,
                     '#000000', 'auto', 'auto',
                     True, False,
                     None, None)._calc_error_zone()

    if numeric:
        return _zones
    else:
        labels = ['A', 'B', 'C', 'D', 'E']
        return [labels[i] for i in _zones]


class _Parkes(object):
    """Internal class for drawing a Parkes consensus error grid plot"""

    def __init__(self, type, reference, test, units,
                 x_title, y_title, graph_title,
                 xlim, ylim,
                 color_grid, color_gridlabels, color_points,
                 grid, percentage,
                 point_kws, grid_kws):
        # variables assignment
        self.type: int = type
        self.reference: np.array = np.asarray(reference)
        self.test: np.array = np.asarray(test)
        self.units = units
        self.graph_title: str = graph_title
        self.x_title: str = x_title
        self.y_title: str = y_title
        self.xlim: list = xlim
        self.ylim: list = ylim
        self.color_grid: str = color_grid
        self.color_gridlabels: str = color_gridlabels
        self.color_points: str = color_points
        self.grid: bool = grid
        self.percentage: bool = percentage
        self.point_kws = {} if point_kws is None else point_kws.copy()
        self.grid_kws = {} if grid_kws is None else grid_kws.copy()

        self._check_params()
        self._derive_params()

    def _check_params(self):
        if self.type != 1 and self.type != 2:
            raise ValueError('Type of Diabetes should either be 1 or 2.')

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

    def _coef(self, x, y, xend, yend):
        if xend == x:
            raise ValueError('Vertical line - function inapplicable')
        return (yend - y) / (xend - x)

    def _endy(self, startx, starty, maxx, coef):
        return (maxx - startx) * coef + starty

    def _endx(self, startx, starty, maxy, coef):
        return (maxy - starty) / coef + startx

    def _calc_error_zone(self):
        # ref, pred
        ref = self.reference
        pred = self.test

        # calculate conversion factor if needed
        n = 18 if self.units == 'mmol' else 1

        maxX = max(max(ref) + 20 / n, 550 / n)
        maxY = max([*(np.array(pred) + 20 / n), maxX, 550 / n])

        # we initialize an array with ones
        # this in fact very smart because all the non-matching values will automatically
        # end up in zone A (which is zero)
        _zones = np.zeros(len(ref))

        if self.type == 1:
            ce = self._coef(35, 155, 50, 550)
            cdu = self._coef(80, 215, 125, 550)
            cdl = self._coef(250, 40, 550, 150)
            ccu = self._coef(70, 110, 260, 550)
            ccl = self._coef(260, 130, 550, 250)
            cbu = self._coef(280, 380, 430, 550)
            cbl = self._coef(385, 300, 550, 450)

            limitE1 = Polygon([(x, y) for x, y in zip([0, 35 / n, self._endx(35 / n, 155 / n, maxY, ce), 0, 0],
                                                      [150 / n, 155 / n, maxY, maxY, 150 / n])])

            limitD1L = Polygon([(x, y) for x, y in zip([250 / n, 250 / n, maxX, maxX, 250 / n],
                                                       [0, 40 / n, self._endy(410 / n, 110 / n, maxX, cdl), 0, 0])])

            limitD1U= Polygon([(x, y) for x, y in zip([0, 25 / n, 50 / n, 80 / n, self._endx(80 / n, 215 / n, maxY, cdu), 0, 0],
                                                      [100 / n, 100 / n, 125 / n, 215 / n, maxY, maxY, 100 / n])])

            limitC1L = Polygon([(x, y) for x, y in zip([120 / n, 120 / n, 260 / n, maxX, maxX, 120 / n],
                                                       [0, 30 / n, 130 / n, self._endy(260 / n, 130 / n, maxX, ccl), 0, 0])])

            limitC1U = Polygon([(x, y) for x, y in zip([0, 30 / n, 50 / n, 70 / n, self._endx(70 / n, 110 / n, maxY, ccu), 0, 0],
                                                       [60 / n, 60 / n, 80 / n, 110 / n, maxY, maxY, 60 / n])])

            limitB1L = Polygon([(x, y) for x, y in zip([50 / n, 50 / n, 170 / n, 385 / n, maxX, maxX, 50 / n],
                                                       [0, 30 / n, 145 / n, 300 / n, self._endy(385 / n, 300 / n, maxX, cbl), 0, 0])])

            limitB1U = Polygon([(x, y) for x, y in zip([0, 30 / n, 140 / n, 280 / n, self._endx(280 / n, 380 / n, maxY, cbu), 0, 0],
                                                       [50 / n, 50 / n, 170 / n, 380 / n, maxY, maxY, 50 / n])])

            for i, points in enumerate(zip(ref, pred)):
                for f, r in zip([limitB1L, limitB1U,
                                 limitC1L, limitC1U,
                                 limitD1L, limitD1U,
                                 limitE1],
                                [1, 1,
                                 2, 2,
                                 3, 3,
                                 4]):
                    if f.contains(Point(points[0], points[1])):
                        _zones[i] = r

            return [int(i) for i in _zones]

        elif self.type == 2:
            ce = self._coef(35, 200, 50, 550)
            cdu = self._coef(35, 90, 125, 550)
            cdl = self._coef(410, 110, 550, 160)
            ccu = self._coef(30, 60, 280, 550)
            ccl = self._coef(260, 130, 550, 250)
            cbu = self._coef(230, 330, 440, 550)
            cbl = self._coef(330, 230, 550, 450)

            limitE2 = Polygon([(x, y) for x, y in zip([0, 35 / n, self._endx(35 / n, 200 / n, maxY, ce), 0, 0],  # x limits E upper
                              [200 / n, 200 / n, maxY, maxY, 200 / n])])  # y limits E upper

            limitD2L = Polygon([(x, y) for x, y in zip([250 / n, 250 / n, 410 / n, maxX, maxX, 250 / n],  # x limits D lower
                               [0, 40 / n, 110 / n, self._endy(410 / n, 110 / n, maxX, cdl), 0, 0])])  # y limits D lower

            limitD2U = Polygon([(x, y) for x, y in zip([0, 25 / n, 35 / n, self._endx(35 / n, 90 / n, maxY, cdu), 0, 0],  # x limits D upper
                                [80 / n, 80 / n, 90 / n, maxY, maxY, 80 / n])]) # y limits D upper

            limitC2L = Polygon([(x, y) for x, y in zip([90 / n, 260 / n, maxX, maxX, 90 / n],  # x limits C lower
                               [0, 130 / n, self._endy(260 / n, 130 / n, maxX, ccl), 0, 0])])  # y limits C lower

            limitC2U = Polygon([(x, y) for x, y in zip([0, 30 / n, self._endx(30 / n, 60 / n, maxY, ccu), 0, 0],  # x limits C upper
                               [60 / n, 60 / n, maxY, maxY, 60 / n])])  # y limits C upper

            limitB2L = Polygon([(x, y) for x, y in zip([50 / n, 50 / n, 90 / n, 330 / n, maxX, maxX, 50 / n],  # x limits B lower
                                [0, 30 / n, 80 / n, 230 / n, self._endy(330 / n, 230 / n, maxX, cbl), 0, 0])])  # y limits B lower

            limitB2U = Polygon([(x, y) for x, y in zip([0, 30 / n, 230 / n, self._endx(230 / n, 330 / n, maxY, cbu), 0, 0],  # x limits B upper
                                [50 / n, 50 / n, 330 / n, maxY, maxY, 50 / n])])  # y limits B upper

            for i, points in enumerate(zip(ref, pred)):
                for f, r in zip([limitB2L, limitB2U,
                                 limitC2L, limitC2U,
                                 limitD2L, limitD2U,
                                 limitE2],
                                [1, 1,
                                 2, 2,
                                 3, 3,
                                 4]):
                    if f.contains(Point(points[0], points[1])):
                        _zones[i] = r

            return [int(i) for i in _zones]


    def plot(self, ax):
        # ref, pred
        ref = self.reference
        pred = self.test

        # calculate conversion factor if needed
        n = 18 if self.units == 'mmol' else 1

        maxX = self.xlim or max(max(ref) + 20 / n, 550 / n)
        maxY = self.ylim or max([*(np.array(pred) + 20 / n), maxX, 550 / n])

        if self.type == 1:
            ce =  self._coef(35, 155, 50, 550)
            cdu = self._coef(80, 215, 125, 550)
            cdl = self._coef(250, 40, 550, 150)
            ccu = self._coef(70, 110, 260, 550)
            ccl = self._coef(260, 130, 550, 250)
            cbu = self._coef(280, 380, 430, 550)
            cbl = self._coef(385, 300, 550, 450)

            _gridlines = [
                ([0, min(maxX, maxY)], [0, min(maxX, maxY)], ':'),

                ([0, 30 / n], [50 / n, 50 / n], '-'),
                ([30 / n, 140 / n], [50 / n, 170 / n], '-'),
                ([140 /n, 280 / n], [170 / n, 380 / n], '-'),
                ([280 / n, self._endx(280 / n, 380 / n, maxY, cbu)], [380 / n, maxY], '-'),

                ([50 / n, 50 / n], [0 / n, 30 / n], '-'),
                ([50 / n, 170 / n], [30 / n, 145 / n], '-'),
                ([170 / n, 385 / n], [145 / n, 300 / n], '-'),
                ([385 / n, maxX], [300 / n, self._endy(385 / n, 300 / n, maxX, cbl)], '-'),

                ([0 / n, 30 / n], [60 / n, 60 / n], '-'),
                ([30 / n, 50 / n], [60 / n, 80 / n], '-'),
                ([50 / n, 70 / n], [80 / n, 110 / n], '-'),
                ([70 / n, self._endx(70/n, 110/n, maxY, ccu)], [110 / n, maxY], '-'),

                ([120 / n, 120 / n], [0 / n, 30 / n], '-'),
                ([120 / n, 260 / n], [30 / n, 130 / n], '-'),
                ([260 / n, maxX], [130 / n, self._endy(260 / n, 130 / n, maxX, ccl)], '-'),

                ([0 / n, 25 / n], [100 / n, 100 / n], '-'),
                ([25 / n, 50 / n], [100 / n, 125 / n], '-'),
                ([50 / n, 80 / n], [125 / n, 215 / n], '-'),
                ([80 / n, self._endx(80 / n, 215 / n, maxY, cdu)], [215 / n, maxY], '-'),

                ([250 / n, 250 / n], [0 / n, 40 / n], '-'),
                ([250 / n, maxX], [40 / n, self._endy(410 / n, 110 / n, maxX, cdl)], '-'),

                ([0 / n, 35 / n], [150 / n, 155 / n], '-'),
                ([35 / n, self._endx(35 / n, 155 / n, maxY, ce)], [155 / n, maxY], '-'),
            ]

        elif self.type == 2:
            ce =  self._coef(35, 200, 50, 550)
            cdu = self._coef(35, 90, 125, 550)
            cdl = self._coef(410, 110, 550, 160)
            ccu = self._coef(30, 60, 280, 550)
            ccl = self._coef(260, 130, 550, 250)
            cbu = self._coef(230, 330, 440, 550)
            cbl = self._coef(330, 230, 550, 450)

            _gridlines = [
                ([0, min(maxX, maxY)], [0, min(maxX, maxY)], ':'),
                ([0, 30/n], [50/n, 50/n], '-'),
                ([30 / n, 230 / n], [50 / n, 330 / n], '-'),
                ([230 / n, self._endx(230/n, 330/n, maxY, cbu)], [330/n, maxY], '-'),
                ([50/n, 50/n], [0/n, 30/n], '-'),
                ([50/n, 90/n], [30/n, 80/n], '-'),
                ([90/n, 330/n], [80/n, 230/n], '-'),
                ([330/n, maxX], [230/n, self._endy(330/n, 230/n, maxX, cbl)], '-'),
                ([0/n, 30/n], [60/n, 60/n], '-'),
                ([30/n, self._endx(30/n, 60/n, maxY, ccu)], [60/n, maxY], '-'),
                ([90/n, 260/n], [0/n, 130/n], '-'),
                ([260/n, maxX], [130/n, self._endy(260/n, 130/n, maxX, ccl)], '-'),
                ([0/n, 25/n], [80/n, 80/n], '-'),
                ([25/n, 35/n], [80/n, 90/n], '-'),
                ([35/n, self._endx(35/n, 90/n, maxY, cdu)], [90/n, maxY], '-'),
                ([250/n, 250/n], [0/n, 40/n], '-'),
                ([250/n, 410/n], [40/n, 110/n], '-'),
                ([410/n, maxX], [110/n, self._endy(410/n, 110/n, maxX, cdl)], '-'),
                ([0/n, 35/n], [200/n, 200/n], '-'),
                ([35/n, self._endx(35/n, 200/n, maxY, ce)], [200/n, maxY], '-'),
            ]

        colors = ['#196600', '#7FFF00', '#FF7B00', '#FF5700', '#FF0000']

        _gridlabels = [
            (600, 600, "A", colors[0]),
            (360, 600, "B", colors[1]),
            (600, 355, "B", colors[1]),
            (165, 600, "C", colors[2]),
            (600, 215, "C", colors[2]),
            (600, 50, "D", colors[3]),
            (75, 600, "D", colors[3]),
            (5, 600, "E", colors[4])
        ]


        # plot individual points
        if self.color_points == 'auto':
            ax.scatter(self.reference,
                       self.test,
                       marker='o',
                       alpha=0.6,
                       c=[colors[i] for i in self._calc_error_zone()], s=8,
                       **self.point_kws)
        else:
            ax.scatter(self.reference,
                       self.test, marker='o',
                       color=self.color_points,
                       alpha=0.6,
                       s=8,
                       **self.point_kws)

        # plot grid lines
        if self.grid:
            for g in _gridlines:
                ax.plot(np.array(g[0]),
                        np.array(g[1]),
                        g[2], color=self.color_grid,
                        **self.grid_kws)

            if self.percentage:
                zones = [['A', 'B', 'C', 'D', 'E'][i] for i in self._calc_error_zone()]

                for l in _gridlabels:
                    ax.text(l[0] / n, l[1] / n, l[2],
                            fontsize=12,
                            fontweight='bold',
                            color=l[3] if self.color_gridlabels == 'auto' else self.color_gridlabels)
                    ax.text(l[0] / n + (18 / n),
                            l[1] / n + (18 / n),
                            "{:.1f}".format((zones.count(l[2])/len(zones))*100),
                            fontsize=9,
                            fontweight='bold',
                            color=l[3] if self.color_gridlabels == 'auto' else self.color_gridlabels)

            else:
                for l in _gridlabels:
                    ax.text(l[0] / n, l[1] / n, l[2],
                            fontsize=12,
                            fontweight='bold',
                            color=l[3] if self.color_gridlabels == 'auto' else self.color_gridlabels)

        # limits and ticks
        _ticks = [70, 100, 150, 180, 240, 300, 350, 400, 450, 500,
                  550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

        ax.set_xticks([round(x/n, 1) for x in _ticks])
        ax.set_yticks([round(x/n, 1) for x in _ticks])
        ax.set_xlim(0, maxX)
        ax.set_ylim(0, maxY)

        # graph labels
        ax.set_ylabel(self.y_title)
        ax.set_xlabel(self.x_title)
        if self.graph_title is not None:
            ax.set_title(self.graph_title)


def parkes(type, reference, test, units,
           x_label=None, y_label=None, title=None,
           xlim=None, ylim=None,
           color_grid='#000000', color_gridlabels='auto', color_points='auto',
           grid=True, percentage=False,
           point_kws=None, grid_kws=None,
           square=False, ax=None):
    """Provide a glucose error grid analyses as designed by Parkes.

    This is an Axis-level function which will draw the Parke-error grid plot.
    onto the current active Axis object unless ``ax`` is provided.


    Parameters
    ----------
    type : int
        Parkes error grid differ for each type of diabetes. This should be either 1 or 2 corresponding
        to the type of diabetes.
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
        Title of the Parkes-error grid plot. If None is provided, no title will be plotted.
    xlim : list, optional
        Minimum and maximum limits for X-axis. Should be provided as list or tuple.
        If not set, matplotlib will decide its own bounds.
    ylim : list, optional
        Minimum and maximum limits for Y-axis. Should be provided as list or tuple.
        If not set, matplotlib will decide its own bounds.
    color_grid : str, optional
        Color of the Clarke error grid lines. Defaults to #000000 which represents the black color.
    color_gridlabels : str, optional
        Color of the grid labels (A, B, C, ..) that will be plotted. Defaults to 'auto' which colors
        the points according to their relative zones.
    color_points : str, optional
        Color of the individual differences that will be plotted. Defaults to 'auto' which colors
        the points according to their relative zones.
    grid : bool, optional
        Enable the grid lines of the Parkes error. Defaults to True.
    percentage : bool, optional
        If True, percentage of the zones will be depicted in the plot.
    square : bool, optional
        If True, set the Axes aspect to "equal" so each cell will be square-shaped.
    point_kws : dict of key, value mappings, optional
        Additional keyword arguments for `plt.scatter`.
    grid_kws : dict of key, value mappings, optional
        Additional keyword arguments for the grid with `plt.plot`.
    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the Parkes error grid plot.

    See Also
    -------
    Parkes, J. L., Slatin S. L. et al. Diabetes Care, vol. 23, no. 8, 2000, pp. 1143-1148.
    Pfutzner, A., Klonoff D. C., et al. J Diabetes Sci Technol, vol. 7, no. 5, 2013, pp. 1275-1281.
    """

    plotter: _Parkes = _Parkes(type, reference, test, units,
                               x_label, y_label, title,
                               xlim, ylim,
                               color_grid, color_gridlabels, color_points,
                               grid, percentage,
                               point_kws, grid_kws)

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()

    if square:
        ax.set_aspect('equal')

    plotter.plot(ax)

    return ax

def parkeszones(type, reference, test, units,
                numeric=False):
    """Provides the error zones as depicted by the
    Parkes error grid analysis for each point in the reference and test datasets.


    Parameters
    ----------
    type : int
        Parkes error grid differ for each type of diabetes. This should be either 1 or 2 corresponding
        to the type of diabetes.
    reference, test : array, or list
        Glucose values obtained from the reference and predicted methods, preferably provided in a np.array.
    units : str
        The SI units which the glucose values are provided in. Options: 'mmol', 'mgdl' or 'mg/dl'.
    numeric : bool, optional
        If this is set to true, returns integers (0 to 4) instead of characters for each of the
        zones.

    Returns
    -------
    parkeszones : list
        Returns a list depicting the zones for each of the reference and test values.

    """

    # obtain zones from a Clarke reference object
    _zones = _Parkes(type, reference, test, units,
                     None, None, None,
                     None, None,
                     True, False,
                     '#000000', 'auto', 'auto',
                     None, None)._calc_error_zone()

    if numeric:
        return _zones
    else:
        labels = ['A', 'B', 'C', 'D', 'E']
        return [labels[i] for i in _zones]


class _SEG(object):
    """Internal class for drawing a surveillance error grid error grid plot"""

    def __init__(self, reference, test, units,
                 x_title, y_title, graph_title,
                 xlim, ylim,
                 color_points,
                 percentage,
                 point_kws):
        # variables assignment
        self.reference: np.array = np.asarray(reference)
        self.test: np.array = np.asarray(test)
        self.units = units
        self.graph_title: str = graph_title
        self.x_title: str = x_title
        self.y_title: str = y_title
        self.xlim: list = xlim
        self.ylim: list = ylim
        self.color_points: str = color_points
        self.percentage: bool = percentage
        self.point_kws = {} if point_kws else point_kws.copy()

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


    def _calc_error_score(self):
        n = 18 if self.units == 'mmol' else 1
        ref = self.reference * n
        pred = self.test * n

        _zones = []
        data = np.loadtxt('../static/seg.csv')
        _zones = np.array([data.T[int(p), int(t)] for p, t in zip(pred, ref)])

        return _zones

    def plot(self, ax):
        # ref, pred
        ref = self.reference
        pred = self.test
        _data = np.loadtxt('../static/seg.csv')

        # calculate conversion factor if needed
        n = 18 if self.units == 'mmol' else 1

        maxX = self.xlim or 600
        maxY = self.ylim or 600

        # Define colormaps
        _colors = [(0, 165 / 256, 0), (0, 255 / 256, 0), (255 / 256, 255 /256, 0),
                           (255 / 256, 0, 0), (128 / 256, 0, 0)]
        _nodes = [0.0, 0.4375, 1.0625, 2.7500, 4.000]

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",
                                                                   list(zip([x/4 for x in _nodes],
                                                                            _colors)))

        # Plot color axes
        cax = ax.imshow(np.flipud(np.array(plt.imread('../static/seg600.png'))),
                        origin='lower',
                        cmap=cmap,
                        vmin=0, vmax=4)

        # Plot color bar
        cbar = plt.colorbar(cax,
                            ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                            orientation='vertical',
                            fraction=0.15,
                            aspect=6)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.set_ylabel('Risk score')

        # Separators
        for s in [0, 0.5, 1.5, 2.5, 3.5, 4]:
            cbar.ax.plot([6, 6.5], [s]*2, '-', color='black', lw=1, alpha=1, clip_on=False)

        # Labels
        for l in [(0.25, 'None'), (1.0, 'Slight'), (2.0, 'Moderate'),
                  (3.0, 'High'), (3.75, 'Extreme')]:
            cbar.ax.text(6.2, l[0]-0.008, l[1], ha='left', va='center', rotation=0, fontsize=10)

        if self.percentage:
            seg_scores = self._calc_error_score()
            _zones_sub = [[] for _ in range(8)]
            edges = list(np.arange(0, 4.5, 0.5))

            for x in range(len(edges)-1):
                _zones_sub[x] = np.array(seg_scores[(seg_scores >= edges[x]) &
                                                    (seg_scores < edges[x+1])])

            perc_zones = [(len(x)/len(seg_scores))*100 for x in _zones_sub]

            for i, x in enumerate(perc_zones):
                cbar.ax.plot([0, 5], [(i*.5) + .5]*2, '--', color='grey', lw=1, alpha=.6)
                if x > 0:
                    _str = "<0.01%" if round(x, 2) == 0 else "{:.2f}%".format(x)
                    cbar.ax.text(2, (i*.5)+.25, _str, ha='center', va='center', fontsize=9)



        ax.scatter(self.reference*n,
                   self.test*n, marker='o',
                   color=self.color_points,
                   alpha=0.6,
                   s=8,
                   **self.point_kws)

        # limits and ticks
        _ticks = [0, 90, 180, 270, 360, 450, 540]

        ax.set_xticks([round(x, 1) for x in _ticks])
        ax.set_yticks([round(x, 1) for x in _ticks])
        ax.set_xticklabels([round(x/n, 1) for x in _ticks])
        ax.set_yticklabels([round(x / n, 1) for x in _ticks])

        ax.set_xlim(0, maxX)
        ax.set_ylim(0, maxY)

        # graph labels
        ax.set_ylabel(self.y_title)
        ax.set_xlabel(self.x_title)
        if self.graph_title is not None:
            ax.set_title(self.graph_title)


def seg(reference, test, units,
        x_label=None, y_label=None, title=None,
        xlim=None, ylim=None,
        color_points='white',
        percentage=False,
        point_kws=None,
        square=False, ax=None):
    """Provide a glucose error grid analyses as designed by the surveillance error grid.

    This is an Axis-level function which will draw the surveillance error grid plot.
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
        Title of the plot. If None is provided, no title will be plotted.
    xlim : list, optional
        Minimum and maximum limits for X-axis. Should be provided as list or tuple.
        If not set, matplotlib will decide its own bounds.
    ylim : list, optional
        Minimum and maximum limits for Y-axis. Should be provided as list or tuple.
        If not set, matplotlib will decide its own bounds.
    color_points : str, optional
        Color of the individual differences that will be plotted. Defaults to 'white'.
    percentage : bool, optional
        If True, percentage of the zones will be depicted in the plot.
    point_kws : dict of key, value mappings, optional
        Additional keyword arguments for `plt.scatter`.
    square : bool, optional
        If True, set the Axes aspect to "equal" so each cell will be square-shaped.
    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the Surveillance error grid plot.

    See Also
    -------
    Klonoff, D. C., Lias, C., et al. J Diabetes Sci Technol, vol. 8, no. 4, 2014, pp 658-672.
    Kovatchev, B. P., Wakeman, C. A., et al. J Diabetes Sci Technol, vol 8, no. 4, 2014, pp. 673-684.
    """

    plotter: _SEG = _SEG(reference, test, units,
                         x_label, y_label, title,
                         xlim, ylim,
                         color_points,
                         percentage,
                         point_kws)

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()

    if square:
        ax.set_aspect('equal')

    plotter.plot(ax)

    return ax

def segscores(reference, test, units):
    """Provides the raw error values as depicted by the
    surveillance error grid analysis for each point in the reference and test datasets.


    Parameters
    ----------
    reference, test : array, or list
        Glucose values obtained from the reference and predicted methods, preferably provided in a np.array.
    units : str
        The SI units which the glucose values are provided in. Options: 'mmol', 'mgdl' or 'mg/dl'.


    Returns
    -------
    segscores : list
        Returns a list with a SEG score for each test, reference pair.

    """

    # obtain zones from a Clarke reference object
    _zones = _SEG(reference, test, units,
                     None, None, None,
                     None, None,
                     True, False,
                     '#000000', 'auto', 'auto')._calc_error_score()

    return _zones