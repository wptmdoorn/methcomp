import matplotlib.pyplot as plt
import math
import numpy as np

__all__ = ["passingbablok"]


class _PassingBablok(object):
    """Internal class for drawing a Passing-Bablok regression plot"""

    def __init__(self, method1, method2,
                 color_points):
        self.method1 = method1
        self.method2 = method2
        self.color_points = color_points
        self.n = len(method1)

        self._check_params()
        self._derive_params()

    def _check_params(self):
        if len(self.method1) != len(self.method2):
            raise ValueError('Length of method 1 and method 2 are not equal.')

    def _derive_params(self):
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

        self.intercept = np.median(self.method2 - self.slope * self.method1)

    def plot(self, ax):
        # plot individual points
        ax.scatter(self.method1, self.method2, s=20, alpha=0.6, color=self.color_points)

        # plot reference line
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--', transform=ax.transAxes)

        # plot PaBa-line
        _yvals = self.intercept + self.slope * np.array(ax.get_xlim())
        ax.plot(np.array(ax.get_xlim()), _yvals, linestyle='-')


def passingbablok(method1, method2,
                  color_points='#000000', ax=None):
    plotter: _PassingBablok = _PassingBablok(method1, method2, color_points)

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()

    plotter.plot(ax)

    return ax
