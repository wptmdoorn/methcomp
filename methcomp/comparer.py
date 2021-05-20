"""Abstract comparer base class
"""
from typing import Any, Dict
from abc import ABC, abstractmethod
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Comparer(ABC):

    """Abstract method comparison base class

    Attributes
    ----------
    calculated : bool
        True if result is calculated
    method1 : np.ndarray
        Values for method 1
    method2 : np.ndarray
        Values for method 2
    n : int
        Length of method value vectors
    result : dict
        Calculation result
    """

    def __init__(self, method1: np.ndarray, method2: np.ndarray):
        """Build comparer

        Parameters
        ----------
        method1 : np.ndarray
            Values for method 1
        method2 : np.ndarray
            Values for method 2
        """
        # Process args
        self._method1 = np.asarray(method1)
        self._method2 = np.asarray(method2)
        self._check_params()

        # Additional members
        self.result = {}
        self._calculated = False
        self._n = len(method1)

    @property
    def calculated(self):
        """True if result is calculated"""
        return self._calculated

    @property
    def n(self):
        """Length of method value vectors"""
        return self._n

    @property
    def method1(self):
        """Values for method 1"""
        return self._method1

    @property
    def method2(self):
        """Values for method 2"""
        return self._method2

    def _check_params(self):
        """Check validity of parameters.

        Raises
        ------
        ValueError
            If method values are of different shape or CI outside of range 0,1
        """
        if self.method1.shape != self.method2.shape:
            raise ValueError("Length of method 1 and method 2 are not equal.")

    @abstractmethod
    def _calculate_impl(self):
        """Parameter calculation implementation.

        This function fills `Comparer.result`

        See Also
        --------
        Comparer.calculate
        """
        pass

    def calculate(self) -> Dict[str, Any]:
        """Calculate parameters.

        Calls `_calculate_impl`

        Returns
        -------
        Dict[str, Any]
            Dictionary of calculated results

        See Also
        --------
        Comparer._calculate_impl
        """
        self._calculate_impl()
        self._calculated = True
        return self.result

    @abstractmethod
    def plot(self, ax: matplotlib.axes.Axes = None) -> matplotlib.axes.Axes:
        """Plot calculated result.

        If necessary perform calculation

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            matplotlib axis object, if not passed, uses gca()

        Returns
        -------
        matplotlib.axes.Axes
            axes for plot
        """
        if not self.calculated:
            self.calculate()
        return ax or plt.gca()
