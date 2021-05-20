"""Abstract comparer base class"""
from typing import Any, Dict
from abc import ABC, abstractmethod
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Comparer(ABC):

    """Abstract comparer base class

    Attributes
    ----------
    calculated : bool
        True if result is calculated
    method1 : np.ndarray
        Values for method 1
    method2 : np.ndarray
        Values for method 2
    n : int
        length of method value vectors
    result : dict
        caclulation result
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
        self.method1 = np.asarray(method1)
        self.method2 = np.asarray(method2)
        self._check_params()

        # Additional members
        self.result = {}
        self.calculated = False
        self.n = len(method1)

    def _check_params(self):
        """Check validity of parameters

        Raises
        ------
        ValueError
            If method values are of different shape or CI outside of range 0,1
        """
        if self.method1.shape != self.method2.shape:
            raise ValueError("Length of method 1 and method 2 are not equal.")

    @abstractmethod
    def calculate_impl(self):
        """Calculation implementation

        See Also
        --------
        calculate
        """

    def calculate(self) -> Dict[str, Any]:
        """Calculate regression parameters.

        Returns
        -------
        Dict[str, Any] : Dictionary of calculation results
        """
        self._calculate_impl()
        self.calculated = True
        return self.result


    @abstractmethod
    def plot(self, ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
        """Plot calculated result.

        If necessary perform calculation

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            matplotlib axis object, if not passed, uses gca()

        Returns
        -------
        matplotlib.axes.Axes : axes for plot
        """
        if not self.calculated:
            self.calculate()

        return ax or plt.gca()