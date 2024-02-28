# -*- coding: utf-8 -*-

"""Abstract comparer base class
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence, Union

import numpy as np


class Comparer(ABC):

    """Abstract method comparison base class"""

    def __init__(
        self,
        method1: Union[Sequence[float], np.ndarray],
        method2: Union[Sequence[float], np.ndarray],
    ):
        """Build comparer

        Parameters
        ----------
        method1 : Union[List[float], np.ndarray]
            Values for method 1
        method2 : Union[List[float], np.ndarray]
            Values for method 2
        """
        # Process args
        self._method1 = np.asarray(method1)
        self._method2 = np.asarray(method2)
        self._check_params()

        # Additional members
        self._result: Dict[str, Any] = {}
        self._calculated = False
        self._n = len(method1)

    @property
    def calculated(self) -> bool:
        """True if result is calculated

        Returns
        -------
        vool
            True if calculated
        """
        return self._calculated

    @property
    def n(self) -> int:
        """Length of method value vectors

        Returns
        -------
        int
            Length of method vectors
        """
        return self._n

    @property
    def method1(self) -> np.ndarray:
        """Values for method 1

        Returns
        -------
        np.ndarray
            Values for method 1
        """
        return self._method1

    @property
    def method2(self) -> np.ndarray:
        """Values for method 2

        Returns
        -------
        np.ndarray
            Values for method 2
        """
        return self._method2

    @property
    def result(self) -> Dict[str, Any]:
        """Get result, calculate if necessary

        Returns
        -------
        Dict[str, Any]
            Calculation result
        """
        if not self.calculated:
            self.calculate()
        return self._result

    def _check_params(self):
        """Check validity of parameters.

        Note: extending classes that need to check parameters should call
        `super()._check_params()`
        Note: this is called from `__init__`, extending classes should
        call `super().__init__(...)` after initialising members in `__init__`

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
        return self._result

    @abstractmethod
    def plot(self):
        """Plot calculated result."""
        pass
