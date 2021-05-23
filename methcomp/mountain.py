# -*- coding: utf-8 -*-

"""Mountain plot.
"""
from typing import List, Union
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from .comparer import Comparer

class Mountain(Comparer):

    def __init__(
        self,
        method1: Union[List[float], np.ndarray],
        method2: Union[List[float], np.ndarray],
        n_percentiles: int=100):
        """Construct a regressor.
        
        Parameters
        ----------
        method1 : Union[List[float], np.ndarray]
            Values for method 1
        method2 : Union[List[float], np.ndarray]
            Values for method 2
        n_percentiles : int, optional
            Number of percentile streps - more give a smoother mountain
            (default n=100)
        """
        self.n_percentiles = n_percentiles
        # Process args
        super().__init__(method1, method2)

    def plot(
        self,
        xlabel: str = "Method difference",
        ylabel: str = "Folded CDF (%)",
        label: str = "Method 1 - Method 2",
        title: str = None,
        ax: matplotlib.axes.Axes = None) -> matplotlib.axes.Axes:
        ax = ax or plt.gca()
        ax.step(
            y=self.result["mountain"], 
            x=self.result["quantile"], 
            where='mid', 
            label=label)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.axvline(self.result["median"], label=f"median={self.result['median']:.2f}", linestyle=":")
        ax.legend(title=f"AUC={self.result['auc']:.2f}")
        return ax

    def _check_params(self):
        """Check validity of parameters

        Raises
        ------
        ValueError
            If method values are of different shape or CI outside of range 0,1
        """
        super()._check_params()
        if self.n_percentiles <= 0:
            raise ValueError("Number of percentile steps should be positive")


    def _calculate_impl(self):
        """Calculate mountain parameters."""

        # quantile values to evaluate
        qrange=np.linspace(0, 1, self.n_percentiles)
        quantile = np.quantile(self.method1-self.method2, qrange)
        # Split qrange in the middle and convert to percentile
        mountain=np.where(qrange<0.5,qrange,1.0-qrange)*100
        # Calcualte area under curve
        auc = (np.diff(quantile)*(mountain[:-1]+mountain[1:])/2).sum()
        # Build result
        self.result = {
            "mountain": mountain,
            "quantile": quantile,
            "auc": auc,
            "median": quantile[self.n_percentiles//2]
        }
