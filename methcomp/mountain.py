# -*- coding: utf-8 -*-

"""Mountain plot.
"""
from typing import List, Union
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from .comparer import Comparer

class Mountain(Comparer):

    """Mountain plot
    
    Attributes
    ----------
    n_percentiles : TYPE
        Description
    result : Dict[str, Any]
        Mountain calculation result with "mountain", "quantile", "auc", "median"

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from methcomp import mountain
    >>> method1 = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    >>> method2 = np.asarray([1.03,2.05,2.79,3.67,5.00,5.82,7.16,7.69,8.53,10.38,11.11,12.17,13.47,13.83,15.15,16.12,16.94,18.09,19.13,19.54,])
    >>> q=mountain.Mountain(method1, method2, 100)
    >>> q.calculate()
    >>> q.plot()
    >>> plt.show()
    """
    
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
        """Plot mountain plot
        
        Parameters
        ----------
        xlabel : str, optional
            The label which is added to the X-axis. (default: "Method difference")
        ylabel : str, optional
            The label which is added to the Y-axis. (default: "Folded CDF (%)")
        label : str, optional
            mountaint line legend label (default:"Method 1 - Method 2" )
        title : str, optional
            figure title, if none there will be no title
            (default: None)
        ax : matplotlib.axes.Axes, optional
            matplotlib axis object, if not passed, uses gca()
        
        Returns
        ------------------
        matplotlib.axes.Axes
            axes object with the plot
        """
        ax = ax or plt.gca()
        ax.step(
            y=self.result["mountain"], 
            x=self.result["quantile"], 
            where='mid', 
            label=label)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title or "")
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
        """Calculate mountain parameters.
        """

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
