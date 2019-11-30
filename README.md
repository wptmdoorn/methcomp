methcomp: method comparison for clinical chemistry
=======================================
<div>
<img src="https://i.ibb.co/qj85D6M/combined.png" width="60%">
</div> 
--------------------------------------

Methcomp is a Python visualization library based 
and built on matplotlib. It provides a high-level interface for 
method comparison for clinical chemisty methods, amongst others.

Documentation
-------------
Online documentation is currently only available through the docstrings
of available functions. Methcomp currently provides two functions for method comparison.

    blandaltman(method1, method2,
                x_label='Mean of methods', y_label='Difference between methods', title=None,
                diff='absolute', limit_of_agreement=1.96, reference=False, CI=0.95,
                color_mean='#008bff', color_loa='#FF7000', color_points='#000000',
                ax=None):
              
Generate a Bland-Altman plot to compare two sets of measurements of the same value.

    passingbablok(method1, method2,
                      x_label='Method 1', y_label='Method 2', title=None,
                      CI=0.95, line_reference=True, line_CI=False, legend=True,
                      color_points='#000000', color_paba='#008bff',
                      square=False, ax=None):
                                          
Generate a Passing-Bablok plot to compare two sets of measurements of the same value.

Dependencies
------------

Seaborn supports Python 2.7 and 3.5+.
Installation requires [numpy](http://www.numpy.org/), 
[scipy](https://www.scipy.org/), 
[pandas](https://pandas.pydata.org/), 
and [matplotlib](https://matplotlib.org/).


Installation
------------

Installation is currently not available yet. Users interested in using this package should 
fork the Github repository and use it locally. 

Testing
-------

N/A.