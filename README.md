methcomp: method comparison for clinical chemistry
=======================================
<div>
<img src="https://wptmdoorn.name/assets/02methcomp/methcomp1.png" height="180">
<img src="https://wptmdoorn.name/assets/02methcomp/methcomp3.png" height="180"> 
<img src="https://wptmdoorn.name/assets/02methcomp/methcomp2.png" height="180">
<img src="https://wptmdoorn.name/assets/02methcomp/methcomp4.png" height="180">
</div> <br>

--------------------------------------

[![PyPI Version](https://img.shields.io/badge/pypi-v1.0.0-blue)](https://pypi.org/project/methcomp/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](https://github.com/wptmdoorn/methcomp/blob/master/LICENSE)
[![Build Status](https://travis-ci.org/wptmdoorn/methcomp.svg?branch=master)](https://travis-ci.org/wptmdoorn/methcomp)
[![Code Coverage](https://codecov.io/gh/wptmdoorn/methcomp/branch/master/graph/badge.svg)](https://codecov.io/gh/wptmdoorn/methcomp)

Methcomp is a Python visualization library based 
and built on matplotlib. It provides a high-level interface for 
method comparison for clinical chemistry methods, amongst others.

Documentation
------------- 
Online documentation is currently only available by clicking on [here](http://wptmdoorn.name/methcomp/docs/).  
Illustrative examples using the methcomp library can be found in examples/ directory and in the blog post 
[here](https://wptmdoorn.name/Method-Comparison/).

Dependencies
------------
Methcomp supports Python 2 and 3.
Installation requires [numpy](http://www.numpy.org/), 
[scipy](https://www.scipy.org/), 
[pandas](https://pandas.pydata.org/), 
and [matplotlib](https://matplotlib.org/).

Installation
------------

The latest stable release (and older versions) can be installed from PyPI:

    pip install methcomp

Otherwise, you may instead want to use the development version from directly through Github:

    pip install git+https://github.com/wptmdoorn/methcomp

Testing
-------
To test the code, run `make pytest` in the source directory. This will run the tests in the tests/ directory. 
