methcomp: method comparison for clinical chemistry
=======================================
<div>
<img src="https://raw.githubusercontent.com/wptmdoorn/methcomp/master/doc/_static/example_1.png" height="180">
<img src="https://raw.githubusercontent.com/wptmdoorn/methcomp/master/doc/_static/example_2.png" height="180">
<img src="https://raw.githubusercontent.com/wptmdoorn/methcomp/master/doc/_static/example_3.png" height="180">
<img src="https://raw.githubusercontent.com/wptmdoorn/methcomp/master/doc/_static/example_4.png" height="180">
</div> <br>

***

[![PyPI Version](https://img.shields.io/badge/pypi-v1.0.0-blue)](https://pypi.org/project/methcomp/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](https://github.com/wptmdoorn/methcomp/blob/master/LICENSE)
[![Build Status](https://travis-ci.org/wptmdoorn/methcomp.svg?branch=master)](https://travis-ci.org/wptmdoorn/methcomp)
[![Code Coverage](https://codecov.io/gh/wptmdoorn/methcomp/branch/master/graph/badge.svg)](https://codecov.io/gh/wptmdoorn/methcomp)

Methcomp is a Python visualization library based
and built on matplotlib. It provides a high-level interface for
method comparison for clinical chemistry methods, amongst others.

## Documentation
Online documentation is currently only available by clicking on [here](https://methcomp.readthedocs.io/en/latest/).
Illustrative examples using the methcomp library can be found in examples/ directory and in the blog post
[here](https://wptmdoorn.name/Method-Comparison/).

## Dependencies
Methcomp supports Python 2 and 3.
Installation requires [numpy](http://www.numpy.org/),
[scipy](https://www.scipy.org/),
[pandas](https://pandas.pydata.org/),
and [matplotlib](https://matplotlib.org/).****

## Installation
The latest stable release (and older versions) can be installed from PyPI:

    pip install methcomp

Otherwise, you may instead want to use the development version from directly through Github:

    pip install git+https://github.com/wptmdoorn/methcomp

## Development

### Testing

To test the code, run `make pytest` in the source directory. This will run the tests in the tests/ directory.

### Pre-commit

This project supprts [Pre-commit](https://pre-commit.com/), to install it run
```sh

pip install pre-commit
pre-commit install
```
at the top level of this repo. After this `git commit` should run the chain of commands listed in `.pre-commit-config.yaml`. To force running the chain to run use

```sh

pre-commit run --all-files
```

A number of basic checks are executed by pre-commit, syntax checks for toml and yaml files, trimming of trailing whitespaces and ensuring newline at end of file, checking for merge conflict strings, warning about python debug statements, and fixing the python encoding pragmas at top of files. Pre-commit also runs [Mypy](https://mypy.readthedocs.io/en/stable/index.html) as a static type checker, [isort](https://pycqa.github.io/isort/) to sort imports, [black](https://black.readthedocs.io/en/stable/) to format code, and [flake8](https://flake8.pycqa.org/en/latest/) as a linter.
