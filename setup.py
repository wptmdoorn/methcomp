#! /usr/bin/env python
#
# Copyright (C) 2019 William P.T.M. van Doorn

DESCRIPTION = "methcomp: method comparison for clinical chemistry"
LONG_DESCRIPTION = """\
Methcomp is a Python visualization library based and built on `matplotlib <https://matplotlib.org/>`_. It provides a high-level interface for method comparison for clinical chemisty methods, amongst others.
Here is some of the functionality that methcomp currently offers:
- Method comparison using Bland-Altman plots;
- Method comparison using regression techniques such as Passing-Bablok, Deming and simple linear regression;
- Method comparison for glucose sensors using Clarke and Parkes error grids.
"""

DISTNAME = 'methcomp'
MAINTAINER = 'William P.T.M. van Doorn'
MAINTAINER_EMAIL = 'wptmdoorn@gmail.com'
URL = 'https://github.com/wptmdoorn/methcomp'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/wptmdoorn/methcomp/archive/v1.0.0.tar.gz'
VERSION = '1.0.0'

INSTALL_REQUIRES = [
    'numpy>=1.17.2',
    'scipy>=1.3.1',
    'matplotlib>=3.1.2',
    'pandas',
    'shapely',
    'statsmodels'
]

PACKAGES = [
    'methcomp',
]

CLASSIFIERS = [
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Visualization",
]

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

if __name__ == "__main__":

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        install_requires=INSTALL_REQUIRES,
        packages=PACKAGES,
        classifiers=CLASSIFIERS
    )
