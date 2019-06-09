#!/usr/bin/env python
import os
from setuptools import find_packages

with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

from numpy.distutils.core import setup, Extension


setup_options = dict(
    name='doe-mps',
    version="0.1.1",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url='https://github.com/kirthevasank/mps/',
    license='MIT',
    author_email='kandasamy@cs.cmu.edu',
    packages=['mps', 'mps.exd', 'mps.policies', 'mps.utils', 'mps.prob'],
    install_requires=[
      'future',
      'numpy',
      'scipy',
      'six',
    ],
    classifiers=[
      "Intended Audience :: Developers",
      "Intended Audience :: Education",
      "Intended Audience :: Science/Research",
      "License :: OSI Approved :: MIT License",
      "Operating System :: MacOS",
      "Operating System :: Microsoft :: Windows",
      "Operating System :: POSIX :: Linux",
      "Operating System :: Unix",
      "Programming Language :: Python :: 2",
      "Programming Language :: Python :: 3",
      "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

setup(**setup_options)

