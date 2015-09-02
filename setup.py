#!/usr/bin/env python

from setuptools import setup

setup(
    author='Bruce Merry',
    author_email='bmerry@ska.ac.za',
    name='cbfsim',
    version='0.1.dev0',
    description='Simulator for MeerKAT correlator',
    install_requires=['katcp', 'trollius', 'tornado>=4.2', 'spead2', 'katpoint', 'numpy',
        'katsdpsigproc', 'pycuda'],
    tests_require=['nose'],
    packages=['katcbfsim']
)
