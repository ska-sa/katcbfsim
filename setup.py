#!/usr/bin/env python

from setuptools import setup

tests_require=['nose', 'scipy', 'mock']

setup(
    author='Bruce Merry',
    author_email='bmerry@ska.ac.za',
    name='katcbfsim',
    package_data={'': ['*.mako']},
    include_package_data=True,
    description='Simulator for MeerKAT correlator',
    scripts=['scripts/cbfsim.py'],
    setup_requires=['katversion'],
    install_requires=['katcp', 'trollius', 'tornado>=4.2', 'spead2', 'katpoint', 'numpy', 'h5py',
        'katsdpsigproc[CUDA]', 'katsdptelstate'],
    tests_require=tests_require,
    extras_require={'test': tests_require, 'doc': ['sphinx>=1.3']},
    packages=['katcbfsim'],
    use_katversion=True
)
