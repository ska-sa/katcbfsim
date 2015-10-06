#!/usr/bin/env python

from setuptools import setup

setup(
    author='Bruce Merry',
    author_email='bmerry@ska.ac.za',
    name='katcbfsim',
    version='0.1.dev0',
    package_data={'': ['*.mako']},
    include_package_data=True,
    description='Simulator for MeerKAT correlator',
    scripts=['scripts/cbfsim.py'],
    install_requires=['katcp', 'trollius', 'tornado>=4.2', 'spead2', 'katpoint', 'numpy', 'h5py',
        'katsdpsigproc', 'katsdptelstate', 'pycuda'],
    tests_require=['nose', 'scipy'],
    packages=['katcbfsim']
)
