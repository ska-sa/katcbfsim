#!/usr/bin/env python3

from setuptools import setup, find_packages

tests_require = ['nose', 'scipy', 'mock', 'asynctest']

setup(
    author='MeerKAT SDP Team',
    author_email='sdpdev+katcbfsim@ska.ac.za',
    name='katcbfsim',
    package_data={'': ['*.mako']},
    include_package_data=True,
    description='Simulator for MeerKAT correlator',
    scripts=['scripts/cbfsim.py'],
    setup_requires=['katversion'],
    install_requires=[
        'aiokatcp', 'spead2>=1.11.1', 'katpoint', 'numpy', 'h5py',
        'katsdpsigproc[CUDA]', 'katsdptelstate', 'netifaces', 'numba',
        'katsdpservices'],
    tests_require=tests_require,
    extras_require={'test': tests_require, 'doc': ['sphinx>=1.3']},
    packages=find_packages(),
    use_katversion=True
)
