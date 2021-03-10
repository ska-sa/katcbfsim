#!/usr/bin/env python3

from setuptools import setup, find_packages

tests_require = ['nose', 'mock', 'asynctest']

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
        'aiokatcp',
        'h5py',
        'katpoint',
        'katsdpservices[argparse,aiomonitor]',
        'katsdpsigproc[cuda]',
        'katsdptelstate',
        'numba',
        'numpy',
        'scipy',
        'spead2>=3.1'
    ],
    tests_require=tests_require,
    extras_require={'test': tests_require, 'doc': ['sphinx>=1.3']},
    packages=find_packages(),
    use_katversion=True
)
