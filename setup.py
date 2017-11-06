#!/usr/bin/env python

from setuptools import setup, find_packages

tests_require=['nose', 'scipy', 'mock', 'asynctest']

setup(
    author='Bruce Merry',
    author_email='bmerry@ska.ac.za',
    name='katcbfsim',
    package_data={'': ['*.mako']},
    include_package_data=True,
    description='Simulator for MeerKAT correlator',
    scripts=['scripts/cbfsim.py'],
    setup_requires=['katversion'],
    install_requires=[
        'aiokatcp', 'spead2>=1.4.0', 'katpoint', 'numpy', 'h5py',
        'katsdpsigproc[CUDA]', 'katsdptelstate', 'jsonschema', 'netifaces', 'numba',
        'katsdpservices'],
    tests_require=tests_require,
    extras_require={'test': tests_require, 'doc': ['sphinx>=1.3']},
    packages=find_packages(),
    use_katversion=True
)
