#!/bin/bash
set -e -x
pip install -U pip setuptools wheel
pip install numpy  # Some requirements need it already installed
pip install -r requirements.txt
pip install .[doc]
rm -rf doc/_build
make -C doc html
