#!/bin/bash
set -e -x
pip install -U pip setuptools wheel
pip install numpy  # Some requirements need it already installed
pip install coverage
pip install -r requirements.txt
pip install .[test]
export CUDA_DEVICE=0
nosetests --no-path-adjustment --with-xunit --with-coverage --cover-erase --cover-xml --cover-inclusive --cover-package=katcbfsim
