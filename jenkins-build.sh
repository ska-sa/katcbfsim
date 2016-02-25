#!/bin/bash
set -e -x
pip install -r ~/docker-base/pre-requirements.txt
install-requirements.py -d ~/docker-base/base-requirements.txt -d ~/docker-base/gpu-requirements.txt \
    -r requirements.txt -r test-requirements.txt
export CUDA_DEVICE=0
nosetests --with-xunit --with-coverage --cover-erase --cover-xml --cover-inclusive --cover-package=katcbfsim
