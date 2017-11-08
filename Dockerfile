FROM sdp-docker-registry.kat.ac.za:5000/docker-base-gpu

MAINTAINER Bruce Merry "bmerry@ska.ac.za"

# Switch to Python 3 environment
ENV PATH="$PATH_PYTHON3" VIRTUALENV="$VIRTUALENV_PYTHON3"

# Install dependencies.
COPY requirements.txt /tmp/install/requirements.txt
RUN install-requirements.py -d ~/docker-base/base-requirements.txt -d ~/docker-base/gpu-requirements.txt -r /tmp/install/requirements.txt

# Install the package
COPY . /tmp/install/katcbfsim
WORKDIR /tmp/install/katcbfsim
RUN python ./setup.py clean && pip install --no-index .
# Sanity test
RUN python -c 'import katcbfsim; print("Successful import")'

EXPOSE 7147
