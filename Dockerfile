FROM sdp-ingest5.kat.ac.za:5000/docker-base-gpu

MAINTAINER Bruce Merry "bmerry@ska.ac.za"

# Install system packages. Python packages are mostly installed here, but
# certain packages are handled by pip because they're not available.
RUN apt-get -y update && apt-get -y install \
    libboost-python1.55-dev \
    libboost-system1.55-dev \
    python-appdirs \
    python-concurrent.futures \
    python-h5py \
    python-mako \
    python-mock \
    python-nose \
    python-numpy \
    python-ply \
    python-py \
    python-pytest \
    python-pytools \
    python-scipy \
    python-unittest2

# Install Python dependencies. Versions are explicitly listed and pinned, so
# that the docker image is reproducible. There were all up-to-date versions
# at the time of writing i.e. there are no currently known reasons not to
# update to newer versions.
RUN pip install --no-deps \
    ProxyTypes==0.9 \
    backports.ssl-match-hostname==3.4.0.2 \
    certifi==2015.9.6.2 \
    pycuda==2015.1.3 \
    pyephem==3.7.6.0 \
    manhole==1.3.0 \
    six==1.9.0 \
    spead2==0.4.4 \
    tornado==4.2 \
    trollius==2.0 \
    git+ssh://git@github.com/ska-sa/katcp-python \
    git+ssh://git@github.com/ska-sa/katpoint \
    git+ssh://git@github.com/ska-sa/katsdpsigproc \
    git+ssh://git@github.com/ska-sa/katsdptelstate

# Install the package
COPY . /tmp/install/katcbfsim
WORKDIR /tmp/install/katcbfsim
RUN python ./setup.py clean && pip install --no-index .

# Run as non-root user
USER kat
WORKDIR /home/kat

EXPOSE 7147

# Sanity test
RUN python -c 'import katcbfsim; print "Successful import"'
