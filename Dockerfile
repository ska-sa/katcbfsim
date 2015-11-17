FROM sdp-ingest5.kat.ac.za:5000/docker-base-gpu:docker-refactor

MAINTAINER Bruce Merry "bmerry@ska.ac.za"

# Install dependencies.
COPY requirements.txt /tmp/install/requirements.txt
RUN install-requirements.py -d ~/docker-base/base-requirements.txt -d ~/docker-base/gpu-requirements.txt -r /tmp/install/requirements.txt

# Install the package
COPY . /tmp/install/katcbfsim
WORKDIR /tmp/install/katcbfsim
RUN python ./setup.py clean && pip install --no-index .
# Sanity test
RUN python -c 'import katcbfsim; print "Successful import"'

EXPOSE 7147
