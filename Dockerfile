ARG KATSDPDOCKERBASE_REGISTRY=sdp-docker-registry.kat.ac.za:5000

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-gpu-build:new-rdma-core as build

# Enable Python 3 environment
ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"

# Install dependencies.
COPY --chown=kat:kat requirements.txt /tmp/install/requirements.txt
RUN install-requirements.py -d ~/docker-base/base-requirements.txt -d ~/docker-base/gpu-requirements.txt -r /tmp/install/requirements.txt

# Install the package
COPY --chown=kat:kat . /tmp/install/katcbfsim
WORKDIR /tmp/install/katcbfsim
RUN python ./setup.py clean
RUN pip install --no-deps .
RUN pip check
# Sanity test
RUN python -c 'import katcbfsim; print("Successful import")'

#######################################################################

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-gpu-runtime:new-rdma-core
LABEL maintainer="sdpdev+katcbfsim@ska.ac.za"

# Install from the build stage
COPY --from=build --chown=kat:kat /home/kat/ve3 /home/kat/ve3
ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"

# Expose katcp port
EXPOSE 7147
