ARG KATSDPDOCKERBASE_REGISTRY=harbor.sdp.kat.ac.za/dpp
# :sdp-docker-registry.kat.ac.za:5000

FROM $KATSDPDOCKERBASE_REGISTRY/base-gpu-build:focaluvpip AS build

# Enable Python 3 environment
ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"

# Install dependencies.
COPY --chown=kat:kat requirements.txt /tmp/install/requirements.txt
#RUN install_pinned.py -r /tmp/install/requirements.txt

RUN chmod -R 777 /tmp/install
COPY requirements.txt /tmp/install/
RUN uv pip compile /tmp/install/requirements.txt \
      -o /tmp/install/requirements.lock && \
    uv pip sync /tmp/install/requirements.lock 

# Install the package
COPY --chown=kat:kat . /tmp/install/katcbfsim
WORKDIR /tmp/install/katcbfsim
RUN python ./setup.py clean
RUN pip install --no-deps .
RUN pip check
# Sanity test
RUN python -c 'import katcbfsim; print("Successful import")'

#######################################################################

FROM $KATSDPDOCKERBASE_REGISTRY/base-gpu-runtime:focaluvpip
LABEL maintainer="sdpdev+katcbfsim@ska.ac.za"

# Install from the build stage
COPY --from=build --chown=kat:kat /home/kat/ve3 /home/kat/ve3
ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"
# Allow raw packets (for ibverbs raw QPs)
USER root
RUN setcap cap_net_raw+p /usr/local/bin/capambel
USER kat

# Expose katcp port
EXPOSE 7147
