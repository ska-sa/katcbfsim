ARG KATSDPDOCKERBASE_REGISTRY=harbor.sdp.kat.ac.za/dpp

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-gpu-build AS build

# Install dependencies.
COPY --chown=kat:kat requirements.txt /tmp/install/requirements.txt
RUN uv pip install --no-deps -r /tmp/install/requirements.txt && \
    uv pip check

# Install the package
COPY --chown=kat:kat . /tmp/install/katcbfsim
RUN cd /tmp/install/katcbfsim && \
    uv pip install --no-deps . && \
    uv pip check

#######################################################################

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-gpu-runtime
LABEL maintainer="sdpdev+katcbfsim@ska.ac.za"

# Install from the build stage
COPY --from=build --chown=kat:kat /home/kat/ve3 /home/kat/ve3

# Allow raw packets (for ibverbs raw QPs)
USER root
RUN setcap cap_net_raw+p /usr/local/bin/capambel
USER kat

# Expose katcp port
EXPOSE 7147
