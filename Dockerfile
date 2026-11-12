ARG KATSDPDOCKERBASE_REGISTRY=harbor.sdp.kat.ac.za/dpp

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-gpu-build AS build

# Enable Python 3 environment
# uv is installed in ~/.local/bin in the base image
# Activate the base image virtualenv and make uv available
ENV PATH="/home/kat/.local/bin:$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"

# Install dependencies.
COPY --chown=kat:kat requirements.txt /tmp/install/requirements.txt
#Replace install_pinned.py with uv 
RUN uv pip compile /tmp/install/requirements.txt \
      -o /tmp/install/requirements.lock && \
    uv pip install --no-deps -r /tmp/install/requirements.lock&&\
    uv pip check

# Install the package
COPY --chown=kat:kat . /tmp/install/katcbfsim
RUN cd /tmp/install/katcbfsim && \
    python3 ./setup.py clean   && \
    uv pip install --no-deps . && \
    uv pip check

#######################################################################

FROM $KATSDPDOCKERBASE_REGISTRY/docker-base-gpu-runtime
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
