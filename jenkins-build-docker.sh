#!/bin/bash
set -e -x
REGISTRY="sdp-ingest5.kat.ac.za:5000"
LABEL="${GIT_BRANCH#origin/}"
if [ "$LABEL" = "master" ]; then
    LABEL=latest
fi
set +x   # Avoid showing password in log
docker login -e "jenkins@gpu-dev.kat.ac.za" -u "$REGISTRY_USERNAME" -p "$REGISTRY_PASSWORD" "$REGISTRY"
set -x
docker build --pull=true -t "$REGISTRY/katcbfsim:$LABEL" .
docker push "$REGISTRY/katcbfsim:$LABEL"
