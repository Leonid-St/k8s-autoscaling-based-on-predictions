#!/usr/bin/env bash

export DOCKER_REPO="<FILL IN>"
export IMAGE_TAG="gatling_c"

gcloud auth configure-docker
docker build --tag=$IMAGE_TAG .
# Create a tag TARGET_IMAGE that refers to SOURCE_IMAGE - https://docs.docker.com/engine/reference/commandline/tag/
docker tag $IMAGE_TAG $DOCKER_REPO/$IMAGE_TAG
docker push $DOCKER_REPO/$IMAGE_TAG
