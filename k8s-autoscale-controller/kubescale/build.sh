#!/usr/bin/env bash

# make sure you have gcloud installed --> brew cask install google-cloud-sdk
# and that you've ran "gcloud auth configure-docker"

export DOCKER_REPO="leonidusername5432/autoscaler-predictor-model"
export IMAGE_TAG="controller-kubescaler"

gcloud auth configure-docker
docker build --tag=$IMAGE_TAG .
# Create a tag TARGET_IMAGE that refers to SOURCE_IMAGE - https://docs.docker.com/engine/reference/commandline/tag/
docker tag $IMAGE_TAG $DOCKER_REPO/$IMAGE_TAG
docker push $DOCKER_REPO/$IMAGE_TAG



