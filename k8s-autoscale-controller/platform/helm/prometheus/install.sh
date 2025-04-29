#!/usr/bin/env bash

# install a release
helm install stable/prometheus -f values.yaml --name prometheus-release --namespace cm --debug --version 11.12.0
