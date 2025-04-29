#!/usr/bin/env bash

# upgrade a release with new configs (values.yaml file)
helm upgrade prometheus-release stable/prometheus -f values.yaml --namespace cm --debug
