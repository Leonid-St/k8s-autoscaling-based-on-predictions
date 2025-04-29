#!/usr/bin/env bash

if [[ -z "${PUBLISH_PORT}" ]]; then
  PORT="9000"
else
  PORT="${PUBLISH_PORT}"
fi
if [[ -z "${WORKERS}" ]]; then
  WORKERS="1"
else
  WORKERS="${WORKERS}"
fi

gunicorn --bind 0.0.0.0:$PORT --workers 16 --backlog 100000 --worker-class sync --timeout 90000 --log-level=DEBUG wsgi
