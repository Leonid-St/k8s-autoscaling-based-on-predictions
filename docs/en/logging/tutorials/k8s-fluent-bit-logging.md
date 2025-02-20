---
title: Transferring {{ managed-k8s-full-name }} cluster logs to {{ cloud-logging-name }}
description: The Fluent Bit log processor allows you to transfer the {{ managed-k8s-name }} cluster logs to {{ cloud-logging-name }}. The Fluent Bit plugin for {{ cloud-logging-full-name }} module is used to transfer logs.
canonical: '{{ link-docs }}/tutorials/container-infrastructure/k8s-fluent-bit-logging'
---

# Transferring {{ managed-k8s-full-name }} cluster logs to {{ cloud-logging-name }}

{% include notitle [k8s-logs-to-logging](../../_tutorials/containers/k8s-logs-to-logging.md) %}


## Sending pod logs and service logs via Fluent Bit {#fluent-bit}

{% include notitle [Setting up Managed K8S Fluent Bit](../../_tutorials/containers/k8s-fluent-bit-logging.md) %}
