---
title: What is {{ si-name }}? {{ si-full-name }} overview
description: '{{ si-name }} is a service used to configure and manage integrations using serverless technologies in {{ yandex-cloud }}.'
keywords:
  - workflows
  - workflow
  - WF
  - eventrouter
  - event router
  - Workflow
  - YaWL specification
  - Glue
  - integrations
  - API Gateway
  - API gateway
  - bus
  - connector
  - rule
---

# {{ si-name }} overview

{{ si-full-name }} is a service used to configure and manage integrations using serverless technologies in {{ yandex-cloud }}.

{{ si-name }} allows you to:
* Develop microservice architectures without having to configure virtual machines.
* Create and automate workflows for security incident response.
* Automate business operations.
* Set up CI/CD pipelines.
* Develop event-driven serverless applications based on orchestration and choreography to coordinate and manage events.

## Available features {#instruments}

### {{ sw-name }} {#workflows}

{% include [workflows-preview-note](../../_includes/serverless-integrations/workflows-preview-note.md) %}

Build and automate workflows using the Yandex Workflows Language (YaWL).

### {{ er-name }} {#eventrouter}

{% include [event-router-preview-note](../../_includes/serverless-integrations/event-router-preview-note.md) %}

Configure event exchange between your services and {{ yandex-cloud }} services using filtering, transformation, and routing.

### {{ api-gw-name }} {#api-gateway}

Create API gateways supporting [OpenAPI 3.0](https://github.com/OAI/OpenAPI-Specification) and the extension pack for engagement with {{ yandex-cloud }} services. For more information, see the [{{ api-gw-name }} documentation](../../api-gateway/).
