---
title: Templating in {{ sw-full-name }}
description: Templating in {{ sw-name }} stands for dynamic generation of field values in a YaWL specification.
keywords:
  - workflows
  - workflow
  - WF
  - runtime process
  - YaWL specification
  - templating
  - templater
  - template
  - template
---


# Templating

For the [YaWL specification](./yawl.md) fields that support templating, values can be generated dynamically using the data obtained from the workflow state. The templating language is `jq`. For more information, see the [jq documentation](https://jqlang.github.io/jq/manual/).

By default, templating is not used for string values of fields; use [string interpolation](https://jqlang.github.io/jq/manual/#string-interpolation) instead.

## Input data for the templater {#input-data}

The templater input data will be different depending on the field.

Field | Payload
--- | ---
`input` | Overall [state of the workflow](workflow.md#state)
`output` | Step output data
Other fields that do not support templating | JSON object filtered by `input`

### Templater usage examples {#templator-examples}

Field value | Templater interpretation
--- | ---
`this is just a string` | Non-templated string
`this is a value from workflow state \(.data[1].some_property)` | `this is a value from workflow state <workflow_state_data>`
`\({x: 1, y: .a.b.c})` | `{“x”: 1, “y”: “<workflow_state_data>”}`

## Templater extensions {#extensions}

Templater extensions allow you to call jq functions implementing non-standard logic.

### {{ lockbox-full-name }} {#lockbox-extension}

A special `lockboxPayload` jq function allows you to get the value of the [{{ lockbox-name }} secret](../../../lockbox/concepts/secret.md) by its ID, key and (optionally) [version](../../../lockbox/concepts/secret.md#version). For the extension to work correctly, the [workflow](./workflow.md) settings must specify a [service account](../../../iam/concepts/users/service-accounts.md) that has permissions to view the secret contents, e.g., the one with the `lockbox.payloadViewer` [role](../../../lockbox/security/index.md#lockbox-payloadViewer).

Description of the `lockboxPayload` jq function arguments:

Argument order | Type | Required | Description
--- | --- | --- | ---
1 | `string` | Yes | Secret ID.
2 | `string` | Yes | Secret key.
3 | `string` | No | Secret version ID.

#### Examples of using the lockboxPayload jq function {#lockbox-examples}

* Getting the latest secret version:

    ```text
    \(lockboxPayload("<secret_ID>"; "<secret_key>"))
    ```
* Getting the specified secret version:

    ```text
    \(lockboxPayload("<secret_ID>"; "<secret_key>"; "<secret_version_ID>"))
    ```