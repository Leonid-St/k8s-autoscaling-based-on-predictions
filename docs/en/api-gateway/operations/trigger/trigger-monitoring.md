---
title: Viewing trigger monitoring charts in {{ api-gw-full-name }}
description: You can view a monitoring chart in {{ api-gw-full-name }} using the management console by selecting {{ api-gw-full-name }} and clicking the trigger of interest.
---

# Viewing trigger monitoring charts in {{ api-gw-full-name }}

 You can monitor triggers using the monitoring tools in the management console. These tools display diagnostic information as charts. Metric values are collected and charts are displayed by [{{ monitoring-name }}](../../../monitoring/). 

The chart update period is 15 seconds.

## Viewing monitoring charts {#charts}

{% list tabs group=instructions %}

- Management console {#console}

    1. In the [management console]({{ link-console-main }}), select the folder containing your trigger.

    1. Open **{{ ui-key.yacloud.iam.folder.dashboard.label_api-gateway }}**.

    1. Select a trigger to view its monitoring charts.

    1. Go to the **{{ ui-key.yacloud.common.monitoring }}** tab.

    1. The following charts will open on the page:

        * **Request latency**: Average time it takes a trigger to process a request.
        * **Read events**: Number of events that have set off a trigger.
        * **Function access errors**: Number of access errors when sending messages to WebSocket connections.
        * **Function call errors**: Number of errors when sending messages to WebSocket connections.

    You can select the time period to display information for: hour, day, week, month, or a custom interval.

{% endlist %}

## Custom metrics {#metrics}

To get started with [metrics](../../../monitoring/concepts/data-model.md#metric), [dashboards](../../../monitoring/concepts/visualization/dashboard.md), and [alerts](../../../monitoring/concepts/alerting.md#alert) in {{ monitoring-name }}, click **{{ ui-key.yacloud.monitoring.button_open-in-monitoring }}** in the top-right corner.

| Metric name | Units | entity type                                                                                                                                                                                                                                                                                                                   | Comment                                                                                                                                                          |
|----|----|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `serverless.triggers.`<br/>`inflight` | Invocations | <ul><li>`request`: Message distribution.</li></ul>                                                                                                                                                                                                                                                                              | Number of concurrent message distributions.                                                                                                          |
| `serverless.triggers.`<br/>`error_per_second` | Errors per second | <ul><li>`request`: Message distribution.</li></ul>                                                                                                              | Frequency of message distribution errors.                                                                                                               |
| `serverless.triggers.`<br/>`access_error_per_second` | Errors per second | <ul><li>`request`: Message distribution.</li></ul>                                                                                                              | Frequency of access errors during message distribution.                                                                                                       |
| `serverless.triggers.`<br/>`read_events_per_second` | Events per second | <ul><li>`incoming`: Events that have set off any trigger other than a trigger for {{ message-queue-full-name }}.</li><li>`message_queue`: Events that have set off a trigger for {{ message-queue-full-name }}.</li></ul> | Frequency of events causing a trigger to fire.                                                                                                  |
| `serverless.triggers.`<br/>`execution_time_milliseconds` | Invocations per second | <ul><li>`request`: Message distribution.</li></ul>                                                                                                                                                                                                                                                                               | Distribution histogram of messaging frequency versus request processing time in milliseconds. Request processing time intervals are provided in the `bin` label. |
| `serverless.triggers.`<br/>`event_size_exceeded_per_second` | Errors per second | <ul><li>`incoming`: Events that have set off any trigger other than a trigger for {{ message-queue-full-name }}.                                                                                                                                                                                                          | Frequency of errors on exceeding the message size limit.                                                                                            |

### Custom metrics labels {#labels}

| Label name | Possible values | Comment |
|----|----|----|
| `trigger` | Trigger name | The label contains the name of the trigger that the metric values refer to. |
| `id` | Trigger ID | The label contains the ID of the trigger the metric values refer to. |
| `type` | entity type | The label contains the entity type the metric values refer to. |
| `bin` | Interval of histogram values | For metrics represented by a histogram, the label value describes the interval of the measured value during which the event occurred. |