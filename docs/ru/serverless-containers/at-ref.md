---
title: Справочник аудитных логов {{ serverless-containers-full-name }} в {{ at-full-name }}
description: На этой странице приведен справочник событий сервиса {{ serverless-containers-name }}, отслеживаемых в {{ at-name }}.
---

# Справочник аудитных логов {{ at-full-name }}

В {{ at-name }} поддерживается отслеживание событий уровня конфигурации (Control Plane) для {{ serverless-containers-full-name }}. Подробнее см. [{#T}](../audit-trails/concepts/format.md).

Общий вид значения поля `event_type` (_тип события_):

```text
{{ at-event-prefix }}.audit.serverless.containers.<имя_события>
```

{% include [serverless-containers-events](../_includes/audit-trails/events/serverless-containers-events.md) %}