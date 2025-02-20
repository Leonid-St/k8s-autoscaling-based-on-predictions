---
title: Справочник аудитных логов {{ cloud-registry-full-name }} в {{ at-full-name }}
description: На этой странице приведен справочник событий сервиса {{ cloud-registry-name }}, отслеживаемых в {{ at-name }}.
---

# Справочник аудитных логов {{ at-full-name }}

В {{ at-name }} поддерживается отслеживание событий уровня конфигурации (Control Plane) для {{ cloud-registry-full-name }}. Подробнее см. [{#T}](../audit-trails/concepts/format.md).

Общий вид значения поля `event_type` (_тип события_):

```text
{{ at-event-prefix }}.audit.cloudregistry.<имя_события>
```

{% include [cloudregistry-events](../_includes/audit-trails/events/cloudregistry-events.md) %}