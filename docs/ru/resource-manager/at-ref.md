---
title: Справочник аудитных логов {{ resmgr-full-name }} в {{ at-full-name }}
description: На этой странице приведен справочник событий сервиса {{ resmgr-name }}, отслеживаемых в {{ at-name }}.
---

# Справочник аудитных логов {{ at-full-name }}

В {{ at-name }} поддерживается отслеживание событий уровня конфигурации (Control Plane) для {{ resmgr-full-name }}. Подробнее см. [{#T}](../audit-trails/concepts/format.md).

Общий вид значения поля `event_type` (_тип события_):

```text
{{ at-event-prefix }}.audit.resourcemanager.<имя_события>
```

{% include [resmgr-events](../_includes/audit-trails/events/resmgr-events.md) %}