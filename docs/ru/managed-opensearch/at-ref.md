---
title: Справочник аудитных логов {{ mos-full-name }} в {{ at-full-name }}
description: На этой странице приведен справочник событий сервиса {{ mos-name }}, отслеживаемых в {{ at-name }}.
---

# Справочник аудитных логов {{ at-full-name }}

В {{ at-name }} поддерживается отслеживание событий уровня конфигурации (Control Plane) для {{ mos-full-name }}. Подробнее см. [{#T}](../audit-trails/concepts/format.md).

Общий вид значения поля `event_type` (_тип события_):

```text
{{ at-event-prefix }}.audit.mdb.opensearch.<имя_события>
```

{% include [mos-events](../_includes/audit-trails/events/mos-events.md) %}
