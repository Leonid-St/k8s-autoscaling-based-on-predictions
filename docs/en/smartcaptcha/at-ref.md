---
title: '{{ captcha-full-name }} event reference in {{ at-full-name }}'
description: This page gives a reference for {{ captcha-name }} events tracked in {{ at-name }}.
---

# {{ at-full-name }} event reference

{{ at-name }} supports tracking control plane events for {{ captcha-full-name }}. For more information, see [{#T}](../audit-trails/concepts/format.md).

The general format of the `event_type` field value is as follows:

```text
{{ at-event-prefix }}.audit.smartcaptcha.<event_name>
```

{% include [captcha-events](../_includes/audit-trails/events/captcha-events.md) %}
