---
title: Инструкция по получению идентификатора облака в {{ yandex-cloud }}
description: Из статьи вы узнаете, как получить идентификатор облака в {{ yandex-cloud }}.
---

# Получение идентификатора облака

{% list tabs group=instructions %}

- Консоль управления {#console}

  1. Перейдите в [консоль управления]({{ link-console-main }}) и [выберите](switch-cloud.md) нужное облако. На открывшейся странице идентификатор облака указан сверху, рядом с именем облака, а также на вкладке **{{ ui-key.yacloud.iam.cloud.switch_overview }}** в строке **{{ ui-key.yacloud.common.id }}**.
 
  1. Чтобы скопировать идентификатор, наведите на него указатель и нажмите значок ![image](../../../_assets/console-icons/copy.svg).

- CLI {#cli}

  Если вы знаете имя облака, задайте его в качестве параметра команды `get`:

  ```
  yc resource-manager cloud get <имя_облака>
  ```
  Результат:

  ```
  id: b1gd129pp9ha********
  ...
  ```

  Если вы не знаете имя облака, получите список облаков с идентификаторами:

  {% include [get-cloud-list](../../../_includes/resource-manager/get-cloud-list.md) %}

- API {#api}

  Чтобы получить список облаков с идентификаторами, воспользуйтесь методом REST API [list](../../api-ref/Cloud/list.md) для ресурса [Cloud](../../api-ref/Cloud/index.md) или вызовом gRPC API [CloudService/List](../../api-ref/grpc/Cloud/list.md).

{% endlist %}
