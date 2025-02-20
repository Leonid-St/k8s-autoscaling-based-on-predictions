---
title: Как изменить шину
description: Следуя данной инструкции, вы сможете изменить шину.
---

# Изменить шину

{% list tabs group=instructions %}

- Консоль управления {#console}

  1. В [консоли управления]({{ link-console-main }}) выберите сервис **{{ ui-key.yacloud.iam.folder.dashboard.label_serverless-integrations }}**.
  1. На панели слева выберите ![image](../../../../_assets/console-icons/object-align-center-vertical.svg) **{{ ui-key.yacloud.serverless-event-router.label_service }}**.
  1. В строке с нужной [шиной](../../../concepts/eventrouter/bus.md) нажмите ![image](../../../../_assets/console-icons/ellipsis.svg) и выберите ![image](../../../../_assets/console-icons/pencil.svg) **{{ ui-key.yacloud.common.edit }}**.
  1. Измените параметры шины.
  1. Нажмите **{{ ui-key.yacloud.common.save }}**.

- CLI {#cli}

  {% include [cli-install](../../../../_includes/cli-install.md) %}

  {% include [default-catalogue](../../../../_includes/default-catalogue.md) %}

  1. Посмотрите описание команды [CLI](../../../../cli/) для обновления параметров [шины](../../../concepts/eventrouter/bus.md):

      ```bash
      yc serverless eventrouter bus update --help
      ```

  1. {% include [get-buses-list](../../../../_includes/serverless-integrations/get-buses-list.md) %}
  1. Укажите в команде параметры, которые необходимо изменить, например имя шины:

      ```bash
      yc serverless eventrouter bus update \
        --name <имя_шины> \
        --new-name <новое_имя_шины>
      ```

      Результат:

      ```text
      id: f66aevm4ithv********
      folder_id: b1g681qpemb4********
      cloud_id: b1gia87mbaom********
      created_at: "2025-02-13T12:36:59.497985Z"
      name: my-bus-43
      description: this is my bus
      labels:
        owner: admin
      deletion_protection: true
      status: ACTIVE
      ```

- {{ TF }} {#tf}

  {% include [terraform-definition](../../../../_tutorials/_tutorials_includes/terraform-definition.md) %}

  {% include [terraform-install](../../../../_includes/terraform-install.md) %}

  Чтобы изменить [шину](../../../concepts/eventrouter/bus.md):

  1. Откройте файл конфигурации {{ TF }} и измените фрагмент с описанием ресурса `yandex_serverless_eventrouter_bus`.

      Пример описания шины в конфигурационном файле {{ TF }}:

      ```hcl
      resource "yandex_serverless_eventrouter_bus" "example_bus" {
        name                = "my-bus"
        description         = "this is my bus"
        deletion_protection = true

        labels = {
          key1 = "value1"
          key2 = "value2"
        }
      }
      ```

      Более подробную информацию о параметрах ресурса `yandex_serverless_eventrouter_bus` см. в [документации провайдера]({{ tf-provider-resources-link }}/serverless_eventrouter_bus).

  1. Примените изменения:

      {% include [terraform-validate-plan-apply](../../../../_tutorials/_tutorials_includes/terraform-validate-plan-apply.md) %}

      {{ TF }} создаст все требуемые ресурсы. Проверить изменения можно в [консоли управления]({{ link-console-main }}) или с помощью команды [CLI](../../../../cli/):

      ```bash
      yc serverless eventrouter bus list
      ```

- API {#api}

  Чтобы изменить [шину](../../../concepts/eventrouter/bus.md), воспользуйтесь методом REST API [Update](../../../../serverless-integrations/eventrouter/api-ref/Bus/update.md) для ресурса [Bus](../../../../serverless-integrations/eventrouter/api-ref/Bus/index.md) или вызовом gRPC API [Bus/Update](../../../../serverless-integrations/eventrouter/api-ref/grpc/Bus/update.md).

{% endlist %}