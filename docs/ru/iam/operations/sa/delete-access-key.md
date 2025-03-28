---
title: Как удалить статические ключи доступа в {{ iam-full-name }}
description: Из статьи вы узнаете, как удалить статические ключи доступа в {{ iam-full-name }} через консоль управления, CLI и API сервиса.
---

# Удаление статических ключей доступа

{% list tabs group=instructions %}

- Консоль управления {#console}

  1. Перейдите в каталог, которому принадлежит сервисный аккаунт.
  1. В списке сервисов выберите **{{ ui-key.yacloud.iam.folder.dashboard.label_iam }}**.
  1. На панели слева выберите ![FaceRobot](../../../_assets/console-icons/face-robot.svg) **{{ ui-key.yacloud.iam.label_service-accounts }}** и выберите нужный сервисный аккаунт.
  1. В блоке **{{ ui-key.yacloud.iam.folder.service-account.overview.section_service-account-keys }}** в строке с ключом, который нужно удалить, нажмите значок ![image](../../../_assets/console-icons/ellipsis.svg) и выберите **{{ ui-key.yacloud.iam.folder.service-account.overview.button_action-delete-api-key }}**.
  1. В открывшемся окне нажмите кнопку **{{ ui-key.yacloud.iam.folder.service-account.overview.popup-confirm_button_delete }}**.

- CLI {#cli}

  {% include [cli-install](../../../_includes/cli-install.md) %}

  1. Получите список с идентификаторами `ID` статических ключей для конкретного сервисного аккаунта. В параметре `--service-account-name` укажите имя сервисного аккаунта:

     ```bash
     yc iam access-key list --service-account-name <имя_сервисного_аккаунта>
     ```

     Результат:

     ```text
     +----------------------+----------------------+----------------------+
     |          ID          |  SERVICE ACCOUNT ID  |        KEY ID        |
     +----------------------+----------------------+----------------------+
     | aje8bdtqec6l******** | ajeedllrkjma******** | R9JK04o1Dfaf******** |
     | ajegqpa91bta******** | ajeedllrkjma******** | cWXGkDoBRho5******** |
     +----------------------+----------------------+----------------------+
     ```

  1. Удалите старый статический ключ. Вместо `<идентификатор>` укажите идентификатор статического ключа:

     ```bash
     yc iam access-key delete <идентификатор>
     ```

- API {#api}

  Чтобы удалить статический ключ, воспользуйтесь методом REST API [delete](../../awscompatibility/api-ref/AccessKey/delete.md) для ресурса [AccessKey](../../awscompatibility/api-ref/AccessKey/index.md) или вызовом gRPC API [AccessKeyService/Delete](../../awscompatibility/api-ref/grpc/AccessKey/delete.md).

{% endlist %}