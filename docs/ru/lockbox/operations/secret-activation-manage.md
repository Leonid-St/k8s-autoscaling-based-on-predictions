---
title: Как деактивировать и активировать секрет в {{ lockbox-full-name }}
description: Из статьи вы узнаете, как деактивировать и активировать секрет в {{ lockbox-full-name }}.
---

# Деактивировать и активировать секрет

Секрет может находиться в активном и деактивированном состояниях. В активном состоянии возможен доступ как к метаданным, так и к содержимому секрета (парам ключ-значение). В деактивированном состоянии возможен доступ только к метаданным, содержимое секрета недоступно.

## Деактивировать секрет {#secret-deactivate}

{% list tabs group=instructions %}

- Консоль управления {#console}

    1. В [консоли управления]({{ link-console-main }}) выберите каталог, которому принадлежит секрет.
    1. В списке сервисов выберите **{{ ui-key.yacloud.iam.folder.dashboard.label_lockbox }}**.
    1. Напротив нужного секрета нажмите ![image](../../_assets/console-icons/ellipsis.svg) и выберите **{{ ui-key.yacloud.lockbox.button_deactivate-secret }}**.
    1. Подтвердите деактивацию.

- CLI {#cli}

  {% include [cli-install](../../_includes/cli-install.md) %}

  {% include [default-catalogue](../../_includes/default-catalogue.md) %}

  1. Посмотрите описание команды CLI для получения информации о деактивации секрета:

      ```bash
      yc lockbox secret deactivate --help
      ```
  1. Деактивируйте секрет, указав его идентификатор или имя:

      ```bash
      yc lockbox secret deactivate <имя_секрета>
      ```
      Результат:

      ```text
      id: e6qkkp3k29jf********
      folder_id: b1go3el0d8fs********
      created_at: "2023-11-08T13:14:34.676Z"
      name: first-secret
      status: INACTIVE
      current_version:
        id: e6qor8pe3ju7********
        secret_id: e6qkkp3k29jf********
        created_at: "2023-11-08T13:14:34.676Z"
        status: ACTIVE
        payload_entry_keys:
          - secret-key
      ```

- API {#api}

  Чтобы деактивировать секрет, воспользуйтесь методом REST API [deactivate](../api-ref/Secret/deactivate.md) для ресурса [Secret](../api-ref/Secret/index.md) или вызовом gRPC API [SecretService/Deactivate](../api-ref/grpc/Secret/deactivate.md).

{% endlist %}

## Активировать секрет {#secret-activate}

{% list tabs group=instructions %}

- Консоль управления {#console}

    1. В [консоли управления]({{ link-console-main }}) выберите каталог, которому принадлежит секрет.
    1. В списке сервисов выберите **{{ ui-key.yacloud.iam.folder.dashboard.label_lockbox }}**.
    1. Напротив нужного секрета нажмите ![image](../../_assets/console-icons/ellipsis.svg) и выберите **{{ ui-key.yacloud.lockbox.button_activate-secret }}**.

- CLI {#cli}

  {% include [cli-install](../../_includes/cli-install.md) %}

  {% include [default-catalogue](../../_includes/default-catalogue.md) %}

  1. Посмотрите описание команды CLI для получения информации об активации секрета:

      ```bash
      yc lockbox secret activate --help
      ```
  1. Активируйте секрет, указав его идентификатор или имя:

      ```bash
      yc lockbox secret activate <имя_секрета>
      ```
      Результат:

      ```text
      id: e6qkkp3k29jf********
      folder_id: b1go3el0d8fs********
      created_at: "2023-11-08T13:14:34.676Z"
      name: first-secret
      status: ACTIVE
      current_version:
        id: e6qor8pe3ju7********
        secret_id: e6qkkp3k29jf********
        created_at: "2023-11-08T13:14:34.676Z"
        status: ACTIVE
        payload_entry_keys:
          - secret-key
      ```

- API {#api}

  Чтобы активировать секрет, воспользуйтесь методом REST API [activate](../api-ref/Secret/activate.md) для ресурса [Secret](../api-ref/Secret/index.md) или вызовом gRPC API [SecretService/Activate](../api-ref/grpc/Secret/activate.md).

{% endlist %}

## См. также {#see-also}

* [{#T}](../concepts/secret.md)
