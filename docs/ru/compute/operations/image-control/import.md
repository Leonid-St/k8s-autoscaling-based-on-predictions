---
title: Как импортировать образ из другого облака или каталога в {{ compute-full-name }}
description: Следуя данной инструкции, вы сможете импортировать образ из другого облака или каталога.
---

# Импортировать образ из другого облака или каталога

Чтобы импортировать образ из другого облака или каталога:

  1. Получите права в облаке или каталоге, в котором находится нужный вам образ:

     * Роль на облако: `resource-manager.clouds.member`.
     * Роль на каталог: `viewer` или `compute.images.user`.

     Подробнее о назначении ролей см. в инструкции [{#T}](../../../iam/operations/roles/grant.md).

  1. Импортируйте копию этого образа в ваш каталог с помощью [CLI](../../../cli/quickstart.md):

     {% include [cli-install](../../../_includes/cli-install.md) %}

     {% include [default-catalogue](../../../_includes/default-catalogue.md) %}

     1. Посмотрите описание команды CLI для создания образа:

        ```bash
        {{ yc-compute }} image create --help
        ```

     1. Получите список доступных образов в исходном облаке или каталоге с помощью команды `{{ yc-compute }} image list --folder-name <имя_исходного_каталога>`. Например:

        ```bash
        {{ yc-compute }} image list --folder-name my-source-folder
        ```

        Результат:

        ```text
        +----------------------+-------------+--------+----------------------+--------+
        |          ID          |    NAME     | FAMILY |     PRODUCT IDS      | STATUS |
        +----------------------+-------------+--------+----------------------+--------+
        | fd8eq6b2fkut******** | first-image |        | f2ehc12fue73******** | READY  |
        +----------------------+-------------+--------+----------------------+--------+
        ```

     1. Выберите идентификатор (`ID`) или имя (`NAME`) импортируемого образа.

     1. Для импорта образа выполните команду `{{ yc-compute }} image create --source-image-id=<идентификатор_исходного_образа>`. Например:

        ```bash
        {{ yc-compute }} image create --source-image-id=fd8o0pt9qfbt********
        ```

        Результат:

        ```yaml
        done (12s)
        id: fd8eq6b2fkut********
        folder_id: b1g07hj5r6i4********
        created_at: "2024-08-14T17:45:44Z"
        storage_size: "2562719744"
        min_disk_size: "21474836480"
        product_ids:
          - f2ehc12fue73********
        status: READY
        os:
          type: LINUX
        ```

   1. Проверьте результат, выполнив команду `{{ yc-compute }} image list --folder-name <имя_целевого_каталога>`. Например:

      ```bash
      {{ yc-compute }} image list --folder-name my-destination-folder
      ```

      Результат:

      ```text
      +----------------------+--------------------+--------+----------------------+--------+
      |          ID          |        NAME        | FAMILY |     PRODUCT IDS      | STATUS |
      +----------------------+--------------------+--------+----------------------+--------+
      | fd8eq6b2fkut******** | first-image        |        | f2ehc12fue73******** | READY  |
      | fd8ghl1n3brd******** | second-image       |        | f2e87com7i95******** | READY  |
      +----------------------+--------------------+--------+----------------------+--------+
      ```