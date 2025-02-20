# Изменить виртуальную машину

После создания ВМ вы можете изменить ее имя, описание, метки, платформу или метаданные.

Как изменить конфигурацию ВМ, читайте в разделе [{#T}](vm-update-resources.md).

{% list tabs group=instructions %}

- Консоль управления {#console}

  Чтобы изменить ВМ:
  1. В [консоли управления]({{ link-console-main }}) выберите каталог, которому принадлежит ВМ.
  1. Выберите сервис **{{ ui-key.yacloud.iam.folder.dashboard.label_compute }}**.
  1. На панели слева выберите ![image](../../../_assets/console-icons/server.svg) **{{ ui-key.yacloud.compute.switch_instances }}** и нажмите на имя нужной ВМ.
  1. Нажмите кнопку ![image](../../../_assets/pencil.svg) **{{ ui-key.yacloud.compute.instance.overview.button_action-edit }}**.
  1. Измените параметры ВМ, например, переименуйте машину, отредактировав поле **{{ ui-key.yacloud.common.name }}**.
  1. Нажмите **{{ ui-key.yacloud.compute.instance.edit.button_update }}**.

- CLI {#cli}

  {% include [cli-install](../../../_includes/cli-install.md) %}

  {% include [default-catalogue](../../../_includes/default-catalogue.md) %}

  1. Посмотрите описание команды CLI для обновления параметров ВМ:

     ```bash
     yc compute instance update --help
     ```

  1. Получите список ВМ в каталоге по умолчанию:

     {% include [compute-instance-list](../../_includes_service/compute-instance-list.md) %}

  1. Выберите идентификатор (`ID`) или имя (`NAME`) нужной машины, например `first-instance`.
  1. Измените параметры ВМ, например, переименуйте машину:

     ```bash
     yc compute instance update first-instance \
       --new-name windows-vm
     ```

- API {#api}

  Чтобы изменить ВМ, воспользуйтесь методом REST API [update](../../api-ref/Instance/update.md) для ресурса [Instance](../../api-ref/Instance/) или вызовом gRPC API [InstanceService/Update](../../api-ref/grpc/Instance/update.md).

{% endlist %}

{% note info %}

При изменении имени ВМ, имя хоста и, соответственно, FQDN не изменяются. Подробнее про генерацию имени FQDN читайте в разделе [{#T}](../../concepts/network.md#hostname).

{% endnote %}

## Примеры {#examples}

### Просмотреть список изменяемых параметров {#viewing-a-list-of-configurable-parameters}

Чтобы просмотреть список изменяемых параметров, выполните команду:

{% list tabs group=instructions %}

- CLI {#cli}

  ```bash
  yc compute instance update --help
  ```

{% endlist %}

### Изменить имя и описание {#changing-the-name-and-description}

Чтобы изменить имя и описание ВМ, выполните следующие шаги:

{% list tabs group=instructions %}

- CLI {#cli}

  {% include [cli-install](../../../_includes/cli-install.md) %}

  {% include [default-catalogue](../../../_includes/default-catalogue.md) %}

  1. Получите список ВМ в каталоге по умолчанию:

     {% include [compute-instance-list](../../_includes_service/compute-instance-list.md) %}

  1. Выберите идентификатор (`ID`) или имя (`NAME`) нужной машины, например `first-instance`.
  1. Измените имя и описание ВМ:

     ```bash
     yc compute instance update first-instance \
       --new-name first-vm \
       --description "changed description vm via CLI"
     ```

- API {#api}

  Воспользуйтесь методом REST API [update](../../api-ref/Instance/update.md) для ресурса [Instance](../../api-ref/Instance/) или вызовом gRPC API [InstanceService/Update](../../api-ref/grpc/Instance/update.md).

{% endlist %}

{% note alert %}

Не изменяйте имя ВМ, если она принадлежит [группе узлов](../../../managed-kubernetes/concepts/index.md#node-group) кластера {{ managed-k8s-name }}. Имя для такой ВМ генерируется автоматически, и его изменение нарушит работу кластера.

{% endnote %}

### Изменить метаданные {#changing-metadata}

{% include [update-metadata-part1](../../../_includes/compute/metadata/update-metadata-part1.md) %}

### Удалить SSH-ключи из метаданных {#delete-keys-from-metadata}

{% include [delete-keys-from-metadata](../../../_includes/compute/delete-keys-from-metadata.md) %}

### Включить доступ по {{ oslogin }} {#enable-oslogin-access}

{% include [update-metadata-part2-oslogin](../../../_includes/compute/metadata/update-metadata-part2-oslogin.md) %}