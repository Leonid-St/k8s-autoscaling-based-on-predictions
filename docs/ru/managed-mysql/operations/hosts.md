---
title: Управление хостами кластера {{ MY }}
description: Из статьи вы узнаете, как управлять хостами кластера {{ MY }} и их настройками.
---

# Управление хостами кластера {{ MY }}

Вы можете добавлять и удалять хосты кластера, а также управлять их настройками. О том, как перенести хосты кластера в другую [зону доступности](../../overview/concepts/geo-scope.md), читайте в [инструкции](host-migration.md).

## Получить список хостов в кластере {#list}

{% list tabs group=instructions %}

- Консоль управления {#console}

  1. Перейдите на [страницу каталога]({{ link-console-main }}) и выберите сервис **{{ ui-key.yacloud.iam.folder.dashboard.label_managed-mysql }}**.
  1. Нажмите на имя нужного кластера, затем выберите вкладку **{{ ui-key.yacloud.mysql.cluster.switch_hosts }}**.

- CLI {#cli}

  {% include [cli-install](../../_includes/cli-install.md) %}

  {% include [default-catalogue](../../_includes/default-catalogue.md) %}

  Чтобы получить список хостов в кластере, выполните команду:

  ```bash
  {{ yc-mdb-my }} host list \
     --cluster-name=<имя_кластера>
  ```

  Результат:

  ```text
  +----------------------------+----------------------+---------+--------+---------------+
  |            NAME            |      CLUSTER ID      |  ROLE   | HEALTH |    ZONE ID    |
  +----------------------------+----------------------+---------+--------+---------------+
  | rc1b...{{ dns-zone }} | c9q5k4ve7ev4******** | MASTER  | ALIVE  | {{ region-id }}-b |
  | rc1a...{{ dns-zone }} | c9q5k4ve7ev4******** | REPLICA | ALIVE  | {{ region-id }}-a |
  +----------------------------+----------------------+---------+--------+---------------+
  ```

  Имя кластера можно запросить со [списком кластеров в каталоге](cluster-list.md#list-clusters).

- REST API {#api}

  1. [Получите IAM-токен для аутентификации в API](../api-ref/authentication.md) и поместите токен в переменную среды окружения:

      {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

  1. Воспользуйтесь методом [Cluster.listHosts](../api-ref/Cluster/listHosts.md) и выполните запрос, например, с помощью {{ api-examples.rest.tool }}:

      ```bash
      curl \
          --request GET \
          --header "Authorization: Bearer $IAM_TOKEN" \
          --url 'https://{{ api-host-mdb }}/managed-mysql/v1/clusters/<идентификатор_кластера>/hosts'
      ```

      Идентификатор кластера можно запросить со [списком кластеров в каталоге](cluster-list.md#list-clusters).

  1. Убедитесь, что запрос был выполнен успешно, изучив [ответ сервера](../api-ref/Cluster/listHosts.md#yandex.cloud.mdb.mysql.v1.ListClusterHostsResponse).

- gRPC API {#grpc-api}

  1. [Получите IAM-токен для аутентификации в API](../api-ref/authentication.md) и поместите токен в переменную среды окружения:

      {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

  1. {% include [grpc-api-setup-repo](../../_includes/mdb/grpc-api-setup-repo.md) %}
  1. Воспользуйтесь вызовом [ClusterService/ListHosts](../api-ref/grpc/Cluster/listHosts.md) и выполните запрос, например, с помощью {{ api-examples.grpc.tool }}:

      ```bash
      grpcurl \
          -format json \
          -import-path ~/cloudapi/ \
          -import-path ~/cloudapi/third_party/googleapis/ \
          -proto ~/cloudapi/yandex/cloud/mdb/mysql/v1/cluster_service.proto \
          -rpc-header "Authorization: Bearer $IAM_TOKEN" \
          -d '{
                "cluster_id": "<идентификатор_кластера>"
              }' \
          {{ api-host-mdb }}:{{ port-https }} \
          yandex.cloud.mdb.mysql.v1.ClusterService.ListHosts
      ```

      Идентификатор кластера можно запросить со [списком кластеров в каталоге](cluster-list.md#list-clusters).

  1. Убедитесь, что запрос был выполнен успешно, изучив [ответ сервера](../api-ref/grpc/Cluster/listHosts.md#yandex.cloud.mdb.mysql.v1.ListClusterHostsResponse).

{% endlist %}

## Добавить хост {#add}

Количество хостов в кластерах {{ mmy-name }} ограничено квотами на количество CPU и объем памяти, которые доступны кластерам БД в вашем облаке. Чтобы проверить используемые ресурсы, откройте страницу [Квоты]({{ link-console-quotas }}) и найдите блок **{{ ui-key.yacloud.iam.folder.dashboard.label_mdb }}**.

{% list tabs group=instructions %}

- Консоль управления {#console}

  1. Перейдите на [страницу каталога]({{ link-console-main }}) и выберите сервис **{{ ui-key.yacloud.iam.folder.dashboard.label_managed-mysql }}**.
  1. Нажмите на имя нужного кластера и перейдите на вкладку **{{ ui-key.yacloud.mysql.cluster.switch_hosts }}**.
  1. Нажмите кнопку **{{ ui-key.yacloud.mdb.cluster.hosts.action_add-host }}**.
  1. Укажите параметры хоста:

     
     * Зону доступности.
     * Подсеть (если нужной подсети в списке нет, [создайте ее](../../vpc/operations/subnet-create.md)).


     * Выберите опцию **{{ ui-key.yacloud.mdb.hosts.dialog.field_public_ip }}**, если хост должен быть доступен извне {{ yandex-cloud }}.
     * Приоритет назначения хоста мастером.
     * Приоритет хоста как {{ MY }}-реплики для создания резервной копии.

- CLI {#cli}

  {% include [cli-install](../../_includes/cli-install.md) %}

  {% include [default-catalogue](../../_includes/default-catalogue.md) %}

  Чтобы добавить хост в кластер:
  1. Запросите список подсетей кластера, чтобы выбрать подсеть для нового хоста:

     ```bash
     yc vpc subnet list
     ```

     Результат:

     ```text
     +----------------------+-----------+-----------------------+---------------+------------------+
     |          ID          |   NAME    |       NETWORK ID      |       ZONE    |      RANGE       |
     +----------------------+-----------+-----------------------+---------------+------------------+
     | b0cl69a2b4c6******** | default-d | enp6rq72rndgr******** | {{ region-id }}-d | [172.**.*.*/20]  |
     | e2lkj9qwe762******** | default-b | enp6rq72rndgr******** | {{ region-id }}-b | [10.**.*.*/16]   |
     | e9b0ph42bn96******** | a-2       | enp6rq72rndgr******** | {{ region-id }}-a | [172.**.**.*/20] |
     | e9b9v22r88io******** | default-a | enp6rq72rndgr******** | {{ region-id }}-a | [172.**.**.*/20] |
     +----------------------+-----------+-----------------------+---------------+------------------+
     ```

     
     Если нужной подсети в списке нет, [создайте ее](../../vpc/operations/subnet-create.md).


  1. Посмотрите описание команды CLI для добавления хостов:

     ```bash
     {{ yc-mdb-my }} host add --help
     ```

  1. Выполните команду добавления хоста (в примере приведены не все доступные параметры):

     ```bash
     {{ yc-mdb-my }} host add \
       --cluster-name=<имя_кластера> \
       --host zone-id=<идентификатор_зоны_доступности>,`
         `subnet-id=<идентификатор_подсети>,`
         `assign-public-ip=<публичный_доступ_к_хосту_подкластера>,`
         `replication-source=<имя_хоста-источника>,`
         `backup-priority=<приоритет_хоста_при_резервном_копировании>,`
         `priority=<приоритет_назначения_хоста_мастером>
     ```

     Где:
     * `--cluster-name` — имя кластера {{ mmy-name }}.
     * `--host` — параметры хоста:
       * `zone-id` — [зона доступности](../../overview/concepts/geo-scope.md).
       * `subnet-id` — [идентификатор подсети](../../vpc/concepts/network.md#subnet). Необходимо указывать, если в выбранной зоне доступности создано две или больше подсетей.
       * `assign-public-ip` — доступность хоста из интернета: `true` или `false.`
       * `replication-source` — источник [репликации](../concepts/replication.md) для хоста.
       * `backup-priority` — приоритет хоста при [резервном копировании](../concepts/backup.md#size): от `0` до `100`.
       * `priority` — приоритет назначения хоста мастером при [выходе из строя основного мастера](../concepts/replication.md#master-failover): от `0` до `100`.

     Имя кластера можно запросить со [списком кластеров в каталоге](cluster-list.md#list-clusters).

- {{ TF }} {#tf}

  1. Откройте актуальный конфигурационный файл {{ TF }} с планом инфраструктуры.

     О том, как создать такой файл, см. в разделе [Создание кластера](cluster-create.md).
  1. Добавьте к описанию кластера {{ mmy-name }} блок `host`:

     ```hcl
     resource "yandex_mdb_mysql_cluster" "<имя_кластера>" {
       ...
       host {
         zone             = "<зона_доступности>"
         subnet_id        = <идентификатор_подсети>
         assign_public_ip = <публичный_доступ_к_хосту>
         priority         = <приоритет_назначения_хоста_мастером>
         ...
       }
     }
     ```

     Где:

     * `assign_public_ip` — публичный доступ к хосту: `true` или `false`.
     * `priority` — приоритет назначения хоста мастером: от `0` до `100`.

  1. Проверьте корректность настроек.

     {% include [terraform-validate](../../_includes/mdb/terraform/validate.md) %}

  1. Подтвердите изменение ресурсов.

     {% include [terraform-apply](../../_includes/mdb/terraform/apply.md) %}

  Подробнее см. в [документации провайдера {{ TF }}]({{ tf-provider-mmy }}).

  {% include [Terraform timeouts](../../_includes/mdb/mmy/terraform/timeouts.md) %}

- REST API {#api}

  1. [Получите IAM-токен для аутентификации в API](../api-ref/authentication.md) и поместите токен в переменную среды окружения:

      {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

  1. Воспользуйтесь методом [Cluster.addHosts](../api-ref/Cluster/addHosts.md) и выполните запрос, например, с помощью {{ api-examples.rest.tool }}:

      ```bash
      curl \
          --request POST \
          --header "Authorization: Bearer $IAM_TOKEN" \
          --header "Content-Type: application/json" \
          --url 'https://{{ api-host-mdb }}/managed-mysql/v1/clusters/<идентификатор_кластера>/hosts:batchCreate' \
          --data '{
                    "hostSpecs": [
                      {
                        "zoneId": "<зона_доступности>",
                        "subnetId": "<идентификатор_подсети>",
                        "assignPublicIp": <публичный_адрес_хоста:_true_или_false>,
                        "replicationSource": "<FQDN_хоста>",
                        "backupPriority": "<приоритет_хоста_при_резервном_копировании>",
                        "priority": "<приоритет_назначения_хоста_мастером>"
                      }
                    ]
                  }'
      ```

      Где `hostSpecs` — массив новых хостов. Один элемент массива содержит настройки для одного хоста и имеет следующую структуру:

      * `zoneId` — зона доступности.
      * `subnetId` — идентификатор подсети.
      * `assignPublicIp` — доступность хоста из интернета по публичному IP-адресу.
      * `replicationSource` — источник репликации для хоста для [ручного управления потоками репликации](../concepts/replication.md#manual-source). В параметре укажите [FQDN хоста](connect.md#fqdn), который будет источником репликации.
      * `backupPriority` — приоритет хоста при [резервном копировании](../concepts/backup.md#size): от `0` до `100`.
      * `priority` — приоритет назначения хоста мастером при [выходе из строя основного мастера](../concepts/replication.md#master-failover): от `0` до `100`.

      Идентификатор кластера можно запросить со [списком кластеров в каталоге](cluster-list.md#list-clusters).

  1. Убедитесь, что запрос был выполнен успешно, изучив [ответ сервера](../api-ref/Cluster/addHosts.md#yandex.cloud.operation.Operation).

- gRPC API {#grpc-api}

  1. [Получите IAM-токен для аутентификации в API](../api-ref/authentication.md) и поместите токен в переменную среды окружения:

      {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

  1. {% include [grpc-api-setup-repo](../../_includes/mdb/grpc-api-setup-repo.md) %}
  1. Воспользуйтесь вызовом [ClusterService/AddHosts](../api-ref/grpc/Cluster/addHosts.md) и выполните запрос, например, с помощью {{ api-examples.grpc.tool }}:

      ```bash
      grpcurl \
          -format json \
          -import-path ~/cloudapi/ \
          -import-path ~/cloudapi/third_party/googleapis/ \
          -proto ~/cloudapi/yandex/cloud/mdb/mysql/v1/cluster_service.proto \
          -rpc-header "Authorization: Bearer $IAM_TOKEN" \
          -d '{
                "cluster_id": "<идентификатор_кластера>",
                "host_specs": [
                  {
                    "zone_id": "<зона_доступности>",
                    "subnet_id": "<идентификатор_подсети>",
                    "assign_public_ip": <публичный_адрес_хоста:_true_или_false>,
                    "replication_source": "<FQDN_хоста>",
                    "backup_priority": "<приоритет_хоста_при_резервном_копировании>",
                    "priority": "<приоритет_назначения_хоста_мастером>"
                  }
                ]
              }' \
          {{ api-host-mdb }}:{{ port-https }} \
          yandex.cloud.mdb.mysql.v1.ClusterService.AddHosts
      ```

      Где `host_specs` — массив новых хостов. Один элемент массива содержит настройки для одного хоста и имеет следующую структуру:

      * `zone_id` — зона доступности.
      * `subnet_id` — идентификатор подсети.
      * `assign_public_ip` — доступность хоста из интернета по публичному IP-адресу.
      * `replication_source` — источник репликации для хоста для [ручного управления потоками репликации](../concepts/replication.md#manual-source). В параметре укажите [FQDN хоста](connect.md#fqdn), который будет источником репликации.
      * `backup_priority` — приоритет хоста при [резервном копировании](../concepts/backup.md#size): от `0` до `100`.
      * `priority` — приоритет назначения хоста мастером при [выходе из строя основного мастера](../concepts/replication.md#master-failover): от `0` до `100`.

      Идентификатор кластера можно запросить со [списком кластеров в каталоге](cluster-list.md#list-clusters).

  1. Убедитесь, что запрос был выполнен успешно, изучив [ответ сервера](../api-ref/grpc/Cluster/create.md#yandex.cloud.operation.Operation).

{% endlist %}


{% note warning %}

Если после добавления хоста к нему невозможно [подключиться](connect.md), убедитесь, что [группа безопасности](../concepts/network.md#security-groups) кластера настроена корректно для подсети, в которую помещен хост.

{% endnote %}


## Изменить хост {#update}

Для каждого хоста в кластере {{ mmy-name }} можно:
* Указать [источник репликации](../concepts/replication.md#manual-source).
* Управлять [публичным доступом](../concepts/network.md#public-access-to-host).
* Задать [приоритет использования](../concepts/backup.md#size) при резервном копировании.
* Задать приоритет назначения хоста мастером при [выходе из строя основного мастера](../concepts/replication.md#master-failover).

{% note info %}

Перезапустить отдельный хост кластера невозможно. Для перезапуска хостов [остановите и запустите кластер](cluster-stop.md).

{% endnote %}

{% list tabs group=instructions %}

- Консоль управления {#console}

  Чтобы изменить параметры хоста в кластере:
  1. Перейдите на [страницу каталога]({{ link-console-main }}) и выберите сервис **{{ ui-key.yacloud.iam.folder.dashboard.label_managed-mysql }}**.
  1. Нажмите на имя нужного кластера и выберите вкладку **{{ ui-key.yacloud.mysql.cluster.switch_hosts }}**.
  1. Нажмите значок ![image](../../_assets/console-icons/ellipsis.svg) в строке нужного хоста и выберите пункт **{{ ui-key.yacloud.common.edit }}**.
  1. Задайте новые настройки для хоста:
     1. Выберите источник репликации для хоста, чтобы вручную управлять потоками репликации.
     1. Включите опцию **{{ ui-key.yacloud.mdb.hosts.dialog.field_public_ip }}**, если хост должен быть доступен извне {{ yandex-cloud }}.
     1. Задайте значение поля **{{ ui-key.yacloud.mysql.field_priority }}**.
     1. Задайте значение поля **{{ ui-key.yacloud.mysql.field_backup_priority }}**.
  1. Нажмите кнопку **{{ ui-key.yacloud.postgresql.hosts.dialog.button_choose }}**.

- CLI {#cli}

  {% include [cli-install](../../_includes/cli-install.md) %}

  {% include [default-catalogue](../../_includes/default-catalogue.md) %}

  Чтобы изменить параметры хоста, выполните команду (в примере приведены не все доступные параметры):

  ```bash
  {{ yc-mdb-my }} host update <имя_хоста> \
    --cluster-name=<имя_кластера> \
    --replication-source=<имя_хоста-источника> \
    --assign-public-ip=<публичный_доступ_к_хосту> \
    --backup-priority=<приоритет_хоста_при_резервном_копировании> \
    --priority=<приоритет_назначения_хоста_мастером>
  ```

  Где:
  * `--cluster-name` — имя кластера {{ mmy-name }}.
  * `--replication-source` — источник [репликации](../concepts/replication.md) для хоста.
  * `--assign-public-ip` — доступность хоста из интернета: `true` или `false`.
  * `--backup-priority` — приоритет хоста при [резервном копировании](../concepts/backup.md#size): от `0` до `100`.
  * `--priority` — приоритет назначения хоста мастером при [выходе из строя основного мастера](../concepts/replication.md#master-failover): от `0` до `100`.

  Имя хоста можно запросить со [списком хостов в кластере](#list), имя кластера — со [списком кластеров в каталоге](cluster-list.md#list-clusters).

- {{ TF }} {#tf}

  Чтобы изменить параметры хоста в кластере:
  1. Откройте актуальный конфигурационный файл {{ TF }} с планом инфраструктуры.

     О том, как создать такой файл, см. в разделе [Создание кластера](cluster-create.md).
  1. Измените в описании кластера {{ mmy-name }} атрибуты блока `host`, соответствующего изменяемому хосту.

     ```hcl
     resource "yandex_mdb_mysql_cluster" "<имя_кластера>" {
       ...
       host {
         replication_source_name = "<источник_репликации>"
         assign_public_ip        = <публичный_доступ_к_хосту>
         priority                = <приоритет_назначения_хоста_мастером>
       }
     }
     ```

     Где:

     * `assign_public_ip` — публичный доступ к хосту: `true` или `false`.
     * `priority` — приоритет назначения хоста мастером: от `0` до `100`.

  1. Проверьте корректность настроек.

     {% include [terraform-validate](../../_includes/mdb/terraform/validate.md) %}

  1. Подтвердите изменение ресурсов.

     {% include [terraform-apply](../../_includes/mdb/terraform/apply.md) %}

  Подробнее см. в [документации провайдера {{ TF }}]({{ tf-provider-mmy }}).

  {% include [Terraform timeouts](../../_includes/mdb/mmy/terraform/timeouts.md) %}

- REST API {#api}

  1. [Получите IAM-токен для аутентификации в API](../api-ref/authentication.md) и поместите токен в переменную среды окружения:

      {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

  1. Воспользуйтесь методом [Cluster.updateHosts](../api-ref/Cluster/updateHosts.md) и выполните запрос, например, с помощью {{ api-examples.rest.tool }}:

      ```bash
      curl \
          --request POST \
          --header "Authorization: Bearer $IAM_TOKEN" \
          --header "Content-Type: application/json" \
          --url 'https://{{ api-host-mdb }}/managed-mysql/v1/clusters/<идентификатор_кластера>/hosts:batchUpdate' \
          --data '{
                    "updateHostSpecs": [
                      {
                        "updateMask": "assignPublicIp,replicationSource,backupPriority,priority",
                        "hostName": "<FQDN_хоста>",
                        "assignPublicIp": <публичный_адрес_хоста:_true_или_false>,
                        "replicationSource": "<FQDN_хоста>",
                        "backupPriority": "<приоритет_хоста_при_резервном_копировании>",
                        "priority": "<приоритет_назначения_хоста_мастером>"
                      }
                    ]
                  }'
      ```

      Где `updateHostSpecs` — массив изменяемых хостов. Один элемент массива содержит настройки для одного хоста и имеет следующую структуру:

      * `updateMask` — перечень изменяемых параметров в одну строку через запятую.
      * `hostName` — [FQDN изменяемого хоста](connect.md#fqdn).
      * `assignPublicIp` — доступность хоста из интернета по публичному IP-адресу.
      * `replicationSource` — источник репликации для хоста для [ручного управления потоками репликации](../concepts/replication.md#manual-source). В параметре укажите [FQDN хоста](connect.md#fqdn), который будет источником репликации.
      * `backupPriority` — приоритет хоста при [резервном копировании](../concepts/backup.md#size): от `0` до `100`.
      * `priority` — приоритет назначения хоста мастером при [выходе из строя основного мастера](../concepts/replication.md#master-failover): от `0` до `100`.

      Идентификатор кластера можно запросить со [списком кластеров в каталоге](cluster-list.md#list-clusters).

  1. Убедитесь, что запрос был выполнен успешно, изучив [ответ сервера](../api-ref/Cluster/updateHosts.md#yandex.cloud.operation.Operation).

- gRPC API {#grpc-api}

  1. [Получите IAM-токен для аутентификации в API](../api-ref/authentication.md) и поместите токен в переменную среды окружения:

      {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

  1. {% include [grpc-api-setup-repo](../../_includes/mdb/grpc-api-setup-repo.md) %}
  1. Воспользуйтесь вызовом [ClusterService/UpdateHosts](../api-ref/grpc/Cluster/updateHosts.md) и выполните запрос, например, с помощью {{ api-examples.grpc.tool }}:

      ```bash
      grpcurl \
          -format json \
          -import-path ~/cloudapi/ \
          -import-path ~/cloudapi/third_party/googleapis/ \
          -proto ~/cloudapi/yandex/cloud/mdb/mysql/v1/cluster_service.proto \
          -rpc-header "Authorization: Bearer $IAM_TOKEN" \
          -d '{
                "cluster_id": "<идентификатор_кластера>",
                "update_host_specs": [
                  {
                    "update_mask": {
                      "paths": [
                        "assign_public_ip", "replication_source", "backup_priority", "priority"
                      ]
                    },
                    "host_name": "<FQDN_хоста>",
                    "assign_public_ip": <публичный_адрес_хоста:_true_или_false>,
                    "replication_source": "<FQDN_хоста>",
                    "backup_priority": "<приоритет_хоста_при_резервном_копировании>",
                    "priority": "<приоритет_назначения_хоста_мастером>"
                  }
                ]
              }' \
          {{ api-host-mdb }}:{{ port-https }} \
          yandex.cloud.mdb.mysql.v1.ClusterService.UpdateHosts
      ```

      Где `update_host_specs` — массив изменяемых хостов. Один элемент массива содержит настройки для одного хоста и имеет следующую структуру:

      * `update_mask` — перечень изменяемых параметров в виде массива строк `paths[]`.
      * `host_name` — [FQDN изменяемого хоста](connect.md#fqdn).
      * `assign_public_ip` — доступность хоста из интернета по публичному IP-адресу.
      * `replication_source` — источник репликации для хоста для [ручного управления потоками репликации](../concepts/replication.md#manual-source). В параметре укажите [FQDN хоста](connect.md#fqdn), который будет источником репликации.
      * `backup_priority` — приоритет хоста при [резервном копировании](../concepts/backup.md#size): от `0` до `100`.
      * `priority` — приоритет назначения хоста мастером при [выходе из строя основного мастера](../concepts/replication.md#master-failover): от `0` до `100`.

      Идентификатор кластера можно запросить со [списком кластеров в каталоге](cluster-list.md#list-clusters).

  1. Убедитесь, что запрос был выполнен успешно, изучив [ответ сервера](../api-ref/grpc/Cluster/create.md#yandex.cloud.operation.Operation).

{% endlist %}


{% note warning %}

Если после изменения хоста к нему невозможно [подключиться](connect.md), убедитесь, что [группа безопасности](../concepts/network.md#security-groups) кластера настроена корректно для подсети, в которую помещен хост.

{% endnote %}


## Удалить хост {#remove}

Вы можете удалить хост из кластера {{ MY }}, если он не является единственным хостом. Чтобы заменить единственный хост, сначала создайте новый хост, а затем удалите старый.

Если хост является мастером в момент удаления, {{ mmy-name }} автоматически назначит мастером следующую по приоритету реплику.

{% list tabs group=instructions %}

- Консоль управления {#console}

  1. Перейдите на [страницу каталога]({{ link-console-main }}) и выберите сервис **{{ ui-key.yacloud.iam.folder.dashboard.label_managed-mysql }}**.
  1. Нажмите на имя нужного кластера и выберите вкладку **{{ ui-key.yacloud.mysql.cluster.switch_hosts }}**.
  1. Нажмите значок ![image](../../_assets/console-icons/ellipsis.svg) в строке нужного хоста и выберите пункт **{{ ui-key.yacloud.common.delete }}**.

- CLI {#cli}

  {% include [cli-install](../../_includes/cli-install.md) %}

  {% include [default-catalogue](../../_includes/default-catalogue.md) %}

  Чтобы удалить хост из кластера, выполните команду:

  ```bash
  {{ yc-mdb-my }} host delete <имя_хоста> \
     --cluster-name=<имя_кластера>
  ```

  Имя хоста можно запросить со [списком хостов в кластере](#list), имя кластера — со [списком кластеров в каталоге](cluster-list.md#list-clusters).

- {{ TF }} {#tf}

  1. Откройте актуальный конфигурационный файл {{ TF }} с планом инфраструктуры.

     О том, как создать такой файл, см. в разделе [Создание кластера](cluster-create.md).
  1. Удалите из описания кластера {{ mmy-name }} блок `host`.
  1. Проверьте корректность настроек.

     {% include [terraform-validate](../../_includes/mdb/terraform/validate.md) %}

  1. Введите слово `yes` и нажмите **Enter**.

     {% include [terraform-apply](../../_includes/mdb/terraform/apply.md) %}

  Подробнее см. в [документации провайдера {{ TF }}]({{ tf-provider-mmy }}).

  {% include [Terraform timeouts](../../_includes/mdb/mmy/terraform/timeouts.md) %}

- REST API {#api}

  1. [Получите IAM-токен для аутентификации в API](../api-ref/authentication.md) и поместите токен в переменную среды окружения:

      {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

  1. Воспользуйтесь методом [Cluster.deleteHosts](../api-ref/Cluster/deleteHosts.md) и выполните запрос, например, с помощью {{ api-examples.rest.tool }}:

      ```bash
      curl \
          --request POST \
          --header "Authorization: Bearer $IAM_TOKEN" \
          --header "Content-Type: application/json" \
          --url 'https://{{ api-host-mdb }}/managed-mysql/v1/clusters/<идентификатор_кластера>/hosts:batchDelete' \
          --data '{
                    "hostNames": [
                      "<FQDN_хоста>"
                    ]
                  }'
      ```

      Где `hostNames` — массив с удаляемым хостом.

      В одном запросе можно передать только один FQDN хоста. Если нужно удалить несколько хостов, выполните запрос для каждого хоста.

      Идентификатор кластера можно запросить со [списком кластеров в каталоге](cluster-list.md#list-clusters).

  1. Убедитесь, что запрос был выполнен успешно, изучив [ответ сервера](../api-ref/Cluster/deleteHosts.md#yandex.cloud.operation.Operation).

- gRPC API {#grpc-api}

  1. [Получите IAM-токен для аутентификации в API](../api-ref/authentication.md) и поместите токен в переменную среды окружения:

      {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

  1. {% include [grpc-api-setup-repo](../../_includes/mdb/grpc-api-setup-repo.md) %}
  1. Воспользуйтесь вызовом [ClusterService/DeleteHosts](../api-ref/grpc/Cluster/deleteHosts.md) и выполните запрос, например, с помощью {{ api-examples.grpc.tool }}:

      ```bash
      grpcurl \
          -format json \
          -import-path ~/cloudapi/ \
          -import-path ~/cloudapi/third_party/googleapis/ \
          -proto ~/cloudapi/yandex/cloud/mdb/mysql/v1/cluster_service.proto \
          -rpc-header "Authorization: Bearer $IAM_TOKEN" \
          -d '{
                "cluster_id": "<идентификатор_кластера>",
                "host_names": [
                  "<FQDN_хоста>"
                ]
              }' \
          {{ api-host-mdb }}:{{ port-https }} \
          yandex.cloud.mdb.mysql.v1.ClusterService.DeleteHosts
      ```

      Где `host_names` — массив с удаляемым хостом.

      В одном запросе можно передать только один FQDN хоста. Если нужно удалить несколько хостов, выполните запрос для каждого хоста.

      Идентификатор кластера можно запросить со [списком кластеров в каталоге](cluster-list.md#list-clusters).

  1. Убедитесь, что запрос был выполнен успешно, изучив [ответ сервера](../api-ref/grpc/Cluster/create.md#yandex.cloud.operation.Operation).

{% endlist %}