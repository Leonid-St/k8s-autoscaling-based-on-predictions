---
title: Какие есть топики и разделы в {{ mkf-full-name }}
description: Из статьи вы узнаете, какие есть топики и разделы в {{ mkf-name }}.
---

# Топики и разделы

## Топики {#topics}

Топик — это способ группировки потоков сообщений по категориям. [Производители](producers-consumers.md) публикуют сообщения определенной категории в топик, а [потребители](producers-consumers.md) подписываются на этот топик и читают из него сообщения. Для каждого топика {{ KF }} ведет лог сообщений, который может быть разбит на несколько разделов.

Например, если производителем данных выступает интернет-магазин, вы можете создать отдельные топики для логирования действий пользователя, для хранения данных о его корзине, для записей о транзакциях и т. д.

{{ mkf-name }} управляет хранением сообщений в топике и обеспечивает:

- Репликацию разделов — при условии, что кластер состоит хотя бы из двух [брокеров](brokers.md) и для топиков задан фактор репликации больше единицы.
- Сжатие сообщений.
- Очистку лога [в соответствии с политикой](../operations/cluster-topics.md#create-topic) при устаревании сообщений в разделах или достижении заданного размера лога.

Подробнее о топиках см. [в документации {{ KF }}](https://kafka.apache.org/documentation/#intro_concepts_and_terms).

### Служебные топики {#service-topics}

В процессе своей работы {{ mkf-name }} может создавать и использовать служебные топики. Записывать пользовательские данные в такие топики нельзя.

Служебный топик `__schema_registry` используется для обеспечения работы [{{ mkf-msr }}](./managed-schema-registry.md).

Служебные топики `__connect-configs`, `__connect-offsets`, `__connect-status` используются для обеспечения работы [{{ mkf-mkc }}](./connectors.md).

## Разделы {#partitions}

Раздел — это последовательность сообщений топика, которые упорядочены в порядке их поступления. Порядковый номер сообщения называют смещением. Потребители читают сообщения от начала к концу раздела, то есть первыми будут прочитаны те сообщения, которые поступили раньше. Чтобы начать читать с определенного сообщения, потребители должны передать брокеру его смещение.

Использование разделов позволяет:

- Распределять нагрузку по хранению сообщений и обработке запросов между несколькими [брокерами {{ KF }}](brokers.md).
- Обеспечивать отказоустойчивость: раздел может реплицироваться на указанное число брокеров.

Подробнее о разделах см. [в документации {{ KF }}](https://kafka.apache.org/documentation/#intro_concepts_and_terms).

## Управление топиками и разделами {#management}

Кластер {{ mkf-name }} позволяет управлять топиками и разделами двумя способами:

* С помощью стандартных интерфейсов {{ yandex-cloud }} (консоль управления, CLI, {{ TF }}, [API](../../glossary/rest-api.md)). Способ подходит, если вы хотите создавать, удалять и настраивать топики и разделы, используя возможности сервиса {{ mkf-name }}.

* С помощью [Admin API {{ KF }}](https://kafka.apache.org/documentation/#adminapi). Способ подходит, если у вас уже есть собственное решение для управления топиками и разделами. При использовании Admin API действуют ограничения:

    {% include [admin-api-limits](../../_includes/mdb/mkf/admin-api-limits.md) %}

Оба способа можно использовать как по отдельности, так и совместно.
