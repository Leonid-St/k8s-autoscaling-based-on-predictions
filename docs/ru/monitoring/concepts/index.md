---
title: '{{ monitoring-full-name }}. Обзор сервиса'
description: Сервис {{ monitoring-full-name }} позволяет собирать и хранить метрики, а также отображать их в виде графиков на дашбордах. {{ monitoring-full-name }} автоматически собирает данные о состоянии ваших ресурсов в {{ yandex-cloud }} и отображает их на сервисных дашбордах. Для загрузки пользовательских метрик доступен API.
---

# Обзор сервиса {{ monitoring-name }}

Сервис {{ monitoring-name }} позволяет собирать и хранить метрики, а также отображать их в виде графиков на дашбордах.

{{ monitoring-name }} автоматически собирает данные о состоянии ваших ресурсов в {{ yandex-cloud }} и отображает их на сервисных дашбордах. Для загрузки пользовательских метрик доступен API.

С помощью сервиса {{ monitoring-name }} вы можете:
* Отслеживать состояние ваших сервисов в {{ yandex-cloud }} на сервисных дашбордах. Это позволяет контролировать текущую нагрузку на ресурсы и планировать увеличение [квот]({{ link-console-quotas }}).
* Загружать собственные метрики, используя API. Вы можете собрать на одном дашборде метрики вашего приложения и метрики используемых им ресурсов {{ yandex-cloud }}.
* Выгружать метрики ваших ресурсов и пользовательские метрики с помощью API.
* Создавать собственные дашборды и графики, чтобы визуализировать метрики наиболее удобным для вас способом.
* Настраивать уведомления (алерты) об изменении метрик. Уведомления можно направлять получателям по различным каналам связи.
* Для критичных изменений метрик можно настраивать последовательность уведомлений — политику эскалаций.

## Принцип работы сервиса {#how-it-works}

Сервис {{ monitoring-name }} собирает и хранит метрики в виде [временных рядов](https://ru.wikipedia.org/wiki/Временной_ряд). Для идентификации и описания характеристик временных рядов используются метки. Они указывают на принадлежность метрики сервису, описывают смысловое значение метрик и т. д. Подробнее в разделе [{#T}](data-model.md).

{{ monitoring-name }} отображает метрики на графиках. Множество связанных между собой графиков можно собрать на дашбордах. Подробнее в разделе [{#T}](visualization/index.md).

## Поставка метрик кластера {{ managed-k8s-full-name }} {#metrics-provider}

Сервис {{ monitoring-name }} позволяет выгружать метрики объектов [кластера {{ managed-k8s-name }}](../../managed-kubernetes/concepts/index.md#kubernetes-cluster). Провайдер преобразует запрос на получение внешних метрик от объекта в кластере {{ managed-k8s-name }} в нужный {{ monitoring-name }} формат, а также выполняет обратное преобразование — от {{ monitoring-name }} до объекта кластера.

{% include [metrics-k8s-tools](../../_includes/managed-kubernetes/metrics-k8s-tools.md) %}

Описание метрик приводится в [Справочнике](../metrics-ref/index.md#managed-kubernetes).