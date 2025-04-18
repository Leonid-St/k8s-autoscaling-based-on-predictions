# Взаимосвязь ресурсов в {{ ml-platform-name }}

{{ ml-platform-name }} работает в рамках [организаций {{ yandex-cloud }}](../../organization/). Все создаваемые сущности {{ ml-platform-name }} являются ресурсами организации. Обмен ресурсами между организациями невозможен.

_[Сообщества](community.md)_ — способ организации групповой работы. Сообщества определяют область видимости проектов и ресурсов {{ ml-platform-name }}.

Сообщество можно создать в одной из [зон доступности](../../overview/concepts/geo-scope.md). Все проекты и ресурсы, созданные в сообществе, также будут размещаться в этой зоне доступности. В другой зоне доступности можно разместить только [ноды](./deploy/index.md#node). После создания сообщество нельзя перенести в другую зону.

_[Проекты](project.md)_ — основное рабочее место пользователя в {{ ml-platform-name }}. В проектах хранятся код, переменные, установленное ПО и прочая информация.

_Ресурсы {{ ml-platform-name }}_ — объекты, которые создаются или используются в проектах: датасеты, Docker-образы, ноды и другие.

## Ресурсы {{ ml-platform-name }} {#resources}

В проектах {{ ml-platform-name }} можно использовать следующие типы ресурсов:

* [Датасеты](dataset.md) — способ хранения информации, который предоставляет быстрый доступ к большим объемам данных внутри проекта.
* [Секреты](secrets.md) — пары ключ-значение, в которых приватные данные (токены, ключи и прочее) хранятся в зашифрованном виде. Секреты создаются в проекте и закрепляются за ним. Созданные секреты можно использовать в коде ячейки как переменные окружения.
* [Docker-образы](docker.md) — окружение операционной системы, в котором собран произвольный набор ПО, библиотек, переменных окружения и конфигурационных файлов.
* [Коннекторы к хранилищам S3](s3-connector.md) — сохраненные конфигурации для подключения [бакетов {{ objstorage-name }}](../../storage/concepts/bucket.md). Бакеты можно монтировать в файловую систему проекта, чтобы облегчить доступ к данным из кода. О том, как создать коннектор S3, см. [{#T}](../operations/data/s3-connectors.md).
* [Ноды](deploy/index.md#node) — сервисы, развернутые для [эксплуатации обученных моделей](deploy/). Сторонние сервисы могут обращаться к нодам по [API](../../glossary/rest-api.md).
* [Алиасы](deploy/index.md#alias) — «надстройка» для публикации сервисов. Алиасы позволяют распределять нагрузку между нодами и обновлять развернутые сервисы во время работы.
* [Шаблоны {{ dataproc-name }}](data-processing-template.md) — готовые конфигурации кластеров {{ dataproc-name }} для автоматического развертывания кластеров из проекта {{ ml-platform-name }}.
* [Модели](models/index.md) — сохраненное состояние интерпретатора, результаты вычислений или обучения. Делятся на обученные в проектах модели и дообученные [фундаментальные модели](models/foundation-models.md).
* [Коннекторы Spark](spark-connector.md) — сохраненные конфигурации для подключения существующих кластеров {{ dataproc-name }} и создания временных кластеров.

## Совместное использование проектов и ресурсов {#sharing}

Для совместного использования проектов и ресурсов {{ ml-platform-name }} предусмотрена возможность публикации ресурсов в сообществах.

Публикация ресурса означает, что все пользователи сообщества получат доступ к ресурсу. Это позволит им использовать ресурс в своих проектах в рамках сообщества. Публиковать ресурсы можно как в сообществе проекта, так и в других сообществах в пределах организации.

Обмен ресурсами между сообществами позволяет использовать Docker-образы, датасеты и другие объекты разными командами внутри одной организации.

Видимость сообществ, проектов и ресурсов {{ ml-platform-name }} ограничена рамками [организации](../../organization/). Обмен ресурсами между организациями невозможен. Также нельзя поделиться ресурсом в сообществе, которое было создано в другой зоне доступности.

Вы можете делиться ресурсами проекта {{ ml-platform-name }}, в котором имеете как минимум роль `Editor`, в любом сообществе организации, в котором вы состоите с минимальной ролью `Developer`. Открыть доступ можно на вкладке **{{ ui-key.yc-ui-datasphere.common.access }}** на странице просмотра ресурса. Подробнее см. [{#T}](../security/index.md).

## Связь ресурсов {{ ml-platform-name }} с сервисами {{ yandex-cloud }} {#ml-cloud-connection}

Сообщества {{ ml-platform-name }} являются ресурсами организации. В одной организации может быть множество сообществ.

Для оплаты сервиса {{ ml-platform-name }} используется [платежный аккаунт](../../billing/concepts/billing-account.md) {{ yandex-cloud }}.

Для доступа к другим сервисам {{ yandex-cloud }} используются [каталоги](../../resource-manager/concepts/resources-hierarchy.md#folder). В них размещены ресурсы конкретного сервиса {{ yandex-cloud }}. Работа с сервисами {{ yandex-cloud }} осуществляется с помощью [сервисных аккаунтов](../../iam/concepts/users/service-accounts.md).
