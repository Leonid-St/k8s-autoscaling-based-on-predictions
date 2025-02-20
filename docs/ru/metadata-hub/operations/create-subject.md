---
title: Создание субъекта
description: Следуя данной инструкции, вы сможете создать субъект.
---

# Создание субъекта

Вы можете создать субъект:

* [отдельно от схемы](#create-new-separate-subject), на вкладке **Субъекты**;
* [при загрузке новой схемы](#create-subject-during-schema-upload).

## Создание субъекта отдельно от схемы {#create-new-separate-subject}

{% list tabs group=instructions %}

- Консоль управления {#console}

  1. В [консоли управления]({{ link-console-main }}) выберите [каталог](../../resource-manager/concepts/resources-hierarchy.md#folder), в котором нужно создать субъект.
  1. Выберите сервис **{{ ui-key.yacloud.iam.folder.dashboard.label_metadata-hub }}**.
  1. Hа панели слева выберите ![image](../../_assets/console-icons/layout-cells.svg) **{{ ui-key.yacloud.iam.folder.dashboard.label_schema-registry }}**.
  1. Выберите пространство имен, в котором вы хотите создать субъект.
  1. На панели слева выберите ![image](../../_assets/console-icons/layers-3-diagonal.svg) **{{ ui-key.yacloud.schema-registry.label_subjects }}**.
  1. В правом верхнем углу нажмите кнопку **{{ ui-key.yacloud.schema-registry.label_create-subject-action }}**.
  1. Укажите следующие параметры:
      * Имя и описание субъекта;
      * **Уровень проверки совместимости**:
          * `BACKWARD`: (значение по умолчанию) потребители, использующие новую схему, могут читать данные, написанные производителями с использованием последней зарегистрированной схемы;
          * `BACKWARD_TRANSITIVE`: потребители, использующие новую схему, могут читать данные, записанные производителями с использованием всех ранее зарегистрированных схем;
          * `FORWARD`: потребители, использующие последнюю зарегистрированную схему, могут читать данные, написанные производителями, использующими новую схему;
          * `FORWARD_TRANSITIVE`: потребители, использующие все ранее зарегистрированные схемы, могут читать данные, написанные производителями с использованием новой схемы;
          * `FULL`: новая схема совместима вперед и назад с последней зарегистрированной схемой;
          * `FULL_TRANSITIVE`: новая схема совместима вперед и назад со всеми ранее зарегистрированными схемами;
          * `NONE`: проверки совместимости схемы отключены.
          Подробнее о типах совместимости схем см. в [документации Confluent](https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html#compatibility-types).
      * В разделе **Схема**:
          * Задайте формат схемы [Protobuf](https://protobuf.dev/), [Avro](https://avro.apache.org/) или [JSON Schema](https://json-schema.org/) и прикрепите файл.
          * Если схема ссылается на другую схему, то в разделе **Референсы** нажмите ![add](../../_assets/console-icons/plus.svg) и введите имя [референса](../../metadata-hub/concepts/schema-registry.md#reference), имя субъекта, под которым зарегистрирована схема для ссылки, и версию схемы зарегистрированного субъекта.
          * Чтобы применить [нормализацию схем данных](https://docs.confluent.io/platform/current/schema-registry/fundamentals/serdes-develop/index.html#schema-normalization), включите настройку **Нормализация**.
          * Если вы хотите пропустить проверку совместимости схем, включите соответствующую настройку.
  1. Нажмите кнопку **{{ ui-key.yacloud.common.create }}**.

{% endlist %}

## Создание субъекта при загрузке схемы {#create-subject-during-schema-upload}

Чтобы создать новый субъект при загрузке схемы, в поле **Субъект** выберите **Новый** и задайте параметры субъекта: имя, описание и уровень проверки совместимости. Подробнее о создании субъекта при загрузке схемы см. в [инструкции](upload-schema-to-subject.md).

