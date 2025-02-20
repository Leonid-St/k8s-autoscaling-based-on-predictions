---
title: Загрузка схемы в новый субъект
description: Следуя данной инструкции, вы сможете загрузить схему и создать субъект.
---

# Загрузка схемы в новый субъект

{% include notitle [preview](../../_includes/note-preview.md) %}

{% list tabs group=instructions %}

- Консоль управления {#console}
  
  1. В [консоли управления]({{ link-console-main }}) выберите [каталог](../../resource-manager/concepts/resources-hierarchy.md#folder), в котором нужно загрузить схему в субъект.
  1. Выберите сервис **{{ ui-key.yacloud.iam.folder.dashboard.label_metadata-hub }}**.
  1. Hа панели слева выберите ![image](../../_assets/console-icons/layout-cells.svg) **{{ ui-key.yacloud.iam.folder.dashboard.label_schema-registry }}**.
  1. Выберите пространство имен, для которого вы хотите загрузить схему.
  1. На панели слева выберите ![image](../../_assets/console-icons/branches-down.svg) **{{ ui-key.yacloud.schema-registry.label_schemas }}**.
  1. Нажмите кнопку **{{ ui-key.yacloud.schema-registry.label_upload-schema-action }}**.
  1. Выберите способ загрузки схемы в новый субъект и укажите следующие параметры:
      * **Имя** — уникальное имя субъекта;
      * (Опционально) Добавьте описание субъекта;
      * **Уровень проверки совместимости** — укажите, если хотите переопределить уровень, указанный для [пространства имен](create-name-space.md). Уровень переопределится только для текущего субъекта.
        * `BACKWARD`: (значение по умолчанию) потребители, использующие новую схему, могут читать данные, написанные производителями с использованием последней зарегистрированной схемы;
        * `BACKWARD_TRANSITIVE`: потребители, использующие новую схему, могут читать данные, записанные производителями с использованием всех ранее зарегистрированных схем;
        * `FORWARD`: потребители, использующие последнюю зарегистрированную схему, могут читать данные, написанные производителями, использующими новую схему;
        * `FORWARD_TRANSITIVE`: потребители, использующие все ранее зарегистрированные схемы, могут читать данные, написанные производителями с использованием новой схемы;
        * `FULL`: новая схема совместима вперед и назад с последней зарегистрированной схемой;
        * `FULL_TRANSITIVE`: новая схема совместима вперед и назад со всеми ранее зарегистрированными схемами;
        * `NONE`: проверки совместимости схемы отключены.
          Подробнее о типах совместимости схем см. в [документации Confluent](https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html#compatibility-types).
  1. Задайте формат данных [Protobuf](https://protobuf.dev/), [Avro](https://avro.apache.org/) или [JSON Schema](https://json-schema.org/) и прикрепите файл.
  1. Если схема ссылается на другую схему, то в разделе **Референсы** нажмите ![add](../../_assets/console-icons/plus.svg) и введите имя [референса](../../metadata-hub/concepts/schema-registry.md#reference), имя субъекта, под которым зарегистрирована схема для ссылки, и версию субъекта, которая содержит схему для референса.
  1. Чтобы применить [нормализацию схем данных](https://docs.confluent.io/platform/current/schema-registry/fundamentals/serdes-develop/index.html#schema-normalization), включите настройку **Нормализация**.
  1. Если вы хотите пропустить проверку совместимости схем, включите соответствующую настройку.
  1. Нажмите кнопку **{{ ui-key.yacloud.schema-registry.label_upload-schema-action }}**.

{% endlist %}
