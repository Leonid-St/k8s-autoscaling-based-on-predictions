1. В [консоли управления]({{ link-console-main }}) выберите [каталог](../../resource-manager/concepts/resources-hierarchy.md#folder), в котором нужно создать подключение.
1. Выберите сервис **{{ ui-key.yacloud.iam.folder.dashboard.label_metadata-hub }}**.
1. Hа панели слева выберите ![image](../../_assets/console-icons/layout-cells.svg) **{{ ui-key.yacloud.iam.folder.dashboard.label_schema-registry }}**.
1. Нажмите кнопку **{{ ui-key.yacloud.schema-registry.label_create-namespace-action }}**.
1. Укажите следующие параметры:
    * **{{ ui-key.yacloud.common.name }}** — уникальное имя пространства имен.
    * Опционально добавьте описание пространства имен.
    * **Уровень проверки совместимости**
        * `BACKWARD`: (значение по умолчанию) потребители, использующие новую схему, могут читать данные, написанные производителями с использованием последней зарегистрированной схемы;
        * `BACKWARD_TRANSITIVE`: потребители, использующие новую схему, могут читать данные, записанные производителями с использованием всех ранее зарегистрированных схем;
        * `FORWARD`: потребители, использующие последнюю зарегистрированную схему, могут читать данные, написанные производителями, использующими новую схему;
        * `FORWARD_TRANSITIVE`: потребители, использующие все ранее зарегистрированные схемы, могут читать данные, написанные производителями с использованием новой схемы;
        * `FULL`: новая схема совместима вперед и назад с последней зарегистрированной схемой;
        * `FULL_TRANSITIVE`: новая схема совместима вперед и назад со всеми ранее зарегистрированными схемами;
        * `NONE`: проверки совместимости схемы отключены.
          Подробнее о типах совместимости схем см. в [документации Confluent](https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html#compatibility-types).
    * **Правила проверки совместимости** — выберите, какие типы проверок схемы вы хотели бы проводить: [Confluent](https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html#compatibility-types) (значение по умолчанию) или [Buf](https://buf-build-git-psachs-docs-and-search-bufbuild.vercel.app/docs/build/usage/).
1. Нажмите кнопку **{{ ui-key.yacloud.common.create }}**.
