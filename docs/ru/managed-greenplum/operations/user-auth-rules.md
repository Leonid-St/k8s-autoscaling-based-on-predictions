# Правила аутентификации пользователей

Правила [аутентификации пользователей](../concepts/user-authentication.md) определяют, какие пользователи и с каких адресов могут подключаться к кластеру и к каким базам данных они могут получить доступ.

Вы можете:

* [Добавить правила](#add-rules).
* [Изменить правила](#edit-rules).
* [Переместить правило](#replace-rule).
* [Удалить правило](#delete-rule).

{% note warning %}

Приоритет правил определяется порядком строк: правила читаются сверху вниз, и применяется первое подходящее. Если по первому подходящему правилу аутентификация не прошла, последующие не рассматриваются.

{% endnote %}

## Добавить правила {#add-rules}

{% list tabs group=instructions %}

- Консоль управления {#console}

    1. Перейдите на [страницу каталога]({{ link-console-main }}) и выберите сервис **{{ ui-key.yacloud.iam.folder.dashboard.label_managed-greenplum }}**.
    1. Нажмите на имя нужного кластера и выберите вкладку ![image](../../_assets/console-icons/persons.svg) **{{ ui-key.yacloud.greenplum.label_user-auth }}**.
    1. Нажмите кнопку **{{ ui-key.yacloud.greenplum.cluster.user-auth.action_edit-rules }}**.
    1. Нажмите кнопку ![image](../../_assets/console-icons/plus.svg) **{{ ui-key.yacloud.greenplum.cluster.user-auth.action_add-rule }}** и задайте его параметры:

        {% include [auth-user-rule-settings](../../_includes/mdb/mgp/auth-user-rule-settings.md) %}

    1. Чтобы добавить еще одно правило, еще раз нажмите кнопку ![image](../../_assets/console-icons/plus.svg) **{{ ui-key.yacloud.greenplum.cluster.user-auth.action_add-rule }}**.
    1. Нажмите кнопку **{{ ui-key.yacloud.common.save }}**.

    {% note info %}

    В конец списка автоматически добавляется правило по умолчанию, которое разрешает аутентификацию всем пользователям во всех базах и со всех хостов с помощью метода `md5` (по паролю).

    {% endnote %}

{% endlist %}

## Изменить правила {#edit-rules}

{% list tabs group=instructions %}

- Консоль управления {#console}

    1. Перейдите на [страницу каталога]({{ link-console-main }}) и выберите сервис **{{ ui-key.yacloud.iam.folder.dashboard.label_managed-greenplum }}**.
    1. Нажмите на имя нужного кластера и выберите вкладку ![image](../../_assets/console-icons/persons.svg) **{{ ui-key.yacloud.greenplum.label_user-auth }}**.
    1. Нажмите кнопку **{{ ui-key.yacloud.greenplum.cluster.user-auth.action_edit-rules }}** и измените параметры правил:

        {% include [auth-user-rule-settings](../../_includes/mdb/mgp/auth-user-rule-settings.md) %}

    1. Нажмите кнопку **{{ ui-key.yacloud.common.save }}**.

{% endlist %}

## Переместить правило {#replace-rule}

{% list tabs group=instructions %}

- Консоль управления {#console}

    1. Перейдите на [страницу каталога]({{ link-console-main }}) и выберите сервис **{{ ui-key.yacloud.iam.folder.dashboard.label_managed-greenplum }}**.
    1. Нажмите на имя нужного кластера и выберите вкладку ![image](../../_assets/console-icons/persons.svg) **{{ ui-key.yacloud.greenplum.label_user-auth }}**.
    1. Нажмите кнопку **{{ ui-key.yacloud.greenplum.cluster.user-auth.action_edit-rules }}**.
    1. Нажмите на значок ![image](../../_assets/console-icons/grip.svg) для нужного правила и переместите его выше или ниже.
    1. Нажмите кнопку **{{ ui-key.yacloud.common.save }}**.

{% endlist %}

## Удалить правило {#delete-rule}

{% list tabs group=instructions %}

- Консоль управления {#console}

    1. Перейдите на [страницу каталога]({{ link-console-main }}) и выберите сервис **{{ ui-key.yacloud.iam.folder.dashboard.label_managed-greenplum }}**.
    1. Нажмите на имя нужного кластера и выберите вкладку ![image](../../_assets/console-icons/persons.svg) **{{ ui-key.yacloud.greenplum.label_user-auth }}**.
    1. Нажмите кнопку **{{ ui-key.yacloud.greenplum.cluster.user-auth.action_edit-rules }}**.
    1. Нажмите на значок ![image](../../_assets/console-icons/ellipsis.svg) для нужного правила и выберите пункт **{{ ui-key.yacloud.common.delete }}**.
    1. Нажмите кнопку **{{ ui-key.yacloud.common.save }}**.

{% endlist %}

{% include [greenplum-trademark](../../_includes/mdb/mgp/trademark.md) %}
