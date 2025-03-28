# Локальные поля задач

Если требуется добавить в задачи новый параметр, которого нет среди существующих в {{ tracker-name }} полей, вы можете добавить локальные поля в вашу очередь.

{% note info %}

Список существующих глобальных полей можно посмотреть на странице [Настройки {{ tracker-name }}]({{ link-admin-fields }}).

{% endnote %}

Локальное поле можно использовать только в задачах той очереди, к которой оно привязано. Преимущество локальных полей в том, что владелец очереди может управлять такими полями без риска повлиять на процессы работы в других очередях. Пользователи, которые работают в других очередях, не будут видеть это поле в своих задачах.

## Добавить локальное поле {#add-local-field}

{% note alert %}

По умолчанию настраивать очередь может [только ее владелец](manager/queue-access.md).

{% endnote %}

1. Откройте [страницу очереди](user/queue.md).

1. В правом верхнем углу страницы нажмите ![](../_assets/tracker/svg/settings-old.svg) **{{ ui-key.startrek.ui_Queues_pages_PageQueue_header.settings }}**.

1. Перейдите на вкладку **Локальные поля**.

1. Нажмите кнопку **Создать поле**.

1. Выберите тип поля и нажмите кнопку **Продолжить**.

1. Задайте параметры нового поля:

    * **Название**. Старайтесь давать полям короткие и емкие названия.

    * **Название на английском**. Это название видно в английском интерфейсе {{ tracker-name }}.

    * **Ключ** генерируется автоматически по названию на английском.

    * **Категория**. Все поля в {{ tracker-name }} группируются по категориям. Начните набирать первые буквы предполагаемой категории и выберите из списка ту, что лучше всего подходит. Например: Системные, Учёт Времени, Agile, Email, SLA.

        Подробнее о полях и категориях читайте в разделе [Стандартные поля задач](user/create-param.md#default-fields).

    * **Множественный выбор** (только для полей типа «Список пользователей», «Выпадающий список», «Поле ввода»).

    * **Тип поля** (только для полей типа «Число», «Дата», «Поле ввода»).

    * **Значения в списке** (только для полей типа «Выпадающий список») — укажите возможные значения и их порядок.
    
1. Нажмите кнопку **Создать поле**. Появится всплывающее уведомление «Новое локальное поле успешно сохранено», а новое поле добавится в список. 

Локальные поля можно переносить в другие категории. Для этого нажмите ![](../_assets/tracker/svg/local-field-move.svg) справа от описания поля в списке и введите название категории назначения.

## Редактировать локальное поле {#edit-local-field}

Редактирование параметров локального поля через веб-интерфейс {{ tracker-name }} не поддерживается. Для этого можно использовать [{{ api-name }}](concepts/queues/edit-local-field.md).


## Удалить локальное поле {#delete-local-field}

Локальное поле удалить невозможно.

Скрыть локальное поле в интерфейсе вы можете с помощью [{{ api-name }}](concepts/queues/edit-local-field.md). Для этого установите следующие значения параметров: `"visible": false, "hidden": true`. 

## Ограничения локальных полей {#restrictions}

#### Особенности использования локальных полей

* Для поиска задач по локальному полю с помощью языка запросов нужно [указать очередь перед ключом или названием поля](user/query-filter.md#local_fields).

* При [переносе](user/move-ticket.md) и [клонировании](user/clone.md) задачи с локальными полями в другую очередь значения локальных полей будут удалены.

* При использовании [макросов](manager/create-macroses.md), [триггеров](user/trigger.md) или [автодействий](user/autoactions.md), чтобы подставить значение локального поля в [комментарий](user/set-action.md#create-comment), [формулу](user/set-action.md#section_calc_field) или [HTTP-запрос](user/set-action.md#create-http) с помощью [переменной](user/vars.md#local-fields), используйте формат записи `{{issue.local.<ключ_поля>}}`. 

#### Где нельзя использовать локальные поля

Локальные поля временно не поддерживаются в некоторых сценариях работы с задачами. Поддержка этих сценариев будет добавлена позже.

* При создании доски задач не получится настроить автоматическое добавление задач по значению локального поля. Это условие можно будет добавить позже, на [странице редактирования](manager/edit-agile-board.md#board-settings) доски с помощью [языка запросов](user/query-filter.md).

* На доске задач локальные поля не отображаются на [карточках](manager/edit-agile-board.md#sec_layout).

* Локальные поля нельзя использовать в [правилах SLA](sla-head.md).

* Значения локальных полей нельзя изменять с помощью [групповых операций](manager/bulk-change.md).

## Как обращаться к локальным полям через API {#local-fields-api}

При работе с локальными полями через [API {{ tracker-name }}](about-api.md) доступно два типа действий:

* Присвоить значение локальному полю.  
  
  Чтобы получить или изменить значение локального поля в задаче через API, в теле [запроса](concepts/issues/patch-issue.md) укажите его идентификатор в формате `603fb94c38bbe658********--<ключ_поля>: "<новое_значение_локального_поля>"`.

  Чтобы узнать идентификатор локального поля, выполните запрос, который позволяет получить [список локальных полей в определенной очереди](concepts/queues/get-local-fields.md).

* Изменить параметры локального поля, например название, описание, или множество доступных значений поля. Подробнее в [справочнике API {{ tracker-name }}](concepts/queues/edit-local-field.md).
