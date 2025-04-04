---
title: 'Знакомство с OLTP: ключевые концепции и сценарии применения'
description: OLTP (OnLine Transaction Processing) — это система оперативной обработки транзакций.
---

# OLTP

OLTP (OnLine Transaction Processing) — система обработки транзакций в реальном времени.

Транзакция — набор запросов, который выполняется в единой очереди. При неудачном или ошибочном завершении любого из запросов отменяется результат работы всего набора запросов.

Реальное время — режим работы системы, когда важны не только правильность обработки и предоставления информации, но и своевременность ее обработки. Информация в системах реального времени (СРВ) должна быть обработана за определенное фиксированное время, либо до наступления какого-либо контрольного события.

## Как работает OLTP {#principles}

OLTP-система обеспечивает непрерывную запись информации в базу данных. Это могут быть сигналы с устройств автоматического сбора данных (например, датчиков технологического процесса), данные о денежных переводах клиентов банка и т.п.

Одновременно OLTP обслуживает выполнение запросов от других систем на чтение данных из базы: от информации об удачном или неудачном завершении определенной транзакции до выборки информации для аналитических отчетов.

Все входящие (на запись) и исходящие (на чтение) запросы инициируются различными пользователями, внешними системами и алгоритмами и выполняются условно параллельно.

## Особенности OLTP {#features}

В отличие от OLAP-систем (OnLine Analytical Processing), OLTP не предусматривает работу со сложной аналитикой, так как:
* база данных OLTP одномерна;
* в базе отсутствует агрегированная (сводная) информация для анализа;
* OLTP-система не оптимизирована для выполнения вложенных запросов на выборку данных из разных таблиц.

Если требуется получить и проанализировать срез данных из OLTP, необходимо:
1. Запустить запрос на чтение из базы данных по интересующему нас фильтру (периоду, объекту и т.п.).
1. Выгрузить результат запроса для автоматизированного анализа в специализированную систему, например OLAP, либо в файл электронной таблицы (`xls` и т.п.) для ручного анализа.

Ориентированность OLTP на высокую скорость обработки простых транзакций имеет и свои слабые стороны:
* Низкая производительность при анализе больших объемов информации (статистические расчеты, Big Data).
  Каждая транзакция в системе для обеспечения принципа изолированности требует выделения отдельных ресурсов. При значительном количестве подобных транзакций (от миллиона и выше) это приведет к постоянным блокировкам записей и таблиц базы данных и общему снижению производительности системы.
* Сложность создания аналитических запросов для выборки данных.
  Высокая степень нормализации OLTP базы данных требует соответствующей высокой вложенности выражений в запросе. К тому же наименования объектов в базе не являются мнемоничными или интуитивно понятными. Пользователю для создания запроса необходимо привлечение разработчика системы или администратора базы данных.
* Снижение производительности системы при запросах к таблицам с избыточным количеством полей и записей.
  Это связано с тем, что структура [реляционной базы данных](relational-databases.md) плохо адаптирована для хранения большого объема информации в одной таблице. Решением проблемы являются изменение архитектуры системы (применение вместо одной большой сводной таблицы с данными нескольких связанных между собой по ключевым полям таблиц) или перенос архивных данных в системы других типов (например, в хранилище или киоск данных).

## Требования к OLTP {#requirements}

Основные требования к OLTP-системам:
1. Достоверность информации. Вся хранящаяся информация должна подчиняться принципу ACID:
   * Atomicity — атомарность или неделимость информации.
     Каждая транзакция в системе или завершается успешно, или полностью откатывается к исходным значениям.
   * Consistency — консистентность или согласованность.
     Информация не берется из ниоткуда и не исчезает в никуда, а просто передается от источника информации к ее приемнику.
   * Isolation — изолированность или блокировка.
     Для обеспечения достоверности информации действия пользователя (чтение или запись) должны быть изолированы от действий других пользователей.
   * Durability — устойчивость.
     Успешно проведенная транзакция должна быть защищена от любого внешнего сбоя, как программного, так и аппаратного.
1. Высокая скорость работы с информацией (как чтение, так и запись в базу данных). Идеальным вариантом является работа системы полностью из оперативной памяти с периодическим сохранением информации на постоянный носитель.
1. Легкая масштабируемость. OLTP система должна строиться по принципу нормализации, когда исключается избыточное хранение данных. Данные хранятся только в одном месте (ячейке таблицы), прочие таблицы при необходимости используют ссылки на эти ячейки без дублирования данных.

При этом требований к хранимой структуре данных не предъявляется, поэтому такая система не предназначена для сложного аналитического учета, который требуется, например, от OLAP.

Под эти условия работы попадают системы, построенные на [реляционных базах данных](relational-databases.md).

## Где применяется OLTP {#usage}

OLTP используется везде, где необходим доступ в реальном времени к большим объемам однотипной информации:
* в автоматизированных системах управления технологическими процессами (SCADA) при оперативной обработке сигналов, поступающих с различных датчиков;
* в банковском секторе при обработке платежных транзакций;
* в ERP-системах предприятий при работе алгоритмов адресного складского хранения;
* в онлайн-магазинах и электронных торговых площадках при работе с заказами и лотами.

## OLTP технологии от {{ yandex-cloud }} {#yc}

Вы можете использовать инструменты {{ yandex-cloud }}, например, при проектировании нового интернет-магазина:
* Облачный сервис [{{ ydb-full-name }}]({{ link-cloud-services }}/ydb) поможет развернуть и поддерживать базу данных YDB в инфраструктуре {{ yandex-cloud }}.
* Распределенная отказоустойчивая OLTP-база данных [YDB](https://ydb.tech/{{ lang }}) обеспечит высокую доступность для клиентов онлайн-магазина.
* Строгая согласованность и поддержка ACID-транзакций в YDB гарантирует сохранность и достоверность информации по заказам клиентов.
* При расширении онлайн-бизнеса YDB в облаке {{ yandex-cloud }} обеспечит простую горизонтальную масштабируемость без дорогостоящей реструктуризации проекта.
* Диалект языка SQL, [{{ yql-short-name }}]({{ ydb.docs }}/yql/reference/), легкий в понимании и простой в освоении, поможет создать и обработать необходимые запросы информации в YDB.

Более подробно работа с YDB описана в [документации](../ydb/).

Чтобы начать работу с сервисом, войдите в свой аккаунт в {{ yandex-cloud }} или [зарегистрируйтесь]({{ link-console-billing }}/create-account).
