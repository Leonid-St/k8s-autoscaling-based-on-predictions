---
title: Версионирование чарта
description: Версионирование чарта — это возможность хранить историю изменений конфигурации чарта с помощью версий. Список версий доступен пользователям с минимальным правом доступа {{ permission-read }} на чарт.
---

# Версионирование

Версионирование чарта — это возможность хранить историю изменений конфигурации чарта с помощью версий. Список версий доступен пользователям с минимальным правом доступа **{{ permission-read }}** на чарт.

{% note info %}

В настоящее время версионирование доступно только для чартов [на основе датасета](#dataset-based-charts).

{% endnote %}

Чтобы перейти к списку версий, в верхней части экрана нажмите на значок ![image](../../../_assets/console-icons/ellipsis.svg) и выберите **История изменений**.

![image](../../../_assets/datalens/concepts/version-list.png)

Чтобы перейти к выбранной версии, нажмите на нее в списке. Номер версии отобразится в значении параметра `revId` строки адреса чарта. При переходе по ссылке, содержащей номер версии в параметре `revId`, сразу откроется эта версия чарта.

При редактировании в пределах текущей версии можно отменить или повторно выполнить внесенные изменения. Для этого в верхней правой части экрана нажмите значок:

* ![image](../../../_assets/console-icons/arrow-uturn-ccw-left.svg) — чтобы отменить изменения;
* ![image](../../../_assets/console-icons/arrow-uturn-cw-right.svg) — чтобы повторно выполнить изменения.

Несохраненные изменения в текущей версии сбрасываются:

* при обновлении страницы;
* при сохранении чарта;
* при переключении на другую версию.

## Виды версий {#version-types}

Различают следующие версии:

* **Актуальная**. Последняя сохраненная версия чарта. Отображается всем пользователям на дашбордах, а также при переходе из навигации или при открытии чарта из контекстного меню на редактирование с дашборда. Актуальной может быть только одна версия чарта.
  
  ![image](../../../_assets/datalens/concepts/current-version.png)

  Если у пользователя есть право доступа **{{ permission-write }}**, он может сделать актуальной любую версию чарта.
  
  {% note warning %}
  
  При актуализации любой версии, кроме черновой, создается новая версия чарта.

  {% endnote %}
  
* **Черновик**. Версия, которая содержит несохраненные изменения чарта. Основные пользователи не видят изменений, которые вносятся в черновик. Это позволяет скрывать редактирование чарта до актуализации версии. Чарт может иметь только один черновик.

  ![image](../../../_assets/datalens/concepts/draft-version.png)

  Чтобы создать черновую версию после редактирования чарта, в правом верхнем углу нажмите значок галочки и выберите **Сохранить как черновик**.

  Черновую версию чарта можно отобразить на дашборде. Для этого [добавьте в параметры](../../operations/chart/add-parameters.md) виджета на дашборде для этого чарта параметр `unreleased` со значением `1`.

* **Неактуальная**. Версия, которая не является актуальной или черновиком.

  ![image](../../../_assets/datalens/concepts/old-version.png)

{% note tip %}

Любой версией чарта можно поделиться: добавьте к ссылке на чарт параметр `revId` (например, `?revId=zac5m4edoaqqr`).

{% endnote %}

## Создание новой версии {#version-create}

Новая версия автоматически создается:

* в режиме редактирования актуальной версии чарта — после нажатия кнопки **Сохранить** (создается новая актуальная версия) или ![chevron-down](../../../_assets/console-icons/chevron-down.svg) → **Сохранить как черновик** (создается новая черновая версия);
* в режиме редактирования черновика или неактуальной версии чарта — после нажатия кнопки **Сохранить как черновик** (создается новая черновая версия) или ![chevron-down](../../../_assets/console-icons/chevron-down.svg) → **Сохранить и сделать актуальной** (создается новая актуальная версия);
* в режиме просмотра черновика или неактуальной версии чарта — после нажатия кнопки **Сделать актуальной**.

Изменения конфигурации чарта, которые приводят к созданию новой версии:

* изменение настроек чарта, доступных при нажатии на значок ![image](../../../_assets/console-icons/gear.svg) вверху экрана;
* добавление, переименование, удаление полей чарта;
* добавление, удаление полей в секции чарта.

## Ограничения {#restrictions}

* История изменений содержит только список версий чарта и включает: вид версии, дату и время сохранения и автора редактирования.
* Версии чарта не содержат изменений прав доступа (эта операция производится отдельно от редактирования самого чарта).
* В версиях не отображается список изменений. Доступен лишь просмотр сохраненного состояния конфигурации чарта.

