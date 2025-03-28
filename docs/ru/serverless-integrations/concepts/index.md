---
title: Что такое {{ si-name }}? Обзор {{ si-full-name }}
description: '{{ si-name }} – это сервис для настройки интеграций и управления ими с помощью serverless-технологий в {{ yandex-cloud }}.'
keywords:
  - workflows
  - workflow
  - воркфлоу
  - eventrouter
  - event router
  - рабочий процесс
  - спецификация YaWL
  - Glue
  - интеграции
  - API Gateway
  - API шлюз
  - шина
  - коннектор
  - правило
---

# Обзор сервиса {{ si-name }}

{{ si-full-name }} — это сервис для настройки интеграций и управления ими с помощью serverless-технологий в {{ yandex-cloud }}.

{{ si-name }} позволяет:
* разрабатывать микросервисные архитектуры без необходимости конфигурировать виртуальные машины;
* создавать и автоматизировать рабочие процессы для реагирования на инциденты безопасности;
* автоматизировать бизнес-операции;
* настраивать пайпланы CI/CD;
* разрабатывать событийно-ориентированные приложения на базе serverless, используя оркестрацию и хореографию, чтобы организовывать взаимодействие между событиями и управлять ими.

## Доступные функциональности {#instruments}

### {{ sw-name }} {#workflows}

{% include [workflows-preview-note](../../_includes/serverless-integrations/workflows-preview-note.md) %}

Выстраивайте и автоматизируйте рабочие процессы при помощи декларативной спецификации Yandex Workflows Language (YaWL).

### {{ er-name }} {#eventrouter}

{% include [event-router-preview-note](../../_includes/serverless-integrations/event-router-preview-note.md) %}

Настраивайте обмен событиями между вашими сервисами и сервисами {{ yandex-cloud }} с возможностью их фильтрации, трансформации и маршрутизации.

### {{ api-gw-name }} {#api-gateway}

Создавайте API-шлюзы, которые поддерживают [спецификацию OpenAPI 3.0](https://github.com/OAI/OpenAPI-Specification) и набор расширений для взаимодействия с сервисами {{ yandex-cloud }}. Подробнее см. [документацию {{ api-gw-name }}](../../api-gateway/).
