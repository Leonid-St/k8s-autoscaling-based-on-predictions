---
title: Правила тарификации для {{ managed-k8s-full-name }}
description: В статье содержатся правила тарификации сервиса {{ managed-k8s-name }}.
editable: false
---


# Правила тарификации для {{ managed-k8s-name }}

{% note tip %}


Чтобы рассчитать стоимость использования сервиса, воспользуйтесь [калькулятором](https://yandex.cloud/ru/prices?state=816ab5d70fb9#calculator) на сайте {{ yandex-cloud }} или ознакомьтесь с тарифами в этом разделе.





{% endnote %}

{% include [link-to-price-list](../_includes/pricing/link-to-price-list.md) %}

В рамках сервиса {{ managed-k8s-name }} тарифицируется использование [мастера](concepts/index.md#master) и исходящий трафик.

Узлы тарифицируются по [правилам тарификации {{ compute-full-name }}](../compute/pricing.md).



## Цены для региона Россия {#prices}



{% include [pricing-diff-regions](../_includes/pricing-diff-regions.md) %}



### Мастер {#master}

Цены за месяц использования формируются из расчета 720 часов в месяц.


{% list tabs group=pricing %}

- Цены в рублях {#prices-rub}

  {% include [rub.md](../_pricing/managed-kubernetes/rub-master.md) %}

- Цены в тенге {#prices-kzt}

  {% include [kzt.md](../_pricing/managed-kubernetes/kzt-master.md) %}

{% endlist %}




{% include [egress-traffic-pricing](../_includes/egress-traffic-pricing.md) %}