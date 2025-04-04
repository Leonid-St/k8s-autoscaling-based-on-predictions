---
title: Сеть в {{ baremetal-full-name }}
description: Из статьи вы узнаете про публичные и приватные сети в {{ baremetal-full-name }}.
---

# Сеть

## Публичная сеть {#public-network}

Сеть c доступом в интернет, к которой физически подключены все серверы. Сетевой трафик между приватной сетью и интернетом на [некоторых](./traffic-restrictions.md) TCP- и UDP-портах ограничен.

## Приватная сеть {#private-network}

Локальная сеть к которой подключены все серверы. Логически объединяет серверы в изолированные пользовательские сети.

### Приватная подсеть {#private-subnet}

Сеть, физически ограниченная сетевым оборудованием одного пула, изолированная как от интернета, так и от сетей других пользователей.

В рамках одного пула между арендованными серверами возможна L2-связность (VLAN) и L3-связность (VRF).  Между серверами, физически размещенными в разных пулах, доступна только L3-связность.

Чтобы настроить сетевое взаимодействие между серверами из разных [пулов](./servers.md), для соответствующих подсетей в блоке **Настройки для маршрутизации** необходимо выбрать одинаковый [VRF](#vrf-segment).

Для адресации в подсетях можно использовать любые CIDR в зарезервированных под частные сети диапазонах `10.0.0.0/8`, `172.16.0.0/12` или `192.168.0.0/16` при условии, что в подсети не менее восьми адресов (префикс CIDR не более `/29`).

### Виртуальный сегмент сети (VRF) {#vrf-segment}

Для обеспечения L3-маршрутизации приватные подсети, в которых настроена маршрутизация, объединяются в виртуальные сегменты сети (VRF).

Серверы из одного или разных пулов, подключенные к разным приватным подсетям, объединенным в VRF, смогут поддерживать между собой сетевой обмен по L3.

## Сервис DHCP в сети {{ baremetal-full-name }} {#dhcp}

Конфигурирование сетевых интерфейсов серверов в публичных и приватных сетях происходит с использованием DHCP.

Сетевому интерфейсу сервера, подключенному к публичной сети, выдается IPv4-адрес из общедоступного диапазона сетей интернет, если при заказе сервера в селекторе **Публичный адрес** было выбрано `Автоматически`.

Если на сервере выключен DHCP, то при конфигурировании сетевого интерфейса публичной сети необходимо учитывать следующее:
* Подсеть, в которой находится публичный IP-адрес сервера, имеет CIDR с префиксом `/31` и состоит из двух адресов: IP-адреса шлюза и IP-адреса хоста.
* IP-адрес хоста указан в поле **Публичный IPv4-адрес** на странице с информацией о сервере.
* IP-адрес шлюза на единицу меньше, чем IP-адрес хоста.

Например:
* Публичный IPv4-адрес сервера — `198.51.100.111`.
* CIDR подсети — `198.51.100.110/31`.
* IP-адрес шлюза — `198.51.100.110`.

Сетевому интерфейсу сервера, подключенному к приватной сети, выдается IP-адрес из CIDR, указанного в настройках для маршрутизации приватной подсети, к которой подключен сервер.

Если на сервер установлена операционная системы из образов {{ marketplace-short-name }}, на всех физических интерфейсах сервера получение IP-адресов по DHCP включено по умолчанию.

В рамках приватной сети DHCP-сервер активируется автоматически, как только будут указаны настройки маршрутизации в параметрах этой подсети. До того, как будут определены настройки для маршрутизации, частная подсеть функционирует как сеть L2, и для ее серверов необходимо вручную назначать IP-адреса.

#### См. также {#see-also}

* [{#T}](./traffic-restrictions.md)