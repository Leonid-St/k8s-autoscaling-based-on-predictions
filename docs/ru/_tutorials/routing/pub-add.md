## Организация публичного соединения {#pub-create}

Чтобы организовать новое публичное соединение в уже созданном транке, создайте [новое обращение в поддержку]({{ link-console-support }}).

### Обращение в поддержку для организации публичное соединения {#pub-ticket}

Обращение должно быть оформлено следующим образом:
```s
Тема: [CIC] Добавить публичное соединение в уже существующий транк.

Текст обращения:
Прошу добавить публичное соединение в уже существующий транк.
Параметры соединения:

trunk_id: euus5dfgchu2********
vlan_id: 101
ipv4_peering:
  peer_bgp_asn: 65001
  #cloud_bgp_asn: {{ cic-bgp-asn }}
allowed-public-services:
  - {{ s3-storage-host }}
  - transcribe.{{ api-host }}
is_nat_extra_ip_required: false
```

где:

* `trunk_id` — идентификатор транка, полученный от поддержки на предыдущем этапе.
* `vlan_id` — идентификатор `VLAN-ID` для данного публичного соединения в 802.1Q транке. Выбирается клиентом. Не может совпадать со значениями `VLAN-ID` ранее настроенных приватных соединений в данном транке.
* `peer_bgp_asn` — номер [BGP ASN](../../interconnect/concepts/priv-con.md#bgp-asn) на оборудовании клиента в формате ASPlain. Выбирается клиентом.
* `allowed-public-services` — список `FQDN API Endpoint` для сервисов [из таблицы](../../interconnect/concepts/pub-con.md#svc-list), к которым нужно предоставить доступ через данное публичное соединение.
* `is_nat_extra_ip_required` — нужно ли клиенту выделить дополнительный `/32` сервисный адрес (префикс) в дополнение к стыковой подсети `/31` для реализации [функций NAT](../../interconnect/concepts/pub-con.md#svc-nat). По умолчанию дополнительный сервисный префикс не выделяется — значение `false`.
* `folder_id` (опционально) — по умолчанию метрики мониторинга для публичного соединения будут сохраняться в каталог, который был указан при создании транкового подключения. При необходимости, можно явно указать каталог для сохранения метрик мониторинга для публичного соединения.

### Ответ поддержки по обращению клиента {#pub-ticket-resp}

По завершении выполнения запрошенных действий по организации публичного соединения поддержка сообщает клиенту идентификатор созданного соединения.

Пример ответа поддержки по обращению на создание публичного соединения (для информации):
```s
id: cf3qdug4fsf7********
ipv4_peering:
  peering_subnet: {{ cic-pbc-subnet }}
  peer_ip: {{ cic-pbc-subnet-client }}
  cloud_ip: {{ cic-pbc-subnet-cloud }}
  peer_bgp_asn: 65001
  #cloud_bgp_asn: {{ cic-bgp-asn }}
allowed-public-services:
  - {{ s3-storage-host }}
  - transcribe.{{ api-host }}
```
где:

* `id` — идентификатор созданного публичного соединения.
* `peering_subnet` — [стыковая подсеть](../../interconnect/concepts/pub-con.md#pub-address) для BGP-пиринга. Выделяется из [адресного пула](../../vpc/concepts/ips.md) {{ yandex-cloud }}.
* `peer_ip` — IP адрес из стыковой (пиринговой) подсети на оборудовании клиента. Назначается {{ yandex-cloud }}.
* `cloud_ip` — IP адрес из стыковой (пиринговой) подсети на оборудовании {{ yandex-cloud }}. Назначается {{ yandex-cloud }}.
* `nat_subnet` — дополнительная подсеть выделенная из адресного пространства {{ yandex-cloud }} для реализации [функций NAT](../../interconnect/concepts/pub-con.md#pub-nat).
* `allowed-public-services` — список `FQDN API Endpoints` из запроса клиента для сервисов, к которым был предоставлен доступ через созданное публичное соединение.

### Контроль состояния публичного соединения {#pub-check}

* Вы самостоятельно отслеживаете переход BGP-сессии публичного соединения на оборудовании {{ yandex-cloud }} в рабочее состояние с помощью сервиса [мониторинга](../../interconnect/concepts/monitoring.md#private-mon).
* Поддержка уведомит вас о завершении процесса конфигурации доступа к запрошенным сервисам {{ yandex-cloud }}. Процесс конфигурации обычно выполняется в течение одного рабочего дня.
* Вы самостоятельно проверяете IP-связность между своим оборудованием и сервисами {{ yandex-cloud }}, которые должны быть доступны через настроенное публичное соединение и сообщаете поддержке о результатах проверки.
* При возникновении проблем с IP-связностью свяжитесь с поддержкой для их диагностики и устранения.

