Если вы создаете кластер из нескольких хостов и не используете [{{ CK }}](../../../../managed-clickhouse/concepts/replication.md#ck), то для хостов {{ ZK }} действуют следующие правила:

* Если в [облачной сети](../../../../vpc/concepts/network.md) кластера есть подсети в каждой из [зон доступности](../../../../overview/concepts/geo-scope.md), а настройки хостов {{ ZK }} не заданы, то в каждую подсеть будет автоматически добавлено по одному такому хосту.

* Если подсети в сети кластера есть только в некоторых зонах доступности, то необходимо указать настройки хостов {{ ZK }} явно.
