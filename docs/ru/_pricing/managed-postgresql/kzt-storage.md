| Услуга                                  | Цена за ГБ в месяц,<br>вкл. НДС                                     |
| ----- | ----: |
| Хранилище на сетевых HDD-дисках         | {{ sku|KZT|mdb.cluster.network-hdd.pg|month|string }}               |
| Хранилище на нереплицируемых SSD-дисках | {{ sku|KZT|mdb.cluster.network-ssd-nonreplicated.pg|month|string }} |
| Хранилище на сетевых SSD-дисках         | {{ sku|KZT|mdb.cluster.network-nvme.pg|month|string }}              |
| Сверхбыстрое сетевое хранилище с тремя репликами (SSD) | {{ sku|KZT|mdb.cluster.network-ssd-io-m3.pg|month|string }} |
| Хранилище на локальных SSD-дисках       | {{ sku|KZT|mdb.cluster.local-nvme.pg|month|string }}                |
| Резервные копии сверх размера хранилища | {{ sku|KZT|mdb.cluster.pg.backup|month|string }}                    |