Посмотрите информацию об устройствах хранения (дисках и созданных на них разделах) сервера:

{% include [fdisk-l-in-rescue](../fdisk-l-in-rescue.md) %}

В примере выше утилита `fdisk` вывела информацию о физических дисках (`/dev/sda` и `/dev/sdb`) и их разделах, а также о разделах в созданном на сервере RAID-массиве (`/dev/md127`, `/dev/md126` и `/dev/md125`).

Диски `/dev/sda` и `/dev/sdb` используются в RAID-массиве, корневая файловая система ОС сервера расположена в разделе `/dev/md125` размером `809.88 GiB`. Этот раздел и необходимо смонтировать.