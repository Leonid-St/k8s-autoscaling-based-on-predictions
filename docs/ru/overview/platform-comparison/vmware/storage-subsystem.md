---
title: Подсистема хранения данных (Storage)
description: В статье сопоставляются модели отказоустойчивости, аварийного восстановления и резервного копирования VMware Сloud Director и {{ yandex-cloud }}. Также вы ознакомитесь с подходами к организации хранения данных.
---

# Подсистема хранения данных (Storage)

## Модели отказоустойчивости и аварийного восстановления {#disaster-recovery-models}

Системы высокой доступности предназначены для защиты от единичных предсказуемых сбоев. Предсказуемыми компонентами могут быть компоненты оборудования: источники питания, процессоры или память, а также ошибки операционной системы и сбои приложений.

Системы обеспечения высокой доступности (High Availability; HA) осуществляют мониторинг компонентов, обнаружение отказов, перезапуск ресурсов. Механизмы High Availability применяются автоматически, без участия человека, однако требуют дополнительных резервных компонентов, на которых будет восстановлено приложение в случае отказа. Так, при отказе одного из серверов виртуализации размещенные на нем ВМ будут перезапущены на другом аналогичном оборудовании. При этом время недоступности будет приближаться ко времени перезагрузки ВМ.

Механизм HA подразумевает наличие общего хранилища данных, пула вычислительных ресурсов и «оркестратора», который определяет, на каком из узлов кластера должна быть запущена ВМ.

Аварийное восстановление (Disaster Recovery; DR) определяется процессами и людьми, которые необходимы для выполнения процедур восстановления после некоего катастрофического сбоя, событием которого может быть отключение электроэнергии, пожар в ЦОД или кибератака.

DR выходит за рамки High Availability и не зависит от центра обработки данных или другого системного уровня, а также подразумевает размещение нагрузки в нескольких географически распределенных местах, например в резервном ЦОД.

Важно то, что DR подразумевает независимость всех компонентов, чтобы избежать невозможности выполнения аварийного восстановления в случаях системной или человеческой ошибки. Таким образом, растянутый между площадками кластер или аналогичная реализация технологии не являются аварийным восстановлением — чаще это можно рассматривать как решение обеспечения непрерывности бизнеса.

vCloud Director и {{ yandex-cloud }} обеспечивают высокий уровень отказоустойчивости и схожи по своей архитектуре, однако в {{ yandex-cloud }} для использования георезервирования не требуются дополнительные модули или настройки, как реализовано в VMware.

## Сравнение моделей отказоустойчивости, аварийного восстановления и резервного копирования  {#model-comparison}

### Модели отказоустойчивости {#fault-tolerance-models}

#|
|| **VMware Cloud Director** | **{{ yandex-cloud }}** ||
|| В VMware в качестве общего хранилища для кластера выступает сущность Datastore, обычно расположенная в сети хранения данных или в общем файловом хранилище, доступ к которому имеют все узлы кластера (гипервизоры ESXi под управлением планировщика vSphere vCenter).

При выходе из строя узла запущенная на нем виртуальная машина сразу же перезапускается на другом узле без затрат времени на перенос данных, так как размещена в хранилище, доступ к которому имеют все узлы кластера.

Таким образом, группа гипервизоров ESXi не является high-availability-решением, так как в данном случае нет оркестратора, который определяет, на каком из узлов запустить нагрузку в автоматическом режиме. vCenter обеспечивает функцию динамической балансировки нагрузки между узлами кластера. 
| Сетевое и серверное оборудование платформы {{ yandex-cloud }} размещается в трех [зонах доступности](../../concepts/geo-scope.md). На них работает базовая инфраструктура платформы: виртуальные сети и машины, сетевые блочные хранилища.

Каждая зона доступности — это инфраструктура, которая является независимой от остальных зон для всех базовых компонентов: сети, ресурсов хранения и вычислительных ресурсов.

Высокая доступность сервиса {{ compute-name }} и построенных на его базе сервисов в рамках зоны обеспечивается схожими механизмами планировщика, который определяет, на каком из тысяч облачных хостов запустить нагрузку или как ее распределить в случае утери одного из хостов из-за сбоя. Более детально о работе планировщика рассказано в [видео](https://www.youtube.com/watch?v=ynkFYOFHn_Q&feature=youtu.be).

Кроме того, доступны механизмы повышения доступности вычислительных ресурсов внутри зоны доступности. Например, [группы размещения](../../../compute/concepts/placement-groups.md) позволяют распределять ВМ таким образом, чтобы каждая из них гарантированно была расположена на отдельной серверной стойке в одной из зон доступности. Если одна из стоек выйдет из строя, другие продолжат работу в обычном режиме.

Диски виртуальных машин создаются и работают в подсистеме Network Block Storage (NBS), которая обеспечивает работу распределенной системы хранения данных (СХД) внутри зоны доступности. В основе сервиса NBS лежит технология Yandex Database ({{ ydb-short-name }}), позволяющая хранить метаданные блоков клиентской нагрузки всех пользователей. Эта распределенная транзакционная СУБД обеспечивает нужную производительность для работы сервиса. На текущий момент NBS предоставляет диски, сравнимые по производительности с корпоративными СХД. Более подробно о доступных типах дисков и их производительности можно почитать в [документации](../../../compute/concepts/limits#compute-limits-vm-disks.md). ||
|#

### Аварийное восстановление (Disaster Recovery) {#disaster-recovery}

#|
|| **VMware Cloud Director** | **{{ yandex-cloud }}** ||
|| В качестве решений Disaster Recovery на платформе VMware чаще всего рассматривают независимый кластер vSphere, использующий отдельные ресурсы хранения для datastore. В этом сценарии резервный ЦОД является полностью независимым и не получает информации о нагрузке и состоянии основного ЦОД.

Синхронизация данных между datastore обеспечивается на уровне систем хранения и обычно происходит в асинхронном режиме. Оркестратором, который определяет логику использования основной или резервной площадки, могут выступать дополнительные решения, интегрированные с платформой vCenter.

Чаще всего для резервного копирования применяются интегрированные с облачным оркестратором vCloud Director решения Veeam DR или vCloud Availability.

В качестве решений непрерывности бизнеса обычно рассматривается «растянутый» (stretched) между площадками метро-кластер с синхронной репликацией данных между площадками на уровне СХД. Синхронная репликация данных подразумевает ряд ограничений, в том числе географического распределения между сайтами, так как с возрастанием расстояний возрастают и задержки отклика СХД.

Решения VMware для обеспечения катастрофоустойчивости обычно применяются для близко расположенных друг от друга площадок, что позволяет обеспечить синхронную репликацию данных СХД с приемлемыми задержками порядка 10 мс RTT (метро-кластер). По этой причине такие решения используются нечасто, так как целесообразнее переложить логику обработки отказа площадки на приложение при cloud-native-подходе. Микросервисная архитектура и контейнерные технологии, такие как Kubernetes, как раз решают эту задачу, что делает решения на базе метро-кластеров с синхронной репликацией данных неактуальными для подавляющего большинства нагрузок. 
| DR платформы {{ yandex-cloud }} можно рассматривать как доступное «из коробки» решение для большинства облачных сервисов. Зоны доступности не имеют общих ресурсов, при этом методы репликации данных зависят от используемых облачных сервисов.

Для сервиса {{ compute-name }} есть возможность создать снимок или образ ВМ, который хранится в объектном хранилище и доступен во всех зонах доступности, чтобы развернуть из него ВМ.

Одним из способов надежного высокодоступного хранения данных в {{ yandex-cloud }} является [{{ objstorage-short-name }}](../../../storage/).

Данные в объектном хранилище {{ objstorage-short-name }} реплицируются между зонами доступности, тем самым позволяя сохранить как сами данные, так и возможность доступа к ним даже при отказе одной из зон доступности.

Доступ к данным в {{ objstorage-short-name }} обеспечивается с помощью AWS S3 совместимого API, что обеспечивает поддержку широкого спектра приложений и клиентов для доступа к данным. Например, одно из решений — [GeeseFS](../../../storage/tools/geesefs.md) — позволяет «монтировать» объектное хранилище как файловую систему POSIX, подменяя таким образом блочное хранение, и создать кластерную файловую систему.

{{ objstorage-short-name }} поддерживает защищенную передачу данных между клиентом и сервисом по протоколу Transport Layer Security (TLS).

Поддержка версионирования, шифрования, списков контроля доступов и блокировки объектов также позволяет рассматривать это хранилище для резервного копирования и архивирования данных. ||
|#

### Резервное копирование {#backup}

#|
|| **VMware Cloud Director** | **{{ yandex-cloud }}** ||
|| В VMware снимки дисков позволяют перенести их на другую инфраструктуру и использовать для создания ВМ. Подобные решения работают на блочном уровне и не гарантируют целостности данных приложений. Гранулярное восстановление отдельных файлов или элементов приложений, таких как СУБД, не гарантируется.

Обеспечение целостности и управление жизненным циклом резервных копий берут на себя специализированные СРК, часто работающие «поверх» технологий, предоставляемых системами виртуализации, такими как снимки.

Чаще всего для работы с файловой системой используется специализированное ПО (агент), которое в том числе управляет работой специализированных приложений и обеспечивает корректную запись данных на диск.

Однако возможны и сценарии «безагентского» резервного копирования, что подразумевает установку ПО РК на хост гипервизора. При этом в гостевой ОС все равно установлен агент системы виртуализации (guest tools), позволяющий использовать механизм quiesce.

Производители систем резервного копирования могут использовать оба подхода.

VMware не рекомендует использовать снимки дисков в чистом виде как замену резервным копиям, так как снимок ВМ связан с диском самой ВМ и не обеспечивает целостности данных. 
| {{ yandex-cloud }} предоставляет возможности для надежного хранения данных сервисов:

  * {{ objstorage-short-name }} — репликация данных, версионирование, object lock.

  * Группа сервисов управляемых баз данных (Managed Databases) — механизмы резервного копирования, нативные для каждой из СУБД. Например, для СУБД [PostgreSQL](../../../managed-postgresql/concepts/backup.md).

  * {{ compute-name }} — сервис {{ backup-full-name }}.

[{{ backup-full-name }}](https://yandex.cloud/ru/services/backup) работает на базе агентской технологии резервного копирования. Доступно копирование и восстановление [виртуальных машин {{ compute-name }}](../../../compute/concepts/vm.md) с [поддерживаемыми операционными системами](../../../backup/concepts/vm-connection.md#os).

[Агент {{ backup-name }}](../../../backup/concepts/agent.md) может быть установлен в гостевую ОС ВМ при создании или после ее развертывания в ручном режиме.

Управляется с помощью [политик резервного копирования](../../../backup/concepts/backup.md#types), которые определяют расписание и тип РК — полный или инкрементальный.

Резервные копии хранятся в {{ objstorage-short-name }}, что обеспечивает высокую надежность хранения данных.

При создании резервных копий для работы агентов РК могут использоваться различные механизмы на уровне операционной системы, такие как VSS или другие, встроенные в ядро или драйвера для ОС Linux. ||
|#

## Подходы к организации хранения данных {#organization-approaches}

### Платформа VMware vСloud Director {#vmware-organization}

Диски виртуальных машин на платформе VMware представлены в виде файлов дисков в различных форматах (VMDK, VHD, RAW) и могут храниться в любых совместимых хранилищах, в качестве которых могут использоваться локальные диски, сети хранения, файловые хранилища и программно определяемая СХД VSAN.

При этом узлы кластера vCenter должны иметь общее хранилище данных (datastore) для обеспечения высокой доступности. Надежность и производительность datastore определяется используемым нижележащим оборудованием СХД. Современное оборудование СХД имеет интеграцию с платформой VMware, что позволяет эффективно предоставлять необходимые ресурсы хранения платформе виртуализации.

Доступные технологии повышения надежности и оптимизации хранения зависят от класса оборудования и могут включать:

* технологии надежного хранения RAID erasure coding, репликацию данных;

* средства оптимизации хранения, такие как tiering: автоматизированное размещение «горячих» данных на быстрых и дорогих носителях и «холодных» данных на медленных и дешевых носителях;

* дедупликацию и сжатие данных, которые позволяют сократить объем повторяющихся данных дисков ВМ, тем самым кратно снизить объемы хранения. Однако следует помнить, что использование этих технологий может влиять на производительность дисковой подсистемы;

* шифрование дисков.

Пользователям платформы vСloud Director доступен выбор уровней хранения, соответствующих преднастроенным конфигурациям datastore. У разных производителей СХД есть собственные методы по ограничению влияния «шумных соседей». Эти методы, как правило, скрыты от пользователей платформы.

Для решения проблемы «шумных соседей» обычно используются различные механизмы гарантий минимальной производительности для отдельных datastore, например storage QoS. В случае если один из клиентов облака VMware сильно утилизирует дисковую подсистему, его диски с данными могут быть перенесены в отдельный datastore, представленный выделенным контроллером и дисками СХД.

При таком переносе данных используется механизм [storage DRS](https://docs.vmware.com/en/VMware-vSphere/7.0/com.vmware.vsphere.resmgmt.doc/GUID-47C8982E-D341-4598-AC71-2CF2ABB644C0.html), который выполняет копирование данных в новый datastore. Эту задачу провайдеры могут решать в ручном или полуавтоматическом режиме. Миграция дисков ВМ занимает некоторое время, что может влиять на производительность работы текущего datastore.

В некоторых случаях пользователям необходимо предоставить отдельный LUN в СХД или локальный диск напрямую в ВМ (raw mapping), не размещая данные в файловой системе datastore. Помимо потенциального повышения производительности таких дисков, у них есть и ограничения:

* Отсутствие кластерной файловой системы не позволяет использовать такие диски несколькими ВМ одновременно.
* ВМ с raw-mapped-диском привязана к хосту, которому предоставлен такой диск, и не может быть мигрирована на другой хост в автоматическом режиме.

### Снимки дисков VMware {#vmware-disk-snapshots}

Снимки дисков VMware не обеспечивают полную копию данных, а лишь перенаправляют новую запись в отдельный файл (COW), который позволяет не затирать исходные данные и в случае необходимости к ним откатиться. По этой причине копирование происходит практически «мгновенно» и сразу после создания снимок не занимает место в СХД. Для работы с СХД VMWare предоставляет специализированное API (VAAI), позволяющее поддержать снимки дисков на аппаратном уровне. Многие вендоры СХД поддерживают интерфейс VAAI.

Технология создания снимков VMware получила широкое распространение и часто именно эта реализация снимков подразумевается клиентами. Производители систем резервного копирования (СРК) используют возможности интеграции с VMware для создания резервных копий (бэкапов). Тем не менее стоит помнить и об ограничениях снимков VMware:

* нет поддержки RDM;

* консистентность данных обеспечивается агентом, который должен контролировать запись на диск из ОС ВМ (quiesces);

* сама по себе технология не является заменой резервным копиям, хотя и используется производителями СРК для выполнения резервного копирования;

* большое количество снимков может влиять на производительность ВМ.

### VMware vSAN {#vsan}

VMware также предлагает отдельный продукт — [vSAN](https://www.vmware.com/products/cloud-infrastructure/vsan), который представляет собой распределенную программную СХД на базе commodity или гиперконвергентного серверного оборудования. Это отдельный продукт VMware, требующий лицензирования и квалифицированной поддержки.

vSAN предоставляет схожие с Enterprise СХД возможности на программном уровне, в том числе синхронную репликацию данных для построения метро-кластера.

### Диски в {{ yandex-cloud }} на базе NBS {#yc-nbs}

{{ yandex-cloud }} использует программную СХД — сервис Network Block Storage (NBS). Эту СХД концептуально можно сравнить с vSAN. В каждой зоне доступности есть свои ресурсы сервиса NBS, что позволяет поддерживать высокую доступность (shared storage) в рамках одной зоны доступности, при этом гарантируя низкие задержки, достаточно высокую производительность и возможность аварийного восстановления данных при отказе одной из зон доступности. Для восстановления данных необходимо использование механизма релокации дисков сервиса {{ compute-name }}, сервиса {{ objstorage-name }}.

Виртуальный диск ВМ состоит из «блоков размещения», каждый из которых представляет собой область на физических дисках. Данные блоков размещения защищены механизмами репликации или erasure coding (кроме дисков типа NRD). Для каждого «блока размещения» каждого типа дисков определен лимит производительности, который искусственно ограничен механизмом [троттлинга](../../../compute/concepts/storage-read-write.md#throttling) со стороны сервиса NBS. Параметры троттлинга определяются типом дисков и их размером (чем больше блоков размещения выделено диску, тем выше его производительность). Подробности об уровне производительности дисков можно прочесть в [документации](../../../compute/concepts/limits.md#compute-limits-disks).

Управление уровнем производительности дисков позволяет исключить появление «шумных соседей» в случаях, когда нагрузка на диск ВМ одного клиента влияет на производительно диска ВМ другого клиента (при размещении на одном хосте).

#### Возможности дисков NBS {#ability-nbs}

1. [Шифрование дисков](../../../compute/concepts/encryption.md)
   
   Поддерживается возможность шифровать диски ВМ ключами шифрования под управлением пользователя. По умолчанию все данные на дисках HDD и SSD шифруются на уровне базы данных хранилища с помощью системного ключа. Также пользователи облака могут дополнительно шифровать диски пользовательскими ключами. Это позволяет контролировать доступ к зашифрованным данным: создавать ключи под конкретного пользователя или конкретную задачу, своевременно деактивировать или удалять конкретные ключи.

1. [Local-SSD](../../../compute/concepts/dedicated-host.md#resource-disks)

   Локальные диски на выделенных хостах предоставляются напрямую виртуальным машинам, размещенным на этом хосте. И вся производительность таких дисков предоставляется без утилизации виртуальной сети. Локальный диск не обеспечивает аппаратной отказоустойчивости, что возлагает задачу обеспечения сохранности данных на пользователя, например, за счет использования программного зеркалирования данных (mdadm) или копирования данных на уровне приложений.

1. [Снимки дисков](../../../compute/concepts/snapshot.md)

   Снимок диска позволяет создать копию данных диска. После создания снимки дисков хранятся в {{ objstorage-name }}, что делает их доступными в любой зоне. Снимок может быть использован независимо — например, как загрузочный образ другой ВМ.

1. Импорт дисков

   Пользователи облака могут импортировать диски из других систем виртуализации и создавать из них ВМ в {{ yandex-cloud }}. Поддерживаются форматы vmdk, qcow, vhd. При импорте дисков происходит их конвертация в формат, совместимый с {{ yandex-cloud }}, — qcow2.

1. Файловые хранилища

   Это виртуальная файловая система, которую можно подключать к нескольким [виртуальным машинам](../../../compute/concepts/vm.md) {{ compute-name }} из одной зоны доступности. С файлами в хранилище можно работать совместно, с нескольких ВМ. Каждое файловое хранилище находится в одной из зон доступности и реплицируется внутри нее, обеспечивая сохранность данных. Между зонами файловые хранилища не реплицируются.

Хранилище подключается к ВМ через интерфейс [Filesystem in Userspace](https://ru.wikipedia.org/wiki/FUSE_(модуль_ядра)) (FUSE) как устройство [virtiofs](https://www.kernel.org/doc/html/latest/filesystems/virtiofs.html), не связанное напрямую с файловой системой хоста.