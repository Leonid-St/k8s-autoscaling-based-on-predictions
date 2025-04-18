---
title: Как выполнить диагностику и устранить проблемы производительности Spark-приложений в {{ dataproc-full-name }}
description: Следуя данной инструкции, вы сможете выполнить диагностику и устранить проблемы производительности Spark-приложений.
---

# Диагностика и устранение проблем производительности Spark-приложений

Если вы столкнулись с медленной работой Spark-приложений:

* [Проверьте их работу](#diagnostics), чтобы установить причину проблем с производительностью.
* Попробуйте воспользоваться одним из [решений распространенных проблем](#troubleshooting).

## Первичная диагностика производительности Spark-приложений {#diagnostics}

Если производительность Spark-приложения низкая, проведите первичную диагностику:

* [Проверьте очередь выполнения приложений](./spark-monitoring.md#queue) и убедитесь, что работа приложения не блокируется другими.
* [Посмотрите подробную информацию о приложении](./spark-monitoring.md#info) и проверьте состояние заданий, моменты фактического начала и завершения их выполнения на диаграмме **Event Timeline**.
* [Проверьте выделенные для приложения ресурсы](./spark-monitoring.md#resources):

    * Убедитесь, что приложению доступно достаточно исполнителей, и доступные исполнители не простаивают.
    * Убедитесь, что ресурсы в рамках одного исполнителя используются сбалансированно.

* [Проверьте планы SQL-запросов](./spark-monitoring.md#sql) и продолжительность выполнения отдельных операций.
* [Проверьте логи приложения](./spark-monitoring.md#logs) на наличие предупреждений о сбоях.

## Устранение распространенных проблем производительности {#troubleshooting}

### Сборка мусора занимает большую долю времени {#gc-time}

Если при [проверке выделенных для приложения ресурсов](./spark-monitoring.md#resources) вы выяснили, что время сборки мусора (**GC Time**) занимает большую долю в общем времени выполнения задач (**Task Time**):

{% include [gc-time-fix](../../_includes/data-processing/gc-time-fix.md) %}

### Множество исполнителей конкурируют за ресурсы CPU {#cpu-wars}

При размещении исполнителей планировщик YARN с настройками по умолчанию не учитывает доступные ресурсы CPU в узле. Поэтому задания, выполняющие интенсивные вычисления, могут замедляться.

Чтобы этого избежать, включите альтернативный алгоритм учета ресурсов при размещении исполнителей, установив следующее [свойство](../concepts/settings-list.md) на уровне кластера:

```text
capacity-scheduler:yarn.scheduler.capacity.resource-calculator=org.apache.hadoop.yarn.util.resource.DominantResourceCalculator
```

Подробнее о работе планировщика YARN см. в [документации Hadoop](https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/CapacityScheduler.html).

### При выполнении заданий возникают систематические ошибки heartbeat {#heartbeat-errors}

При выполнении заданий Spark исполнители регулярно отправляют драйверу специальные _heartbeat_-сообщения с информацией о своем состоянии и о прогрессе выполнения операций. Если драйвер некоторое время не получает heartbeat-сообщение от исполнителя, то считает этого исполнителя неработоспособным и запрашивает для него аварийное завершение у ресурсного менеджера YARN. При этом в логах драйвера фиксируется сообщение вида:

```text
23/02/23 20:22:09 WARN TaskSetManager: Lost task 28.0 in stage 13.0 (TID 242) 
        (rc1c-dataproc-*****.{{ dns-zone }} executor 5): ExecutorLostFailure 
        (executor 5 exited caused by one of the running tasks) 
        Reason: Executor heartbeat timed out after 138218 ms
```

Такие ошибки могут возникать из-за проблем сетевого взаимодействия в кластере, но на практике heartbeat-сообщения чаще всего теряются из-за того, что у исполнителя закончилась свободная оперативная память. При этом соответствующие ошибки (`java.lang.OutOfMemoryError`) могут не фиксироваться в логах задания из-за сбоя логирования, вызванного той же нехваткой памяти.

Если при выполнении заданий возникают систематические ошибки heartbeat, и нет других признаков сетевых ошибок, увеличьте количество оперативной памяти, выделяемой на одну параллельную операцию. Для этого измените в кластере [свойства компонентов](../concepts/settings-list.md):

* Уменьшите количество процессорных ядер на одного исполнителя в параметре `spark.executor.cores`.
* Увеличьте объем резервируемой оперативной памяти для каждого исполнителя в параметре `spark.executor.memory`.

Подробнее об этих параметрах см. в [документации Spark](https://spark.apache.org/docs/latest/configuration.html#available-properties).
