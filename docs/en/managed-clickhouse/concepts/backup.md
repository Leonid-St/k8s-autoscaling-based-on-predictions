---
title: '{{ CH }} backups'
description: '{{ mch-short-name }} supports automatic and manual database backups. Backups take up space in the storage allocated to the cluster. Backups are automatically created every day.'
keywords:
  - back up
  - backup
  - backups
  - '{{ CH }} backups'
  - backup {{ CH }}
  - '{{ CH }}'
---

# Backups in {{ mch-name }}

{{ mch-short-name }} supports automatic and manual database backups.

A backup is automatically created every day. You cannot disable automatic backups or change the retention period.

To restore a cluster from a backup, follow [this guide](../operations/cluster-backups.md#restore).

## Creating backups {#size}

You can create both automatic and manual backups. In both cases, the incremental method is used:

* When creating a new backup, [data chunks]({{ ch.docs }}/engines/table-engines/mergetree-family/mergetree/#mergetree-data-storage) are checked for uniqueness.
* If identical [data chunks]({{ ch.docs }}/engines/table-engines/mergetree-family/mergetree/#mergetree-data-storage) are already present in one of the existing backups and they are not older than {{ mch-dedup-retention }} days, they are not duplicated. For cold data in [hybrid storage](storage.md#hybrid-storage-features), this period is {{ mch-backup-retention }} days.

Backups are created separately for each individual cluster [shard](./sharding.md). They are also restored by individual shard. You can restore:

* One or more shard backups in an individual cluster.
* The entire cluster by specifying backups of all cluster shards.

Backup data is only stored for the `MergeTree` engine family. For other engines, backups only store table schemas. Learn more about engines in the [{{ CH }} documentation]({{ ch.docs }}/engines/table-engines/).

Backups are created based on a random replica host. This is why, if there is no cluster host data consistency, restoring clusters from backups does not guarantee complete data recovery. For example, this may occur in the following cases:

* [The tables are not replicated](replication.md#replicated-tables) on all shard hosts.
* The tables are not replicated and are only hosted on some of the shard hosts.

When [creating](../operations/cluster-create.md) or [updating](../operations/update.md#change-additional-settings) a cluster, you can set the time interval during which the backup will start. Default time: `22:00 - 23:00` UTC (Coordinated Universal Time).

Backups are only created on running clusters. If you are not using your {{ mch-short-name }} cluster 24/7, check the [settings of backup start time](../operations/update.md#change-additional-settings).

For more information about creating a backup manually, see [Managing backups](../operations/cluster-backups.md).

## Storing backups {#storage}

* Backups of [local](storage.md) and [network](storage.md) storage devices are stored in a separate {{ objstorage-name }} bucket and take up no space in the cluster storage. If there are N GB of free space in the cluster, the first N GB of backups are stored free of charge.

* Backups of cold data from [hybrid storage](storage.md#hybrid-storage-features) are stored in the same {{ objstorage-name }} bucket as the regular data. The cost of using Object Storage considers both the space used by the backups and the space used by the data itself.

    For more information, see [Pricing policy](../pricing.md#rules-storage).

* Backups of cold data from [hybrid storage](storage.md#hybrid-storage-features) contain only the increment, i.e., the history of changes to data chunks for the last {{ mch-backup-retention }} days. Backups of data that has not been modified are provided by {{ objstorage-name }}.

* Backups are stored as binary files and encrypted using [GPG](https://en.wikipedia.org/wiki/GNU_Privacy_Guard). Each cluster has its own encryption keys.

* The retention time for backups of an existing cluster depends on the method used to create such backups:

    * Automatic backups are stored for {{ mch-backup-retention }} days by default. When [creating](../operations/cluster-create.md) a cluster or [editing](../operations/update.md#change-additional-settings) its settings, you can specify a different retention period between 3 and 60 days.

    * Manually created backups are stored with no time limit.

* After you delete a cluster, all its backups are kept for {{ mch-backup-retention }} days.

* {% include [no-quotes-no-limits](../../_includes/mdb/backups/no-quotes-no-limits.md) %}

## Checking backup recovery {#capabilities}

To test how backup works, [restore a cluster from a backup](../operations/cluster-backups.md) and check the integrity of your data.

## Deleting a backup {#deletion}

You can delete only manual backups. To delete such a backup, [follow this guide](../operations/cluster-backups.md#delete-backup).

{% include [clickhouse-disclaimer](../../_includes/clickhouse-disclaimer.md) %}
