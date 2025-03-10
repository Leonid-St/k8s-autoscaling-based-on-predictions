---
title: Elasticsearch disk types
description: Elasticsearch allows you to use network and local storage drives for clusters. Network drives are based on network blocks, which are virtual disks in the {{ yandex-cloud }} infrastructure.
keywords:
  - Elasticsearch storage
  - Elasticsearch storage types
  - Elasticsearch
---

# {{ ES }} disk types

{% include [Elasticsearch-end-of-service](../../_includes/mdb/mes/note-end-of-service.md) %}


{{ mes-name }} allows you to use network and local storage drives for database clusters. Network drives are based on network blocks, which are virtual disks in the {{ yandex-cloud }} infrastructure. Local disks are physically located on the cluster servers.

{% include [storage-type-nrd](../../_includes/mdb/mes/storage-type.md) %}

## Selecting disk type when creating a cluster  {#storage-type-selection}

The number of hosts with the _Data node_ role that can be created along with an {{ ES }} cluster depends on the selected disk type:

* With local SSDs (`local-ssd`) or non-replicated SSDs (`network-ssd-nonreplicated`), you can create a cluster with three or more hosts.

  This cluster will be fault-tolerant. To ensure fault tolerance, you can also set up index [sharding and replication](scalability-and-resilience.md).

  Local SSD storage has an effect on how much a cluster will cost: you pay for it even if it is stopped. For more information, see [Pricing policy](../pricing.md).

* With network HDD (`network-hdd`) or network SSD (`network-ssd`) storage, you can add any number of hosts within the current quota.

For more information about limits on the number of hosts per cluster, see [Quotas and limits](./limits.md).

