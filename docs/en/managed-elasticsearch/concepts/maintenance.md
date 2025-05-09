---
title: Maintenance in {{ mes-name }}
description: Maintenance means automatic installation of {{ ES }} updates and fixes for hosts (including for disabled clusters), changing host class and storage size, and other maintenance activities.
---

# Maintenance in {{ mes-name }}

{% include [Elasticsearch-end-of-service](../../_includes/mdb/mes/note-end-of-service.md) %}

Maintenance means:

* Automatic installation of {{ ES }} updates and revisions for hosts (including disabled clusters).
* Other maintenance activities.

Maintenance includes changes within one {{ ES }} major version. For information about migrating between major versions, see [{#T}](../operations/cluster-version-update.md).

## Maintenance window {#maintenance-window}

You can set the preferred maintenance time when [creating a cluster](../operations/cluster-create.md) or [updating its settings](../operations/cluster-update.md):

{% include [Maintenance window](../../_includes/mdb/maintenance-window.md) %}

## Maintenance procedure {#maintenance-order}

In single-host {{ mes-name }} clusters, a single host with the [_Data node_ role](./hosts-roles.md#data-node) undergoes maintenance. This means, if a cluster needs to be restarted during maintenance, it will become unavailable.

In multi-host clusters, hosts with the _Data node_ role undergo maintenance one by one. The hosts are queued randomly. If a host needs to be restarted during maintenance, it becomes unavailable while being restarted. If you access a cluster using the [FQDN of the {{ ES }} host](../operations/cluster-connect.md#fqdn), the cluster may become unavailable. To make your application continuously available, access the cluster using a [special FQDN](../operations/cluster-connect.md#automatic-host-selection) always pointing to the available host.
