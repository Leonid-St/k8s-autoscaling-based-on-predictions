#### Can I restore a cluster backup to a running {{ mpg-short-name }} instance in a different cloud network? {#diff-network}

You can only recover a cluster from a backup within a single cloud network.

To migrate data between cloud networks, follow the [migration guide](../../managed-postgresql/tutorials/replication-overview.md).

#### Can I change the retention period of automatic backups? {#backup-retain-days}

When [creating](../../managed-postgresql/operations/cluster-create.md) or [modifying](../../managed-postgresql/operations/update.md#change-additional-settings) a cluster, you can set the retention period for automatic backups.

#### Do I need the wal2json plugin if data is only replicated but not copied? {#wal2json}

Yes, you need the plugin even if you do not copy data.

#### Can I make a local dump of a database backup? {#backup-local-dump}

You cannot use the service to make a local dump of a DB backup from {{ yandex-cloud }} but you can use the [pg_dump utility]({{ pg.docs.org }}/current/app-pgdump.html).

#### How do I move a local {{ PG }} database dump to {{ yandex-cloud }}? {#dump-to-yc}

Follow the steps described in [Migrating databases](../../managed-postgresql/tutorials/data-migration.md).

#### Why does a data transfer through creating and restoring a logical dump fail with an error? {#backup-error}

[Restoring a logical dump](../../managed-postgresql/tutorials/data-migration.md#backup) may fail with one of the following errors:

* `ERROR:  role "<source_username>" does not exist`
* `ERROR:  must be member of role "<source_username>"`

The errors occur because the target cluster does not have the user (or the privileges of the user) who created the logical dump in the source cluster.

To resolve the errors:

1. In the target cluster, [add a user](../../managed-postgresql/operations/cluster-users.md#adduser) with access to the migrated database and the same name as the user who created the logical dump in the source cluster.
1. Use this user to [restore the logical dump](../../managed-postgresql/tutorials/data-migration.md#restore) or [grant their privileges](../../managed-postgresql/operations/grant.md#grant-privilege) to the user who is restoring the logical dump.

#### What should I do if I get the `replication slot already exists` error when performing logical replication? {#repl-slot-exists}

By default, when you [create a subscription](../../tutorials/dataplatform/postgresql-data-migration.md#create-publication-subscription), a replication slot is also created. The `replication slot already exists` error means that a replication slot already exists.

You can fix this error by doing one of the following:

1. Link your subscription to an existing replication slot. To do this, add the `create_slot = false` parameter to the request to create a subscription.
1. [Delete the existing replication slot](../../managed-postgresql/operations/replication-slots.md#delete) and try creating the subscription again.
