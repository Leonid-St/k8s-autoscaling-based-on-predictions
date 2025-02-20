# {{ backup-name }} agent

The {{ backup-name }} agent enables protected resources to work with {{ backup-name }} and the [backup provider](index.md#providers). It is installed on [{{ compute-full-name }}](../../compute/index.yaml) VMs or [{{ baremetal-full-name }}](../../baremetal/index.yaml) servers.

The {{ backup-name }} agent performs the following actions on the protected resource:

* Registers the protected resource in {{ backup-name }} under a [service account](vm-connection.md#sa).
* Reports the resource's connection status to {{ backup-name }}.
* Backs up the resource according to the [backup policy](policy.md).
* Recovers the resource from a backup.
* Disconnects the resource from {{ backup-name }}.

{% note info %}

To perform operations in {{ backup-name }}, including deleting backups, your [folder](../../resource-manager/concepts/resources-hierarchy.md#folder) must have at least one active protected resource (VM in `RUNNING` status or {{ baremetal-name }} server in `READY` status) with the {{ backup-name }} agent installed. If there are no such VMs or servers, [create](../operations/create-vm.md) a temporary new VM connected to {{ backup-name }} to take the actions you need.

{% endnote %}

You can install the {{ backup-name }} agent on a VM [manually](vm-connection.md#self-install) or create a VM from {{ marketplace-full-name }} [images](vm-connection.md#os) with a pre-installed {{ backup-name }} agent.

{{ baremetal-name }} servers only support manual installation of the {{ backup-name }} agent. For more information, see [Connecting a {{ baremetal-name }} server to {{ backup-name }}](../operations/backup-baremetal/backup-baremetal.md).


#### See also {#see-also}

* [{#T}](../operations/connect-vm-linux.md)
* [{#T}](../operations/connect-vm-oslogin-linux.md)
* [{#T}](../operations/connect-vm-windows.md)
* [{#T}](../operations/refresh-connection.md)
* [{#T}](../operations/update-backup-agent.md)
* [{#T}](../operations/refresh-connection-oslogin-linux.md)
* [{#T}](../operations/backup-baremetal/backup-baremetal.md)
* [{#T}](../operations/backup-baremetal/refresh-connection.md)