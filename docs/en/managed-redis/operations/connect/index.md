---
title: Setting up a connection in {{ mrd-full-name }}
description: Follow this guide to connect to a {{ VLK }} cluster.
---

# Setting up a connection

Available connection methods depend on whether the cluster [sharding](../../concepts/sharding.md) is enabled:

* [Connecting to a non-sharded cluster](./non-sharded.md).
* [Connecting to a sharded cluster](./sharded.md).

## Accessing cluster hosts {#connect}

You can connect to {{ mrd-name }} cluster hosts:

* Via the internet if the following conditions are met:

    * [Public access to hosts](../hosts.md#public-access) is configured.
    * An SSL connection is used.
    * Your cluster was created with TLS support.

* From {{ yandex-cloud }} virtual machines located in the same cloud network.

    
    1. [Create a virtual machine](../../../compute/operations/vm-create/create-linux-vm.md) with a public IP in the same virtual network as the cluster.
    1. [Connect](../../../compute/operations/vm-connect/ssh.md) to the created VM via SSH.
    1. From this VM, connect to {{ VLK }} using one of the sample connection strings.



## Encryption support {#tls-support}

Encrypted SSL connections are supported for {{ mrd-short-name }} clusters. To use SSL, enable **{{ ui-key.yacloud.redis.field_tls-support }}** when [creating a cluster](../cluster-create.md).

By default, {{ VLK }} uses host IP addresses, not their [FQDNs](../../concepts/network.md#hostname). This may [prevent connection to {{ VLK }} hosts](../../concepts/network.md#fqdn-ip-setting) in clusters with TLS support. To be able to connect to hosts, do one of the following:

* Enable the use of FQDNs instead of IP addresses to replace a host's IP address with its FQDN. You can enable this setting when [creating](../cluster-create.md) or [updating](../update.md#configure-fqdn-ip-behavior) a cluster.

    This will allow the [{{ VLK }} client](../../concepts/supported-clients.md) to connect to {{ VLK }} hosts both from {{ yandex-cloud }} VMs and over the internet, as well as request verification of the host's FQDN against the certificate, if required.

    {% include [fqdn-option-compatibility-note](../../../_includes/mdb/mrd/connect/fqdn-option-compatibility-note.md) %}

* Disable verification of the host's FQDN against the certificate on the {{ VLK }} client side.

    This will enable you to connect to {{ VLK }} hosts from {{ yandex-cloud }} VMs.


## Configuring security groups {#configuring-security-groups}

{% include [Security groups notice](../../../_includes/mdb/sg-rules-connect.md) %}

{% include [Security groups rules for VM](../../../_includes/mdb/mrd/connect/sg-rules-for-vm.md) %}

Security group settings for sharded and non-sharded clusters differ.

{% list tabs group=cluster %}

- Non-sharded cluster {#non-sharded}

    [Configure all the cluster security groups](../../../vpc/operations/security-group-add-rule.md) to allow incoming traffic from the security group where the VM is located on port `{{ port-mrd }}` for direct connections to the master host or `{{ port-mrd-sentinel }}` for connections via Sentinel. If you created your cluster with SSL encryption support, specify port `{{ port-mrd-tls }}` for direct encrypted connections to the master or `{{ port-mrd-sentinel }}` for unencrypted connections using Sentinel.

    {% note warning %}

    Connecting to port `{{ port-mrd-sentinel }}` enables you to request cluster information without authenticating. To restrict unauthorized cluster access with host public access enabled, do not specify this port in your security group settings.

    {% endnote %}

    To do this, create the following rule for incoming traffic:

    * **{{ ui-key.yacloud.vpc.network.security-groups.forms.field_sg-rule-port-range }}**: create a separate rule for each port:

        * `{{ port-mrd }}`: For direct unencrypted host connections.
        * `{{ port-mrd-tls }}`: For direct host connections using SSL encryption.
        * `{{ port-mrd-sentinel }}`: For cluster communication via Sentinel.

            To connect to a cluster using Sentinel, you must also create a rule enabling connections via port `{{ port-mrd }}` or `{{ port-mrd-tls }}`.

    * **{{ ui-key.yacloud.vpc.network.security-groups.forms.field_sg-rule-protocol }}**: `{{ ui-key.yacloud.common.label_tcp }}`.
    * **{{ ui-key.yacloud.vpc.network.security-groups.forms.field_sg-rule-source }}**: `{{ ui-key.yacloud.vpc.network.security-groups.forms.value_sg-rule-destination-sg }}`.
    * **{{ ui-key.yacloud.vpc.network.security-groups.forms.field_sg-rule-sg-type }}**: Security group assigned to the VM. If it is the same as the configured group, specify **{{ ui-key.yacloud.vpc.network.security-groups.forms.value_sg-rule-sg-type-self }}**.

- Sharded cluster {#sharded}

    [Configure all the cluster security groups](../../../vpc/operations/security-group-add-rule.md) to allow incoming traffic on port `{{ port-mrd }}` from the security group where the VM is located. If a cluster is created with SSL encryption support, you should only specify port `{{ port-mrd-tls }}`.

    To do this, create the following rule for incoming traffic:

    * **{{ ui-key.yacloud.vpc.network.security-groups.forms.field_sg-rule-port-range }}**: `{{ port-mrd }}` or only `{{ port-mrd-tls }}` for clusters with SSL encryption support.
    * **{{ ui-key.yacloud.vpc.network.security-groups.forms.field_sg-rule-protocol }}**: `{{ ui-key.yacloud.common.label_tcp }}`.
    * **{{ ui-key.yacloud.vpc.network.security-groups.forms.field_sg-rule-source }}**: `{{ ui-key.yacloud.vpc.network.security-groups.forms.value_sg-rule-destination-sg }}`.
    * **{{ ui-key.yacloud.vpc.network.security-groups.forms.field_sg-rule-sg-type }}**: If your cluster and VM are in the same security group, select `{{ ui-key.yacloud.vpc.network.security-groups.forms.value_sg-rule-sg-type-self }}`. Otherwise, specify the VM security group.

{% endlist %}

{% note info %}

You can specify more detailed rules for your security groups, e.g., to allow traffic only in specific subnets.

You must configure security groups correctly for all subnets in which the cluster hosts will reside. If security group settings are incomplete or incorrect, you may lose access to the cluster if the master is switched [manually](../failover.md) or [automatically](../../concepts/replication.md#availability).

{% endnote %}

For more information about security groups, see [{#T}](../../concepts/network.md#security-groups).


## Getting an SSL certificate {#get-ssl-cert}

To use an encrypted SSL connection, get an SSL certificate:

{% include [install-certificate](../../../_includes/mdb/mrd/install-certificate.md) %}

{% include [ide-ssl-cert](../../../_includes/mdb/mdb-ide-ssl-cert.md) %}

## {{ VLK }} host FQDN {#fqdn}

To connect to a host, you need its fully qualified domain name ([FQDN](../../concepts/network.md#hostname)). You can obtain it in one of the following ways:

* [Request a list of cluster hosts](../hosts.md#list-hosts).
* In the [management console]({{ link-console-main }}), copy the command for connecting to the cluster. This command contains the host FQDN. To get the command, go to the cluster page and click **{{ ui-key.yacloud.mdb.clusters.button_action-connect }}**.
* Look up the FQDN in the management console:

   1. Go to the cluster page.
   1. Go to **{{ ui-key.yacloud.mdb.cluster.hosts.label_title }}**.
   1. Copy the **{{ ui-key.yacloud.mdb.cluster.hosts.host_column_name }}** column value.
