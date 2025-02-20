#### How do I connect to a cluster? {#how-to-connect}

View the connection examples in the [documentation](../../managed-mysql/operations/connect.md#connection-string) or on the cluster page in the [management console]({{ link-console-main }}) (click **{{ ui-key.yacloud.mdb.clusters.button_action-connect }}** in the top panel).

{{ MY }} hosts with public access only support connections with an [SSL certificate](../../managed-mysql/operations/connect.md#get-ssl-cert).

There are also special FQDNs pointing to the [current master](../../managed-mysql/operations/connect.md#fqdn-master) and the cluster's [least lagging replica](../../managed-mysql/operations/connect.md#fqdn-replica).

#### Why cannot I connect from the internet? {#from-internet}

Check whether there is public access to the host. To do this, in the [management console]({{ link-console-main }}):
1. Go to the folder page and select **{{ ui-key.yacloud.iam.folder.dashboard.label_managed-mysql }}**.
1. Click the name of the cluster you need and select the **{{ ui-key.yacloud.mysql.cluster.switch_hosts }}** tab.
1. Look at the value in the specific host's **{{ ui-key.yacloud.mdb.cluster.hosts.host_column_public-ip }}** column.

{{ MY }} hosts with public access only support connections with an [SSL certificate](../../managed-mysql/operations/connect.md#get-ssl-cert).

Additional information:
* If public access is only configured for certain hosts in your cluster, automatic master change may make the master unavailable over the internet.
* If you are using [special FQDNs](../../managed-mysql/operations/connect.md#fqdn-master), check the host list to see that the current master or replica have public access.
* If you are using [{#T}](../../managed-mysql/concepts/network.md#security-groups), check their [settings](../../managed-mysql/operations/connect.md#configure-security-groups).

#### Why cannot I connect from {{ yandex-cloud }}? {#from-yc}

Please check that you are connecting from a {{ yandex-cloud }} VM located on the same virtual network as the {{ MY }} cluster.

To do this, in the management console:
1. Go to the folder page, select **{{ ui-key.yacloud.iam.folder.dashboard.label_managed-mysql }}**, and click the name of the specific cluster.
1. Check the value of the **{{ ui-key.yacloud.mdb.cluster.overview.label_network }}** parameter and click the network name link to see its subnets.
1. Make sure the virtual machine is located on one of the network's subnets.

Additional information:
* If you are connecting to a host with public access, a connection can only be established with an [SSL certificate](../../managed-mysql/operations/connect.md#get-ssl-cert).
* If you are using [special FQDNs](../../managed-mysql/operations/connect.md#fqdn-master), check the host list to see that the current master or replica have public access.
* If you are using [{#T}](../../managed-mysql/concepts/network.md#security-groups), check their settings.

#### Why cannot I connect to a multi-host cluster? {#to-multi-host-cluster}

If public access is only configured for certain hosts in your cluster, automatic master change may make the master unavailable over the internet.

Check whether there is public access to the host. To do this, in the [management console]({{ link-console-main }}):
1. Go to the folder page and select **{{ ui-key.yacloud.iam.folder.dashboard.label_managed-mysql }}**.
1. Click the name of the cluster you need and select the **{{ ui-key.yacloud.mysql.cluster.switch_hosts }}** tab.
1. Look at the value in the specific host's **{{ ui-key.yacloud.mdb.cluster.hosts.host_column_public-ip }}** column.

Additional information:

* If you are using [special FQDNs](../../managed-mysql/operations/connect.md#fqdn-master), check the host list to see that the current master or replica have public access.


* If you cannot connect to the host you added, check that the cluster [security group](../../managed-mysql/concepts/network.md#security-groups) is configured correctly for the host's subnet.


#### Can I connect to cluster hosts via SSH or get superuser permissions on hosts? {#connect-ssh}

{% include [connect-via-ssh](../../_includes/mdb/connect-via-ssh.md) %}

#### What should I do if I get the revocation check error when using PowerShell to obtain an SSL certificate? {#get-ssl-error}

Here is the full text of the error:

```text
curl: (35) schannel: next InitializeSecurityContext failed: Unknown error (0x80092012)
The revocation function was unable to check revocation for the certificate
```
This means, when connecting to the website, the service failed to check whether or not the website’s certificate is on the list of revoked certificates.

To fix this error:

* Make sure the corporate network settings do not block the check.
* Run the command with the `--ssl-no-revoke` parameter.

   ```powershell
   mkdir $HOME\.mysql; curl.exe --ssl-no-revoke -o $HOME\.mysql\root.crt {{ crt-web-path }}
   ```

#### Why would the connection limit be exceeded? {#connection-limit}

The maximum number of concurrent connections to a {{ mmy-short-name }} cluster host is set by the `max_connections` parameter, which by default is `<MB_of_RAM_per_host> ÷ 32`, but not less than 100.

For example, for a {{ s1-micro }} class host, the default `max_connections` parameter value is: 8,192 ÷ 32 = 256.

You can [edit](../../managed-mysql/operations/update.md#change-mysql-config) the **Max connections** value in the cluster settings.
