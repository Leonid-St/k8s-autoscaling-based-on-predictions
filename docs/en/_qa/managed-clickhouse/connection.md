#### Can I connect to individual {{ CH }} hosts? {#connect-node}

Yes. You can connect to {{ CH }} cluster hosts:

* Using the [HTTPS interface]({{ ch.docs }}/interfaces/http/):
    * Via an encrypted SSL connection on port 8443.
    * Without encryption through port 8123.

* Using the [command-line client]({{ ch.docs }}/interfaces/cli/):
    * Via an encrypted SSL connection on port 9440.
    * Without encryption through port 9000.

SSH connections are not supported.

#### Why cannot I connect to a host from the internet? {#fail-connection}

Most likely, no public access is enabled for the cluster, so you can only connect to it from a VM in {{ yandex-cloud }}. You can only request public access when [creating a new host](../../managed-clickhouse/concepts/network.md#public-access-to-a-host) in your cluster.

#### How do I connect to a non-public host in {{ yandex-cloud }}? {#private-host}

[Connect](../../managed-clickhouse/operations/connect/clients.md) to a host from a VM in {{ yandex-cloud }} hosted in the same cloud network, or [add](../../managed-clickhouse/operations/hosts.md#add-host) a new cluster host with public access and connect to a non-public host through it.

#### Can I connect to a public cluster without SSL? {#without-ssl}

No. You can only connect to public hosts using an SSL connection. For more information, see the [documentation](../../managed-clickhouse/operations/connect/index.md).

#### Can I connect to cluster hosts via SSH or get superuser permissions on hosts? {#connect-ssh}

{% include [connect-via-ssh](../../_includes/mdb/connect-via-ssh.md) %}

#### What do I do if I get the revocation check error when using PowerShell to obtain an SSL certificate? {#get-ssl-error}

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
    mkdir -Force $HOME\.yandex; `
    curl.exe {{ crt-web-path-root }} `
      --ssl-no-revoke `
      --output $HOME\.yandex\{{ crt-local-file-root }}; `
    curl.exe {{ crt-web-path-int }} `
      --ssl-no-revoke `
      --output $HOME\.yandex\{{ crt-local-file-int }}; `
    Import-Certificate `
      -FilePath $HOME\.yandex\{{ crt-local-file-root }} `
      -CertStoreLocation cert:\CurrentUser\Root; `
    Import-Certificate `
      -FilePath $HOME\.yandex\{{ crt-local-file-int }} `
      -CertStoreLocation cert:\CurrentUser\Root
    ```
