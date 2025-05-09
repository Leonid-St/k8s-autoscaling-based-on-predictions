# Pushing a Helm chart to a registry

You can push a [Helm chart](https://helm.sh/docs/topics/charts/) to a [registry](../../concepts/registry.md) in {{ cloud-registry-name }}. {{ cloud-registry-name }} stores Helm charts as [Docker images](../../concepts/docker-image.md).

{% note info %}

If you are using a Helm version lower than 3.7.1, re-upload the charts to the {{ cloud-registry-name }} registry when upgrading to a newer version.

{% endnote %}

To push a Helm chart:

{% list tabs group=instructions %}

- CLI {#cli}

  1. [Install](https://helm.sh/docs/intro/install/) the Helm client version 3.8.0 or higher.

     {% note info %}

     When installing Helm, environment variables are not updated automatically. To run `helm` commands, use the installation directory or manually add Helm to environment variables.

     {% endnote %}

  1. If you are using a Helm version below 3.8.0, enable [Open Container Initiative](https://opencontainers.org/) support in the Helm client:

     ```bash
     export HELM_EXPERIMENTAL_OCI=1
     ```

  1. Authenticate your Helm client in the {{ cloud-registry-name }} registry using one of the available methods.
     * With an OAuth token:
       1. If you do not have an OAuth token yet, get one by following [this link]({{ link-cloud-oauth }}).
       1. Run this command:

          ```bash
          helm registry login {{ cloud-registry }} -u oauth
          Password: <OAut_token>
          ```

     * With an IAM token:
       1. Get an [IAM token](../../../iam/operations/iam-token/create.md).
       1. Run this command:

          ```bash
          helm registry login {{ cloud-registry }} -u iam
          Password: <IAM_token>
          ```

     Result:

     ```text
     Login succeeded
     ```

  1. Create a Helm chart:
  
     ```bash
     helm create <Helm_chart_name>
     ```

     The name must meet the following requirements:

     {% include [name-format](../../../_includes/name-format.md) %}

     Result:

     ```text
     Creating <Helm_chart_name>
     ```

  1. Build a Helm chart to upload:

     ```bash
     helm package <Helm_chart_name>/. --version <Helm_chart_version>
     ```

     Result:

     ```text
     Successfully packaged chart and saved it to: <path>/<Helm_chart_name>-<version>.tgz
     ```

  1. Push the Helm chart to {{ cloud-registry-name }}:

     ```bash
     helm push <Helm_chart_name>-<version>.tgz oci://{{ cloud-registry }}/<registry_ID>
     ```

     Result:

     ```text
     Pushed: {{ cloud-registry }}/crp3h07fgv9b********/<Helm_chart_name>:<version>
     Digest: <SHA256...>
     ```

{% endlist %}

## Examples {#examples}

{% list tabs group=instructions %}

- CLI {#cli}

  1. Create a Helm chart:

     ```bash
     helm create my-chart
     ```

     Result:

     ```text
     Creating my-chart
     ```

  1. Build a Helm chart to upload:

     ```bash
     helm package my-chart/. --version 3.11.2
     ```

     Result:

     ```text
     Successfully packaged chart and saved it to: C:/my-chart-3.11.2.tgz
     ```

  1. Push the Helm chart to {{ cloud-registry-name }}:

     ```bash
     helm push my-chart-3.11.2.tgz oci://{{ cloud-registry}}/<registry_ID>
     ```

     Result:

     ```text
     Pushed: {{ cloud-registry }}/crp3h07fgv9b********/my-chart:3.11.2
     Digest: sha256:dc44a4e8b686b043b8a88f77ef9dcb998116fab422e8c892a2370da0********
     ```

{% endlist %}