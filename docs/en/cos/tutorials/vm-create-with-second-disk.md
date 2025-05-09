# Creating a VM from a {{ coi }} with an additional volume for a Docker container

In this tutorial, you will create a [VM](../../compute/concepts/vm.md) from a [{{ coi }}](../concepts/index.md) that contains a Docker container with a running MongoDB instance and an additional 10 GB [volume](../../container-registry/concepts/docker-volume.md) attached.

## Getting started {#before-you-begin}

If the required Docker image is pushed to {{ container-registry-full-name }}, create a [service account](../../iam/operations/sa/create.md) with the [{{ roles-cr-puller }}](../../container-registry/security/index.md#choosing-roles) role for the [registry](../../container-registry/concepts/registry.md) in use. A {{ coi }} VM will pull the Docker image from the registry on behalf of this account.

## Create a VM from a {{ coi }} with an additional volume for a Docker container {#create-vm}

{% list tabs group=instructions %}

- CLI {#cli}

  {% include [cli-install](../../_includes/cli-install.md) %}

  {% include [default-catalogue](../../_includes/default-catalogue.md) %}

  1. See the description of the CLI command for creating a VM:

     ```bash
     yc compute instance create-with-container --help
     ```

  1. Prepare the [Docker Compose specification](../concepts/coi-specifications.md#compose-spec). Save the following data to the `docker-compose.yaml` file:

     ```yaml
     version: '3.4'
     services:
       app2:
         container_name: container-name
         image: "mongo:latest"
         ports:
           - "27017:27017"
         restart: always
         volumes:
           - /home/yc-user/coi-data:/data
     x-yc-disks:
       - device_name: coi-data
         fs_type: ext4
         host_path: /home/yc-user/coi-data
     ```

     When creating a VM via the CLI, a default user is created: `yc-user`.
  1. Create a VM with multiple disks:

     ```bash
     yc compute instance create-with-container \
       --name coi-vm \
       --zone {{ region-id }}-a \
       --create-boot-disk size=30 \
       --create-disk name=data-disk,size=10,device-name=coi-data \
       --network-interface subnet-name=<subnet_name>,nat-ip-version=ipv4 \
       --ssh-key <path_to_public_part_of_SSH_key> \
       --docker-compose-file docker-compose.yaml
     ```

     Where:
     * `--name`: VM name.
     * `--zone`: [Availability zone](../../overview/concepts/geo-scope.md).
     * `--create-boot-disk`: VM [disk](../../compute/concepts/disk.md) parameters.

        {% include [min-disk-size](../../_includes/cos/min-disk-size.md) %}

     * `--network-interface`: [Network](../../vpc/concepts/network.md#network) parameters:
       * `subnet-name`: Name of the [subnet](../../vpc/concepts/network.md#subnet) to host the VM.
       * `nat-ip-version`: [Public IPv4 address](../../vpc/concepts/ips.md) assignment method.
     * `--ssh-key`: Path to the file with the [public key](../../compute/operations/vm-connect/ssh.md#creating-ssh-keys).
     * `--docker-compose-file`: YAML file with container specification.

     Once created, the VM will appear in the VM list under **{{ ui-key.yacloud.iam.folder.dashboard.label_compute }}** in the [management console]({{ link-console-main }}).
  1. Check the result.
     1. [Connect to the VM via SSH](../../compute/operations/vm-connect/ssh.md).
     1. Get the ID of the running Docker container:

        ```bash
        sudo docker ps -a
        ```

        Result:

        ```text
        CONTAINER ID  IMAGE         COMMAND                 CREATED        STATUS        PORTS      NAMES
        1f71********  mongo:latest  "docker-entrypoint.s…"  5 minutes ago  Up 5 minutes  27017/tcp  container-name
        ```

     1. Connect to the running Docker container:

        ```bash
        sudo docker exec -it 1f71******** bash
        ```

     1. View the list of attached disks. Pay attention to the mounted `/dev/vdb 11G 24M 9.9G 1% /data` disk:

        ```bash
        df -H
        ```

        Result:

        ```text
        Filesystem  Size  Used  Avail  Use%  Mounted on
        overlay     32G   4.0G    27G   14%  /
        tmpfs       68M      0    68M    0%  /dev
        ...
        tmpfs       1.1G     0   1.1G    0%  /proc/acpi
        tmpfs       1.1G     0   1.1G    0%  /proc/scsi
        tmpfs       1.1G     0   1.1G    0%  /sys/firmware
        ```

{% endlist %}

For more information about working with VMs, see our [step-by-step guides](../../compute/operations/index.md).