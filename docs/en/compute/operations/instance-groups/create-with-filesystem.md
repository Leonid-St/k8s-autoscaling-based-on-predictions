---
title: Creating an instance group connected to a file storage
description: Follow this guide to create an instance group connected to a file storage.
---

# Creating an instance group connected to a file storage


One of the ways to handle [stateful workloads](../../concepts/instance-groups/stateful-workload.md) is to store the application state in a [file storage](../../concepts/filesystem.md) independent of the instance group.

To create an instance group that will automatically connect a common file storage to each of its instances:

1. {% include [sa.md](../../../_includes/instance-groups/sa.md) %}
1. If you do not have a file storage, [create one](../filesystem/create.md).
1. To be able to create, update, and delete VMs in the group, [assign](../../../iam/operations/sa/assign-role-for-sa.md) the [compute.editor](../../security/index.md#compute-editor) role to the service account.
1. Create an instance group:

    {% list tabs group=instructions %}

    - CLI {#cli}

      {% include [cli-install.md](../../../_includes/cli-install.md) %}

      {% include [default-catalogue.md](../../../_includes/default-catalogue.md) %}

      1. See the description of the [CLI](../../../cli/) command for creating an instance group:

         ```bash
         {{ yc-compute-ig }} create --help
         ```

      1. Check whether the [folder](../../../resource-manager/concepts/resources-hierarchy.md#folder) contains any [networks](../../../vpc/concepts/network.md#network):

         ```bash
         yc vpc network list
         ```

         If there are no networks, [create one](../../../vpc/operations/network-create.md).
      1. Select one of the {{ marketplace-full-name }} public images, e.g., [Ubuntu 22.04 LTS](/marketplace/products/yc/ubuntu-22-04-lts).

         {% include [standard-images.md](../../../_includes/standard-images.md) %}

      1. Prepare a file with the [YAML specification](../../concepts/instance-groups/specification.md) of the instance group and give it a name, e.g., `specification.yaml`.

          To connect a file storage to instances in the instance group, add the following to the specification:

          * In the `instance_template` field, a nested `filesystem_specs` field containing the description of the file storage:

              ```yml
              instance_template:
                ...
                filesystem_specs:
                  - mode: READ_WRITE
                    device_name: <instance_device_name>
                    filesystem_id: <file_storage_ID>
              ```

              Where:
              * `mode`: File storage access mode, `READ_WRITE` (read and write).
              * `device_name`: Device name for connecting the [file storage](../../concepts/filesystem.md) to the VM instance, e.g., `sample-fs`. It can only contain lowercase Latin letters, numbers, and hyphens. The first character must be a letter. The last character cannot be a hyphen. The name can be up to 63 characters long.
              * `filesystem_id`: File storage ID. You can view the ID in the [management console]({{ link-console-main }}) or using the `yc compute filesystem list` CLI command.

          * In the `#cloud-config` section of the `instance_template.metadata.user-data` field, commands for mounting the file storage to the VM:

              ```yml
              instance_template:
                ...
                metadata:
                  user-data: |-
                    #cloud-config
                    ...
                    runcmd:
                      - mkdir <instance_mount_point>
                      - mount -t virtiofs <instance_device_name> <instance_mount_point>
                      - echo "test-fs <instance_mount_point> virtiofs    rw    0   0" | tee -a /etc/fstab
              ```

              Where:
              * `<instance_mount_point>`: Instance directory to mount the connected file storage to, e.g., `/mnt/vfs0`.
              * `<instance_device_name>`: Device name for connecting the file storage to the VM instance. The value must match the one specified earlier in the `instance_template.filesystem_specs.device_name` field.

          Here is a YAML specification example:

          ```yml
          name: my-vm-group-with-fs
          service_account_id: <service_account_ID>
          description: "This instance group was created using a YAML configuration file."
          instance_template:
            platform_id: standard-v3
            resources_spec:
              memory: 2g
              cores: 2
            boot_disk_spec:
              mode: READ_WRITE
              disk_spec:
                image_id: fd8dlvgiatiqd8tt2qke
                type_id: network-hdd
                size: 32g
            network_interface_specs:
              - network_id: enp9mji1m7b3********
                primary_v4_address_spec: {
                  one_to_one_nat_spec: {
                    ip_version: IPV4
                  }
                }
                security_group_ids:
                  - enpuatgvejtn********
            filesystem_specs:
              - mode: READ_WRITE
                device_name: sample-fs
                filesystem_id: epdccsrlalon********
            metadata:
              user-data: |-
                #cloud-config
                datasource:
                 Ec2:
                  strict_id: false
                ssh_pwauth: no
                users:
                - name: my-user
                  sudo: ALL=(ALL) NOPASSWD:ALL
                  shell: /bin/bash
                  ssh_authorized_keys:
                  - <public_SSH_key>
                runcmd:
                - mkdir /mnt/vfs0
                - mount -t virtiofs sample-fs /mnt/vfs0
                - echo "sample-fs /mnt/vfs0 virtiofs    rw    0   0" | tee -a /etc/fstab
          deploy_policy:
            max_unavailable: 1
            max_expansion: 0
          scale_policy:
            fixed_scale:
              size: 2
          allocation_policy:
            zones:
              - zone_id: {{ region-id }}-a
                instance_tags_pool:
                - first
                - second
          ```

          This example shows a specification for [creating a fixed-size instance group](./create-fixed-group.md) with a file storage connected to the instances.

          For more information about the instance group specification parameters, see [{#T}](../../concepts/instance-groups/specification.md).

      1. Create an instance group in the default folder:

          ```bash
          {{ yc-compute-ig }} create --file specification.yaml
          ```

         This command will create a group of two same-type instances with the following configuration:
         * Name: `my-vm-group-with-fs`.
         * OS: `Ubuntu 22.04 LTS`.
         * Availability zone: `{{ region-id }}-a`.
         * vCPU: 2, RAM: 2 GB.
         * Network [HDD](../../concepts/disk.md#disks-types): 32 GB.
         * Connected to a file storage. The file storage will be mounted to the `/mnt/vfs0` directory of the group instances.
         * {% include [ssh-connection-internal-ip](../../../_includes/instance-groups/ssh-connection-internal-ip.md) %}

    - {{ TF }} {#tf}

      {% include [terraform-install](../../../_includes/terraform-install.md) %}

      1. In the configuration file, define the parameters of the resources you want to create:

          ```hcl
          resource "yandex_iam_service_account" "ig-sa" {
            name        = "ig-sa"
            description = "Service account for managing the instance group"
          }

          resource "yandex_resourcemanager_folder_iam_member" "compute_editor" {
            folder_id  = "<folder_ID>"
            role       = "compute.editor"
            member     = "serviceAccount:${yandex_iam_service_account.ig-sa.id}"
            depends_on = [
              yandex_iam_service_account.ig-sa,
            ]
          }

          resource "yandex_compute_instance_group" "ig-1" {
            name                = "fixed-ig"
            folder_id           = "<folder_ID>"
            service_account_id  = "${yandex_iam_service_account.ig-sa.id}"
            deletion_protection = "<deletion_protection>"
            depends_on          = [yandex_resourcemanager_folder_iam_member.compute_editor]
            instance_template {
              platform_id = "standard-v3"
              resources {
                memory = <RAM_in_GB>
                cores  = <number_of_vCPUs>
              }

              boot_disk {
                mode = "READ_WRITE"
                initialize_params {
                  image_id = "<image_ID>"
                }
              }

              filesystem {
                mode = "READ_WRITE"
                device_name = "<instance_device_name>"
                filesystem_id = "<file_storage_ID>"
              }

              network_interface {
                network_id         = "${yandex_vpc_network.network-1.id}"
                subnet_ids         = ["${yandex_vpc_subnet.subnet-1.id}"]
                security_group_ids = ["<list_of_security_group_IDs>"]
                nat                = true
              }

              metadata = {
                user-data = "#cloud-config\n  datasource:\n   Ec2:\n    strict_id: false\n  ssh_pwauth: no\n  users:\n  - name: <instance_username>\n    sudo: ALL=(ALL) NOPASSWD:ALL\n    shell: /bin/bash\n    ssh_authorized_keys:\n    - <public_SSH_key>\n  runcmd:\n    - mkdir <instance_mount_point>\n    - mount -t virtiofs <instance_device_name> <instance_mount_point>\n    - echo \"sample-fs <instance_mount_point> virtiofs    rw    0   0\" | tee -a /etc/fstab"
              }
            }

            scale_policy {
              fixed_scale {
                size = <number_of_instances_in_group>
              }
            }

            allocation_policy {
              zones = ["{{ region-id }}-a"]
            }

            deploy_policy {
              max_unavailable = 1
              max_expansion   = 0
            }
          }

          resource "yandex_vpc_network" "network-1" {
            name = "network1"
          }

          resource "yandex_vpc_subnet" "subnet-1" {
            name           = "subnet1"
            zone           = "{{ region-id }}-a"
            network_id     = "${yandex_vpc_network.network-1.id}"
            v4_cidr_blocks = ["192.168.10.0/24"]
          }
          ```

          Where:
          * `yandex_iam_service_account`: [Service account](../../../iam/concepts/users/service-accounts.md) description. All operations in {{ ig-name }} are performed on behalf of the service account.

            {% include [sa-dependence-brief](../../../_includes/instance-groups/sa-dependence-brief.md) %}

          * `yandex_resourcemanager_folder_iam_member`: Description of access permissions for the [folder](../../../resource-manager/concepts/resources-hierarchy.md#folder) the service account belongs to. To be able to create, update, and delete VM instances in the instance group, assign the `compute.editor` [role](../../security/index.md#compute-editor) to the service account.
          * `yandex_compute_instance_group`: Instance group description:
            * General information about the instance group:
              * `name`: Instance group name.
              * `folder_id`: Folder ID.
              * `service_account_id`: Service account ID.
              * `deletion_protection`: Instance group protection against deletion, `true` or `false`. You cannot delete an instance group with this option enabled. The default value is `false`.
            * [Instance template](../../concepts/instance-groups/instance-template.md):
              * `platform_id`: [Platform](../../concepts/vm-platforms.md).
              * `resources`: Number of vCPUs and amount of RAM available to the VM instance. The values must match the selected [platform](../../concepts/vm-platforms.md).
              * `boot_disk`: Boot [disk](../../concepts/disk.md) settings.
                * `mode`: Disk access mode, `READ_ONLY` or `READ_WRITE`.
                * `image_id`: ID of the selected image. You can get the image ID from the [list of public images](../images-with-pre-installed-software/get-list.md).
              * `filesystem`: [File storage](../../concepts/filesystem.md) settings.
                * `mode`: File storage access mode, `READ_WRITE` (read and write).
                * `device_name`: Device name for connecting the file storage to the VM instance, e.g., `sample-fs`. It can only contain lowercase Latin letters, numbers, and hyphens. The first character must be a letter. The last character cannot be a hyphen. The name can be up to 63 characters long.
                * `filesystem_id`: File storage ID. You can view the ID in the [management console]({{ link-console-main }}) or using the `yc compute filesystem list` CLI command.
              * `network_interface`: [Network](../../../vpc/concepts/network.md#network) settings. Specify the IDs of your network, [subnet](../../../vpc/concepts/network.md#subnet), and [security groups](../../../vpc/concepts/security-groups.md).
              * `metadata`: In [metadata](../../concepts/vm-metadata.md), provide the following:
                * Instance username and public key to enable this user to access the instance via SSH. 
                * Instance mount point for the file storage, i.e., instance directory to mount the connected file storage to, e.g., `/mnt/vfs0`.
                * Instance device name, i.e., device name for connecting the file storage to the VM instance. The value must match the one specified earlier in the `device_name` field of the `filesystem` section.

                For more information, see [{#T}](../../concepts/vm-metadata.md).
            * [Policies](../../concepts/instance-groups/policies/index.md):
              * `deploy_policy`: Instance [deployment policy](../../concepts/instance-groups/policies/deploy-policy.md) for the group.
              * `scale_policy`: Instance [scaling policy](../../concepts/instance-groups/policies/scale-policy.md) for the group.
              * `allocation_policy`: [Policy for allocating](../../concepts/instance-groups/policies/allocation-policy.md) instances across [availability zones](../../../overview/concepts/geo-scope.md) and regions.
          * `yandex_vpc_network`: Cloud network description.
          * `yandex_vpc_subnet`: Description of the subnet to connect the instance group to.

            {% note info %}

            If you already have suitable resources, such as a service account, cloud network, and subnet, you do not need to redefine them. Specify their names and IDs in the appropriate parameters.

            {% endnote %}

          For more information about the resources you can create with {{ TF }}, see the [relevant provider documentation]({{ tf-provider-link }}).

      1. Create the resources:

          {% include [terraform-validate-plan-apply](../../../_tutorials/_tutorials_includes/terraform-validate-plan-apply.md) %}

          All the resources you need will then be created in the specified folder. You can check the new resources and their settings using the [management console]({{ link-console-main }}).

          {% include [ssh-connection-internal-ip](../../../_includes/instance-groups/ssh-connection-internal-ip.md) %}

    - API {#api}

      Use the [create](../../instancegroup/api-ref/InstanceGroup/create.md) REST API method for the [InstanceGroup](../../instancegroup/api-ref/InstanceGroup/index.md) resource or the [InstanceGroupService/Create](../../instancegroup/api-ref/grpc/InstanceGroup/create.md) gRPC API call.

    {% endlist %}

Make sure the file storage is connected to the group instances. For this, [connect](../vm-connect/ssh.md#vm-connect) to the instances via SSH and navigate to the directory you specified as the mount point.