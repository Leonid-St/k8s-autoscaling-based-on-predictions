---
title: Uploading organization audit logs to {{ yds-full-name }}
description: Follow this guide to upload organization audit logs to {{ yds-name }}.
---

# Uploading organization audit logs to {{ yds-name }}


Follow this guide to create a new [trail](../concepts/trail.md) that will upload management event audit logs of all resources and, optionally, data event audit logs of selected services of an organization to a {{ yds-name }} stream.


## Prepare the environment {#before-you-begin}

To export organization audit logs:

1. [Create](../../data-streams/operations/manage-streams.md#create-data-stream) a data stream to upload audit logs.
1. [Create](../../iam/operations/sa/create.md) a service account.
1. Assign roles to the service account:

    {% list tabs group=instructions %}

    - CLI {#cli}

      {% include [cli-install](../../_includes/cli-install.md) %}

      {% include [default-catalogue](../../_includes/default-catalogue.md) %}

      * `audit-trails.viewer` for the organization whose audit logs will be collected:

        ```
        yc organization-manager organization add-access-binding \
          --role audit-trails.viewer \
          --id <organization_ID> \
          --service-account-id <service_account_ID>
        ```

        Where:
        * `--role`: Role you want to assign.
        * `--id`: [ID](../../organization/operations/organization-get-id.md) of the organization from whose resources audit logs will be collected.
        * `--service-account-id`: Service account ID.

      * `yds.writer` for the folder to host the trail:

        ```
        yc resource-manager folder add-access-binding \
          --role yds.writer \
          --id <folder_ID> \
          --service-account-id <service_account_ID>
        ```

        Where:
        * `--role`: Role you want to assign.
        * `--id`: ID of the folder to host the trail.
        * `--service-account-id`: Service account ID.

    {% endlist %}

1. On the [Access permissions]({{ link-console-access-management }}) page, make sure you have the following roles:
    * `iam.serviceAccounts.user` for the service account.
    * `audit-trails.editor` for the folder to host the trail.
    * `audit-trails.viewer` for the organization whose audit logs will be collected.
    * `yds.viewer` for the data stream in {{ yds-name }}.


## Creating a trail {#the-trail-creation}

To create a trail that exports organization audit logs:

{% list tabs group=instructions %}

- Management console {#console}

  1. In the [management console]({{ link-console-main }}), select the folder to host the trail.
  1. Select **{{ ui-key.yacloud.iam.folder.dashboard.label_audit-trails }}**.

  1. Click **{{ ui-key.yacloud.audit-trails.button_create-trail }}** and specify:

      * **{{ ui-key.yacloud.common.name }}**: Name of the new trail.
      * **{{ ui-key.yacloud.common.description }}**: Trail description (optional).

  1. Under **{{ ui-key.yacloud.audit-trails.label_destination }}**, configure the destination object:

      * **{{ ui-key.yacloud.audit-trails.label_destination }}**: `{{ ui-key.yacloud.audit-trails.label_dataStream }}`.
      * **{{ ui-key.yacloud.audit-trails.label_stream-name }}**: Select the data stream. You can also create a new data stream by clicking **{{ ui-key.yacloud.common.create-new }}** and [specifying](../../data-streams/operations/manage-streams.md#create-data-stream) the data stream settings.

  1. Under **{{ ui-key.yacloud.audit-trails.label_service-account }}**, select the service account that the trail will use to upload audit log files to the data stream.

  1. Under **{{ ui-key.yacloud.audit-trails.label_path-filter-section }}**, configure the collection of management event audit logs:

      * **{{ ui-key.yacloud.audit-trails.label_collecting-logs }}**: Select `{{ ui-key.yacloud.common.enabled }}`.
      * **{{ ui-key.yacloud.audit-trails.label_resource-type }}**: Select `{{ ui-key.yacloud.audit-trails.label_organization-manager.organization }}`.
      * **{{ ui-key.yacloud.audit-trails.label_organization-manager.organization }}**: Automatically populated field containing the name of the current organization.
      * **{{ ui-key.yacloud.audit-trails.label_resource-manager.cloud }}**: Keep the default value, `{{ ui-key.yacloud.common.all }}`.

  1. {% include [data-plane-on-console](../../_includes/audit-trails/data-plane-on-console.md) %}

  1. Click **{{ ui-key.yacloud.common.create }}**.

- CLI {#cli}

  {% include [cli-install](../../_includes/cli-install.md) %}

  {% include [default-catalogue](../../_includes/default-catalogue.md) %}

  1. View the description of the [CLI](../../cli/) command for creating a trail:

      ```bash
      yc audit-trails trail create --help
      ```

  1. Run the command to create a trail to collect [management event](../concepts/format.md) audit logs in the organization:

      ```bash
      yc audit-trails trail create \
        --name <trail_name> \
        --description "<trail_description>" \
        --service-account-id <service_account_ID> \
        --destination-yds-stream <data_stream_name> \
        --destination-yds-database-id <YDB_DB_ID> \
        --filter-from-organisation-id <organization_ID> \
        --filter-some-cloud-ids <list_of_clouds_in_organization>
      ```

      Where:

      {% include [trail-create-cli-descs_part1](../../_includes/audit-trails/trail-create-cli-descs-part1.md) %}

      {% include [trail-create-cli-descs-yds](../../_includes/audit-trails/trail-create-cli-descs-yds.md) %}

      {% include [trail-create-cli-descs_org](../../_includes/audit-trails/trail-create-cli-descs-org.md) %}

      {% cut "Result:" %}

      ```yml
      done (1s)
      id: cnpl3n5g00po********
      folder_id: b1g9d2k0itu4********
      created_at: "2024-04-01T10:32:18.570Z"
      updated_at: "2024-04-01T10:32:18.570Z"
      name: sample-trail
      description: My very first trail
      destination:
        data_stream:
          database_id: etnsamrdlkiv********
          stream_name: sample-stream
      service_account_id: ajeee339l4m5********
      status: ACTIVE
      filter:
        path_filter:
          root:
            some_filter:
              resource:
                id: bpfaidqca8vd********
                type: organization-manager.organization
              filters:
                - any_filter:
                    resource:
                      id: b1glti4eser3********
                      type: resource-manager.cloud
                - any_filter:
                    resource:
                      id: b1gssd27h7ra********
                      type: resource-manager.cloud
        event_filter: {}
      cloud_id: b1glti4eser3********
      ```

      {% endcut %}

  1. To create a trail to collect management event and [data event](../concepts/format-data-plane.md) audit logs, use the trail's YAML specification file.

      1. Create a trail configuration file, e.g., `specification.yaml`:

          {% cut "specification.yaml" %}

          {% include [cli-spec-org-yds](../../_includes/audit-trails/cli-spec-org-yds.md) %}

          {% endcut %}

          {% cut "Specification description" %}

          {% include [trail-create-clispec-descs-part1](../../_includes/audit-trails/trail-create-clispec-descs-part1.md) %}

          {% include [trail-create-clispec-descs-yds](../../_includes/audit-trails/trail-create-clispec-descs-yds.md) %}

          {% include [trail-create-clispec-descs-part2](../../_includes/audit-trails/trail-create-clispec-descs-part2.md) %}

          {% endcut %}

      1. Run the command, specifying the path to the created trail specification file:

          ```bash
          yc audit-trails trail create --file <specification_file_path>
          ```

          {% cut "Result" %}

          ```yml
          done (1s)
          id: cnphmspp6fe2********
          folder_id: b1g9d2k0itu4********
          created_at: "2024-04-02T04:13:41.024Z"
          updated_at: "2024-04-02T04:13:41.024Z"
          name: sample-trail
          description: My very first trail
          destination:
            data_stream:
              database_id: etnneuucptkf********
              stream_name: sample-stream
          service_account_id: ajeee339l4m5********
          status: ACTIVE
          filter:
            path_filter:
              root:
                some_filter:
                  resource:
                    id: bpfaidqca8vd********
                    type: organization-manager.organization
                  filters:
                    - any_filter:
                        resource:
                          id: b1glti4eser3********
                          type: resource-manager.cloud
                    - any_filter:
                        resource:
                          id: b1gssd27h7ra********
                          type: resource-manager.cloud
            event_filter:
              filters:
                - service: mdb.postgresql
                  categories:
                    - plane: DATA_PLANE
                      type: WRITE
                  path_filter:
                    root:
                      some_filter:
                        resource:
                          id: bpfaidqca8vd********
                          type: organization-manager.organization
                        filters:
                          - any_filter:
                              resource:
                                id: b1glti4eser3********
                                type: resource-manager.cloud
                          - any_filter:
                              resource:
                                id: b1gssd27h7ra********
                                type: resource-manager.cloud
                - service: dns
                  categories:
                    - plane: DATA_PLANE
                      type: READ
                  path_filter:
                    root:
                      any_filter:
                        resource:
                          id: bpfaidqca8vd********
                          type: organization-manager.organization
          cloud_id: b1glti4eser3********
          ```

          {% endcut %}

      For more information about the `yc audit-trails trail create` command, see the [CLI reference](../../cli/cli-ref/audit-trails/cli-ref/trail/create.md).

- {{ TF }} {#tf}

  {% include [terraform-definition](../../_tutorials/_tutorials_includes/terraform-definition.md) %}

  {% include [terraform-install](../../_includes/terraform-install.md) %}

  1. In the configuration file, describe the parameters of the trail that will collect audit events of the organization's resources:

      ```hcl
      resource "yandex_audit_trails_trail" "basic_trail" {
        name = "<trail_name>"
        folder_id   = "<folder_ID>"
        description = "<trail_description>"
        
        labels = {
          key = "value"
        }
        
        service_account_id = "<service_account_ID>"
        
        data_stream_destination {
          stream_name   = "<stream_name>"
          database_id = "<YDB_DB_ID>"
        }
        
        filter {
          path_filter {
            some_filter {
              resource_id   = "<organization_ID>"
              resource_type = "<parent_resource_type>"
              any_filters {
                  resource_id   = "<ID_of_cloud_1>"
                  resource_type = "<child_resource_type>"
              }
              any_filters {
                  resource_id   = "<ID_of_cloud_2>"
                  resource_type = "<child_resource_type>"
              }
            }
          }
          event_filters {
            service = "<ID_of_service_1>"
            categories {
              plane = "DATA_PLANE"
              type  = "<action_type>"
            }
            path_filter {
              any_filter {
                resource_id = "<organization_ID>"
                resource_type = "<resource_type>"
              }
            }
          }
          event_filters {
            service = "<ID_of_service_2>"
            categories {
              plane = "DATA_PLANE"
              type  = "<action_type>"
            }
            path_filter {
              any_filter {
                resource_id = "<organization_ID>"
                resource_type = "<resource_type>"
              }
            }
          }
        }
      }
      ```

      Where:

      {% include [trail-create-tf-descs_part1](../../_includes/audit-trails/trail-create-tf-descs-part1.md) %}

      {% include [trail-create-tf-descs-yds](../../_includes/audit-trails/trail-create-tf-descs-yds.md) %}

      {% include [trail-create-tf-descs_part2](../../_includes/audit-trails/trail-create-tf-descs-part2.md) %}

      For more information about the `yandex_audit_trails_trail` resource parameters in {{ TF }}, see the [provider documentation]({{ tf-provider-resources-link }}/audit_trails_trail).

  1. Create resources:

      {% include [terraform-validate-plan-apply](../../_tutorials/_tutorials_includes/terraform-validate-plan-apply.md) %}
      
      {{ TF }} will create all the required resources. You can check the new resources and their settings using the [management console]({{ link-console-main }}) or this [CLI](../../cli/) command:

      ```bash
      yc audit-trails trail get <trail_name>
      ```

- API {#api}

  Use the [create](../api-ref/Trail/create.md) REST API method for the [Trail](../api-ref/Trail/index.md) resource or the [TrailService/Create](../api-ref/grpc/Trail/create.md) gRPC API call.

{% endlist %}

The trail will be created and will begin uploading audit logs to the data stream.


## What's next {#whats-next}

* Learn more about the [audit log format](../concepts/format.md).
* Find out about the procedure for [uploading audit logs to SIEM](../concepts/export-siem.md).
* Learn more about [searching audit logs in buckets](../tutorials/search-bucket.md).
