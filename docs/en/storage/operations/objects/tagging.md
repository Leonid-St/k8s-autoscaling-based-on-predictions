---
title: Object label management in {{ objstorage-full-name }}
description: Follow this guide to manage object labels in an {{ objstorage-name }} bucket.
---

# Object label management

An [object label](../../concepts/tags.md) is a key-value pair used for logical object labeling.

{% include [labels-tags](../../../_includes/storage/labels-tags.md) %}

## Adding or updating labels {#add-edit-tag}

{% list tabs group=instructions %}

- {{ yandex-cloud }} CLI {#cli}

  {% include [cli-install](../../../_includes/cli-install.md) %}

  {% include [default-catalogue](../../../_includes/default-catalogue.md) %}

  1. View the description of the CLI command to change object labels in a bucket:

      ```bash
      yc storage s3api put-object-tagging --help
      ```

  1. Get a list of buckets in the default folder:

      ```bash
      yc storage bucket list
      ```

      Result:

      ```text
      +------------------+----------------------+-------------+-----------------------+---------------------+
      |       NAME       |      FOLDER ID       | MAX SIZE    | DEFAULT STORAGE CLASS |     CREATED AT      |
      +------------------+----------------------+-------------+-----------------------+---------------------+
      | my-bucket        | b1g5dk66rc1u******** | 53687091200 | STANDARD              | 2024-12-17 07:17:04 |
      +------------------+----------------------+-------------+-----------------------+---------------------+
      ```

  1. In the terminal, run this command that overwrites any existing object labels:

      ```bash
      yc storage s3api put-object-tagging \
        --bucket <bucket_name> \
        --key <object_key> \
        --version-id <version_ID> \
        --tagging TagSet=[{Key=<label_key>,Value=<label_value>},{Key=<label_key>,Value=<label_value>}]
      ```

      Where:
      * `--bucket`: Bucket name.
      * `--key`: Bucket object [key](../../concepts/object.md#key).
      * `--version-id`: Object [version](../../concepts/versioning.md). This is an optional parameter.
      * `--tagging`: Array of object labels, where:
        * `Key`: Label key, the `string` type.
        * `Value`: Label value, the `string` type.

      Result:

      ```text
      request_id: ac9066e5********
      ```

- AWS CLI {#aws-cli}

  If you do not have the AWS CLI yet, [install and configure it](../../tools/aws-cli.md).

  In the terminal, run this command that overwrites any existing object labels:

  ```bash
  aws s3api put-object-tagging \
    --bucket <bucket_name> \
    --key <object_key> \
    --tagging 'TagSet=[{Key=<label_key>,Value=<label_value>},{Key=<label_key>,Value=<label_value>}]' \
    --endpoint-url=https://{{ s3-storage-host }}
  ```

  Where:
  * `--bucket`: Bucket name.
  * `--key`: Bucket object [key](../../concepts/object.md#key).
  * `--tagging`: Array of object labels, where:
    * `Key`: Label key, the `string` type.
    * `Value`: Label value, the `string` type.
  * `--endpoint-url`: {{ objstorage-name }} endpoint.

- {{ TF }} {#tf}

  {% include [terraform-role](../../../_includes/storage/terraform-role.md) %}

  {% include [terraform-install](../../../_includes/terraform-install.md) %}

  1. Open the {{ TF }} configuration file and add the `tags` section to the object description fragment:

      ```hcl
      resource "yandex_storage_object" "test-object" {
        access_key       = "<static_key_ID>"
        secret_key       = "<secret_key>"
        bucket           = "<bucket_name>"
        key              = "<object_name>"
        source           = "<path_to_file>"
        tags             = {
          <label_1_key> = "<label_1_value>"
          <label_2_key> = "<label_2_value>"
        }
      }
      ```

      Where `tags` is an array of object labels formatted as `<key> = "<value>"`.

      For more information about the `yandex_storage_object` resource in Terraform, see the [provider documentation]({{ tf-provider-resources-link }}/storage_object).

  1. Apply the changes:

      {% include [terraform-validate-plan-apply](../../../_tutorials/_tutorials_includes/terraform-validate-plan-apply.md) %}

  That will add the labels to the object. You can check the new labels and the object's configuration using the [AWS CLI or S3 API](#get-tag).

- API {#api}

  To add or edit object labels, use the [putObjectTagging](../../s3/api-ref/object/putobjecttagging.md) S3 API method.

{% endlist %}

## Viewing labels {#get-tag}

{% list tabs group=instructions %}

- {{ yandex-cloud }} CLI {#cli}

  {% include [cli-install](../../../_includes/cli-install.md) %}

  {% include [default-catalogue](../../../_includes/default-catalogue.md) %}

  1. View the description of the CLI command to view object labels in a bucket:

      ```bash
      yc storage s3api get-object-tagging --help
      ```

  1. Get a list of buckets in the default folder:

      ```bash
      yc storage bucket list
      ```

      Result:

      ```text
      +------------------+----------------------+-------------+-----------------------+---------------------+
      |       NAME       |      FOLDER ID       | MAX SIZE    | DEFAULT STORAGE CLASS |     CREATED AT      |
      +------------------+----------------------+-------------+-----------------------+---------------------+
      | my-bucket        | b1g5dk66rc1u******** | 53687091200 | STANDARD              | 2024-12-17 07:17:04 |
      +------------------+----------------------+-------------+-----------------------+---------------------+
      ```

  1. Get information about object labels:

      ```bash
      yc storage s3api get-object-tagging \
        --bucket <bucket_name> \
        --key <object_key> \
        --version-id <version_ID>
      ```

      Where:
      * `--bucket`: Bucket name.
      * `--key`: Bucket object [key](../../concepts/object.md#key).
      * `--version-id`: Object [version](../../concepts/versioning.md). This is an optional parameter.

      Result:

      ```text
      request_id: 2defc7b2********
      tag_set:
        - key: key-for-my-object-2
          value: second-value
        - key: key-for-my-object-1
          value: first-value
      ```

- AWS CLI {#aws-cli}

  If you do not have the AWS CLI yet, [install and configure it](../../tools/aws-cli.md).

  In the terminal, run this command:

  ```bash
  aws s3api get-object-tagging \
    --bucket <bucket_name> \
    --key <object_key> \
    --endpoint-url=https://{{ s3-storage-host }}
  ```

  Where:
  * `--bucket`: Bucket name.
  * `--key`: Bucket object [key](../../concepts/object.md#key).
  * `--endpoint-url`: {{ objstorage-name }} endpoint.

  Result:

  ```json
  {
      "TagSet": [
          {
              "Key": "test-key-1",
              "Value": "test-value-1"
          },
          {
              "Key": "test-key-2",
              "Value": "test-value-2"
          }
      ]
  }
  ```

- API {#api}

  To view object labels, use the [getObjectTagging](../../s3/api-ref/object/getobjecttagging.md) S3 API method.

{% endlist %}

## Deleting labels {#delete-tag}

{% list tabs group=instructions %}

- {{ yandex-cloud }} CLI {#cli}

  {% include [cli-install](../../../_includes/cli-install.md) %}

  {% include [default-catalogue](../../../_includes/default-catalogue.md) %}

  1. View the description of the CLI command to delete object labels in a bucket:

      ```bash
      yc storage s3api delete-object-tagging --help
      ```

  1. Get a list of buckets in the default folder:

      ```bash
      yc storage bucket list
      ```

      Result:

      ```text
      +------------------+----------------------+-------------+-----------------------+---------------------+
      |       NAME       |      FOLDER ID       | MAX SIZE    | DEFAULT STORAGE CLASS |     CREATED AT      |
      +------------------+----------------------+-------------+-----------------------+---------------------+
      | my-bucket        | b1g5dk66rc1u******** | 53687091200 | STANDARD              | 2024-12-17 07:17:04 |
      +------------------+----------------------+-------------+-----------------------+---------------------+
      ```

  1. Delete object labels in a bucket:

      ```bash
      yc storage s3api delete-object-tagging \
        --bucket <bucket_name> \
        --key <object_key> \
        --version-id <version_ID>
      ```

      Where:
      * `--bucket`: Bucket name.
      * `--key`: Bucket object [key](../../concepts/object.md#key).
      * `--version-id`: Object [version](../../concepts/versioning.md). This is an optional parameter.

      Result:

      ```text
      request_id: 1131dfc4********
      ```

- AWS CLI {#aws-cli}

  If you do not have the AWS CLI yet, [install and configure it](../../tools/aws-cli.md).

  In the terminal, run this command:

  ```bash
  aws s3api delete-object-tagging \
    --bucket <bucket_name> \
    --key <object_key> \
    --endpoint-url=https://{{ s3-storage-host }}
  ```

  Where:
  * `--bucket`: Bucket name.
  * `--key`: Bucket object [key](../../concepts/object.md#key).
  * `--endpoint-url`: {{ objstorage-name }} endpoint.

- {{ TF }} {#tf}

  {% include [terraform-install](../../../_includes/terraform-install.md) %}

  1. Open the {{ TF }} configuration file and delete the `tags` section from the fragment describing the object.

     {% cut "Sample object tag description in a {{ TF }} configuration" %}

     ```hcl
      resource "yandex_storage_object" "test-object" {
        ...
        tags             = {
          <label_1_key> = "<label_1_value>"
          <label_2_key> = "<label_2_value>"
        }
      }
     ...
     ```

     {% endcut %}

  1. Apply the changes:

      {% include [terraform-validate-plan-apply](../../../_tutorials/_tutorials_includes/terraform-validate-plan-apply.md) %}

  That will delete the object's labels. You can check the deletion of labels and the object's configuration using the [AWS CLI or S3 API](#get-tag).

- API {#api}

  To delete object labels, use the [deleteObjectTagging](../../s3/api-ref/object/deleteobjecttagging.md) S3 API method.

{% endlist %}
