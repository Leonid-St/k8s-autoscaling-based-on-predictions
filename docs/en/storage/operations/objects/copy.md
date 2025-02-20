---
title: Copying objects in {{ objstorage-full-name }}
description: Follow this guide to copy an object from a bucket in {{ objstorage-name }}.
---

# Copying objects

{{ objstorage-name }} supports _server-side object copy_.

If [encryption](../../concepts/encryption.md) in a [bucket](../../concepts/bucket.md) is disabled, [objects](../../concepts/object.md) will be copied to server-side buckets. The host on which the copy operation is run and {{ objstorage-name }} exchange nothing but object [keys](../../concepts/object.md#key). No fee is charged for the copying traffic in this case because the traffic is internal for the object storage. However, you are [charged](../../pricing.md#prices-operations) for copy requests.

If bucket objects are encrypted, first they will be copied to the host that executes the command and then uploaded to the target bucket.

Large objects uploaded using [multipart uploads](../../concepts/multipart.md) are stored in the bucket in parts. They are copied by invoking [copyPart](../../s3/api-ref/multipart/copypart.md) for each part. Therefore, multipart objects cost more to copy than regular ones.

You can copy either the [entire bucket contents](#copy-from-bucket-to-bucket) or an [individual bucket object](#copy-single-object).


{% include [encryption-roles](../../../_includes/storage/encryption-roles.md) %}


## Copying a single object {#copy-single-object}

{% list tabs group=instructions %}

- {{ yandex-cloud }} CLI {#cli}

  {% include [cli-install](../../../_includes/cli-install.md) %}

  {% include [default-catalogue](../../../_includes/default-catalogue.md) %}

  1. View the description of the CLI command to upload an object to a bucket:

      ```bash
      yc storage s3api copy-object --help
      ```

  1. Get a list of buckets in the default folder:

      ```bash
      yc storage bucket list
      ```

      Result:

      ```text
      +------------------+----------------------+-------------+-----------------------+---------------------+
      |       NAME       |      FOLDER ID       |  MAX SIZE   | DEFAULT STORAGE CLASS |     CREATED AT      |
      +------------------+----------------------+-------------+-----------------------+---------------------+
      | first-bucket     | b1gmit33ngp6******** | 53687091200 | STANDARD              | 2022-12-16 13:58:18 |
      +------------------+----------------------+-------------+-----------------------+---------------------+
      ```

  1. Run this command:

      ```bash
      yc storage s3api copy-object \
        --copy-source <source_bucket>/<object_key> \
        --bucket <target_bucket> \
        --key <object_key>
      ```

      Where:

      * `--copy-source`: Name of the source bucket and object [key](../../concepts/object.md#key) you need to copy.
      * `--bucket`: Name of the bucket to copy the object to.
      * `--key`: Key to use for storing the object in the bucket.

      Result:

      ```bash
      copy_object_result:
        etag: '"d41d8cd98f00b204e9800998********"'
        last_modified_at: "2024-10-08T14:21:41.628Z"
      request_id: 61523025********
      copy_source_version_id: "null"
      ```

- AWS CLI {#aws-cli}

  1. If you do not have the AWS CLI yet, [install and configure it](../../tools/aws-cli.md).
  1. Run this command:

      ```bash
      aws --endpoint-url=https://{{ s3-storage-host }}/ \
        s3 cp s3://<source_bucket>/<object_key> s3://<target_bucket>/<object_key>
      ```

      Where:

      * `--endpoint-url`: {{ objstorage-name }} endpoint.
      * `s3 cp`: Copy object command.

      Result:

      ```text
      copy: s3://<source_bucket>/<object_key> to s3://<target_bucket>/<object_key>
      ```

      For more information about the `aws s3 cp` command, see the [AWS CLI Command Reference](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/sns/publish.html).

{% endlist %}

## Copying all bucket objects to another bucket {#copy-from-bucket-to-bucket}

{% list tabs group=instructions %}

- AWS CLI {#cli}

  1. If you do not have the AWS CLI yet, [install and configure it](../../tools/aws-cli.md).
  1. Run this command:

      ```bash
      aws --endpoint-url=https://{{ s3-storage-host }}/ \
        s3 cp --recursive s3:<source_bucket>/ s3:<target_bucket>/
      ```

      Where:

      * `--endpoint-url`: {{ objstorage-name }} endpoint.
      * `s3 cp`: Copy object command.
      * `--recursive`: Parameter for copying all objects from the source bucket.

      For more information about the `aws s3 cp` command, see the [AWS CLI Command Reference](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/sns/publish.html).

  All objects from the source bucket will be copied to the target bucket.

  {% note info %}

  You can copy objects between buckets either within a single cloud or between different clouds. To do this, make sure your [service account](../../../iam/concepts/users/service-accounts.md) has write permissions for both buckets.

  {% endnote %}


{% endlist %}

