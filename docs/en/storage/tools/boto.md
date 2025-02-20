---
title: boto3 and boto
description: In this tutorial, you will learn about boto3 and boto, how to install and configure them, and will see some examples of operations.
---

# boto3 and boto


[boto3](https://github.com/boto/boto3) and [boto](https://github.com/boto/boto) are software development kits (SDKs) for the Python 2.x and 3.x programming languages. The SDKs are designed for working with AWS services.


## Getting started {#before-you-begin}

{% include [aws-tools-prepare](../../_includes/aws-tools/aws-tools-prepare.md) %}

{% include [access-bucket-sa](../../_includes/storage/access-bucket-sa.md) %}

## Installation {#installation}

{% include [note-boto-versions](../../_includes/aws-tools/note-boto-versions.md) %}

To install boto3 version 1.35.99, run the following command in the terminal:

```bash
pip3 install boto3==1.35.99
```

{% include [install-boto](../../_includes/aws-tools/install-boto.md)%}

## Setup {#setup}

{% list tabs group=instructions %}

- Locally {#locally}

  {% include [storage-sdk-setup](../_includes_service/storage-sdk-setup-storage-url.md) %}

- {{ sf-full-name }} {#functions}
  
  [Add environment variables](../../functions/operations/function/version-manage#version-env) to a function in {{ sf-name }}:

  * `AWS_ACCESS_KEY_ID`: Static service account key ID.
  * `AWS_SECRET_ACCESS_KEY`: Secret key.
  * `AWS_DEFAULT_REGION`: Region ID.

  Use the {{ objstorage-name }} address to access `{{ s3-storage-host }}`.

{% endlist %}


## Example {#boto-example}


{% list tabs group=instructions %}

- Locally {#locally}
  
  boto3: 

  {% include [boto3-example](../../_includes/storage/boto3-example.md) %}

  {% cut "boto" %}

  {% include [boto-example](../../_includes/storage/boto-example.md) %}

  {% endcut %}

- {{ sf-full-name }} {#functions}

  For an example, see this [video conversion guide](../tutorials/video-converting-queue.md).

{% endlist %}

