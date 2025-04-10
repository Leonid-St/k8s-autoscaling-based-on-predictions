---
title: Getting started with {{ cdn-full-name }}
description: Follow this guide to create and set up a CDN resource.
---

# Getting started with {{ cdn-full-name }}

Configure content sharing via a CDN using a {{ objstorage-name }} bucket as an example:

1. [Create a CDN resource](#create-cdn-resource).
1. [Upload your content to the CDN](#upload-content-to-CDN).
1. [Configure a CNAME record for your domain](#setup-cname).
1. [Test the CDN](#check-cdn-working).

## Getting started {#before-you-begin}

1. Make sure you have a domain name and can access the DNS settings on the website of your DNS hosting provider. This is usually the company that registered your domain.
1. Go to the [management console]({{ link-console-main }}) and log in to {{ yandex-cloud }} or sign up if not signed up yet.
1. If you do not have a folder yet, create one:

   {% include [create-folder](../_includes/create-folder.md) %}

1. Go to the folder page and select **{{ ui-key.yacloud.iam.folder.dashboard.label_cdn }}**.
1. {% include [activate-provider](../_includes/cdn/activate-provider.md) %}
1. If you do not have a {{ objstorage-name }} bucket yet:

   1. [Create a bucket](../storage/operations/buckets/create.md).
   1. [Enable public access](../storage/operations/buckets/bucket-availability.md) to the objects in the bucket and the object list.
   1. [Upload content](../storage/operations/objects/upload.md) to the bucket.

## Create a CDN resource {#create-cdn-resource}

1. Go to the folder page and select **{{ ui-key.yacloud.iam.folder.dashboard.label_cdn }}**.
1. In the **{{ ui-key.yacloud.cdn.label_resources-list }}** tab, click **{{ ui-key.yacloud.cdn.button_resource-create }}**.
1. Configure your CDN resource as follows:

   * **{{ ui-key.yacloud.cdn.label_content-query-type }}**: Select `{{ ui-key.yacloud.cdn.value_query-type-one-origin }}`.
   * **{{ ui-key.yacloud.cdn.label_source-type }}**: Select `{{ ui-key.yacloud.cdn.value_source-type-bucket }}`.
   * **{{ ui-key.yacloud.cdn.label_bucket }}**: Select the bucket created in {{ objstorage-name }}.
   * **{{ ui-key.yacloud.cdn.label_section-domain }}**: Specify the primary domain name that you will use in your website links to content posted on the CDN, e.g., `cdn.example.com`.

     {% note alert %}

     You cannot change the primary domain name used for content distribution after creating a CDN resource.

     {% endnote %}

   * Under **{{ ui-key.yacloud.cdn.label_section-additional }}**:

     1. In the **{{ ui-key.yacloud.cdn.label_protocol }}** field, select `{{ ui-key.yacloud.common.label_http }}`.
     1. In the **{{ ui-key.yacloud.cdn.label_host-header }}** field, select `{{ ui-key.yacloud.cdn.value_host-header-custom }}`.
     1. In the **{{ ui-key.yacloud.cdn.label_custom-host-header }}** field, specify the domain name from the [bucket URL](../storage/concepts/bucket.md#bucket-url) in `<bucket_name>.{{ s3-storage-host }}` format. You do not need to specify the schema (`http` or `https`). For example:

        ```
        my-bucket.{{ s3-storage-host }}
        ```

        {% note alert %}

        If an invalid `Host` header is configured, {{ objstorage-name }} will respond to CDN server requests with errors.

        {% endnote %}

1. Click **{{ ui-key.yacloud.common.create }}**.

## Upload your content to the CDN {#upload-content-to-CDN}

{% note tip %}

We recommend that you preload large content to the CDN (for example, files larger than 200 MB). This is an optional step. If you skip uploading content at this point, it will be uploaded to the CDN automatically when a user tries to access it for the first time.

{% endnote %}

To preload content to CDN servers:

1. Go to the **{{ ui-key.yacloud.cdn.label_resource-content }}** tab.
1. Click **{{ ui-key.yacloud.cdn.button_resource-content-prefetch-cache }}**.
1. In the **{{ ui-key.yacloud.cdn.label_resource-content-prefetch-cache-paths }}** field, enter the names of the files stored in the bucket, omitting the bucket name, e.g.:

    ```text
    /index.html
    /static/styles.css
    /static/app.js
    ```

1. Click **{{ ui-key.yacloud.cdn.button_resource-content-prefetch-cache }}**.


## Set up a CNAME record for your domain {#setup-cname}

1. In the **{{ ui-key.yacloud.common.overview }}** tab, under **{{ ui-key.yacloud.cdn.label_dns-settings_title }}**, copy to the clipboard the URL generated on the `.edgecdn.ru` domain.
1. Go to your domain's DNS settings on the site of your DNS hosting provider.
1. Edit the relevant CNAME record so that it points to the previously copied URL on the `.edgecdn.ru` domain. For example, if the domain name you specified when creating your CDN resource is `cdn.example.com`, you need to create the following CNAME record or replace with it the existing record for `cdn`:

    ```http
    cdn CNAME cl-********.edgecdn.ru.
    ```

    {% include [note-dns-aname](../_includes/cdn/note-dns-aname.md) %}

## Test the CDN {#check-cdn-working}

1. Wait until the DNS records are updated. This may take several hours.
1. Make sure the `cdn` CNAME record in the DNS server's cache points to the URL generated by the service, e.g., `cl-********.edgecdn.ru`.
1. Try opening the site's URL in the browser, for example:

    ```http
    http://cdn.example.com/index.html
    ```
