---
title: How to search in {{ search-api-full-name }} using API v2 in synchronous mode
description: Follow this guide to learn how to use the API v2 interface in {{ search-api-name }} to send search queries and get search results in XML or HTML format in synchronous mode.
---

# Performing search queries using API v2 in synchronous mode

With {{ search-api-name }}'s [API v2](../concepts/index.md#api-v2), you can perform text search through the Yandex search database and get search results in [XML](../concepts/response.md) or [HTML](../concepts/html-response.md) format in synchronous mode. You can run queries using the [gPRC API](../api-ref/grpc/). The search results you get depend on the parameters specified in your query.

## Getting started {#before-you-begin}

{% include [before-begin](../../_tutorials/_tutorials_includes/before-you-begin.md) %}


To use the examples, install [gRPCurl](https://github.com/fullstorydev/grpcurl) and [jq](https://stedolan.github.io/jq).


## Get your cloud ready {#initial-setup}

{% include [prepare-cloud-v2](../../_includes/search-api/prepare-cloud-v2.md) %}

## Send a search query {#execute-request}

1. Send a query and get a Base64-encoded result:

    {% list tabs group=instructions %}

    - gRPC API {#grpc-api}

      1. Create a file with the request body, e.g., `body.json`:

          **body.json**

          {% include [grpc-body-v2](../../_includes/search-api/grpc-body-v2.md) %}

          {% cut "Description of fields" %}

          {% include [grpc-v2-body-params](../../_includes/search-api/grpc-v2-body-params.md) %}

          {% endcut %}

      1. Run an gRPC call by specifying the IAM token you got earlier:

          ```bash
          grpcurl \
            -rpc-header "Authorization: Bearer <IAM_token>" \
            -d @ < body.json \
            searchapi.{{ api-host }}:443 yandex.cloud.searchapi.v2.WebSearchService/Search \
            > result.json
          ```

        Eventually the search query result will be saved to a file named `result.json` containing a [Base64-encoded](https://en.wikipedia.org/wiki/Base64) [XML](../concepts/response.md) or [HTML](../concepts/html-response.md) response in the `rawData` field.

    {% endlist %}

1. Depending on the requested response format, decode the result from `Base64`:

    {% list tabs group=search_api_request %}

    - XML {#xml}

      ```bash
      echo "$(< result.json)" | \
        jq -r .rawData | \
        base64 --decode > result.xml
      ```

      The XML response to the query will be saved to a file named `result.xml`.

    - HTML {#html}

      ```bash
      echo "$(< result.json)" | \
        jq -r .rawData | \
        base64 --decode > result.html
      ```

      The HTML response to the query will be saved to a file named `result.html`.

    {% endlist %}

#### See also {#see-also}

* [{#T}](./web-search.md)
* [{#T}](../concepts/web-search.md)
* [{#T}](../api-ref/authentication.md)
* [{#T}](../concepts/response.md)
* [{#T}](../concepts/html-response.md)