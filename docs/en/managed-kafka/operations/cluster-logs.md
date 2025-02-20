# Viewing {{ KF }} cluster logs

{{ mkf-name }} allows you to [get a cluster log snippet](#get-log) for the selected period and [view logs in real time](#get-log-stream).

{% note info %}

Here, the log is the system log of the cluster and its hosts. This log is not related to the partition log for the {{ KF }} topic where the broker writes messages received from message producers.

{% endnote %}

{% include [log-duration](../../_includes/mdb/log-duration.md) %}

## Getting a cluster log {#get-log}

{% list tabs group=instructions %}

- Management console {#console}

    1. In the [management console]({{ link-console-main }}), go to the relevant folder.
    1. In the services list, select **{{ ui-key.yacloud.iam.folder.dashboard.label_managed-kafka }}**.
    1. Click the name of the cluster you need and select the ![image](../../_assets/console-icons/receipt.svg) **{{ ui-key.yacloud.common.logs }}** tab.
    1. Select **{{ ui-key.yacloud.kafka.label_filter_origin }}**, **{{ ui-key.yacloud.mdb.cluster.logs.label_hosts }}**, and **{{ ui-key.yacloud.mdb.cluster.logs.label_severity }}**.
    1. Specify the time period for which you want to display the log.

- CLI {#cli}

    {% include [cli-install](../../_includes/cli-install.md) %}

    {% include [default-catalogue](../../_includes/default-catalogue.md) %}

    1. View the description of the CLI command to view cluster logs:

        ```bash
        {{ yc-mdb-kf }} cluster list-logs --help
        ```

    1. Run the following command to get cluster logs (our example does not contain a complete list of available parameters):

        ```bash
        {{ yc-mdb-kf }} cluster list-logs <cluster_name_or_ID> \
           --limit <entry_number_limit> \
           --columns <log_columns_list> \
           --filter <entry_filtration_settings> \
           --since <time_range_left_boundary> \
           --until <time_range_right_boundary>
        ```

        Where:

        * {% include [logs output limit](../../_includes/cli/logs/limit.md) %}
        * `--columns`: List of log columns to draw data from.
            * `hostname`: [Host name](cluster-hosts.md).
            * `message`: Message output by the component.
            * `severity`: Logging level. Output example: `INFO`.
            * `origin`: Message origin. Output examples: `kafka_server` or `kafka_controller`.

        * {% include [logs filter](../../_includes/cli/logs/filter.md) %}
        * {% include [logs since time](../../_includes/cli/logs/since.md) %}
        * {% include [logs until time](../../_includes/cli/logs/until.md) %}

    You can request the cluster name and ID with the [list of clusters in the folder](cluster-list.md#list-clusters).

- REST API {#api}

    1. [Get an IAM token for API authentication](../api-ref/authentication.md) and put it into the environment variable:

        {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

    1. Use the [Cluster.listLogs](../api-ref/Cluster/listLogs.md) method and send the following request, e.g., via {{ api-examples.rest.tool }}:

        ```bash
        curl \
            --request GET \
            --header "Authorization: Bearer $IAM_TOKEN" \
            --url 'https://{{ api-host-mdb }}/managed-kafka/v1/clusters/<cluster_ID>:logs' \
            --url-query columnFilter=<list_of_data_columns> \
            --url-query fromTime=<time_range_left_boundary> \
            --url-query toTime=<time_range_right_boundary>
        ```

        Where:

        * `columnFilter`: List of data columns:

            {% include [column-filter-list](../../_includes/mdb/api/column-filter-list.md) %}

            {% include [column-filter-rest](../../_includes/mdb/api/column-filter-rest.md) %}

        {% include [from-time-rest](../../_includes/mdb/api/from-time-rest.md) %}

        * `toTime`: Right boundary of a time range, the format is the same as for `fromTime`.

        
        You can get the cluster ID with a [list of clusters in the folder](./cluster-list.md#list-clusters).


    1. View the [server response](../api-ref/Cluster/listLogs.md#yandex.cloud.mdb.kafka.v1.ListClusterLogsResponse) to make sure the request was successful.

- gRPC API {#grpc-api}

    1. [Get an IAM token for API authentication](../api-ref/authentication.md) and put it into the environment variable:

        {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

    1. {% include [grpc-api-setup-repo](../../_includes/mdb/grpc-api-setup-repo.md) %}

    1. Use the [ClusterService/ListLogs](../api-ref/grpc/Cluster/listLogs.md) call and send the following request, e.g., via {{ api-examples.grpc.tool }}:

        ```bash
        grpcurl \
            -format json \
            -import-path ~/cloudapi/ \
            -import-path ~/cloudapi/third_party/googleapis/ \
            -proto ~/cloudapi/yandex/cloud/mdb/kafka/v1/cluster_service.proto \
            -rpc-header "Authorization: Bearer $IAM_TOKEN" \
            -d '{
                    "cluster_id": "<cluster_ID>",
                    "column_filter": [<list_of_data_columns>],
                    "from_time": "<time_range_left_boundary>" \
                    "to_time": "<time_range_right_boundary>"
                }' \
            {{ api-host-mdb }}:{{ port-https }} \
            yandex.cloud.mdb.kafka.v1.ClusterService.ListLogs
        ```

        Where:

        * `service_type`: Type of the service to request logs for. The only valid value is `CLICKHOUSE`.
        * `column_filter`: List of data columns:

            {% include [column-filter-list](../../_includes/mdb/api/column-filter-list.md) %}

            {% include [column-filter-grpc](../../_includes/mdb/api/column-filter-grpc.md) %}

        {% include [from-time-grpc](../../_includes/mdb/api/from-time-grpc.md) %}

        * `to_time`: Right boundary of a time range, the format is the same as for `from_time`.

        
        You can get the cluster ID with a [list of clusters in the folder](./cluster-list.md#list-clusters).


    1. View the [server response](../api-ref/grpc/Cluster/listLogs.md#yandex.cloud.mdb.kafka.v1.ListClusterLogsResponse) to make sure the request was successful.

{% endlist %}

## Getting a log entry stream {#get-log-stream}

This method allows you to get cluster logs in real time.

{% list tabs group=instructions %}

- CLI {#cli}

    {% include [cli-install](../../_includes/cli-install.md) %}

    {% include [default-catalogue](../../_includes/default-catalogue.md) %}

    To view cluster logs as they become available, run this command:

    ```bash
    {{ yc-mdb-kf }} cluster list-logs <cluster_name_or_ID> --follow
    ```

    You can request the cluster name and ID with the [list of clusters in the folder](cluster-list.md#list-clusters).

- REST API {#api}

    1. [Get an IAM token for API authentication](../api-ref/authentication.md) and put it into the environment variable:

        {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

    1. Use the [Cluster.streamLogs](../api-ref/Cluster/streamLogs.md) method and send the following request, e.g., via {{ api-examples.rest.tool }}:

        ```bash
        curl \
            --request GET \
            --header "Authorization: Bearer $IAM_TOKEN" \
            --url 'https://{{ api-host-mdb }}/managed-kafka/v1/clusters/<cluster_ID>:stream_logs' \
            --url-query columnFilter=<list_of_data_columns> \
            --url-query fromTime=<time_range_left_boundary> \
            --url-query toTime=<time_range_right_boundary> \
            --url-query filter=<log_filter>
        ```

        Where:

        * `columnFilter`: List of data columns:

            {% include [column-filter-list](../../_includes/mdb/api/column-filter-list.md) %}

            {% include [column-filter-rest](../../_includes/mdb/api/column-filter-rest.md) %}

        {% include [from-time-rest](../../_includes/mdb/api/from-time-rest.md) %}

        * `toTime`: Right boundary of a time range, the format is the same as for `fromTime`.

            {% include [tail-f-semantics](../../_includes/mdb/api/tail-f-semantics.md) %}

        * `filter`: Log filter. You can filter logs so that the stream contains only the logs you need.

            For more information about filters and their syntax, see the [API reference](../api-ref/Cluster/streamLogs.md#query_params).

            {% include [stream-logs-filter](../../_includes/mdb/api/stream-logs-filter.md) %}

        
        You can get the cluster ID with a [list of clusters in the folder](./cluster-list.md#list-clusters).


    1. View the [server response](../api-ref/Cluster/streamLogs.md#yandex.cloud.mdb.kafka.v1.StreamLogRecord) to make sure the request was successful.

- gRPC API {#grpc-api}

    1. [Get an IAM token for API authentication](../api-ref/authentication.md) and put it into the environment variable:

        {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

    1. {% include [grpc-api-setup-repo](../../_includes/mdb/grpc-api-setup-repo.md) %}

    1. Use the [ClusterService/StreamLogs](../api-ref/grpc/Cluster/streamLogs.md) call and send the following request, e.g., via {{ api-examples.grpc.tool }}:

        ```bash
        grpcurl \
            -format json \
            -import-path ~/cloudapi/ \
            -import-path ~/cloudapi/third_party/googleapis/ \
            -proto ~/cloudapi/yandex/cloud/mdb/kafka/v1/cluster_service.proto \
            -rpc-header "Authorization: Bearer $IAM_TOKEN" \
            -d '{
                    "cluster_id": "<cluster_ID>",
                    "column_filter": [<list_of_data_columns>],
                    "from_time": "<time_range_left_boundary>",
                    "to_time": "<time_range_right_boundary>",
                    "filter": "<log_filter>"
                }' \
            {{ api-host-mdb }}:{{ port-https }} \
            yandex.cloud.mdb.kafka.v1.ClusterService.StreamLogs
        ```

        Where:

        * `column_filter`: List of data columns:

            {% include [column-filter-list](../../_includes/mdb/api/column-filter-list.md) %}

            {% include [column-filter-grpc](../../_includes/mdb/api/column-filter-grpc.md) %}

        {% include [from-time-grpc](../../_includes/mdb/api/from-time-grpc.md) %}

        * `to_time`: Right boundary of a time range, the format is the same as for `from_time`.

            {% include [tail-f-semantics](../../_includes/mdb/api/tail-f-semantics.md) %}

        * `filter`: Log filter. You can filter logs so that the stream contains only the logs you need.

            {% include [stream-logs-filter](../../_includes/mdb/api/stream-logs-filter.md) %}

            For more information about filters and their syntax, see the [API reference](../api-ref/grpc/Cluster/streamLogs.md).

        
        You can get the cluster ID with a [list of clusters in the folder](./cluster-list.md#list-clusters).


    1. View the [server response](../api-ref/grpc/Cluster/streamLogs.md#yandex.cloud.mdb.kafka.v1.StreamLogRecord) to make sure the request was successful.

{% endlist %}
