# Managing machine learning models in {{ mch-name }}

{{ mch-short-name }} allows you to analyze data by applying [CatBoost](https://catboost.ai/) machine learning models without additional tools.

To apply a model, add it to your cluster and call it in an SQL query using the built-in `catboostEvaluate()` function. After running this query, you get model predictions for each row of input data.

Read more about the `catboostEvaluate()` function in the [{{ CH }} documentation]({{ ch.docs }}/sql-reference/functions/other-functions/#catboostevaluatepath_to_model-feature_1-feature_2--feature_n).

## Before adding a model {#prereq}

{{ mch-short-name }} only works with readable models uploaded to {{ objstorage-full-name }}:


1. To link your [service account](../../iam/concepts/users/service-accounts.md) to the cluster, [make sure](../../iam/operations/roles/get-assigned-roles.md) your {{ yandex-cloud }} account has the [iam.serviceAccounts.user](../../iam/security/index.md#iam-serviceAccounts-user) role or higher.
1. [Upload](../../storage/operations/objects/upload.md) the trained model file to {{ objstorage-full-name }}.
1. [Connect the service account to the cluster](s3-access.md#connect-service-account). You will use your [service account](../../iam/concepts/users/service-accounts.md) to configure permissions to access the model file.
1. [Assign](s3-access.md#configure-acl) the `storage.viewer` role to the service account.
1. In the bucket's ACL, [add the `READ` permission](../../storage/operations/buckets/edit-acl.md) to the service account.
1. [Get a link](s3-access.md#get-link-to-object) to the model file.


## Getting a list of models in a cluster {#list}

{% list tabs group=instructions %}

- Management console {#console}

    1. In the [management console]({{ link-console-main }}), go to the folder page and select **{{ ui-key.yacloud.iam.folder.dashboard.label_managed-clickhouse }}**.
    1. Click the cluster name and select the **{{ ui-key.yacloud.clickhouse.cluster.switch_ml-models }}** tab in the left-hand panel.

- CLI {#cli}

    {% include [cli-install](../../_includes/cli-install.md) %}

    {% include [default-catalogue](../../_includes/default-catalogue.md) %}

    To get a list of models in a cluster, run the command:

    ```bash
    {{ yc-mdb-ch }} ml-model list --cluster-name=<cluster_name>
    ```

    You can request the cluster name with a [list of clusters in the folder](cluster-list.md#list-clusters).

- REST API {#api}

    1. [Get an IAM token for API authentication](../api-ref/authentication.md) and put it into the environment variable:

        {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

    1. Use the [MlModel.List](../api-ref/MlModel/list.md) method and send the following request, e.g., via {{ api-examples.rest.tool }}:

        ```bash
        curl \
            --request GET \
            --header "Authorization: Bearer $IAM_TOKEN" \
            --url 'https://{{ api-host-mdb }}/managed-clickhouse/v1/clusters/<cluster_ID>/mlModels'
        ```

        You can get the cluster ID with a [list of clusters in the folder](cluster-list.md#list-clusters).

    1. View the [server response](../api-ref/MlModel/list.md#yandex.cloud.mdb.clickhouse.v1.ListMlModelsResponse) to make sure the request was successful.

- gRPC API {#grpc-api}

    1. [Get an IAM token for API authentication](../api-ref/authentication.md) and put it into the environment variable:

        {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

    1. {% include [grpc-api-setup-repo](../../_includes/mdb/grpc-api-setup-repo.md) %}
    1. Use the [MlModelService.List](../api-ref/grpc/MlModel/list.md) call and send the following request, e.g., via {{ api-examples.grpc.tool }}:

        ```bash
        grpcurl \
            -format json \
            -import-path ~/cloudapi/ \
            -import-path ~/cloudapi/third_party/googleapis/ \
            -proto ~/cloudapi/yandex/cloud/mdb/clickhouse/v1/ml_model_service.proto \
            -rpc-header "Authorization: Bearer $IAM_TOKEN" \
            -d '{
                    "cluster_id": "<cluster_ID>"
                }' \
            {{ api-host-mdb }}:{{ port-https }} \
            yandex.cloud.mdb.clickhouse.v1.MlModelService.List
        ```

        You can get the cluster ID with a [list of clusters in the folder](cluster-list.md#list-clusters).

    1. View the [server response](../api-ref/grpc/MlModel/list.md#yandex.cloud.mdb.clickhouse.v1.ListMlModelsResponse) to make sure the request was successful.

{% endlist %}

## Getting detailed information about a model {#get}

{% list tabs group=instructions %}

- Management console {#console}

    1. In the [management console]({{ link-console-main }}), go to the folder page and select **{{ ui-key.yacloud.iam.folder.dashboard.label_managed-clickhouse }}**.
    1. Click the cluster name and select the **{{ ui-key.yacloud.clickhouse.cluster.switch_ml-models }}** tab in the left-hand panel.

- CLI {#cli}

    {% include [cli-install](../../_includes/cli-install.md) %}

    {% include [default-catalogue](../../_includes/default-catalogue.md) %}

    To get model details, run this command:

    ```bash
    {{ yc-mdb-ch }} ml-model get <model_name> \
      --cluster-name=<cluster_name>
    ```

    You can request the model name with a [list of cluster models](#list) and the cluster name with a [list of clusters in the folder](cluster-list.md#list-clusters).

- REST API {#api}

    1. [Get an IAM token for API authentication](../api-ref/authentication.md) and put it into the environment variable:

        {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

    1. Use the [MlModel.Get](../api-ref/MlModel/get.md) method and send the following request, e.g., via {{ api-examples.rest.tool }}:

        ```bash
        curl \
            --request GET \
            --header "Authorization: Bearer $IAM_TOKEN" \
            --url 'https://{{ api-host-mdb }}/managed-clickhouse/v1/clusters/<cluster_ID>/mlModels/<model_name>'
        ```

        You can request the cluster ID with a [list of clusters in the folder](cluster-list.md#list-clusters) and model name with a [list of models](#list) in the cluster.

    1. View the [server response](../api-ref/MlModel/get.md#yandex.cloud.mdb.clickhouse.v1.MlModel) to make sure the request was successful.

- gRPC API {#grpc-api}

    1. [Get an IAM token for API authentication](../api-ref/authentication.md) and put it into the environment variable:

        {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

    1. {% include [grpc-api-setup-repo](../../_includes/mdb/grpc-api-setup-repo.md) %}
    1. Use the [MlModelService.Get](../api-ref/grpc/MlModel/get.md) call and send the following request, e.g., via {{ api-examples.grpc.tool }}:

        ```bash
        grpcurl \
            -format json \
            -import-path ~/cloudapi/ \
            -import-path ~/cloudapi/third_party/googleapis/ \
            -proto ~/cloudapi/yandex/cloud/mdb/clickhouse/v1/ml_model_service.proto \
            -rpc-header "Authorization: Bearer $IAM_TOKEN" \
            -d '{
                    "cluster_id": "<cluster_ID>",
                    "ml_model_name": "<model_name>"
                }' \
            {{ api-host-mdb }}:{{ port-https }} \
            yandex.cloud.mdb.clickhouse.v1.MlModelService.Get
        ```

        You can request the cluster ID with a [list of clusters in the folder](cluster-list.md#list-clusters) and model name with a [list of models](#list) in the cluster.

    1. View the [server response](../api-ref/grpc/MlModel/get.md#yandex.cloud.mdb.clickhouse.v1.MlModel) to make sure the request was successful.

{% endlist %}

## Creating a model {#add}

{% note info %}

The only supported model type is CatBoost: `ML_MODEL_TYPE_CATBOOST`.

{% endnote %}

{% list tabs group=instructions %}

- Management console {#console}

    1. Select the cluster:

        1. In the [management console]({{ link-console-main }}), go to the folder page and select **{{ ui-key.yacloud.iam.folder.dashboard.label_managed-clickhouse }}**.
        1. Click the cluster name and select the **{{ ui-key.yacloud.clickhouse.cluster.switch_ml-models }}** tab in the left-hand panel.
        1. Click **{{ ui-key.yacloud.clickhouse.cluster.ml-models.button-action_add-ml-model }}**.

    1. Configure the model parameters:

        * **{{ ui-key.yacloud.clickhouse.cluster.ml-models.field_ml-model-type }}**: `ML_MODEL_TYPE_CATBOOST`.
        * **{{ ui-key.yacloud.clickhouse.cluster.ml-models.field_ml-model-name }}**: Model name. Model name is one of the arguments of the `catboostEvaluate()` function, which is used to call the model in {{ CH }}.
        * **{{ ui-key.yacloud.clickhouse.cluster.ml-models.field_ml-model-uri }}**: Model address in {{ objstorage-full-name }}.

    1. Click **{{ ui-key.yacloud.clickhouse.cluster.ml-models.label_add-ml-model }}** and wait for the model to be created.

- CLI {#cli}

    {% include [cli-install](../../_includes/cli-install.md) %}

    {% include [default-catalogue](../../_includes/default-catalogue.md) %}

    To create a model, run this command:

    ```bash
    {{ yc-mdb-ch }} ml-model create <model_name> \
      --cluster-name=<cluster_name> \
      --type=ML_MODEL_TYPE_CATBOOST \
      --uri=<link_to_model_file_in_Object_Storage>
    ```

    You can request the cluster name with a [list of clusters in the folder](cluster-list.md#list-clusters).

- {{ TF }} {#tf}

    1. Open the current {{ TF }} configuration file with an infrastructure plan.

        For more information about creating this file, see [Creating clusters](cluster-create.md).

    1. To the {{ mch-name }} cluster description, add the `ml_model` block with a description of the added machine learning model:

        ```hcl
        resource "yandex_mdb_clickhouse_cluster" "<cluster_name>" {
          ...
          ml_model {
            name = "<model_name>"
            type = "ML_MODEL_TYPE_CATBOOST"
            uri  = "<link_to_model_file_in_Object_Storage>"
          }
        }
        ```

    1. Make sure the settings are correct.

        {% include [terraform-validate](../../_includes/mdb/terraform/validate.md) %}

    1. Confirm updating the resources.

        {% include [terraform-apply](../../_includes/mdb/terraform/apply.md) %}

    For more information, see the [{{ TF }} provider documentation]({{ tf-provider-mch }}).

    {% include [Terraform timeouts](../../_includes/mdb/mch/terraform/timeouts.md) %}

- REST API {#api}

    1. [Get an IAM token for API authentication](../api-ref/authentication.md) and put it into the environment variable:

        {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

    1. Use the [MlModel.Create](../api-ref/MlModel/create.md) method and send the following request, e.g., via {{ api-examples.rest.tool }}:

        ```bash
        curl \
            --request POST \
            --header "Authorization: Bearer $IAM_TOKEN" \
            --header "Content-Type: application/json" \
            --url 'https://{{ api-host-mdb }/managed-clickhouse/v1/clusters/<cluster_ID>/mlModels' \
            --data '{
                      "mlModelName": "<model_name>",
                      "type": "ML_MODEL_TYPE_CATBOOST",
                      "uri": "<file_link>"
                    }'
        ```

        Where:

        * `mlModelName`: Model name.
        * `type`: Model type, always takes the `ML_MODEL_TYPE_CATBOOST` value.
        * `uri`: Link to the model file in {{ objstorage-name }}.

        You can get the cluster ID with a [list of clusters in the folder](cluster-list.md#list-clusters).

    1. View the [server response](../api-ref/MlModel/create.md#yandex.cloud.operation.Operation) to make sure the request was successful.

- gRPC API {#grpc-api}

    1. [Get an IAM token for API authentication](../api-ref/authentication.md) and put it into the environment variable:

        {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

    1. {% include [grpc-api-setup-repo](../../_includes/mdb/grpc-api-setup-repo.md) %}
    1. Use the [MlModelService.Create](../api-ref/grpc/MlModel/create.md) call and send the following request, e.g., via {{ api-examples.grpc.tool }}:

        ```bash
        grpcurl \
            -format json \
            -import-path ~/cloudapi/ \
            -import-path ~/cloudapi/third_party/googleapis/ \
            -proto ~/cloudapi/yandex/cloud/mdb/clickhouse/v1/ml_model_service.proto \
            -rpc-header "Authorization: Bearer $IAM_TOKEN" \
            -d '{
                    "cluster_id": "<cluster_ID>",
                    "ml_model_name": "<model_name>",
                    "type": "ML_MODEL_TYPE_CATBOOST",
                    "uri": "<file_link>"
                }' \
            {{ api-host-mdb }}:{{ port-https }} \
            yandex.cloud.mdb.clickhouse.v1.MlModelService.Create
        ```

        Where:

        * `ml_model_name`: Model name.
        * `type`: Model type, always takes the `ML_MODEL_TYPE_CATBOOST` value.
        * `uri`: Link to the model file in {{ objstorage-name }}.

        You can get the cluster ID with a [list of clusters in the folder](cluster-list.md#list-clusters).

    1. View the [server response](../api-ref/grpc/MlModel/create.md#yandex.cloud.operation.Operation) to make sure the request was successful.

{% endlist %}

## Applying a model {#apply}

To apply the model to data stored in a {{ CH }} cluster:

1. [Connect to the cluster](connect/clients.md).
1. Execute an SQL query in the format:

   ```sql
   SELECT 
       catboostEvaluate('<path_to_model_file>', 
                     <column_1_name>,
                     <column_2_name>,
                     ...
                     <column_N_name>)
   FROM <table_name>
   ```

As the `catboostEvaluate()` function arguments, specify the following:

   * Path to the model file in `/var/lib/clickhouse/models/<model_name>.bin` format.
   * Names of columns containing the input data.

The result of the query execution will be a column with model predictions for each row of the source table.

## Updating a model {#update}

{{ mch-name }} does not track changes in the model file located in the {{ objstorage-full-name }} bucket.

To update the contents of a model that is already connected to the cluster:


1. [Upload the file](../../storage/operations/objects/upload.md) with the current model to {{ objstorage-full-name }}.
1. [Get a link](s3-access.md#get-link-to-object) to this file.
1. Change the parameters of the model connected to {{ mch-name }} by providing a new link to the model file.


{% list tabs group=instructions %}

- Management console {#console}

    1. In the [management console]({{ link-console-main }}), go to the folder page and select **{{ ui-key.yacloud.iam.folder.dashboard.label_managed-clickhouse }}**.
    1. Click the cluster name and select the **{{ ui-key.yacloud.clickhouse.cluster.switch_ml-models }}** tab in the left-hand panel.
    1. Select the appropriate model, click ![image](../../_assets/console-icons/ellipsis-vertical.svg), and select **{{ ui-key.yacloud.clickhouse.cluster.ml-models.button_action-edit-ml-model }}**.

- CLI {#cli}

    {% include [cli-install](../../_includes/cli-install.md) %}

    {% include [default-catalogue](../../_includes/default-catalogue.md) %}

    To change the link to the model file in the {{ objstorage-full-name }} bucket, run the command:

    ```bash
    {{ yc-mdb-ch }} ml-model update <model_name> \
      --cluster-name=<cluster_name> \
      --uri=<new_link_to_file_in_Object_Storage>
    ```

    You can request the model name with a [list of cluster models](#list) and the cluster name with a [list of clusters in the folder](cluster-list.md#list-clusters).

- {{ TF }} {#tf}

    1. Open the current {{ TF }} configuration file with an infrastructure plan.

        For more information about creating this file, see [Creating clusters](cluster-create.md).

    1. In the {{ mch-name }} cluster description, change the `uri` parameter value under `ml_model`:

        ```hcl
        resource "yandex_mdb_clickhouse_cluster" "<cluster_name>" {
        ...
          ml_model {
            name = "<model_name>"
            type = "ML_MODEL_TYPE_CATBOOST"
            uri  = "<new_link_to_model_file_in_Object_Storage>"
          }
        }
        ```

    1. Make sure the settings are correct.

        {% include [terraform-validate](../../_includes/mdb/terraform/validate.md) %}

    1. Confirm updating the resources.

        {% include [terraform-apply](../../_includes/mdb/terraform/apply.md) %}

    For more information, see the [{{ TF }} provider documentation]({{ tf-provider-mch }}).

    {% include [Terraform timeouts](../../_includes/mdb/mch/terraform/timeouts.md) %}

- REST API {#api}

    1. [Get an IAM token for API authentication](../api-ref/authentication.md) and put it into the environment variable:

        {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

    1. Use the [MlModel.Update](../api-ref/MlModel/update.md) method and send the following request, e.g., via {{ api-examples.rest.tool }}:

        {% include [note-updatemask](../../_includes/note-api-updatemask.md) %}

        ```bash
        curl \
            --request PATCH \
            --header "Authorization: Bearer $IAM_TOKEN" \
            --header "Content-Type: application/json" \
            --url 'https://{{ api-host-mdb }/managed-clickhouse/v1/clusters/<cluster_ID>/mlModels/<model_name>' \
            --data '{
                      "updateMask": "uri",
                      "uri": "<file_link>"
                    }'
        ```

        Where:

        * `updateMask`: List of parameters to update as a single string, separated by commas.

            Here only one parameter is specified: `uri`.

        * `uri`: Link to the new model file in {{ objstorage-name }}.

        You can get the cluster ID with a [list of clusters in the folder](cluster-list.md#list-clusters).

    1. View the [server response](../api-ref/MlModel/update.md#yandex.cloud.operation.Operation) to make sure the request was successful.

- gRPC API {#grpc-api}

    1. [Get an IAM token for API authentication](../api-ref/authentication.md) and put it into the environment variable:

        {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

    1. {% include [grpc-api-setup-repo](../../_includes/mdb/grpc-api-setup-repo.md) %}
    1. Use the [MlModelService.Update](../api-ref/grpc/MlModel/update.md) call and send the following request, e.g., via {{ api-examples.grpc.tool }}:

        {% include [note-grpc-updatemask](../../_includes/note-grpc-api-updatemask.md) %}

        ```bash
        grpcurl \
            -format json \
            -import-path ~/cloudapi/ \
            -import-path ~/cloudapi/third_party/googleapis/ \
            -proto ~/cloudapi/yandex/cloud/mdb/clickhouse/v1/ml_model_service.proto \
            -rpc-header "Authorization: Bearer $IAM_TOKEN" \
            -d '{
                    "cluster_id": "<cluster_ID>",
                    "ml_model_name": "<schema_name>",
                    "update_mask": {
                      "paths": ["uri"]
                    },
                    "uri": "<file_link>"
                }' \
            {{ api-host-mdb }}:{{ port-https }} \
            yandex.cloud.mdb.clickhouse.v1.MlModelService.Create
        ```

        Where:

        * `ml_model_name`: Model name.
        * `update_mask`: List of parameters to update as an array of `paths[]` strings.

            Here only one parameter is specified: `uri`.

        * `uri`: Link to the new model file in {{ objstorage-name }}.

        You can get the cluster ID with a [list of clusters in the folder](cluster-list.md#list-clusters).

    1. View the [server response](../api-ref/grpc/MlModel/update.md#yandex.cloud.operation.Operation) to make sure the request was successful.

{% endlist %}

## Disabling a model {#disable}

{% note info %}


After disabling a model, the corresponding object is kept in the {{ objstorage-full-name }} bucket. If you no longer need this model object, you can [delete](../../storage/operations/objects/delete.md) it.


{% endnote %}

{% list tabs group=instructions %}

- Management console {#console}

    1. In the [management console]({{ link-console-main }}), go to the folder page and select **{{ ui-key.yacloud.iam.folder.dashboard.label_managed-clickhouse }}**.
    1. Click the cluster name and select the **{{ ui-key.yacloud.clickhouse.cluster.switch_ml-models }}** tab in the left-hand panel.
    1. Select the appropriate model, click ![image](../../_assets/console-icons/ellipsis-vertical.svg), and select **{{ ui-key.yacloud.mdb.clusters.button_action-delete }}**.

- CLI {#cli}

    {% include [cli-install](../../_includes/cli-install.md) %}

    {% include [default-catalogue](../../_includes/default-catalogue.md) %}

    To disable a model, run the command:

    ```bash
    {{ yc-mdb-ch }} ml-model delete <model_name> \
      --cluster-name=<cluster_name>
    ```

    You can request the model name with a [list of cluster models](#list) and the cluster name with a [list of clusters in the folder](cluster-list.md#list-clusters).

- {{ TF }} {#tf}

    1. Open the current {{ TF }} configuration file with an infrastructure plan.

        For more information about creating this file, see [Creating clusters](cluster-create.md).

    1. Delete the description block of the appropriate `ml_model` model from the {{ mch-name }} cluster description.

    1. Make sure the settings are correct.

        {% include [terraform-validate](../../_includes/mdb/terraform/validate.md) %}

    1. Confirm updating the resources.

        {% include [terraform-apply](../../_includes/mdb/terraform/apply.md) %}

    For more information, see the [{{ TF }} provider documentation]({{ tf-provider-mch }}).

    {% include [Terraform timeouts](../../_includes/mdb/mch/terraform/timeouts.md) %}

- REST API {#api}

    1. [Get an IAM token for API authentication](../api-ref/authentication.md) and put it into the environment variable:

        {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

    1. Use the [MlModel.Delete](../api-ref/MlModel/delete.md) method and send the following request, e.g., via {{ api-examples.rest.tool }}:

        ```bash
        curl \
            --request DELETE \
            --header "Authorization: Bearer $IAM_TOKEN" \
            --url 'https://{{ api-host-mdb }}/managed-clickhouse/v1/clusters/<cluster_ID>/mlModels/<model_name>'
        ```

        You can request the cluster ID with a [list of clusters in the folder](cluster-list.md#list-clusters) and model name with a [list of models](#list) in the cluster.

    1. View the [server response](../api-ref/MlModel/delete.md#yandex.cloud.operation.Operation) to make sure the request was successful.

- gRPC API {#grpc-api}

    1. [Get an IAM token for API authentication](../api-ref/authentication.md) and put it into the environment variable:

        {% include [api-auth-token](../../_includes/mdb/api-auth-token.md) %}

    1. {% include [grpc-api-setup-repo](../../_includes/mdb/grpc-api-setup-repo.md) %}
    1. Use the [MlModelService.Delete](../api-ref/grpc/MlModel/delete.md) call and send the following request, e.g., via {{ api-examples.grpc.tool }}:

        ```bash
        grpcurl \
            -format json \
            -import-path ~/cloudapi/ \
            -import-path ~/cloudapi/third_party/googleapis/ \
            -proto ~/cloudapi/yandex/cloud/mdb/clickhouse/v1/ml_model_service.proto \
            -rpc-header "Authorization: Bearer $IAM_TOKEN" \
            -d '{
                    "cluster_id": "<cluster_ID>",
                    "ml_model_name": "<schema_name>"
                }' \
            {{ api-host-mdb }}:{{ port-https }} \
            yandex.cloud.mdb.clickhouse.v1.MlModelService.Delete
        ```

        You can get the cluster ID with a [list of clusters in the folder](cluster-list.md#list-clusters).

    1. View the [server response](../api-ref/grpc/MlModel/delete.md#yandex.cloud.operation.Operation) to make sure the request was successful.

{% endlist %}

## Example {#example-ml-model}

If you do not have a suitable data set or model to process it, you can test machine learning in {{ mch-short-name }} using this example. We prepared a data file for it and trained a model to analyze it. You can upload data to {{ CH }} and see model predictions for different rows of the table.

{% note info %}

In this example, we are going to use public data from the [Amazon Employee Access Challenge](https://www.kaggle.com/c/amazon-employee-access-challenge). The model is trained to predict values in the `ACTION` column. The same data and model are used on [GitHub](https://github.com/ClickHouse/clickhouse-presentations/blob/master/tutorials/catboost_with_clickhouse_ru.md).

{% endnote %}

To upload data to {{ CH }} and test the model:

1. In the [management console]({{ link-console-main }}), add the test model:

    * **{{ ui-key.yacloud.clickhouse.cluster.ml-models.field_ml-model-type }}**: `ML_MODEL_TYPE_CATBOOST`.
    * **{{ ui-key.yacloud.clickhouse.cluster.ml-models.field_ml-model-name }}**: `ml_test`.
    * **{{ ui-key.yacloud.clickhouse.cluster.ml-models.field_ml-model-uri }}**: `https://{{ s3-storage-host-mch }}/catboost_model.bin`.


1. [Download the file with data](https://{{ s3-storage-host }}/doc-files/managed-clickhouse/train.csv) to analyze.


1. [Connect to the cluster](connect/clients.md).

1. Create a test table:

    ```sql
    CREATE TABLE
                ml_test_table (date Date MATERIALIZED today(), 
                              ACTION UInt8, 
                              RESOURCE UInt32, 
                              MGR_ID UInt32, 
                              ROLE_ROLLUP_1 UInt32, 
                              ROLE_ROLLUP_2 UInt32, 
                              ROLE_DEPTNAME UInt32, 
                              ROLE_TITLE UInt32, 
                              ROLE_FAMILY_DESC UInt32, 
                              ROLE_FAMILY UInt32, 
                              ROLE_CODE UInt32) 
                ENGINE = MergeTree() 
    PARTITION BY date 
    ORDER BY date;
    ```

1. Upload the data to the table:

    ```sql
    INSERT INTO ml_test_table FROM INFILE '<path_to_file>/train.csv' FORMAT CSVWithNames;
    ```

1. Test the model:

    * Get predicted values in the `ACTION` column for the first 10 rows in the table:

        ```sql
        SELECT
            catboostEvaluate('/var/lib/clickhouse/models/ml_test.bin',
                            RESOURCE,
                            MGR_ID,
                            ROLE_ROLLUP_1,
                            ROLE_ROLLUP_2,
                            ROLE_DEPTNAME,
                            ROLE_TITLE,
                            ROLE_FAMILY_DESC,
                            ROLE_FAMILY,
                            ROLE_CODE) > 0 AS prediction,
            ACTION AS target
        FROM ml_test_table
        LIMIT 10;
        ```

    * Get the probability prediction for the first 10 rows in the table:

        ```sql
        SELECT
            catboostEvaluate('/var/lib/clickhouse/models/ml_test.bin',
                            RESOURCE,
                            MGR_ID,
                            ROLE_ROLLUP_1,
                            ROLE_ROLLUP_2,
                            ROLE_DEPTNAME,
                            ROLE_TITLE,
                            ROLE_FAMILY_DESC,
                            ROLE_FAMILY,
                            ROLE_CODE) AS prediction,
            1. / (1 + exp(-prediction)) AS probability,
            ACTION AS target
        FROM ml_test_table
        LIMIT 10;
        ```

{% include [clickhouse-disclaimer](../../_includes/clickhouse-disclaimer.md) %}
