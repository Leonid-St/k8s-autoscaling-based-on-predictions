---
title: Create function
description: Follow this guide to create a function.
---

# Creating a function

{% list tabs group=instructions %}

- Management console {#console}

	1. In the [management console]({{ link-console-main }}), select the folder where you want to create a function.
	1. Select **{{ ui-key.yacloud.iam.folder.dashboard.label_serverless-functions }}**.
	1. Click **{{ ui-key.yacloud.serverless-functions.list.button_create }}**.
	1. Enter a name and description for the function. The name format is as follows:

		{% include [name-format](../../../_includes/name-format.md) %}

    1. Click **{{ ui-key.yacloud.common.create }}**.

- CLI {#cli}

    {% include [cli-install](../../../_includes/cli-install.md) %}

    {% include [default-catalogue](../../../_includes/default-catalogue.md) %}

    To create a function, run the command:

    ```
    yc serverless function create --name=<function_name>
    ```

    Result:

    ```
    id: b09bhaokchn9********
    folder_id: aoek49ghmknn********
    created_at: "2019-06-14T10:03:37.475Z"
    name: python_function
    log_group_id: eolm8aoq9vcp********
    http_invoke_url: https://{{ sf-url }}/b09bhaokchn9********
    status: ACTIVE
    ```

- {{ TF }} {#tf}

    {% include [terraform-definition](../../../_tutorials/_tutorials_includes/terraform-definition.md) %}

    {% include [terraform-install](../../../_includes/terraform-install.md) %}

    To create a function:

    1. In the configuration file, define the parameters of the resources you want to create:

       * `yandex_function`: Description of the new function and its source code:
         * `name`: Function name.
         * `folder_id`: Folder ID.
         * `description`: Text description of the function.
         * `labels`: Function labels in `key:value` format.
         * `user_hash`: Any string to identify the function version. When the function changes, update this string, too. The function will update when this string is updated.
         * `runtime`: Function [runtime environment](../../concepts/runtime/index.md).
         * `entrypoint`: Function name in the source code that will serve as an entry point to applications.
         * `memory`: Amount of memory allocated for the function, in MB.
         * `execution_timeout`: Function execution timeout.
         * `service_account_id`: ID of the service account to call the function under.
         * `environment`: Environment variables in `key:value` format.
         * `tags`: Function tags.
         * `version`: Function version.
         * `image_size`: Size of the image for the function.
         * `loggroup_id`: ID of the log group for the function.
         * `package`: Package with the function version source code. You can only use either the `package` or the `content` field.
         * `package.0.sha_256`: SHA256 hash of the package.
         * `package.0.bucket_name`: Name of the {{ objstorage-name }} bucket that stores the function version source code.
         * `package.0.object_name`: Name of the {{ objstorage-name }} object containing the function version source code.
         * `content`: Function source code. You can only use either the `content` or the `package` field.
         * `content.0.zip_filename`: Name of the ZIP archive containing the function source code.

        Here is an example of the configuration file structure:

        ```
        provider "yandex" {
            token     = "<service_account_OAuth_token_or_static_key>"
            folder_id = "<folder_ID>"
            zone      = "{{ region-id }}-a"
        }
             
        resource "yandex_function" "test-function" {
            name               = "test-function"
            description        = "Test function"
            user_hash          = "first-function"
            runtime            = "python37"
            entrypoint         = "main"
            memory             = "128"
            execution_timeout  = "10"
            service_account_id = "<service_account_ID>"
            tags               = ["my_tag"]
            content {
                zip_filename = "<path_to_ZIP_archive>"
            }
        }

        output "yandex_function_test-function" {
            value = "${yandex_function.test-function.id}"
        }
        ```

        For more information about the `yandex_function` resource properties, see the [provider documentation]({{ tf-provider-resources-link }}/function).

    1. Check the configuration using this command:
        
       ```
       terraform validate
       ```

       If the configuration is correct, you will get this message:
        
       ```
       Success! The configuration is valid.
       ```

    1. Run this command:

       ```
       terraform plan
       ```
        
       The terminal will display a list of resources with their parameters. No changes will be made at this step. If the configuration contains any errors, {{ TF }} will point them out.
         
    1. Apply the configuration changes:

       ```
       terraform apply
       ```
    1. Confirm the changes: type `yes` into the terminal and press **Enter**.
      
       You can check the new resources and their settings using the [management console]({{ link-console-main }}) or this [CLI](../../../cli/quickstart.md) command:

       ```
       yc serverless function list
       ```

- API {#api}

    To create a function, use the [create](../../functions/api-ref/Function/create.md) REST API method for the [Function](../../functions/api-ref/Function/index.md) resource or the [FunctionService/Create](../../functions/api-ref/grpc/Function/create.md) gRPC API call.

- {{ yandex-cloud }} Toolkit {#yc-toolkit}

    You can create a list of function versions using the [{{ yandex-cloud }} Toolkit plugin](https://github.com/yandex-cloud/ide-plugin-jetbrains/blob/master/README.en.md) for the IDE family on the [JetBrains](https://www.jetbrains.com/) [IntelliJ platform](https://www.jetbrains.com/opensource/idea/).

{% endlist %}
