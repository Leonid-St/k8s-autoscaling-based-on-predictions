---
title: Cloud setup guide for individuals
description: In this tutorial, you will learn how to set up a cloud as an individual.
---

# Configure your cloud

When a user registers with {{ yandex-cloud }}, a _cloud_ is created for the user. The cloud is a separate workspace with this user as the owner. In this cloud, the `default` folder and `default` network will be created.

The owner can create new folders and resources in this cloud, and manage access rights to them.

## Create a folder

{% list tabs group=instructions %}

- Management console {#console}

  {% include [create-folder](../../_includes/create-folder.md) %}

- CLI {#cli}

  1. View the description of the create folder command:

      ```
      $ yc resource-manager folder create --help
      ```

  1. Create a new folder:

      * with a name and without a description:
          ```
          $ yc resource-manager folder create \
              --name new-folder
          ```

          The folder naming requirements are as follows:

          {% include [name-format](../../_includes/name-format.md) %}

      * with a name and description:

          ```
          $ yc resource-manager folder create \
              --name new-folder \
              --description "my first folder with description"
          ```

- API {#api}

  To create a folder, use the [create](../../resource-manager/api-ref/Folder/create.md) method for the [Folder](../../resource-manager/api-ref/Folder/index.md).

{% endlist %}

## Update a folder {#change-folder}

The management console only allows you to change the name of a folder. To change its description, use the CLI or API.

{% list tabs group=instructions %}

- Management console {#console}

  1. On the management console [home page]({{ link-console-main }}), select the folder. This page displays folders for the selected cloud. You can [switch to another cloud](../../resource-manager/operations/cloud/switch-cloud.md), if required.
  1. Click ![image](../../_assets/console-icons/ellipsis.svg) next to the folder and select **{{ ui-key.yacloud.common.edit }}**.
  1. Enter a new name for the folder.
  1. Click **{{ ui-key.yacloud.iam.cloud.folders.popup-edit_button_save }}**.

- CLI {#cli}

  1. View the description of the update folder command:

      ```
      $ yc resource-manager folder update --help
      ```
  1. If you know the folder ID or name, proceed to the next step. Otherwise, use one of these methods to get them:

      * Get a list of folders:

          ```
          $ yc resource-manager folder list
          +----------------------+--------------------+--------+--------+-------------+
          |          ID          |        NAME        | LABELS | STATUS | DESCRIPTION |
          +----------------------+--------------------+--------+--------+-------------+
          | b1gppulhhm2aaufq9eug | yet-another-folder |        | ACTIVE |             |
          | b1gvmob95yysaplct532 | default            |        | ACTIVE |             |
          +----------------------+--------------------+--------+--------+-------------+
          ```

      * If you know the ID of the resource that belongs to the required folder, you can get the folder ID from the information about that resource:

          ```
          $ yc <SERVICE-NAME> <RESOURCE> get <RESOURCE-ID>
          ```

          Where:
          * `<SERVICE-NAME>`: Service name, e.g., `compute`.
          * `<RESOURCE>`: Resource category, e.g., `instance`.
          * `<RESOURCE-ID>`: Resource ID.

          For example, the `fhmp74bfis2aim728p2a` VM belongs to the `b1gpvjd9ir42nsng55ck` folder:

          ```
          $ yc compute instance get fhmp74bfis2ais728p2a
          id: fhmp74bfis2ais728p2a
          folder_id: b1gpvjd9ia42nsng55ck
          ...
          ```
  1. Change the folder parameters, e.g., name and description. You can specify the folder to update by its name or ID.

      ```
      $ yc resource-manager folder update default \
          --new-name myfolder \
          --description "this is my default-folder"
      ```

      The command will rename the `default` folder to `myfolder` and update its description.

      The folder naming requirements are as follows:

      {% include [name-format](../../_includes/name-format.md) %}



- API {#api}

  To edit the folder, use the [update](../../resource-manager/api-ref/Folder/update.md) method for the [Folder](../../resource-manager/api-ref/Folder/index.md) resource.
  
{% endlist %}