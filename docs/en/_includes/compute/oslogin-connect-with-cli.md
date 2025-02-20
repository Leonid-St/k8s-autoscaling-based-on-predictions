To connect to a VM via {{ oslogin }} with an SSH certificate using the YC CLI:

1. {% include [oslogin-connect-cert-enable-in-org](../../_includes/compute/oslogin-connect-cert-enable-in-org.md) %}
1. View the description of the CLI command to connect to a VM:

    ```bash
    yc compute ssh --help
    ```
1. {% include [os-login-cli-organization-list](../../_includes/organization/os-login-cli-organization-list.md) %}
1. {% include [os-login-cli-profile-list](../../_includes/organization/os-login-cli-profile-list.md) %}
1. {% include [oslogin-connect-instr-list-vms](../../_includes/compute/oslogin-connect-instr-list-vms.md) %}
1. Connect to the VM:

    ```bash
    yc compute ssh \
      --name <VM_name>
      --login <user_or_service_account_login>
      --internal-address
    ```

    Where:
    * `--name`: Previously obtained VM name. You can specify the VM ID instead of its name by using the `--id` parameter.
    * `--login`: Previously obtained user or service account login, as set in the {{ oslogin }} profile. This is an optional parameter. If this parameter is not specified, the connection will use the SSH certificate of the user or service account currently authorized in the YC CLI profile.
    * (Optional) `--internal-address`: To connect using an internal IP address.

    You can also see the command for VM connection in the [management console]({{ link-console-main }}). On the **{{ ui-key.yacloud.compute.instance.overview.label_title }}** page of the VM, under **Connect to VM**, expand the **Connect via the {{ yandex-cloud }} CLI interface** section and select the **Certificate** tab.