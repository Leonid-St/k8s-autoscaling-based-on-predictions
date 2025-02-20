# Migrating services from a NLB with target resources from VMs to an L7 ALB


This tutorial covers a situation where a [{{ network-load-balancer-full-name }}](../../network-load-balancer/)'s incoming traffic is distributed among VMs located in target groups downstream of the load balancer.

To migrate a service from a network load balancer to an L7 load balancer:

1. [See recommendations for service migration](#recommendations).
1. [Complete the prerequisite steps](#before-you-begin).
1. [Create a {{ sws-name }} profile](#create-profile-sws).
1. [Create an L7 load balancer](#create-alb). At this step, you will connect your {{ sws-name }} profile to a virtual host of the L7 load balancer.
1. [Migrate user load from the network load balancer to the L7 load balancer](#migration-nlb-to-alb).

## Service migration recommendations {#recommendations}

1. In addition to DDoS protection at level L7 of the OSI model using [{{ sws-full-name }}](../../smartwebsecurity/), we recommend enabling DDoS protection at L3-L4. To do this, [reserve a public static IP address with DDoS protection](../../vpc/operations/enable-ddos-protection.md#enable-on-reservation) in advance and use this address for the L7 load balancer's listener.

    If the network load balancer's listener already uses a public IP address with DDoS protection, you can save it and use it for the L7 load balancer.

    If the network load balancer's listener uses a public IP address without DDoS protection, the only way to enable L7 load balancer DDoS protection at level L3-L4 is to change the public IP address for your service.

    When using L3-L4 DDoS protection, configure a trigger threshold for the L3-L4 protection mechanisms aligned with the amount of legitimate traffic to the protected resource. To set up this threshold, contact [support]({{ link-console-support }}).

    Also, set the MTU value to `1450` for the target resources downstream of the load balancer. For more information, see [Setting up MTU when enabling DDoS protection](../../vpc/operations/adjust-mtu-ddos-protection.md).

1. We recommend performing migration during the hours when user load is at its lowest. If you plan to keep your public IP address, bear in mind that migration involves moving this IP address from the load balancer to the L7 load balancer. Your service will be unavailable during this period. Under normal conditions, this may last for several minutes.

1. When using an L7 load balancer, requests to backends come with the source IP address from the range of internal IP addresses of the subnets specified when creating the L7 load balancer. The original IP address of the request source (user) is specified in the `X-Forwarded-For` header. If you want to log public IP addresses of users on the web server, reconfigure it.

1. See the [autoscaling and resource units](../../application-load-balancer/concepts/application-load-balancer.md#lcu-scaling) in the L7 load balancer.

## Getting started {#before-you-begin}

1. [Create subnets](../../vpc/operations/subnet-create.md) in three availability zones. These will be used for the L7 load balancer.

1. Create [security groups](../../application-load-balancer/concepts/application-load-balancer.md#security-groups) that allow the L7 load balancer to receive incoming traffic and send it to the target resources, and allow the target resources to receive incoming traffic from the load balancer.

1. When using HTTPS, [add your service's TLS certificate](../../certificate-manager/operations/import/cert-create.md#create-certificate) to [{{ certificate-manager-full-name }}](../../certificate-manager/).

1. [Reserve a static public IP address with DDoS protection](../../vpc/operations/get-static-ip.md) at level L3-L4 for the L7 load balancer. See [service migration recommendations](#recommendations).

## Create a {{ sws-name }} security profile {#create-profile-sws}

[Create](../../smartwebsecurity/operations/profile-create.md) a {{ sws-name }} security profile by selecting **{{ ui-key.yacloud.smart-web-security.title_default-template }}**.

Use these settings when creating the profile:

* In the **{{ ui-key.yacloud.smart-web-security.form.label_default-action }}** field, select `{{ ui-key.yacloud.smart-web-security.form.label_action-allow }}`.
* For the **{{ ui-key.yacloud.smart-web-security.overview.label_smart-protection-rule }}** rule, enable **{{ ui-key.yacloud.smart-web-security.overview.column_dry-run-rule }} (dry run)**.

These settings are limited to logging the info about the traffic without applying any actions to it. This will reduce the risk of disconnecting users due to profile configuration issues. As you move along, you will be able to turn **{{ ui-key.yacloud.smart-web-security.overview.column_dry-run-rule }} (dry run)** off and configure some prohibiting rules for your use case in the security profile.

## Create an L7 load balancer {#create-alb}

1. [Create a target group](../../application-load-balancer/operations/target-group-create.md) for the L7 load balancer. Under **{{ ui-key.yacloud.alb.label_targets }}**, select the VMs in your network load balancer's target group.

1. [Create a group of backends](../../application-load-balancer/operations/backend-group-create.md) with the following parameters:

    1. Select `{{ ui-key.yacloud.alb.label_proto-http }}` as the backend group type.
    1. If your service requires requests to be processed within a single user session by the same backend resource, enable [session affinity](../../application-load-balancer/concepts/backend-group.md#session-affinity) for the backend group.
    1. Under **{{ ui-key.yacloud.alb.label_backends }}**, click **{{ ui-key.yacloud.common.add }}** and set up the backend:

        * **{{ ui-key.yacloud.common.type }}**: `{{ ui-key.yacloud.alb.label_target-group }}`.
        * **{{ ui-key.yacloud.alb.label_target-groups }}**: Target group you created earlier.
        * **{{ ui-key.yacloud.alb.label_port }}**: Your service's TCP port the VMs are accepting incoming traffic on.
        * Under **{{ ui-key.yacloud.alb.label_protocol-settings }}**, select a protocol, `{{ ui-key.yacloud.alb.label_proto-http-plain }}` or `{{ ui-key.yacloud.alb.label_proto-http-tls }}`, based on your service.
        * Under **HTTP health check**, configure the health check using [these recommendations](../../application-load-balancer/concepts/best-practices.md).
        * (Optional) Set other settings as per [this guide](../../application-load-balancer/operations/backend-group-create.md).

1. [Create an HTTP router](../../application-load-balancer/operations/http-router-create.md). Under **{{ ui-key.yacloud.alb.label_virtual-hosts }}**, click **{{ ui-key.yacloud.alb.button_virtual-host-add }}** and specify the virtual host settings:

    * **{{ ui-key.yacloud.alb.label_authority }}**: Your service domain name.
    * **{{ ui-key.yacloud.alb.label_security-profile-id }}**: {{ sws-name }} profile you created earlier.

        {% note warning %}

        Linking your security profile to a virtual host of the L7 load balancer is the key step to connecting {{ sws-name }}.

        {% endnote %}

    * Click **{{ ui-key.yacloud.alb.button_add-route }}** and specify the route settings:

        * **{{ ui-key.yacloud.alb.label_path }}**: `Starts with ` `/`.
        * **{{ ui-key.yacloud.alb.label_route-action }}**: `{{ ui-key.yacloud.alb.label_route-action-route }}`.
        * **{{ ui-key.yacloud.alb.label_backend-group }}**: Backend group you created earlier.

1. [Create an L7 load balancer](../../application-load-balancer/operations/application-load-balancer-create.md) by selecting **{{ ui-key.yacloud.alb.label_alb-create-form }}**:

    * Specify the previously created security group.
    * Under **{{ ui-key.yacloud.alb.section_allocation-settings }}**, select the subnets in three availability zones for the load balancer nodes. Enable traffic in these subnets.
    * Under **{{ ui-key.yacloud.alb.section_autoscale-settings }}**, specify the [minimum number of resource units](../../application-load-balancer/concepts/application-load-balancer.md#lcu-scaling-settings) per availability zone based on expected load.

        We recommend selecting the number of resource units based on load expressed in:

        * Number of requests per second (RPS)
        * Number of concurrent active connections
        * Number of new connections per second
        * Traffic processed per second

    * Under **{{ ui-key.yacloud.alb.label_listeners }}**, click **{{ ui-key.yacloud.alb.button_add-listener }}** and set up the listener:

        * Under **{{ ui-key.yacloud.alb.section_external-address-specs }}**, specify:

            * **{{ ui-key.yacloud.alb.label_port }}**: Your service's TCP port the VMs are accepting incoming traffic on.
            * **{{ ui-key.yacloud.common.type }}**: `{{ ui-key.yacloud.alb.label_address-list }}`. Select from the list a public IP address with DDoS protection at L3-L4. For more information, see [service migration recommendations](#recommendations).
        * Under **{{ ui-key.yacloud.alb.section_common-address-specs }}**, specify:

            * **{{ ui-key.yacloud.alb.label_listener-type }}**: `{{ ui-key.yacloud.alb.label_listener-type-http }}`.
            * **{{ ui-key.yacloud.alb.label_protocol-type }}**: Depending on your service, select `{{ ui-key.yacloud.alb.label_proto-http-plain }}` or `{{ ui-key.yacloud.alb.label_proto-http-tls }}`.
            * If you select `{{ ui-key.yacloud.alb.label_proto-http-tls }}`, specify the TLS certificate you added to {{ certificate-manager-name }} earlier in the **{{ ui-key.yacloud.alb.label_certificate }}** field.
            * **{{ ui-key.yacloud.alb.label_http-router }}**: HTTP router you created earlier.

1. Wait until the L7 load balancer goes `Active`.

1. Go to the new L7 load balancer and select **{{ ui-key.yacloud.alb.label_healthchecks }}** on the left. Make sure you get `HEALTHY` for all the L7 load balancer's health checks.

1. Run a test request to the service through the L7 load balancer, for example, using one of these methods:

    * Add this record to the `hosts` file on your workstation: `<L7_load_balancer_public_IP_address> <service_domain_name>`. Delete the record after the test.
    * Execute the request using {{ api-examples.rest.tool }} depending on the protocol type:

        ```bash
        curl http://<service_domain_name> \
            --resolve <service_domain_name>:<service_port>:<public_IP_address_of_L7_load_balancer>
        ```

        ```bash
        curl https://<service_domain_name> \
            --resolve <service_domain_name>:<service_port>:<public_IP_address_of_L7_load_balancer>
        ```

## Migrate user load from the network load balancer to the L7 load balancer {#migration-nlb-to-alb}

Select one of the migration options:

* [Keep the public IP address for your service](#save-public-ip).
* [Do not keep public IP address for your service.](#not-save-public-ip)

### Keep the public IP address for your service {#save-public-ip}

1. If your external network load balancer uses a dynamic public IP address, [convert it to static](../../vpc/operations/set-static-ip.md).

1. [Delete the listener](../../network-load-balancer/operations/listener-remove.md) in the network load balancer to release the static public IP address. This will make your service unavailable through the network load balancer.

1. In the L7 load balancer, assign to the listener the public IP address previously assigned to the network load balancer:

    {% list tabs group=instructions %}

    * CLI {#cli}

        {% include [include](../../_includes/cli-install.md) %}

        {% include [default-catalogue](../../_includes/default-catalogue.md) %}

        To change a public IP address, run this command:

        ```bash
        yc application-load-balancer load-balancer update-listener <load_balancer_name> \
           --listener-name <listener_name> \
           --external-ipv4-endpoint address=<service_public_IP_address>,port=<service_port>
        ```

        Where `address` is the public IP address previously assigned to the network load balancer.

    * {{ TF }} {#tf}

        1. Open the current {{ TF }} configuration file with an infrastructure plan.

            For how to create this file, see [Creating an L7 load balancer](../../application-load-balancer/operations/application-load-balancer-create.md).

            For more information about the `yandex_alb_load_balancer` resource parameters in {{ TF }}, see the [provider documentation]({{ tf-provider-resources-link }}/alb_load_balancer).

        1. In the load balancer description, change the `address` parameter value under `listener.endpoint.address.external_ipv4_address`:

            ```hcl
            resource "yandex_alb_load_balancer" "<load_balancer_name>" {
              ...
              listener {
                ...
                endpoint {
                  address {
                    external_ipv4_address {
                      address = <service_public_IP_address>
                    }
                  }
                  ports = [ <service_port> ]
                }
              }
            }
            ```

            Where `address` is the public IP address previously assigned to the network load balancer.

        1. Apply the changes:

            {% include [terraform-validate-plan-apply](../../_tutorials/_tutorials_includes/terraform-validate-plan-apply.md) %}

    {% endlist %}

1. After the IP addresses changes, your service will again be available through the L7 load balancer. Monitor the L7 load balancer's user load from the [load balancer statistics](../../application-load-balancer/operations/application-load-balancer-get-stats.md) charts.

1. Delete the now free static public IP address you selected when creating the L7 load balancer.

1. (Optional) [Delete the network load balancer](../../network-load-balancer/operations/load-balancer-delete.md) after migrating user load to the L7 load balancer.

### Do not keep the public IP address for your service {#not-save-public-ip}

1. To migrate user load from a network load balancer to an L7 load balancer, in the DNS service of your domain's public zone, change the A record value for the service domain name to the public IP address of the L7 load balancer. If the public domain zone was created in [{{ dns-full-name }}](../../dns/), change the record using this guide.

    {% note info %}

    The propagation of DNS record updates depends on the time-to-live (TTL) value and the number of links in the DNS request chain. This process can take a long time.

    {% endnote %}

1. As the DNS record updates propagate, follow the increase of requests to the L7 load balancer from the [load balancer statistics](../../application-load-balancer/operations/application-load-balancer-get-stats.md) charts.

1. Follow the decrease of the network load balancer load using the `processed_bytes` and `processed_packets` [load balancer metrics](../../monitoring/metrics-ref/network-load-balancer-ref.md). You can [create a dashboard](../../monitoring/operations/dashboard/create.md) to visualize these metrics. The absence of load on the network load balancer for a prolonged period of time indicates that the user load has been transfered to the L7 load balancer.

1. (Optional) [Delete the network load balancer](../../network-load-balancer/operations/load-balancer-delete.md) after migrating user load to the L7 load balancer.
