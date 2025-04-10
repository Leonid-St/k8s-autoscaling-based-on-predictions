# Creating a tunnel between two subnets using OpenVPN Access Server

When hosting computing resources in a public cloud, one often needs to ensure secure connections between two different subnets, such as an office network and a test farm in {{ yandex-cloud }}. The best way to handle this is using a VPN to:

* Connect geographically remote networks.
* Connect freelancers to the office network.
* Set up an encrypted connection over an open Wi-Fi network.

This tutorial describes how to create a VPN tunnel using the OpenVPN technology.

[OpenVPN Access Server](/marketplace/products/yc/openvpn-access-server) is compatible with the OpenVPN [open-source version](https://github.com/OpenVPN) and built on its basis. This product provides clients for Windows, Mac, Android, and iOS and is used to manage connections using a web interface.

In this example, we are going to create a tunnel that connects two different subnets into a single network. It will be working between two VPN gateways, one of them being OpenVPN Access Server and the other, a VM instance with the OpenVPN client. To test the VPN tunnel, configure gateways on both sides of it. In our example, one subnet is hosted in {{ yandex-cloud }}, while the other may reside both in {{ yandex-cloud }} and in an external network.

To create a tunnel between two different subnets:

1. [Prepare your cloud](#before-you-begin).
1. [Create a network and subnets](#create-environment).
1. [Create VMs you want to link](#create-target-vm).
1. [Create a VM gateway](#create-vm-gateway).
1. [Start the VPN server](#create-vpn-server).
1. [Configure network traffic permissions](#network-settings).
1. [Get the administrator password](#get-admin-password).
1. [Create an OpenVPN user for the tunnel](#configure-openvpn).
1. [Set up the second subnet's gateway to access the OpenVPN server](#configure-second-end-of-the-tunnel).
1. [Test the tunnel](#test-vpn-tunnel).

If you no longer need the VPN server, [delete the created VMs](#clear-out).

## Prepare your cloud {#before-you-begin}

{% include [before-you-begin](../_tutorials_includes/before-you-begin.md) %}


### Required paid resources {#paid-resources}

The cost of infrastructure support for OpenVPN includes:

* Fee for the disks and continuously running VMs (see [{{ compute-full-name }} pricing](../../compute/pricing.md)).
* Fee for using a dynamic or static external IP address (see [{{ vpc-full-name }} pricing](../../vpc/pricing.md)).
* Fee for the OpenVPN Access Server license (when using more than two connections).


## Create a network and subnets {#create-environment}

To connect cloud resources to the internet, make sure you have a [network](../../vpc/concepts/network.md) and [subnets](../../vpc/concepts/network.md#subnet).

### Create a network {#create-network}

{% list tabs group=instructions %}

- Management console {#console}

  1. In the [management console]({{ link-console-main }}), select the folder where you want to create a cloud network.
  1. In the list of services, select **{{ vpc-name }}**.
  1. Click **Create network**.
  1. Enter a name for the network, e.g., `ovpn-network`.
  1. Disable the **Create subnets** option.
  1. Click **Create network**.

{% endlist %}

### Create subnets {#create-subnets}

{% list tabs group=instructions %}

- Management console {#console}

  1. Select the `ovpn-network` network.
  1. Click **Add subnet**.
  1. Enter a name for the subnet, e.g., `ovpn-left`.
  1. Select an [availability zone](../../overview/concepts/geo-scope.md) from the drop-down list.
  1. Enter the subnet CIDR: `10.128.0.0/24`.
  1. Click **Create subnet**.
  1. Repeat steps 2 to 6 for the second subnet named `ovpn-right` with the `10.253.11.0/24` CIDR.

{% endlist %}

### Create VMs you want to link {#create-target-vm}

{% list tabs group=instructions %}

- Management console {#console}

  1. On the [folder page](../../resource-manager/concepts/resources-hierarchy.md#folder) in the [management console]({{ link-console-main }}), click **{{ ui-key.yacloud.iam.folder.dashboard.button_add }}** and select `{{ ui-key.yacloud.iam.folder.dashboard.value_compute }}`.
  1. Under **{{ ui-key.yacloud.compute.instances.create.section_image }}** in the **{{ ui-key.yacloud.compute.instances.create.placeholder_search_marketplace-product }}** field, select an image for the VM.
  1. Under **{{ ui-key.yacloud.k8s.node-groups.create.section_allocation-policy }}**, select the [availability zone](../../overview/concepts/geo-scope.md) the `ovpn-left` subnet resides in.
  1. Under **{{ ui-key.yacloud.compute.instances.create.section_network }}**:

      * In the **{{ ui-key.yacloud.component.compute.network-select.field_subnetwork }}** field, select the network named `ovpn-network` and the subnet named `ovpn-left`.
      * In the **{{ ui-key.yacloud.component.compute.network-select.field_external }}** field, select `{{ ui-key.yacloud.component.compute.network-select.switch_none }}`.
      * Expand the **{{ ui-key.yacloud.component.compute.network-select.section_additional }}** section:

          * In the **{{ ui-key.yacloud.component.internal-v4-address-field.field_internal-ipv4-address }}** field, select `{{ ui-key.yacloud.component.compute.network-select.switch_manual }}`.
          * In the input field that appears, enter `10.128.0.4`.

  1. Under **{{ ui-key.yacloud.compute.instances.create.section_access }}**, select **{{ ui-key.yacloud.compute.instance.access-method.label_oslogin-control-ssh-option-title }}** and specify the access credentials for the VM:

      * In the **{{ ui-key.yacloud.compute.instances.create.field_user }}** field, enter the username: `yc-user`.
      * {% include [access-ssh-key](../../_includes/compute/create/access-ssh-key.md) %}

  1. Under **{{ ui-key.yacloud.compute.instances.create.section_base }}**, specify the VM name: `ao-openvpn-test`.
  1. Click **{{ ui-key.yacloud.compute.instances.create.button_create }}**.
  1. Repeat steps 1 to 7 to create the second VM named `vm-ovpn-host` with internal address `10.253.11.110`, hosted in the `ovpn-right` subnet.

{% endlist %}

## Create a VM gateway {#create-vm-gateway}

{% list tabs group=instructions %}

- Management console {#console}

  1. On the [folder page](../../resource-manager/concepts/resources-hierarchy.md#folder) in the [management console]({{ link-console-main }}), click **{{ ui-key.yacloud.iam.folder.dashboard.button_add }}** and select `{{ ui-key.yacloud.iam.folder.dashboard.value_compute }}`.
  1. Under **{{ ui-key.yacloud.compute.instances.create.section_image }}** in the **{{ ui-key.yacloud.compute.instances.create.placeholder_search_marketplace-product }}** field, select an image for the VM.
  1. Under **{{ ui-key.yacloud.k8s.node-groups.create.section_allocation-policy }}**, select the [availability zone](../../overview/concepts/geo-scope.md) the `ovpn-right` subnet resides in.
  1. Under **{{ ui-key.yacloud.compute.instances.create.section_network }}**:

      * In the **{{ ui-key.yacloud.component.compute.network-select.field_subnetwork }}** field, select the network named `ovpn-network` and the subnet named `ovpn-right`.
      * In the **{{ ui-key.yacloud.component.compute.network-select.field_external }}** field, select `{{ ui-key.yacloud.component.compute.network-select.switch_auto }}` or `{{ ui-key.yacloud.component.compute.network-select.switch_list }}`.

          Either use static public IP addresses [from the list](../../vpc/operations/get-static-ip) or [convert](../../vpc/operations/set-static-ip) the VM IP address to static. Dynamic IP addresses may change after the VM reboots and the connections will no longer work.

      * Expand the **{{ ui-key.yacloud.component.compute.network-select.section_additional }}** section; in the **{{ ui-key.yacloud.component.internal-v4-address-field.field_internal-ipv4-address }}** field, select `{{ ui-key.yacloud.component.compute.network-select.switch_manual }}`.
      * In the input field that appears, enter `10.253.11.19`.

  1. Under **{{ ui-key.yacloud.compute.instances.create.section_access }}**, select **{{ ui-key.yacloud.compute.instance.access-method.label_oslogin-control-ssh-option-title }}** and specify the VM access data:

      * In the **{{ ui-key.yacloud.compute.instances.create.field_user }}** field, enter the username: `yc-user`.
      * {% include [access-ssh-key](../../_includes/compute/create/access-ssh-key.md) %}

  1. Under **{{ ui-key.yacloud.compute.instances.create.section_base }}**, specify the VM name: `vm-ovpn-gw`.
  1. Click **{{ ui-key.yacloud.compute.instances.create.button_create }}**.

{% endlist %}

## Start the VPN server {#create-vpn-server}

Create a VM to be the gateway for VPN connections:

{% list tabs group=instructions %}

- Management console {#console}

  1. On the [folder page](../../resource-manager/concepts/resources-hierarchy.md#folder) in the [management console]({{ link-console-main }}), click **{{ ui-key.yacloud.iam.folder.dashboard.button_add }}** and select `{{ ui-key.yacloud.iam.folder.dashboard.value_compute }}`.
  1. Under **{{ ui-key.yacloud.compute.instances.create.section_image }}**, enter `OpenVPN Access Server` in the **{{ ui-key.yacloud.compute.instances.create.placeholder_search_marketplace-product }}** field and select the [OpenVPN Access Server](/marketplace/products/yc/openvpn-access-server) image.
  1. Under **{{ ui-key.yacloud.k8s.node-groups.create.section_allocation-policy }}**, select the [availability zone](../../overview/concepts/geo-scope.md) the `ovpn-left` subnet resides in.
  1. Under **{{ ui-key.yacloud.compute.instances.create.section_storages }}**, enter `10 {{ ui-key.yacloud.common.units.label_gigabyte }}` as your boot [disk](../../compute/concepts/disk.md) size.
  1. Under **{{ ui-key.yacloud.compute.instances.create.section_platform }}**, navigate to the `{{ ui-key.yacloud.component.compute.resources.label_tab-custom }}` tab and specify the required [platform](../../compute/concepts/vm-platforms.md), number of vCPUs, and amount of RAM:

      * **{{ ui-key.yacloud.component.compute.resources.field_platform }}**: `Intel Ice Lake`.
      * **{{ ui-key.yacloud.component.compute.resources.field_cores }}**: `2`.
      * **{{ ui-key.yacloud.component.compute.resources.field_core-fraction }}**: `100%`.
      * **{{ ui-key.yacloud.component.compute.resources.field_memory }}**: `2 {{ ui-key.yacloud.common.units.label_gigabyte }}`.

  1. Under **{{ ui-key.yacloud.compute.instances.create.section_network }}**:

      * In the **{{ ui-key.yacloud.component.compute.network-select.field_subnetwork }}** field, select the network named `ovpn-network` and the subnet named `ovpn-left`.
      * In the **{{ ui-key.yacloud.component.compute.network-select.field_external }}** field, select `{{ ui-key.yacloud.component.compute.network-select.switch_auto }}` or `{{ ui-key.yacloud.component.compute.network-select.switch_list }}`.

          Either use static public IP addresses [from the list](../../vpc/operations/get-static-ip.md) or [convert](../../vpc/operations/set-static-ip.md) the VM IP address to static. Dynamic IP addresses may change after the VM reboots and the connections will no longer work.

      * In the **{{ ui-key.yacloud.component.compute.network-select.field_security-groups }}** field, select a [security group](../../vpc/concepts/security-groups.md). If you leave this field empty, the [default security group](../../vpc/concepts/security-groups.md#default-security-group) will be assigned.
      * Expand the **{{ ui-key.yacloud.component.compute.network-select.section_additional }}** section; in the **{{ ui-key.yacloud.component.internal-v4-address-field.field_internal-ipv4-address }}** field, select `{{ ui-key.yacloud.component.compute.network-select.switch_manual }}`.
      * In the input field that appears, enter `10.128.0.3`.

  1. Under **{{ ui-key.yacloud.compute.instances.create.section_access }}**, select **{{ ui-key.yacloud.compute.instance.access-method.label_oslogin-control-ssh-option-title }}** and specify the VM access data:

      * In the **{{ ui-key.yacloud.compute.instances.create.field_user }}** field, enter the username: `yc-user`.
      * {% include [access-ssh-key](../../_includes/compute/create/access-ssh-key.md) %}

  1. Under **{{ ui-key.yacloud.compute.instances.create.section_base }}**, specify the VM name: `vpn-server`.
  1. Click **{{ ui-key.yacloud.compute.instances.create.button_create }}**.
  1. A window will open informing you of the pricing type, which is BYOL (Bring Your Own License).
  1. Click **Create**.

{% endlist %}

## Configure network traffic permissions {#network-settings}

{% include [openvpn-network-settings](../_tutorials_includes/openvpn-network-settings.md) %}

## Get the administrator password {#get-admin-password}

{% include [openvpn-get-admin-password](../_tutorials_includes/openvpn-get-admin-password.md) %}

## Create an OpenVPN user for the tunnel {#configure-openvpn}

OpenVPN Access Server provides two web interfaces:

1. Client Web UI at `https://<VM_public_IP_address>:943/`. This interface is used by regular users to download client applications and configuration profiles.
1. Admin Web UI at `https://<VM_public_IP_address>:943/admin/`. This interface is used to configure the server.

{% note info %}

By default, the server has a self-signed certificate installed. If you need to replace this certificate, follow the steps described [here](https://openvpn.net/vpn-server-resources/installing-a-valid-ssl-web-certificate-in-access-server/).

{% endnote %}

Once you deploy OpenVPN Access Server on the {{ yandex-cloud }} VM that will be working as a gateway, you will have specific IP addresses and accounts as follows (the addresses below are provided for indicative purposes, yours may be different):

1. Internal IP of the `vpn-server` gateway: `10.128.0.3`.
1. Public IP address of the `vpn-server` VM: `<VM_public_IP_address>`
1. Admin Web UI: `https://<VM_public_IP_address>:943/admin`
1. Account for accessing the Admin UI: `openvpn/<admin password>`
1. Client Web UI: `https://<VM_public_IP_address>:943`

On the server side, create an OpenVPN user the second subnet's gateway will use to access the OpenVPN server to enable the tunnel. To create a user, log in to the Admin Web UI admin panel:

1. Open `https://<VM_public_IP_address>:943/admin` in your browser.
1. Enter the `openvpn` username and password (see [this section](#get-admin-password) on how to get the admin password).
1. Click **Agree**. This will open the home screen of the OpenVPN admin panel.
1. Go to the **User management** tab and select **User permissions**.
1. In the user list, enter the name of the new user in the **New Username** field, e.g., `as-gw-user`.
1. Click the pencil icon in the **More Settings** column and set the new user's password in the **Local Password** field.
1. In the **Access Control** field, select **User Routing** and specify the current local subnet where OpenVPN Access Server is deployed, e.g., `10.128.0.0/24`.
1. In the **VPN Gateway** field, select **Yes** and specify another local subnet to connect to via the tunnel, e.g., `10.253.11.0/24`.
1. Click **Save settings**.
1. Click **Update running server**.
1. Log in to the user panel under the new `as-gw-user` account, save the connection profile in a file named `as-gw-user.conf`, and move this file to the VM that will act as a gateway for the OpenVPN tunnel in the other subnet.

## Set up the second subnet's gateway to access the OpenVPN server {#configure-second-end-of-the-tunnel}

Run the following commands in the `vm-ovpn-gw` console:

```bash
sudo apt update
sudo apt install openvpn
cp as-gw-user.conf /etc/openvpn/client/
echo -e "as-gw-user\n<password>" > /etc/openvpn/client/param.txt
```

As a result, a file named `param.txt` should appear in the `/etc/openvpn/client/` folder. Copy to the same folder the previously created `as-gw-user.conf` file of the OpenVPN user you created to establish the tunnel:

```bash
ls -lh /etc/openvpn/client/
```

Result:

```
total 16K
-rw-rw-r-- 1 root root 9.7K Nov 10 14:37 as-gw-user.conf
-rw-r--r-- 1 root root 24 Nov 10 14:31 param.txt
```

In the `auth-user-pass` string of the `/etc/openvpn/as-gw-user.conf` file, specify the `param.txt` file name:

```
dev tun
dev-type tun
remote-version-min 1.2
reneg-seq 604800
auth-user-pass param.txt
verb 3
push-peer-info
```

Run the following commands:

```bash
sudo systemctl enable openvpn-client@as-gw-user
sudo systemctl start openvpn-client@as-gw-user
sudo systemctl status openvpn-client@as-gw-user
```

The result should look like this:

```
● openvpn-client@as-gw-user.service - OpenVPN tunnel for as/gw/user
    Loaded: loaded (/lib/systemd/system/openvpn-client@.service; enabled; vendor preset:
enabled)
    Active: active (running) since Fri 2022-11-11 20:12:49 UTC; 1h 6min ago
        Docs: man:openvpn(8)
            https://community.openvpn.net/openvpn/wiki/Openvpn24ManPage
            https://community.openvpn.net/openvpn/wiki/HOWTO
    Main PID: 2626 (openvpn)
    Status: "Initialization Sequence Completed"
        Tasks: 1 (limit: 2237)
    Memory: 2.0M
        CPU: 157ms
    CGroup: /system.slice/system-openvpn\x2dclient.slice/openvpn-client@as-gw-user.service
            └─2626 /usr/sbin/openvpn --suppress-timestamps --nobind --config as-gw-user.conf
```

To enable packet transfers from other hosts, run these commands:

```bash
vm-ovpn-gw:~$ sudo bash -c "echo 'net.ipv4.ip_forward = 1' >> /etc/sysctl.conf"
vm-ovpn-gw:~$ sudo sysctl -p
```

Check that there is a route to `10.253.11.0/24` at the `vpn-server` gateway:
    
```bash
vpn-server:~$ sudo ip route
```

Result:
    
```
default via 10.128.0.1 dev eth0 proto dhcp src 10.128.0.3 metric 100
10.128.0.0/24 dev eth0 proto kernel scope link src 10.128.0.3
10.128.0.1 dev eth0 proto dhcp scope link src 10.128.0.3 metric 100
10.253.11.0/24 dev as0t2 proto static
172.27.224.0/22 dev as0t0 proto kernel scope link src 172.27.224.1
172.27.228.0/22 dev as0t1 proto kernel scope link src 172.27.228.1
172.27.232.0/22 dev as0t2 proto kernel scope link src 172.27.232.1
172.27.236.0/22 dev as0t3 proto kernel scope link src 172.27.236.1
```

Check the route to `10.128.0.0/24` at the `vm-ovpn-gw` VM:

```bash
sudo ip route
```

Result:

```
default via 10.253.11.1 dev ens18 proto dhcp src 10.253.11.19 metric 100
10.128.0.0/24 via 172.27.232.1 dev tun0 metric 101
10.253.11.0/24 dev ens18 proto kernel scope link src 10.253.11.19 metric 100
10.253.11.1 dev ens18 proto dhcp scope link src 10.253.11.19 metric 100
172.27.224.0/20 via 172.27.232.1 dev tun0 metric 101
172.27.232.0/22 dev tun0 proto kernel scope link src 172.27.232.5
178.154.226.72 via 10.253.11.1 dev ens18
```

## Test the tunnel {#test-vpn-tunnel}

To test the tunnel, you will need the test VMs we mentioned above. These must reside in both subnets and be different from the tunnel gateways.

For these two VMs to exchange data, they both must see the static routes to the other subnet. `ao-openvpn-test` to `10.253.11.0/24`, and `vm-ovpn-host` to `10.128.0.0/24`.

Run the following command on `vm-ovpn-host`:

```bash
sudo ip route add 10.128.0.0./24 via 10.253.11.19
```

On the test VM in {{ yandex-cloud }}, adding a static route within the VM will not help. In {{ yandex-cloud }}, static routes for VMs should be specified [in a different way](../../vpc/concepts/routing.md).

In {{ yandex-cloud }}, the `ao-openvpn-as` VMs (OpenVPN server) and `ao-openvpn-test` VMs reside in the same `default` subnet. In the settings of this subnet, add a static route with the following parameters:
    
```
Name: office-net
Prefix: 10.253.11.0/24
Next hop: 10.128.0.3
```

To apply this static route to the `ao-openvpn-test` VM, shut it down and start it again.

Now use the `ping` command to test the tunnel from the `vm-ovpn-host` VM to the other test VM:
    
```bash
ping 10.128.0.4
```

Result:
    
```
PING 10.128.0.4 (10.128.0.4) 56(84) bytes of data.
64 bytes from 10.128.0.4: icmp_seq=1 ttl=61 time=7.45 ms
64 bytes from 10.128.0.4: icmp_seq=2 ttl=61 time=5.61 ms
64 bytes from 10.128.0.4: icmp_seq=3 ttl=61 time=5.65 ms
^C
--- 10.128.0.4 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 2003ms
rtt min/avg/max/mdev = 5.613/6.235/7.446/0.855 ms
```

Do the same from the other end of the tunnel, from the `ao-openvpn-test` test VM:

```bash
ping 10.253.11.110
```

Result:
    
```
PING 10.253.11.110 (10.253.11.110) 56(84) bytes of data.
64 bytes from 10.253.11.110: icmp_seq=1 ttl=61 time=6.23 ms
64 bytes from 10.253.11.110: icmp_seq=2 ttl=61 time=5.90 ms
64 bytes from 10.253.11.110: icmp_seq=3 ttl=61 time=6.09 ms
64 bytes from 10.253.11.110: icmp_seq=4 ttl=61 time=5.69 ms
^C
--- 10.253.11.110 ping statistics ---
4 packets transmitted, 4 received, 0% packet loss, time 3005ms
rtt min/avg/max/mdev = 5.688/5.976/6.229/0.203 ms
```

## How to delete the resources you created {#clear-out}

To free up folder resources, [delete](../../compute/operations/vm-control/vm-delete.md) the `vpn-server` VM and the test VM.

If you reserved a public static IP address, [delete it](../../vpc/operations/address-delete.md).

#### See also {#see-also}

* [OpenVPN Project Wiki](https://community.openvpn.net/openvpn/wiki)
* [{#T}](../../certificate-manager/operations/managed/cert-get-content.md)
* [Connecting to Access Server](https://openvpn.net/vpn-server-resources/connecting-to-access-server-with-linux/#openvpn-open-source-openvpn-cli-program)
