# Configuring an SFTP server based on CentOS 7

In this tutorial, you will create VMs for the [SFTP](https://en.wikipedia.org/wiki/SSH_File_Transfer_Protocol) server and client using {{ compute-short-name }}, connect to them over SSH, create an SFTP user, and set up data backups.

To deploy an infrastructure:

1. [Prepare your cloud](#before-you-begin).
1. [Create a VM for the SFTP server](#create-vm-sftp-server).
1. [Configure the SFTP server](#config-sftp-server).
1. [Create an SFTP user](#create-sftp-user).
1. [Create and configure a VM for the SFTP client](#create-vm-sftp-client).
1. [Back up configuration files to the SFTP server](#backup).
1. [Test the backup](#check-backup).
1. [Set up a backup schedule](#schedule).
1. [Restore settings from a backup](#restore).
1. [Check whether the settings are restored correctly](#check-restore).

If you no longer need the resources you created, [delete them](#cleanup).

## Prepare your cloud {#before-you-begin}

{% include [before-you-begin](../_tutorials_includes/before-you-begin.md) %}


### Required paid resources {#paid-resources}

The infrastructure support costs include:

* Fee for two continuously running VMs (see [{{ compute-full-name }} pricing](../../compute/pricing.md)):
  * VM for the SFTP client.
  * VM for the SFTP server.
* Fee for using a dynamic or static external IP address (see [{{ vpc-full-name }} pricing](../../vpc/pricing.md)).


## Create a VM for the SFTP server {#create-vm-sftp-server}

To create a VM:

{% list tabs group=instructions %}

- Management console {#console}

  1. In the [management console]({{ link-console-main }}), select the [folder](../../resource-manager/concepts/resources-hierarchy.md#folder) to create your VM.
  1. In the list of services, select **{{ ui-key.yacloud.iam.folder.dashboard.label_compute }}**.
  1. In the left-hand panel, select ![image](../../_assets/console-icons/server.svg) **{{ ui-key.yacloud.compute.switch_instances }}**.
  1. Click **{{ ui-key.yacloud.compute.instances.button_create }}**.
  1. Under **{{ ui-key.yacloud.compute.instances.create.section_image }}** select a public [CentOS 7](/marketplace/products/yc/centos-7) image.
  1. Under **{{ ui-key.yacloud.k8s.node-groups.create.section_allocation-policy }}**, select an [availability zone](../../overview/concepts/geo-scope.md) to place your VM in.
  1. Under **{{ ui-key.yacloud.compute.instances.create.section_platform }}**, navigate to the **{{ ui-key.yacloud.component.compute.resources.label_tab-custom }}** tab and specify the parameters as follows:

      * **{{ ui-key.yacloud.component.compute.resources.field_platform }}**: `Intel Ice Lake`.
      * **{{ ui-key.yacloud.component.compute.resources.field_cores }}**: `2`.
      * **{{ ui-key.yacloud.component.compute.resources.field_core-fraction }}**: `20%`.
      * **{{ ui-key.yacloud.component.compute.resources.field_memory }}**: `2 {{ ui-key.yacloud.common.units.label_gigabyte }}`

  1. Under **{{ ui-key.yacloud.compute.instances.create.section_network }}**:

      * In the **{{ ui-key.yacloud.component.compute.network-select.field_subnetwork }}** field, enter the ID of a subnet in the new VM’s availability zone. Alternatively, you can select a [cloud network](../../vpc/concepts/network.md#network) from the list.

          * Each network must have at least one [subnet](../../vpc/concepts/network.md#subnet). If there is no subnet, create one by selecting **{{ ui-key.yacloud.component.vpc.network-select.button_create-subnetwork }}**.
          * If you do not have a network, click **{{ ui-key.yacloud.component.vpc.network-select.button_create-network }}** to create one:

              * In the window that opens, enter the network name and select the folder to host the network.
              * (Optional) Select the **{{ ui-key.yacloud.vpc.networks.create.field_is-default }}** option to automatically create subnets in all availability zones.
              * Click **{{ ui-key.yacloud.vpc.networks.create.button_create }}**.

      * In the **{{ ui-key.yacloud.component.compute.network-select.field_external }}** field, select `{{ ui-key.yacloud.component.compute.network-select.switch_auto }}` to assign the VM a random external IP address from the {{ yandex-cloud }} pool. To ensure the external IP address does not change after the VM is stopped, [convert it to static](../../vpc/operations/set-static-ip.md).

  1. Under **{{ ui-key.yacloud.compute.instances.create.section_access }}**, specify the information required to access the VM:

      * In the **{{ ui-key.yacloud.compute.instances.create.field_user }}** field, enter a name for the user you want to create on the VM, e.g., `yc-user`.
      * {% include [access-ssh-key](../../_includes/compute/create/access-ssh-key.md) %}

      {% note alert %}

      Once created, the VM will get an IP address and a host name (FQDN) for connections. If you selected the **{{ ui-key.yacloud.component.compute.network-select.switch_none }}** option in the **{{ ui-key.yacloud.component.compute.network-select.field_external }}** field, you will not be able to access the VM from the internet.

      {% endnote %}

  1. Under **{{ ui-key.yacloud.compute.instances.create.section_base }}**, specify the VM name: `sftp-server`.
  1. Click **{{ ui-key.yacloud.compute.instances.create.button_create }}**.

{% endlist %}

It may take a few minutes to create a VM. 

## Configure the SFTP server {#config-sftp-server}

SFTP server functionality is included in the standard SSH program that comes with the CentOS 7 distribution. To configure the SFTP server, edit the `/etc/ssh/sshd_config` configuration file:

1. [Log in to the SFTP server VM via SSH](../../compute/operations/vm-connect/ssh.md#vm-connect).
1. Open the configuration file with the vi editor. The editor comes with the distribution and does not require installation. If you are not familiar with this editor, you can learn more in the [official documentation](https://www.vim.org/docs.php).

   ```bash
   sudo vi /etc/ssh/sshd_config
   ```

1. Add the following lines at the end of the file:

   ```text
   Match User fuser
   ForceCommand internal-sftp
   PasswordAuthentication no
   ChrootDirectory /var/sftp
   PermitTunnel no
   AllowAgentForwarding no
   AllowTcpForwarding no
   X11Forwarding no
   ```

   Where:

   * `Match User fuser`: Means that all subsequent rows will be applied only when connecting the `fuser` user.
   * `ForceCommand internal-sftp`: Only connects the user in SFTP mode and disables access to the shell.
   * `PasswordAuthentication no`: Disables login and password-based access.
   * `ChrootDirectory /var/sftp`: Only allows user access to the `/var/sftp` directory.
   * `PermitTunnel no`, `AllowAgentForwarding no`, `AllowTcpForwarding no`, and `X11Forwarding no`: Disable tunneling and port and graphical app forwarding via an SSH session.

1. Save the file.
1. Output the configuration file without commented or empty lines:

    ```bash
    sudo cat /etc/ssh/sshd_config | grep -v -e '^#' -e '^$'
    ```

1. Make sure the output of the previous command matches the following lines:

   ```text
   HostKey /etc/ssh/ssh_host_rsa_key
   HostKey /etc/ssh/ssh_host_ecdsa_key
   HostKey /etc/ssh/ssh_host_ed25519_key
   SyslogFacility AUTHPRIV
   AuthorizedKeysFile .ssh/authorized_keys
   PasswordAuthentication no
   ChallengeResponseAuthentication no
   GSSAPIAuthentication yes
   GSSAPICleanupCredentials no
   UsePAM yes
   X11Forwarding yes
   AcceptEnv LANG LC_CTYPE LC_NUMERIC LC_TIME LC_COLLATE LC_MONETARY LC_MESSAGES
   AcceptEnv LC_PAPER LC_NAME LC_ADDRESS LC_TELEPHONE LC_MEASUREMENT
   AcceptEnv LC_IDENTIFICATION LC_ALL LANGUAGE
   AcceptEnv XMODIFIERS
   Subsystem sftp  /usr/libexec/openssh/sftp-server
   Match User fuser
   ForceCommand internal-sftp
   PasswordAuthentication no
   ChrootDirectory /var/sftp
   PermitTunnel no
   AllowAgentForwarding no
   AllowTcpForwarding no
   X11Forwarding no
   ```

1. Restart the SFTP server for the settings to take effect:

   ```bash
   sudo systemctl restart sshd
   ```

   Once restarted, [log in to the SFTP server VM via SSH](../../compute/operations/vm-connect/ssh.md#vm-connect) again.

1. Create a group for SFTP users:

   ```bash
   sudo groupadd ftpusers
   ```

1. Create directories to save files to:

   ```bash
   sudo mkdir -p /var/sftp/backups
   ```

   * `sftp`: Root directory of the SFTP server.
   * `backups`: Directory to store backups on the SFTP server.

1. Set folder permissions so that all users in the `ftpusers` group can write and read files on the SFTP server:

   ```bash
   sudo chown root:ftpusers /var/sftp/backups
   sudo chmod 770 /var/sftp/backups
   ```

1. Check whether the set permissions are correct:

   ```bash
   ls -la /var | grep sftp
   ls -la /var/sftp
   ```

   Result:

   ```bash
   drwxr-xr-x.  4 root root   37 Aug  7 11:35 sftp
   drwxrwx---. 2 root ftpusers 80 Aug  7 08:41 backups
   ```

## Create an SFTP user {#create-sftp-user}

On the SFTP server VM:

1. Create an SFTP user, such as `fuser`:

   ```bash
   sudo useradd fuser
   ```

1. Create a password for the SFTP user:

   ```bash
   sudo passwd fuser
   ```

1. Create SSH keys for the `fuser` user. For this, run the `ssh-keygen` command under `fuser`:

   ```bash
   sudo runuser -l  fuser -c 'ssh-keygen'
   ```

   For the key generation process, see below. Leave the `passphrase` field blank.

   ```text
   Generating public/private rsa key pair.
   Enter file in which to save the key (/home/fuser/.ssh/id_rsa): 
   Created directory '/home/fuser/.ssh'.
   Enter passphrase (empty for no passphrase): 
   Enter same passphrase again: 
   Your identification has been saved in /home/fuser/.ssh/id_ed25519.
   Your public key has been saved in /home/fuser/.ssh/id_ed25519.pub.
   The key fingerprint is:
   SHA256:sXiE7EfPl8mo9mZCG+ta7fBxwdwdhbjNux63P8EIYNs fuser@ftp-server.{{ region-id }}.internal
   The key's randomart image is:
   +--[ED25519 256]--+
   |             . ..|
   |     . . o  . . .|
   |      o = +  + . |
   |     . + * E.+o..|
   |      o S + X +..|
   |       ooo . o.o |
   |       .=+o . ..o|
   |       o+=oo  .+.|
   |      .o.++  ...+|
   +----[SHA256]-----+
   ```

1. Create a file to save the SFTP client's public SSH keys to. Set the required permissions: 

   ```bash
   sudo touch /home/fuser/.ssh/authorized_keys
   sudo chmod 600 /home/fuser/.ssh/authorized_keys
   sudo chown fuser:fuser /home/fuser/.ssh/authorized_keys
   ```

1. Make sure the permissions are set correctly:

   ```bash
   ls -la /home/fuser/.ssh/
   ```

   Result:

   ```bash
   -rw-------. 1 fuser fuser  421 Aug  7 08:31 authorized_keys
   -rw-------. 1 fuser fuser  419 Aug  7 08:29 id_ed25519
   -rw-r--r--. 1 fuser fuser  107 Aug  7 08:29 id_ed25519.pub
   ```

1. Add the SFTP user to the SFTP group:

   ```bash
   sudo usermod -G ftpusers fuser
   ```

## Create and configure a VM for the SFTP client {#create-vm-sftp-client}

The process for creating a VM for the SFTP client is exactly the same as the one for the SFTP server. 

1. Complete steps 1-11 of the [Create a VM for the SFTP server](#create-vm-sftp-server) section; this time, however, name your VM as `sftp-client`.

1. [Log in to the SFTP client VM via SSH](../../compute/operations/vm-connect/ssh.md#vm-connect).
1. Create an SSH key pair on the SFTP client. This is done in a similar way to what you did for the `fuser` user in the [previous section](#create-sftp-user):

   ```bash
   ssh-keygen
   ```

1. Output the public key on the SFTP client screen:

   ```bash
   cat ~/.ssh/id_rsa.pub
   ```

1. [Log in to the SFTP server VM via SSH](../../compute/operations/vm-connect/ssh.md#vm-connect).
1. Open the `/home/fuser/.ssh/authorized_keys` file: 

   ```bash
   sudo vi /home/fuser/.ssh/authorized_keys
   ```

1. Copy the SSH key received on the SFTP client to the end of the file.
1. Save the file.
1. Make sure the SFTP client VM is accessible from the SFTP server and vice versa:

   1. [Log in to the SFTP server VM via SSH](../../compute/operations/vm-connect/ssh.md#vm-connect).
   1. Find the public or internal IP address of the SFTP client in the {{ yandex-cloud }} console under VM settings. 

      {% note warning %}

      The internal addresses of the SFTP client and server must be located in the same subnet or be linked via routing settings.

      {% endnote %}

   1. Enter the following command in the SFTP server terminal and provide the appropriate value:

      ```bash
      ping -c 3 <SFTP_client_IP_address>
      ```

   1. Make sure the packages are sent and received successfully: 

      ```bash
      ping -c 3 84.201.170.171
      ```

      Result:

      ```bash
      PING 84.201.170.171 (84.201.170.171) 56(84) bytes of data.
      64 bytes from 84.201.170.171: icmp_seq=1 ttl=55 time=8.59 ms
      64 bytes from 84.201.170.171: icmp_seq=2 ttl=55 time=6.32 ms
      64 bytes from 84.201.170.171: icmp_seq=3 ttl=55 time=5.95 ms

      --- 84.201.170.171 ping statistics ---
      3 packets transmitted, 3 received, 0% packet loss, time 2003ms
      rtt min/avg/max/mdev = 5.955/6.959/8.595/1.168 ms
      ```

   1. Repeat the check on the SFTP client by specifying the SFTP server IP in the command.

## Make a backup of configuration files on the SFTP server {#backup}

This guide describes how to back up configuration (`.conf`) files from the `/etc` folder. 

The backup process is as follows:

1. Archive all configuration files you need.
1. Send the archive to the SFTP server.
1. Delete the archive on the SFTP client. 

To set up the backup process:

1. [Log in to the SFTP client VM via SSH](../../compute/operations/vm-connect/ssh.md#vm-connect).
1. Set environment variables for the script to work properly. To do this, open the `~/.bash_profile` file:

   ```bash
   vi ~/.bash_profile
   ```

1. Add the following lines at the end of the file and provide the appropriate values:

   ```bash
   export SFTP_SERVER=<SFTP_server_IP_address>
   export SFTP_USER='fuser'
   ```

1. Apply the settings:

   ```bash
   source ~/.bash_profile
   ```

1. Make sure the variables are there:

   ```bash
   env | grep SFTP
   ```

   Result:

   ```text
   SFTP_USER=fuser
   SFTP_SERVER=10.128.0.5
   ```

1. Compress all configuration files into a single archive:

   ```bash
   sudo find /etc -type f -name *.conf -print0 | sudo tar -czf backup.tar.gz --null -T -
   ```

   Where:

   * `sudo find /etc -type f -name *.conf -print0`: Searching for all `.conf` files from `/etc`.
   * `sudo tar -czf backup.tar.gz --null -T -`: Moving the configuration files to the `backup.tar.gz` archive.

1. Send the resulting archive to the SFTP server:

   ```bash
   curl \
     --upload-file backup.tar.gz sftp://$SFTP_SERVER/backups/backup_$(hostname)_$(date "+%Y%m%d_%H%M%S").tar.gz \
     --insecure \
     --user $SFTP_USER:
   ```

   Where:

   * `-T`: Uploads the `backup.tar.gz` file to the remote server.
   * `$SFTP_SERVER`: Variable that automatically takes the value of the SFTP server IP address.
   * `backup_$(hostname)_$(date "+%Y%m%d_%H%M%S").tar.gz`: Adds the name of the computer to the archive name and the date and time when the archive was created. This will help you navigate the list of backups on the server.

     For example, the name of the archive on the server might look like this: `backup_ftp-server.{{ region-id }}.internal_20190803_180228.tar.gz`. 

   * `--insecure`: Disables SSL certificate verification by the SFTP server. In this case, the traffic within the SSH session is still encrypted.
   * `$SFTP_USER`: Variable that automatically takes the SFTP user value.
   * `:`: Empty password. No password is requested.

1. Delete the archive on the SFTP client:

   ```bash
   sudo rm -f backup.tar.gz
   ```

You can perform all actions for creating a backup with a single command in the SFTP client terminal:

```bash
sudo find /etc -type f -name *.conf -print0 | sudo tar -czf backup.tar.gz --null -T -&& curl --upload-file backup.tar.gz sftp://$SFTP_SERVER/backups/backup_$(hostname)_$(date "+%Y%m%d_%H%M%S").tar.gz --insecure --user $SFTP_USER: && sudo rm -f backup.tar.gz
```

## Check whether the backup is working properly {#check-backup}

To make sure the backup is being created properly, run the backup, and find the copy on the server:

1. [Log in to the SFTP client VM via SSH](../../compute/operations/vm-connect/ssh.md#vm-connect) and run the backup command:

   ```bash
   sudo find /etc -type f -name *.conf -print0 | sudo tar -czf backup.tar.gz --null -T -&& curl --upload-file backup.tar.gz sftp://$SFTP_SERVER/backups/backup_$(hostname)_$(date "+%Y%m%d_%H%M%S").tar.gz --insecure --user $SFTP_USER: && sudo rm -f backup.tar.gz
   ```

1. [Log in to the SFTP server VM via SSH](../../compute/operations/vm-connect/ssh.md#vm-connect) and make sure there is a file named like `backup_ftp-server.{{ region-id }}.internal_20190803_180228.tar.gz` in the SFTP user's home directory. To do this, run the following command on the SFTP server:

   ```bash
   sudo ls /var/sftp/backups
   ```

## Set up a schedule for backups {#schedule}

To create regular backups of your settings, you can use `crontab`, a built-in utility.

1. [Log in to the SFTP client VM via SSH](../../compute/operations/vm-connect/ssh.md#vm-connect) and open the `crontab` file for editing:

   ```bash
   crontab -e
   ```

1. Add the following lines to run backups daily at 11:00 pm UTC:

   ```text
   SFTP_SERVER=<SFTP_server_IP_address>
   SFTP_USER='fuser'

   0 23 * * * sudo find /etc -type f -name *.conf -print0 | sudo tar -czf backup.tar.gz --null -T -&& curl --upload-file backup.tar.gz sftp://$SFTP_SERVER/backups/backup_$(hostname)_$(date "+\%Y\%m\%d_\%H\%M\%S").tar.gz --insecure --user $SFTP_USER: && sudo rm -f backup.tar.gz
   ```

   * The VM's time is UTC by default. Keep the time zone difference in mind when setting up the schedule.
   * In the command you type in crontab, all `%` characters should be escaped with `\`.


## Restore settings from a backup {#restore}

To restore settings, do the following:

1. Download the backup from the SFTP server to the SFTP client.
1. Unpack the archive.
1. Copy the configuration files from the archive to the system.
1. Delete the archive. 

To restore the settings from the backup:

1. On the SFTP server, in the `/var/sftp/backups` directory, select the backup from which you want to restore the configuration files. For example, let’s assume you select `backup_ftp-server.{{ region-id }}.internal_20190803_180228.tar.gz`.

1. [Log in to the SFTP client VM via SSH](../../compute/operations/vm-connect/ssh.md#vm-connect).
1. Set an environment variable for the backup file name:

   ```bash
   SFTP_BACKUP='backup_ftp-server.{{ region-id }}.internal_20190803_180228.tar.gz'
   ```

1. Download the backup from the SFTP server:

   ```bash
   sftp $SFTP_USER@$SFTP_SERVER:/backups/$SFTP_BACKUP .
   ```

1. Unpack the archive:

   ```bash
   tar -xzf $SFTP_BACKUP
   ```

1. Copy the configuration files from the archive to the system. Use `yes` when running the command to skip confirmation when overwriting files:

   ```bash
   yes | sudo cp -rfp etc / 
   ```

1. Delete the archive and unpacked content:

   ```bash
   rm -f $SFTP_BACKUP
   rm -rfd etc
   ```

You can perform all actions required to restore settings from a backup with a single command in the SFTP client terminal:

```bash
sftp $SFTP_USER@$SFTP_SERVER:/backups/$SFTP_BACKUP . && tar -xzf $SFTP_BACKUP && yes | sudo cp -rfp etc / && rm -rfd etc && rm -f $SFTP_BACKUP
```

## Check whether the settings are restored correctly {#check-restore}
  
On the SFTP client VM:

1. To make sure the configuration files from the archive successfully get into the file system, add a verification section to the command above:

   ```bash
   sftp $SFTP_USER@$SFTP_SERVER:/backups/$SFTP_BACKUP . && tar -xzf $SFTP_BACKUP && echo "## this is from backup" >> etc/yum.conf && yes | sudo cp -rfp etc / && rm -rfd etc && rm -f $SFTP_BACKUP
   ```

   The `echo "## this is from backup" >> etc/yum.conf` command writes the _## this is from backup_ test phrase at the end of the `etc/yum.conf` file unpacked from the archive.

1. After restoring the backup, run the following command:

   ```bash
   cat /etc/yum.conf | grep backup
   ```

1. Make sure the test phrase is displayed on the screen:

   ```bash
   ## this is from backup
   ```

## How to delete the resources you created {#cleanup}

If you no longer need the SFTP server and client:

* [Delete the VMs](../../compute/operations/vm-control/vm-delete.md) for the SFTP client and server (in our example, `sftp-server` and `sftp-client`).
* [Delete the static IP address](../../vpc/operations/address-delete.md) if you reserved one.
