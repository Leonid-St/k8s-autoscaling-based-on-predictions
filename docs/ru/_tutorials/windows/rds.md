# Развертывание Microsoft Remote Desktop Services


{% include [ms-disclaimer](../../_includes/ms-disclaimer.md) %}



В сценарии описывается развертывание Microsoft Windows Server Datacenter с предустановленной службой Remote Desktop Services в {{ yandex-cloud }}. Инсталляция Microsoft Windows Server with Remote Desktop Services будет состоять из одного сервера, на котором будут установлены службы Remote Desktop Services и Active Directory. Образы представлены с подготовленными квотами на 5, 10, 25, 50 и 100 пользователей. Выберите версию с необходимой квотой. Все примеры приводятся для сервера с квотой на 5 пользователей.

{% note warning %}

Для увеличения квоты необходимо пересоздать ВМ.

{% endnote %}

Чтобы развернуть инфраструктуру Remote Desktop Services:

1. [Подготовьте облако к работе](#before-you-begin).
1. [Создайте облачную сеть и подсети](#create-network).
1. [Создайте скрипт для управления локальной учетной записью администратора](#admin-script).
1. [Создайте ВМ для Remote Desktop Services](#add-vm).
1. [Установите и настройте службу контроллера домена (Active Directory)](#install-ad).
1. [Настройте правила файрвола](#firewall).
1. [Настройте сервер лицензирования в домене](#license-server).
1. [Настройте роль Remote Desktop Session Host](#rdsh).
1. [Создайте пользователей](#create-users).

Если созданные ресурсы вам больше не нужны, [удалите их](#clear-out).

## Подготовьте облако к работе {#before-you-begin}

{% include [before-you-begin](../_tutorials_includes/before-you-begin.md) %}

{% include [ms-additional-data-note](../_tutorials_includes/ms-additional-data-note.md) %}

### Необходимые платные ресурсы {#paid-resources}

В стоимость инсталляции Microsoft Windows Server with Remote Desktop Services входят:

* плата за постоянно запущенные виртуальные машины (см. [тарифы {{ compute-full-name }}](../../compute/pricing.md));
* плата за использование динамических или статических публичных IP-адресов (см. [тарифы {{ vpc-full-name }}](../../vpc/pricing.md));
* стоимость исходящего трафика из {{ yandex-cloud }} в интернет (см. [тарифы {{ compute-full-name }}](../../compute/pricing.md)).

## Создайте облачную сеть и подсети {#create-network}

Создайте облачную сеть `my-network` с подсетями во всех зонах доступности, где будут находиться виртуальные машины.

1. Создайте облачную сеть:

   {% list tabs group=instructions %}

   - Консоль управления {#console}

     Чтобы создать [облачную сеть](../../vpc/concepts/network.md):
     1. Откройте раздел **{{ vpc-name }}** в каталоге, где требуется создать облачную сеть.
     1. Нажмите кнопку **Создать сеть.**
     1. Задайте имя сети: `my-network`.
     1. Нажмите кнопку **Создать сеть**.

   - CLI {#cli}

     {% include [cli-install](../../_includes/cli-install.md) %}
  
     {% include [default-catalogue](../../_includes/default-catalogue.md) %} 

     Чтобы создать облачную сеть, выполните команду:

     ```
     yc vpc network create --name my-network
     ```

   - API {#api}

     Воспользуйтесь методом REST API [create](../../vpc/api-ref/Network/create.md) для ресурса [Network](../../vpc/api-ref/Network/index.md) или вызовом gRPC API [NetworkService/Create](../../vpc/api-ref/grpc/Network/create.md).

   {% endlist %}

2. Создайте подсеть в сети `my-network`:

   {% list tabs group=instructions %}

     - Консоль управления {#console}

       Чтобы создать подсеть:
       1. Откройте раздел **{{ vpc-name }}** в каталоге, где требуется создать подсеть.
       1. Нажмите на имя облачной сети.
       1. Нажмите кнопку **Добавить подсеть**.
       1. Заполните форму: введите имя подсети `my-subnet-d`, выберите зону доступности `{{ region-id }}-d` из выпадающего списка.
       1. Введите CIDR подсети: IP-адрес и маску подсети: `10.1.0.0/16`. Подробнее про диапазоны IP-адресов в подсетях читайте в разделе [Облачные сети и подсети](../../vpc/concepts/network.md).
       1. Нажмите кнопку **Создать подсеть**.

     - CLI {#cli}

       Чтобы создать подсеть, выполните команду:

       ```
       yc vpc subnet create \
         --name my-subnet-d \
         --zone {{ region-id }}-d \
         --network-name my-network \
         --range 10.1.0.0/16
       ```
     
     - API {#api}

       Воспользуйтесь методом REST API [create](../../vpc/api-ref/Subnet/create.md) для ресурса [Subnet](../../vpc/api-ref/Subnet/index.md) или вызовом gRPC API [SubnetService/Create](../../vpc/api-ref/grpc/Subnet/create.md).

   {% endlist %}


## Создайте скрипт для управления локальной учетной записью администратора {#admin-script}

Создайте файл `setpass`, содержащий скрипт, который будет устанавливать пароль для локальной учетной записи администратора при создании виртуальных машин через CLI:

{% list tabs group=programming_language %}

- PowerShell {#powershell}

    ```
    #ps1
    Get-LocalUser | Where-Object SID -like *-500 | Set-LocalUser -Password (ConvertTo-SecureString "<ваш пароль>" -AsPlainText -Force)
    ```
{% endlist %}

Пароль должен соответствовать [требованиям к сложности]({{ ms.docs }}/windows/security/threat-protection/security-policy-settings/password-must-meet-complexity-requirements#справочные-материалы).

Подробные рекомендации по защите Active Directory читайте на [сайте разработчика]({{ ms.docs }}/windows-server/identity/ad-ds/plan/security-best-practices/best-practices-for-securing-active-directory).

## Создайте ВМ для Windows Server with Remote Desktop Services {#add-vm}

Создайте виртуальную машину для Windows Server with Remote Desktop Services. Эта машина будет иметь доступ в интернет.

{% list tabs group=instructions %}

- Консоль управления {#console}

  1. На странице каталога в [консоли управления]({{ link-console-main }}) нажмите кнопку **{{ ui-key.yacloud.iam.folder.dashboard.button_add }}** и выберите `{{ ui-key.yacloud.iam.folder.dashboard.value_compute }}`.
  1. В блоке **{{ ui-key.yacloud.compute.instances.create.section_image }}** в поле **{{ ui-key.yacloud.compute.instances.create.placeholder_search_marketplace-product }}** введите `RDS` и выберите подходящий образ [RDS](/marketplace?tab=software&search=windows+rds): 
  1. В блоке **{{ ui-key.yacloud.k8s.node-groups.create.section_allocation-policy }}** выберите [зону доступности](../../overview/concepts/geo-scope.md) `{{ region-id }}-d`.
  1. В блоке **{{ ui-key.yacloud.compute.instances.create.section_storages }}** задайте размер загрузочного [диска](../../compute/concepts/disk.md) `50 {{ ui-key.yacloud.common.units.label_gigabyte }}`.
  1. В блоке **{{ ui-key.yacloud.compute.instances.create.section_platform }}** перейдите на вкладку `{{ ui-key.yacloud.component.compute.resources.label_tab-custom }}` и укажите необходимую [платформу](../../compute/concepts/vm-platforms.md), количество vCPU и объем RAM:

      * **{{ ui-key.yacloud.component.compute.resources.field_platform }}** — `Intel Ice Lake`.
      * **{{ ui-key.yacloud.component.compute.resources.field_cores }}** — `4`.
      * **{{ ui-key.yacloud.component.compute.resources.field_core-fraction }}** — `100%`.
      * **{{ ui-key.yacloud.component.compute.resources.field_memory }}** — `8 {{ ui-key.yacloud.common.units.label_gigabyte }}`.
  1. В блоке **{{ ui-key.yacloud.compute.instances.create.section_network }}** укажите:

      * **{{ ui-key.yacloud.component.compute.network-select.field_subnetwork }}** — сеть `my-network` и подсеть `my-subnet-d`.
      * **{{ ui-key.yacloud.component.compute.network-select.field_external }}** — `{{ ui-key.yacloud.component.compute.network-select.switch_auto }}`.
  1. В блоке **{{ ui-key.yacloud.compute.instances.create.section_base }}** задайте имя ВМ: `my-rds-vm`.
  1. Нажмите кнопку **{{ ui-key.yacloud.compute.instances.create.button_create }}**.

  {% include [vm-reset-password-windows-operations](../../_includes/compute/reset-vm-password-windows-operations.md) %}

- CLI {#cli}

  ```
   yc compute instance create \
     --name my-rds-vm \
     --hostname my-rds-vm \
     --memory 8 \
     --cores 4 \
     --zone {{ region-id }}-d \
     --network-interface subnet-name=my-subnet-d,ipv4-address=10.1.0.3,nat-ip-version=ipv4 \
     --create-boot-disk image-folder-id=standard-images,image-family=windows-2022-dc-gvlk-rds-5 \
     --metadata-from-file user-data=setpass
  ```

  {% include [cli-metadata-variables-substitution-notice](../../_includes/compute/create/cli-metadata-variables-substitution-notice.md) %}

- API {#api}

  Воспользуйтесь методом REST API [create](../../compute/api-ref/Instance/create.md) для ресурса [Instance](../../compute/api-ref/Instance/) или вызовом gRPC API [InstanceService/Create](../../compute/api-ref/grpc/Instance/create.md).

{% endlist %}

## Установите и настройте службу контроллера домена (Active Directory) {#install-ad}

1. Перезапустите ВМ `my-rds-vm`:

   {% list tabs group=instructions %}
   
   - Консоль управления {#console}
   
     1. На странице каталога в [консоли управления]({{ link-console-main }}) выберите сервис **{{ compute-name }}**.
     1. Выберите виртуальную машину `my-rds-vm`.
     1. Нажмите ![image](../../_assets/console-icons/ellipsis.svg) и выберите пункт **Перезапустить**.
   
   - CLI {#cli}
   
     ```
     yc compute instance restart my-rds-vm
     ```
   
   - API {#api}
   
     Воспользуйтесь методом REST API [restart](../../compute/api-ref/Instance/restart.md) для ресурса [Instance](../../compute/api-ref/Instance/) или вызовом gRPC API [InstanceService/Restart](../../compute/api-ref/grpc/Instance/restart.md).
   
   {% endlist %}

1. Подключитесь к ВМ `my-rds-vm` с [помощью RDP](../../compute/operations/vm-connect/rdp.md). Используйте логин `Administrator` и ваш пароль.
1. Установите роли Active Directory:

    {% list tabs group=programming_language %}
    
    - PowerShell {#powershell}

        ```powershell
        Install-WindowsFeature AD-Domain-Services -IncludeManagementTools
        Restart-Computer -Force
        ```
      
    {% endlist %}

1. Создайте лес Active Directory:

    {% list tabs group=programming_language %}
    
    - PowerShell {#powershell}

        ```powershell
        Install-ADDSForest -DomainName 'yantoso.net' -Force:$true
        ```
      
    {% endlist %}

   Windows перезапустится автоматически. Снова подключитесь к ВМ `my-rds-vm`. Используйте логин `yantoso\Administrator` и ваш пароль. Снова откройте PowerShell.

## Настройте правила файрвола {#firewall}

1. Добавьте правила файрвола, защищающие службу Active Directory от запросов из внешних сетей:
    
    {% list tabs group=programming_language %}
    
    - PowerShell {#powershell}

        ```powershell
        Set-NetFirewallRule `
          -DisplayName 'Active Directory Domain Controller - LDAP (UDP-In)' `
          -RemoteAddress:Intranet
        
        Set-NetFirewallRule `
          -DisplayName 'Active Directory Domain Controller - LDAP (TCP-In)' `
          -RemoteAddress:Intranet
        
        Set-NetFirewallRule `
          -DisplayName 'Active Directory Domain Controller - Secure LDAP (TCP-In)' `
          -RemoteAddress:Intranet
        ```
      
    {% endlist %}

## Настройте сервер лицензирования в домене {#license-server}

1. Добавьте системного пользователя Network Service в Terminal Server License Servers, группу безопасности Active Directory:
    
    {% list tabs group=programming_language %}
    
    - PowerShell {#powershell}

        ```powershell
        net localgroup "Terminal Server License Servers" /Add 'Network Service'
        ```
      
    {% endlist %}
   
1. Установите тип лицензирования.
    
    {% note info %}
    
    Доступны только `User CAL` лицензии.
    
    {% endnote %}

    {% list tabs group=programming_language %}
    
    - PowerShell {#powershell}

        ```powershell
        New-ItemProperty `
        -Path 'HKLM:\SOFTWARE\Policies\Microsoft\Windows NT\Terminal Services' `
        -Name 'LicensingMode' `
        -Value 4 `
        -PropertyType 'DWord'
        ```
      
    {% endlist %}
    
1. Укажите службу лицензирования RDS:

    {% list tabs group=programming_language %}
    
    - PowerShell {#powershell}

        ```powershell
        New-ItemProperty `
        -Path 'HKLM:\SOFTWARE\Policies\Microsoft\Windows NT\Terminal Services' `
        -Name 'LicenseServers' `
        -Value 'localhost' `
        -PropertyType 'String'
        ```
      
    {% endlist %}

1. (Опционально) Ограничьте количество разрешенных одновременных сессий к серверу:

    {% list tabs group=programming_language %}
    
    - PowerShell {#powershell}
        
      ```powershell
      New-ItemProperty `
      -Path 'HKLM:\SOFTWARE\Policies\Microsoft\Windows NT\Terminal Services' `
      -Name 'MaxInstanceCount' `
      -Value 5 `
      -PropertyType 'DWord'
      ```
   
    {% endlist %}

## Настройте роль Remote Desktop Session Host {#rdsh}

Установите роль Remote Desktop Session Host на сервер:

{% list tabs group=programming_language %}

- PowerShell {#powershell}

    ```powershell
    Install-WindowsFeature RDS-RD-Server -IncludeManagementTools
    Restart-Computer -Force
    ```

{% endlist %}


## Добавьте сервер в группу безопасности AD и зарегистрируйте его как SCP {#ad-sg-scp}

Добавьте сервер в Terminal Server License Servers, группу безопасности Active Directory, и зарегистрируйте его как точку подключения пользователей к сервису лицензирования (service connection point, SCP):

{% list tabs group=operating_system %}

- Windows Server {#windows}

  1. Нажмите **Start**.
  1. В поле поиска введите `Remote Desktop Licensing Manager`.
  1. Нажмите правой кнопкой мыши на сервер в списке и выберите **Review Configuration...**
  1. Напротив первого предупреждения, о группе `Terminal Server License Servers`, нажмите **Add to Group**, а затем — **Continue**.
  1. Напротив второго предупреждения, о точке подключения к сервису, нажмите **Register as SCP**.
  1. Нажмите **OK**.
  1. Перезапустите ВМ.

{% endlist %}


## Создайте пользователей {#create-users}

1. Создайте тестовых пользователей:

    {% list tabs group=programming_language %}
    
    - PowerShell {#powershell}
    
        ```powershell
        New-ADUser `
          -Name ru1 `
          -PasswordNeverExpires $true `
          -Enabled $true `
          -AccountPassword ("P@ssw0rd!1" | ConvertTo-SecureString -AsPlainText -Force )
        New-ADUser `
          -Name ru2 `
          -PasswordNeverExpires $true `
          -Enabled $true `
          -AccountPassword ("P@ssw0rd!1" | ConvertTo-SecureString -AsPlainText -Force )
        New-ADUser `
          -Name ru3 `
          -PasswordNeverExpires $true `
          -Enabled $true `
          -AccountPassword ("P@ssw0rd!1" | ConvertTo-SecureString -AsPlainText -Force )
        New-ADUser `
          -Name ru4 `
          -PasswordNeverExpires $true `
          -Enabled $true `
          -AccountPassword ("P@ssw0rd!1" | ConvertTo-SecureString -AsPlainText -Force )
        New-ADUser `
          -Name ru5 `
          -PasswordNeverExpires $true `
          -Enabled $true `
          -AccountPassword ("P@ssw0rd!1" | ConvertTo-SecureString -AsPlainText -Force )
        ```
      
    {% endlist %}

1. Выдайте пользователям права `Remote Desktop Users`:

    {% list tabs group=programming_language %}
    
    - PowerShell {#powershell}

        ```powershell
        Add-ADGroupMember -Members 'ru1' -Identity 'Remote Desktop Users'
        Add-ADGroupMember -Members 'ru2' -Identity 'Remote Desktop Users'
        Add-ADGroupMember -Members 'ru3' -Identity 'Remote Desktop Users'
        Add-ADGroupMember -Members 'ru4' -Identity 'Remote Desktop Users'
        Add-ADGroupMember -Members 'ru5' -Identity 'Remote Desktop Users'
        ```

    {% endlist %}

1. Настройте права доступа по RDP для группы `Remote Desktop Users`:

    {% list tabs group=programming_language %}
    
    - PowerShell {#powershell}
    
        ```powershell
        & secedit /export /cfg sec_conf_export.ini  /areas user_rights
        $secConfig = Get-Content sec_conf_export.ini
        $SID = 'S-1-5-32-555'
        $secConfig = $secConfig -replace '^SeRemoteInteractiveLogonRight .+', "`$0,*$SID"
        $secConfig | Set-Content sec_conf_import.ini
        & secedit /configure /db secedit.sdb /cfg sec_conf_import.ini /areas user_rights
        Remove-Item sec_conf_import.ini
        Remove-Item sec_conf_export.ini
        ```
      
    {% endlist %}

## Как удалить созданные ресурсы {#clear-out}

Если вам больше не нужны созданные ресурсы, удалите [виртуальные машины](../../compute/operations/vm-control/vm-delete.md) и [сети](../../vpc/operations/network-delete.md).