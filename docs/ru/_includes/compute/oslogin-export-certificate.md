Чтобы экспортировать SSH-сертификат пользователя организации {{ org-name }} или [сервисного аккаунта](../../iam/concepts/users/service-accounts.md) на локальный компьютер:

{% list tabs group=instructions %}


- CLI {#cli}

  {% include [cli-install](../cli-install.md) %}

  {% include [default-catalogue](../default-catalogue.md) %}

  1. Посмотрите описание команды CLI для экспорта SSH-сертификата в локальную директорию:

      ```bash
      yc compute ssh certificate export --help
      ```
  1. {% include [os-login-cli-organization-list](../../_includes/organization/os-login-cli-organization-list.md) %}
  1. {% include [os-login-cli-profile-list](../../_includes/organization/os-login-cli-profile-list.md) %}
  1. Экспортируйте сертификат:

      ```bash
      yc compute ssh certificate export \
          --login <логин_пользователя_или_сервисного_аккаунта> \
          --organization-id <идентификатор_организации> \
          --directory <путь_к_директории>
      ```

      Где:
      * `--login` — полученный ранее логин пользователя или сервисного аккаунта, заданный в профиле {{ oslogin }}. Необязательный параметр. Если параметр не задан, SSH-сертификат будет выгружен для пользователя или сервисного аккаунта, авторизованного в текущий момент в профиле {{ yandex-cloud }} CLI.
      * `--organization-id` — полученный ранее [идентификатор](../../organization/operations/organization-get-id.md) организации, из которой нужно экспортировать SSH-сертификат. Необязательный параметр. Если параметр не задан, сертификат будет выгружен из организации, к которой относится каталог по умолчанию.
      * `--directory` — путь к локальной директории, в которой будет сохранен экспортированный SSH-сертификат. Необязательный параметр. Если параметр не задан, сертификат будет по умолчанию сохранен в директории `.ssh` в домашней директории текущего пользователя компьютера (`~/.ssh/`).

      Результат:

      ```text
      Identity: /home/user1/.ssh/yc-cloud-id-b1gia87mbaom********-<логин_в_профиле_OS_Login>
      Certificate: /home/user1/.ssh/yc-cloud-id-b1gia87mbaom********-<логин_в_профиле_OS_Login>-cert.pub
      ```

      При сохранении экспортированного сертификата в директорию, отличную от директории по умолчанию, убедитесь что доступ к сохраненным файлам сертификата разрешен только текущему пользователю. При необходимости измените разрешения с помощью команды `chmod` в Linux и macOS или на вкладке **Безопасность** свойств файлов в Проводнике Windows.

{% endlist %}