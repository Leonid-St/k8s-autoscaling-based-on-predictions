1. В блоке **{{ ui-key.yacloud.compute.instance.overview.section_network }}** на странице ВМ в [консоли управления]({{ link-console-main }}) найдите публичный IP-адрес ВМ.
1. [Подключитесь](../../compute/operations/vm-connect/ssh.md) к ВМ по протоколу SSH.
1. Выдайте права на запись для вашего пользователя на директорию `/var/www/html`:

   ```bash
   sudo chown -R "$USER":apache /var/www/html
   ```

1. Загрузите на ВМ файлы сайта с помощью [протокола SCP](https://ru.wikipedia.org/wiki/SCP).

   {% list tabs group=operating_system %}

   - Linux/macOS {#linux-macos}

     Используйте утилиту командной строки `scp`:

     ```bash
     scp -r <путь_до_директории_с_файлами> <имя_пользователя_ВМ>@<IP-адрес_ВМ>:/var/www/html
     ```

   - Windows {#windows}

     С помощью программы [WinSCP](https://winscp.net/eng/download.php) скопируйте локальную директорию с файлами в директорию `/var/www/html` на ВМ.

   {% endlist %}