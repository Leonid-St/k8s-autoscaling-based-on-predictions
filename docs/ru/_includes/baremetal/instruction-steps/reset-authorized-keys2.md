Замените содержимое файла `authorized_keys`, добавив в него новый публичный SSH-ключ:

```bash
echo "<новый_SSH-ключ>" | tee authorized_keys
```

Если вы хотите добавить новый ключ, не удаляя старый, передайте в команду `tee` параметр `-a`:

```bash
echo "<новый_SSH-ключ>" | tee -a authorized_keys
```

{% note info %}

На этом этапе вы также можете [изменить](../../../baremetal/operations/servers/reset-password.md) пароль root-пользователя для доступа на сервер или изменить любые другие настройки операционной системы сервера, которые могут препятствовать корректной загрузке или подключению к нему, а также провести необходимые работы по диагностике и исправлению ошибок.

{% endnote %}