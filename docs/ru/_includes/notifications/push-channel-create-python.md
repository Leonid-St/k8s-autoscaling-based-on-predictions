```python
response = client.create_platform_application(
    Name="<имя_канала>",
    Platform="<тип_платформы>",
    Attributes= {
         "<параметр1>": "<значение1>",
         ...
         "<параметрN>": "<значениеN>"
    }
)
print ("Platform application ARN:", response['PlatformApplicationArn'])
```

Где:

* `Name` — имя канала уведомлений, задается пользователем. Должно быть уникальным в [облаке](../../resource-manager/concepts/resources-hierarchy.md#cloud). Может содержать строчные и заглавные буквы латинского алфавита, цифры, подчеркивания, дефисы и точки. Допустимая длина — от 1 до 256 символов. Рекомендуется для каналов APNs указывать в имени идентификатор приложения (Bundle ID), для FCM и HMS — полное название пакета приложения (Package name).
* `Platform` — тип мобильной платформы:

  * `APNS` и `APNS_SANDBOX` — Apple Push Notification service (APNs). Для тестирования приложения используйте `APNS_SANDBOX`.
  * `GCM` — Firebase Cloud Messaging (FCM).
  * `HMS` — Huawei Mobile Services (HMS).

* `Attributes` — параметры аутентификации на мобильной платформе в формате `"<параметр>": "<значение>"` через запятую. Значения зависят от типа платформы:

  * APNs:

    * Аутентификация с токеном:

      * `PlatformPrincipal` — путь к файлу с ключом подписи токена, полученный в Apple.
      * `PlatformCredential` — идентификатор ключа подписи (Key ID).
      * `ApplePlatformTeamID` — идентификатор разработчика (Team ID).
      * `ApplePlatformBundleID` — идентификатор приложения (Bundle ID).

    * Аутентификация с сертификатом:

      * `PlatformPrincipal` — SSL-сертификат в формате `.pem`.
      * `PlatformCredential` — закрытый ключ сертификата в формате `.pem`.

          {% include [convert-p12-to-pem](convert-p12-to-pem.md) %}

    Аутентификация с токеном является предпочтительной, как более современная.

  * FCM: `PlatformCredential` — ключ сервисного аккаунта Google Cloud в формате JSON для аутентификации с помощью HTTP v1 API или API-ключ (server key) для аутентификации с помощью Legacy API.

    Версия HTTP v1 API является предпочтительной, так как с июня 2024 года Legacy API [не будет поддерживаться FCM](https://firebase.google.com/docs/cloud-messaging/migrate-v1).

  * HMS:

    * `PlatformPrincipal` — идентификатор ключа.
    * `PlatformCredential` — API-ключ.

В результате вы получите идентификатор (ARN) канала уведомлений.