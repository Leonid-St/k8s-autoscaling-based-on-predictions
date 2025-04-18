---
editable: false
sourcePath: en/_api-ref/vpc/v1/api-ref/Address/get.md
---

# Virtual Private Cloud API, REST: Address.Get

Returns the specified Address resource.

To get the list of all available Address resources, make a [List](/docs/vpc/api-ref/Address/list#List) request.

## HTTP request

```
GET https://vpc.{{ api-host }}/vpc/v1/addresses/{addressId}
```

## Path parameters

#|
||Field | Description ||
|| addressId | **string**

Required field. ID of the Address resource to return.

To get Address resource ID make a [AddressService.List](/docs/vpc/api-ref/Address/list#List) request. ||
|#

## Response {#yandex.cloud.vpc.v1.Address}

**HTTP Code: 200 - OK**

```json
{
  "id": "string",
  "folderId": "string",
  "createdAt": "string",
  "name": "string",
  "description": "string",
  "labels": "object",
  // Includes only one of the fields `externalIpv4Address`
  "externalIpv4Address": {
    "address": "string",
    "zoneId": "string",
    "requirements": {
      "ddosProtectionProvider": "string",
      "outgoingSmtpCapability": "string"
    }
  },
  // end of the list of possible fields
  "reserved": "boolean",
  "used": "boolean",
  "type": "string",
  "ipVersion": "string",
  "deletionProtection": "boolean",
  "dnsRecords": [
    {
      "fqdn": "string",
      "dnsZoneId": "string",
      "ttl": "string",
      "ptr": "boolean"
    }
  ]
}
```

An Address resource. For more information, see [Address](/docs/vpc/concepts/address).

#|
||Field | Description ||
|| id | **string**

ID of the address. Generated at creation time. ||
|| folderId | **string**

ID of the folder that the address belongs to. ||
|| createdAt | **string** (date-time)

Creation timestamp.

String in [RFC3339](https://www.ietf.org/rfc/rfc3339.txt) text format. The range of possible values is from
`0001-01-01T00:00:00Z` to `9999-12-31T23:59:59.999999999Z`, i.e. from 0 to 9 digits for fractions of a second.

To work with values in this field, use the APIs described in the
[Protocol Buffers reference](https://developers.google.com/protocol-buffers/docs/reference/overview).
In some languages, built-in datetime utilities do not support nanosecond precision (9 digits). ||
|| name | **string**

Name of the address.
The name is unique within the folder.
Value must match the regular expression ``\\|[a-zA-Z]([-_a-zA-Z0-9]{0,61}[a-zA-Z0-9])?``. ||
|| description | **string**

Description of the address. 0-256 characters long. ||
|| labels | **object** (map<**string**, **string**>)

Address labels as `key:value` pairs.
No more than 64 per resource.
The maximum string length in characters for each value is 63.
Each value must match the regular expression `[-_0-9a-z]*`.
The string length in characters for each key must be 1-63.
Each key must match the regular expression `[a-z][-_0-9a-z]*`. ||
|| externalIpv4Address | **[ExternalIpv4Address](#yandex.cloud.vpc.v1.ExternalIpv4Address)**

Includes only one of the fields `externalIpv4Address`.

External ipv4 address specification. ||
|| reserved | **boolean**

Specifies if address is reserved or not. ||
|| used | **boolean**

Specifies if address is used or not. ||
|| type | **enum** (Type)

Type of the IP address.

- `TYPE_UNSPECIFIED`
- `INTERNAL`: Internal IP address.
- `EXTERNAL`: Public IP address. ||
|| ipVersion | **enum** (IpVersion)

Version of the IP address.

- `IP_VERSION_UNSPECIFIED`
- `IPV4`: IPv4 address.
- `IPV6`: IPv6 address. ||
|| deletionProtection | **boolean**

Specifies if address protected from deletion. ||
|| dnsRecords[] | **[DnsRecord](#yandex.cloud.vpc.v1.DnsRecord)**

Optional DNS record specifications ||
|#

## ExternalIpv4Address {#yandex.cloud.vpc.v1.ExternalIpv4Address}

#|
||Field | Description ||
|| address | **string**

Value of address. ||
|| zoneId | **string**

Availability zone from which the address will be allocated. ||
|| requirements | **[AddressRequirements](#yandex.cloud.vpc.v1.AddressRequirements)**

Parameters of the allocated address, for example DDoS Protection. ||
|#

## AddressRequirements {#yandex.cloud.vpc.v1.AddressRequirements}

#|
||Field | Description ||
|| ddosProtectionProvider | **string**

DDoS protection provider ID. ||
|| outgoingSmtpCapability | **string**

Capability to send SMTP traffic. ||
|#

## DnsRecord {#yandex.cloud.vpc.v1.DnsRecord}

#|
||Field | Description ||
|| fqdn | **string**

DNS record name (absolute or relative to the DNS zone in use). ||
|| dnsZoneId | **string**

ID of the public DNS zone. ||
|| ttl | **string** (int64)

TTL of record. ||
|| ptr | **boolean**

If the PTR record is required, this parameter must be set to "true". ||
|#