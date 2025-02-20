---
editable: false
sourcePath: en/_api-ref-grpc/certificatemanager/v1/privateca/api-ref/grpc/Policy/update.md
---

# Certificate Manager Private CA API, gRPC: PolicyService.Update

Updates the specified policy.

Only the fields specified in the `update_mask` will be updated.

## gRPC request

**rpc Update ([UpdatePolicyRequest](#yandex.cloud.certificatemanager.v1.privateca.UpdatePolicyRequest)) returns ([operation.Operation](#yandex.cloud.operation.Operation))**

## UpdatePolicyRequest {#yandex.cloud.certificatemanager.v1.privateca.UpdatePolicyRequest}

```json
{
  "id": "string",
  "name": "string",
  "subject_policy": {
    "country": "string",
    "organization": "string",
    "organizational_unit": "string",
    "distinguished_name_qualifier": "string",
    "state_or_province": "string",
    "common_name": "string",
    "email_address": "string",
    "serial_number": "string",
    "locality": "string",
    "title": "string",
    "surname": "string",
    "given_name": "string",
    "initials": "string",
    "generation_qualifier": "string"
  },
  "san_policy": {
    "other_name_oid": "string",
    "rfc_822_name": "string",
    "dns_name": "string",
    "directory_name": "string",
    "edi_party_name": {
      "name_assigner": "string",
      "party_name": "string"
    },
    "uniform_resource_identifier": "string",
    "ip_address": "string",
    "registered_id": "string"
  },
  // Includes only one of the fields `key_usages_white_list`, `key_usages_black_list`
  "key_usages_white_list": {
    "key_usage": [
      "KeyUsageExtension"
    ]
  },
  "key_usages_black_list": {
    "key_usage": [
      "KeyUsageExtension"
    ]
  },
  // end of the list of possible fields
  // Includes only one of the fields `extended_key_usages_white_list`, `extended_key_usages_black_list`
  "extended_key_usages_white_list": {
    "extended_key_usage": [
      "ExtendedKeyUsageExtension"
    ]
  },
  "extended_key_usages_black_list": {
    "extended_key_usage": [
      "ExtendedKeyUsageExtension"
    ]
  },
  // end of the list of possible fields
  // Includes only one of the fields `extension_white_list`, `extension_black_list`
  "extension_white_list": {
    "extension_oid": [
      "string"
    ]
  },
  "extension_black_list": {
    "extension_oid": [
      "string"
    ]
  },
  // end of the list of possible fields
  "update_mask": "google.protobuf.FieldMask"
}
```

Request message for updating a policy.

#|
||Field | Description ||
|| id | **string**

Required field. ID of the policy to update. ||
|| name | **string**

New name for the policy. ||
|| subject_policy | **[SubjectPolicy](#yandex.cloud.certificatemanager.v1.privateca.SubjectPolicy)**

New policy for the subject field of the certificate. ||
|| san_policy | **[SubjectAlternativeNamePolicy](#yandex.cloud.certificatemanager.v1.privateca.SubjectAlternativeNamePolicy)**

New policy for the subject alternative name (SAN) field of the certificate. ||
|| key_usages_white_list | **[KeyUsagePolicy](#yandex.cloud.certificatemanager.v1.privateca.KeyUsagePolicy)**

Whitelist of allowed key usages.

Includes only one of the fields `key_usages_white_list`, `key_usages_black_list`.

New policy for key usages. ||
|| key_usages_black_list | **[KeyUsagePolicy](#yandex.cloud.certificatemanager.v1.privateca.KeyUsagePolicy)**

Blacklist of disallowed key usages.

Includes only one of the fields `key_usages_white_list`, `key_usages_black_list`.

New policy for key usages. ||
|| extended_key_usages_white_list | **[ExtendedKeyUsagePolicy](#yandex.cloud.certificatemanager.v1.privateca.ExtendedKeyUsagePolicy)**

Whitelist of allowed extended key usages.

Includes only one of the fields `extended_key_usages_white_list`, `extended_key_usages_black_list`.

New policy for extended key usages. ||
|| extended_key_usages_black_list | **[ExtendedKeyUsagePolicy](#yandex.cloud.certificatemanager.v1.privateca.ExtendedKeyUsagePolicy)**

Blacklist of disallowed extended key usages.

Includes only one of the fields `extended_key_usages_white_list`, `extended_key_usages_black_list`.

New policy for extended key usages. ||
|| extension_white_list | **[ExtensionPolicy](#yandex.cloud.certificatemanager.v1.privateca.ExtensionPolicy)**

Whitelist of allowed extensions.

Includes only one of the fields `extension_white_list`, `extension_black_list`.

New policy for extensions. ||
|| extension_black_list | **[ExtensionPolicy](#yandex.cloud.certificatemanager.v1.privateca.ExtensionPolicy)**

Blacklist of disallowed extensions.

Includes only one of the fields `extension_white_list`, `extension_black_list`.

New policy for extensions. ||
|| update_mask | **[google.protobuf.FieldMask](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/field-mask)**

Field mask that specifies which fields of the policy are going to be updated.

Only the fields specified in the mask will be updated. ||
|#

## SubjectPolicy {#yandex.cloud.certificatemanager.v1.privateca.SubjectPolicy}

Policy for the subject field of a certificate. Each string field could be a regexp

#|
||Field | Description ||
|| country | **string**

Allowed country name in the subject field. ||
|| organization | **string**

Allowed organization name in the subject field. ||
|| organizational_unit | **string**

Allowed organizational unit name in the subject field. ||
|| distinguished_name_qualifier | **string**

Allowed distinguished name qualifier in the subject field. ||
|| state_or_province | **string**

Allowed state or province name in the subject field. ||
|| common_name | **string**

Allowed common name in the subject field. ||
|| email_address | **string**

Allowed email address in the subject field. ||
|| serial_number | **string**

Allowed serial number in the subject field. ||
|| locality | **string**

Allowed locality name in the subject field. ||
|| title | **string**

Allowed title in the subject field. ||
|| surname | **string**

Allowed surname in the subject field. ||
|| given_name | **string**

Allowed given name in the subject field. ||
|| initials | **string**

Allowed initials in the subject field. ||
|| generation_qualifier | **string**

Allowed generation qualifier in the subject field. ||
|#

## SubjectAlternativeNamePolicy {#yandex.cloud.certificatemanager.v1.privateca.SubjectAlternativeNamePolicy}

Policy for the Subject Alternative Name (SAN) field of a certificate. Each string field could be a regexp

#|
||Field | Description ||
|| other_name_oid | **string**

Allowed OIDs of the other name in the SAN field. ||
|| rfc_822_name | **string**

Allowed RFC 822 names (email addresses) in the SAN field. ||
|| dns_name | **string**

Allowed DNS names in the SAN field. ||
|| directory_name | **string**

Allowed directory names in the SAN field. ||
|| edi_party_name | **[EdiPartyName](#yandex.cloud.certificatemanager.v1.privateca.EdiPartyName)**

Allowed EDI party names in the SAN field. ||
|| uniform_resource_identifier | **string**

Allowed URIs in the SAN field. ||
|| ip_address | **string**

Allowed IP addresses in the SAN field. ||
|| registered_id | **string**

Allowed registered IDs in the SAN field. ||
|#

## EdiPartyName {#yandex.cloud.certificatemanager.v1.privateca.EdiPartyName}

#|
||Field | Description ||
|| name_assigner | **string**

Specifies the entity or authority that assigned the partyName ||
|| party_name | **string**

The actual identifier of the EDI party ||
|#

## KeyUsagePolicy {#yandex.cloud.certificatemanager.v1.privateca.KeyUsagePolicy}

Policy for key usages in a certificate.

#|
||Field | Description ||
|| key_usage[] | enum **KeyUsageExtension**

List of key usages that are allowed or disallowed.

- `KEY_USAGE_EXTENSION_UNSPECIFIED`
- `DIGITAL_SIGNATURE`
- `CONTENT_COMMITMENT`
- `KEY_ENCIPHERMENT`
- `DATA_ENCIPHERMENT`
- `KEY_AGREEMENT`
- `KEY_CERT_SIGN`
- `CRL_SIGN`
- `ENCIPHER_ONLY`
- `DECIPHER_ONLY` ||
|#

## ExtendedKeyUsagePolicy {#yandex.cloud.certificatemanager.v1.privateca.ExtendedKeyUsagePolicy}

Policy for extended key usages in a certificate.

#|
||Field | Description ||
|| extended_key_usage[] | enum **ExtendedKeyUsageExtension**

List of extended key usages that are allowed or disallowed.

- `EXTENDED_KEY_USAGE_EXTENSION_UNSPECIFIED`
- `SERVER_AUTH`
- `CLIENT_AUTH`
- `CODE_SIGNING`
- `EMAIL_PROTECTION`
- `TIME_STAMPING`
- `OCSP_SIGNING`
- `IPSEC_IKE`
- `MS_CODE_IND`
- `MS_CODE_COM`
- `MS_CTL_SIGN`
- `MS_EFS` ||
|#

## ExtensionPolicy {#yandex.cloud.certificatemanager.v1.privateca.ExtensionPolicy}

Policy for certificate extensions.

#|
||Field | Description ||
|| extension_oid[] | **string**

List of extension OIDs that are allowed or disallowed. ||
|#

## operation.Operation {#yandex.cloud.operation.Operation}

```json
{
  "id": "string",
  "description": "string",
  "created_at": "google.protobuf.Timestamp",
  "created_by": "string",
  "modified_at": "google.protobuf.Timestamp",
  "done": "bool",
  "metadata": {
    "id": "string"
  },
  // Includes only one of the fields `error`, `response`
  "error": "google.rpc.Status",
  "response": {
    "id": "string",
    "certificate_authority_id": "string",
    "name": "string",
    "subject_policy": {
      "country": "string",
      "organization": "string",
      "organizational_unit": "string",
      "distinguished_name_qualifier": "string",
      "state_or_province": "string",
      "common_name": "string",
      "email_address": "string",
      "serial_number": "string",
      "locality": "string",
      "title": "string",
      "surname": "string",
      "given_name": "string",
      "initials": "string",
      "generation_qualifier": "string"
    },
    "san_policy": {
      "other_name_oid": "string",
      "rfc_822_name": "string",
      "dns_name": "string",
      "directory_name": "string",
      "edi_party_name": {
        "name_assigner": "string",
        "party_name": "string"
      },
      "uniform_resource_identifier": "string",
      "ip_address": "string",
      "registered_id": "string"
    },
    // Includes only one of the fields `key_usages_white_list`, `key_usages_black_list`
    "key_usages_white_list": {
      "key_usage": [
        "KeyUsageExtension"
      ]
    },
    "key_usages_black_list": {
      "key_usage": [
        "KeyUsageExtension"
      ]
    },
    // end of the list of possible fields
    // Includes only one of the fields `extended_key_usages_white_list`, `extended_key_usages_black_list`
    "extended_key_usages_white_list": {
      "extended_key_usage": [
        "ExtendedKeyUsageExtension"
      ]
    },
    "extended_key_usages_black_list": {
      "extended_key_usage": [
        "ExtendedKeyUsageExtension"
      ]
    },
    // end of the list of possible fields
    // Includes only one of the fields `extension_white_list`, `extension_black_list`
    "extension_white_list": {
      "extension_oid": [
        "string"
      ]
    },
    "extension_black_list": {
      "extension_oid": [
        "string"
      ]
    },
    // end of the list of possible fields
    "created_at": "google.protobuf.Timestamp",
    "updated_at": "google.protobuf.Timestamp"
  }
  // end of the list of possible fields
}
```

An Operation resource. For more information, see [Operation](/docs/api-design-guide/concepts/operation).

#|
||Field | Description ||
|| id | **string**

ID of the operation. ||
|| description | **string**

Description of the operation. 0-256 characters long. ||
|| created_at | **[google.protobuf.Timestamp](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#timestamp)**

Creation timestamp. ||
|| created_by | **string**

ID of the user or service account who initiated the operation. ||
|| modified_at | **[google.protobuf.Timestamp](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#timestamp)**

The time when the Operation resource was last modified. ||
|| done | **bool**

If the value is `false`, it means the operation is still in progress.
If `true`, the operation is completed, and either `error` or `response` is available. ||
|| metadata | **[UpdatePolicyMetadata](#yandex.cloud.certificatemanager.v1.privateca.UpdatePolicyMetadata)**

Service-specific metadata associated with the operation.
It typically contains the ID of the target resource that the operation is performed on.
Any method that returns a long-running operation should document the metadata type, if any. ||
|| error | **[google.rpc.Status](https://cloud.google.com/tasks/docs/reference/rpc/google.rpc#status)**

The error result of the operation in case of failure or cancellation.

Includes only one of the fields `error`, `response`.

The operation result.
If `done == false` and there was no failure detected, neither `error` nor `response` is set.
If `done == false` and there was a failure detected, `error` is set.
If `done == true`, exactly one of `error` or `response` is set. ||
|| response | **[Policy](#yandex.cloud.certificatemanager.v1.privateca.Policy)**

The normal response of the operation in case of success.
If the original method returns no data on success, such as Delete,
the response is [google.protobuf.Empty](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#google.protobuf.Empty).
If the original method is the standard Create/Update,
the response should be the target resource of the operation.
Any method that returns a long-running operation should document the response type, if any.

Includes only one of the fields `error`, `response`.

The operation result.
If `done == false` and there was no failure detected, neither `error` nor `response` is set.
If `done == false` and there was a failure detected, `error` is set.
If `done == true`, exactly one of `error` or `response` is set. ||
|#

## UpdatePolicyMetadata {#yandex.cloud.certificatemanager.v1.privateca.UpdatePolicyMetadata}

Metadata for the UpdatePolicy operation.

#|
||Field | Description ||
|| id | **string**

ID of the policy being updated. ||
|#

## Policy {#yandex.cloud.certificatemanager.v1.privateca.Policy}

A policy defines the rules and constraints for issuing certificates within a certificate authority.

#|
||Field | Description ||
|| id | **string**

ID of the policy. ||
|| certificate_authority_id | **string**

ID of the certificate authority the policy belongs to. ||
|| name | **string**

Name of the policy. ||
|| subject_policy | **[SubjectPolicy](#yandex.cloud.certificatemanager.v1.privateca.SubjectPolicy2)**

Policy for the subject field of the certificate. ||
|| san_policy | **[SubjectAlternativeNamePolicy](#yandex.cloud.certificatemanager.v1.privateca.SubjectAlternativeNamePolicy2)**

Policy for the Subject Alternative Name (SAN) extension of the certificate. ||
|| key_usages_white_list | **[KeyUsagePolicy](#yandex.cloud.certificatemanager.v1.privateca.KeyUsagePolicy2)**

Whitelist of allowed key usages.

Includes only one of the fields `key_usages_white_list`, `key_usages_black_list`.

Policy for key usages. ||
|| key_usages_black_list | **[KeyUsagePolicy](#yandex.cloud.certificatemanager.v1.privateca.KeyUsagePolicy2)**

Blacklist of disallowed key usages.

Includes only one of the fields `key_usages_white_list`, `key_usages_black_list`.

Policy for key usages. ||
|| extended_key_usages_white_list | **[ExtendedKeyUsagePolicy](#yandex.cloud.certificatemanager.v1.privateca.ExtendedKeyUsagePolicy2)**

Whitelist of allowed extended key usages.

Includes only one of the fields `extended_key_usages_white_list`, `extended_key_usages_black_list`.

Policy for extended key usages. ||
|| extended_key_usages_black_list | **[ExtendedKeyUsagePolicy](#yandex.cloud.certificatemanager.v1.privateca.ExtendedKeyUsagePolicy2)**

Blacklist of disallowed extended key usages.

Includes only one of the fields `extended_key_usages_white_list`, `extended_key_usages_black_list`.

Policy for extended key usages. ||
|| extension_white_list | **[ExtensionPolicy](#yandex.cloud.certificatemanager.v1.privateca.ExtensionPolicy2)**

Whitelist of allowed extensions.

Includes only one of the fields `extension_white_list`, `extension_black_list`.

Policy for certificate extensions. ||
|| extension_black_list | **[ExtensionPolicy](#yandex.cloud.certificatemanager.v1.privateca.ExtensionPolicy2)**

Blacklist of disallowed extensions.

Includes only one of the fields `extension_white_list`, `extension_black_list`.

Policy for certificate extensions. ||
|| created_at | **[google.protobuf.Timestamp](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#timestamp)**

Timestamp when the policy was created. ||
|| updated_at | **[google.protobuf.Timestamp](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#timestamp)**

Timestamp when the policy was last updated. ||
|#

## SubjectPolicy {#yandex.cloud.certificatemanager.v1.privateca.SubjectPolicy2}

Policy for the subject field of a certificate. Each string field could be a regexp

#|
||Field | Description ||
|| country | **string**

Allowed country name in the subject field. ||
|| organization | **string**

Allowed organization name in the subject field. ||
|| organizational_unit | **string**

Allowed organizational unit name in the subject field. ||
|| distinguished_name_qualifier | **string**

Allowed distinguished name qualifier in the subject field. ||
|| state_or_province | **string**

Allowed state or province name in the subject field. ||
|| common_name | **string**

Allowed common name in the subject field. ||
|| email_address | **string**

Allowed email address in the subject field. ||
|| serial_number | **string**

Allowed serial number in the subject field. ||
|| locality | **string**

Allowed locality name in the subject field. ||
|| title | **string**

Allowed title in the subject field. ||
|| surname | **string**

Allowed surname in the subject field. ||
|| given_name | **string**

Allowed given name in the subject field. ||
|| initials | **string**

Allowed initials in the subject field. ||
|| generation_qualifier | **string**

Allowed generation qualifier in the subject field. ||
|#

## SubjectAlternativeNamePolicy {#yandex.cloud.certificatemanager.v1.privateca.SubjectAlternativeNamePolicy2}

Policy for the Subject Alternative Name (SAN) field of a certificate. Each string field could be a regexp

#|
||Field | Description ||
|| other_name_oid | **string**

Allowed OIDs of the other name in the SAN field. ||
|| rfc_822_name | **string**

Allowed RFC 822 names (email addresses) in the SAN field. ||
|| dns_name | **string**

Allowed DNS names in the SAN field. ||
|| directory_name | **string**

Allowed directory names in the SAN field. ||
|| edi_party_name | **[EdiPartyName](#yandex.cloud.certificatemanager.v1.privateca.EdiPartyName2)**

Allowed EDI party names in the SAN field. ||
|| uniform_resource_identifier | **string**

Allowed URIs in the SAN field. ||
|| ip_address | **string**

Allowed IP addresses in the SAN field. ||
|| registered_id | **string**

Allowed registered IDs in the SAN field. ||
|#

## EdiPartyName {#yandex.cloud.certificatemanager.v1.privateca.EdiPartyName2}

#|
||Field | Description ||
|| name_assigner | **string**

Specifies the entity or authority that assigned the partyName ||
|| party_name | **string**

The actual identifier of the EDI party ||
|#

## KeyUsagePolicy {#yandex.cloud.certificatemanager.v1.privateca.KeyUsagePolicy2}

Policy for key usages in a certificate.

#|
||Field | Description ||
|| key_usage[] | enum **KeyUsageExtension**

List of key usages that are allowed or disallowed.

- `KEY_USAGE_EXTENSION_UNSPECIFIED`
- `DIGITAL_SIGNATURE`
- `CONTENT_COMMITMENT`
- `KEY_ENCIPHERMENT`
- `DATA_ENCIPHERMENT`
- `KEY_AGREEMENT`
- `KEY_CERT_SIGN`
- `CRL_SIGN`
- `ENCIPHER_ONLY`
- `DECIPHER_ONLY` ||
|#

## ExtendedKeyUsagePolicy {#yandex.cloud.certificatemanager.v1.privateca.ExtendedKeyUsagePolicy2}

Policy for extended key usages in a certificate.

#|
||Field | Description ||
|| extended_key_usage[] | enum **ExtendedKeyUsageExtension**

List of extended key usages that are allowed or disallowed.

- `EXTENDED_KEY_USAGE_EXTENSION_UNSPECIFIED`
- `SERVER_AUTH`
- `CLIENT_AUTH`
- `CODE_SIGNING`
- `EMAIL_PROTECTION`
- `TIME_STAMPING`
- `OCSP_SIGNING`
- `IPSEC_IKE`
- `MS_CODE_IND`
- `MS_CODE_COM`
- `MS_CTL_SIGN`
- `MS_EFS` ||
|#

## ExtensionPolicy {#yandex.cloud.certificatemanager.v1.privateca.ExtensionPolicy2}

Policy for certificate extensions.

#|
||Field | Description ||
|| extension_oid[] | **string**

List of extension OIDs that are allowed or disallowed. ||
|#