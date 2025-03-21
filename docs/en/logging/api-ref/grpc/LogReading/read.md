---
editable: false
sourcePath: en/_api-ref-grpc/logging/v1/api-ref/grpc/LogReading/read.md
---

# Cloud Logging Service, gRPC: LogReadingService.Read

Read log entries from the specified log group.

## gRPC request

**rpc Read ([ReadRequest](#yandex.cloud.logging.v1.ReadRequest)) returns ([ReadResponse](#yandex.cloud.logging.v1.ReadResponse))**

## ReadRequest {#yandex.cloud.logging.v1.ReadRequest}

```json
{
  // Includes only one of the fields `page_token`, `criteria`
  "page_token": "string",
  "criteria": {
    "log_group_id": "string",
    "resource_types": [
      "string"
    ],
    "resource_ids": [
      "string"
    ],
    "since": "google.protobuf.Timestamp",
    "until": "google.protobuf.Timestamp",
    "levels": [
      "Level"
    ],
    "filter": "string",
    "stream_names": [
      "string"
    ],
    "page_size": "int64",
    "max_response_size": "int64"
  }
  // end of the list of possible fields
}
```

#|
||Field | Description ||
|| page_token | **string**

Page token. To get the next page of results, set `page_token` to the
[ReadResponse.next_page_token](#yandex.cloud.logging.v1.ReadResponse) or [ReadResponse.previous_page_token](#yandex.cloud.logging.v1.ReadResponse) returned by a previous read request.

Includes only one of the fields `page_token`, `criteria`.

Read selector. ||
|| criteria | **[Criteria](#yandex.cloud.logging.v1.Criteria)**

Read criteria.

See [Criteria](#yandex.cloud.logging.v1.Criteria) for details.

Includes only one of the fields `page_token`, `criteria`.

Read selector. ||
|#

## Criteria {#yandex.cloud.logging.v1.Criteria}

Read criteria. Should be used in initial [ReadRequest](#yandex.cloud.logging.v1.ReadRequest).

#|
||Field | Description ||
|| log_group_id | **string**

Required field. ID of the log group to return.

To get a log group ID make a [LogGroupService.List](/docs/logging/api-ref/grpc/LogGroup/list#List) request. ||
|| resource_types[] | **string**

List of resource types to limit log entries to.

Empty list disables filter. ||
|| resource_ids[] | **string**

List of resource IDs to limit log entries to.

Empty list disables filter. ||
|| since | **[google.protobuf.Timestamp](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#timestamp)**

Lower bound of log entries timestamps. ||
|| until | **[google.protobuf.Timestamp](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#timestamp)**

Upper bound of log entries timestamps. ||
|| levels[] | enum **Level**

List of log levels to limit log entries to.

Empty list disables filter.

- `LEVEL_UNSPECIFIED`: Default log level.

  Equivalent to not specifying log level at all.
- `TRACE`: Trace log level.

  Possible use case: verbose logging of some business logic.
- `DEBUG`: Debug log level.

  Possible use case: debugging special cases in application logic.
- `INFO`: Info log level.

  Mostly used for information messages.
- `WARN`: Warn log level.

  May be used to alert about significant events.
- `ERROR`: Error log level.

  May be used to alert about errors in infrastructure, logic, etc.
- `FATAL`: Fatal log level.

  May be used to alert about unrecoverable failures and events. ||
|| filter | **string**

Filter expression. For details about filtering, see [documentation](/docs/logging/concepts/filter). ||
|| stream_names[] | **string**

List of stream names to limit log entries to.

Empty list disables filter. ||
|| page_size | **int64**

The maximum number of results per page to return. ||
|| max_response_size | **int64**

Limits response to maximum size in bytes. Prevents gRPC resource exhaustion.

Default value for max response size is 3.5 MiB ||
|#

## ReadResponse {#yandex.cloud.logging.v1.ReadResponse}

```json
{
  "log_group_id": "string",
  "entries": [
    {
      "uid": "string",
      "resource": {
        "type": "string",
        "id": "string"
      },
      "timestamp": "google.protobuf.Timestamp",
      "ingested_at": "google.protobuf.Timestamp",
      "saved_at": "google.protobuf.Timestamp",
      "level": "Level",
      "message": "string",
      "json_payload": "google.protobuf.Struct",
      "stream_name": "string"
    }
  ],
  "next_page_token": "string",
  "previous_page_token": "string"
}
```

#|
||Field | Description ||
|| log_group_id | **string**

Log group ID the read was performed from. ||
|| entries[] | **[LogEntry](#yandex.cloud.logging.v1.LogEntry)**

List of matching log entries. ||
|| next_page_token | **string**

Token for getting the next page of the log entries.

After getting log entries initially with [Criteria](#yandex.cloud.logging.v1.Criteria), you can use `next_page_token` as the value
for the [ReadRequest.page_token](#yandex.cloud.logging.v1.ReadRequest) parameter in the next read request.

Each subsequent page will have its own `next_page_token` to continue paging through the results. ||
|| previous_page_token | **string**

Token for getting the previous page of the log entries.

After getting log entries initially with [Criteria](#yandex.cloud.logging.v1.Criteria), you can use `previous_page_token` as the value
for the [ReadRequest.page_token](#yandex.cloud.logging.v1.ReadRequest) parameter in the next read request.

Each subsequent page will have its own `next_page_token` to continue paging through the results. ||
|#

## LogEntry {#yandex.cloud.logging.v1.LogEntry}

#|
||Field | Description ||
|| uid | **string**

Unique entry ID.

Useful for logs deduplication. ||
|| resource | **[LogEntryResource](#yandex.cloud.logging.v1.LogEntryResource)**

Entry resource specification.

May contain information about source service and resource ID.
Also may be provided by the user. ||
|| timestamp | **[google.protobuf.Timestamp](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#timestamp)**

Timestamp of the entry. ||
|| ingested_at | **[google.protobuf.Timestamp](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#timestamp)**

Entry ingestion time observed by [LogIngestionService](/docs/logging/api-ref/grpc/LogIngestion/). ||
|| saved_at | **[google.protobuf.Timestamp](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#timestamp)**

Entry save time.

Entry is ready to be read since this moment. ||
|| level | enum **Level**

Entry severity.

See [LogLevel.Level](/docs/logging/api-ref/grpc/Export/run#yandex.cloud.logging.v1.LogLevel.Level) for details.

- `LEVEL_UNSPECIFIED`: Default log level.

  Equivalent to not specifying log level at all.
- `TRACE`: Trace log level.

  Possible use case: verbose logging of some business logic.
- `DEBUG`: Debug log level.

  Possible use case: debugging special cases in application logic.
- `INFO`: Info log level.

  Mostly used for information messages.
- `WARN`: Warn log level.

  May be used to alert about significant events.
- `ERROR`: Error log level.

  May be used to alert about errors in infrastructure, logic, etc.
- `FATAL`: Fatal log level.

  May be used to alert about unrecoverable failures and events. ||
|| message | **string**

Entry text message. ||
|| json_payload | **[google.protobuf.Struct](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/struct)**

Entry annotation. ||
|| stream_name | **string**

Entry stream name. ||
|#

## LogEntryResource {#yandex.cloud.logging.v1.LogEntryResource}

Log entry resource specification.

May be used either by services and by user.

#|
||Field | Description ||
|| type | **string**

Resource type, i.e., `serverless.function` ||
|| id | **string**

Resource ID, i.e., ID of the function producing logs. ||
|#