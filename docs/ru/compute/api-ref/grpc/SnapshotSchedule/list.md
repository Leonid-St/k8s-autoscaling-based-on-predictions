---
editable: false
sourcePath: en/_api-ref-grpc/compute/v1/api-ref/grpc/SnapshotSchedule/list.md
---

# Compute Cloud API, gRPC: SnapshotScheduleService.List

Retrieves the list of snapshot schedules in the specified folder.

## gRPC request

**rpc List ([ListSnapshotSchedulesRequest](#yandex.cloud.compute.v1.ListSnapshotSchedulesRequest)) returns ([ListSnapshotSchedulesResponse](#yandex.cloud.compute.v1.ListSnapshotSchedulesResponse))**

## ListSnapshotSchedulesRequest {#yandex.cloud.compute.v1.ListSnapshotSchedulesRequest}

```json
{
  "folder_id": "string",
  "page_size": "int64",
  "page_token": "string",
  "filter": "string",
  "order_by": "string"
}
```

#|
||Field | Description ||
|| folder_id | **string**

Required field. ID of the folder to list snapshot schedules in.

To get the folder ID, make a [yandex.cloud.resourcemanager.v1.FolderService.List](/docs/resource-manager/api-ref/grpc/Folder/list#List) request. ||
|| page_size | **int64**

The maximum number of results per page to return. If the number of available
results is larger than `page_size`, the service returns a [ListSnapshotSchedulesResponse.next_page_token](#yandex.cloud.compute.v1.ListSnapshotSchedulesResponse)
that can be used to get the next page of results in subsequent list requests. ||
|| page_token | **string**

Page token. To get the next page of results, set `page_token` to the
[ListSnapshotSchedulesResponse.next_page_token](#yandex.cloud.compute.v1.ListSnapshotSchedulesResponse) returned by a previous list request. ||
|| filter | **string**

A filter expression that filters snapshot schedules listed in the response.

The expression must specify:
1. The field name. Currently you can use filtering only on [SnapshotSchedule.name](#yandex.cloud.compute.v1.SnapshotSchedule) field.
2. An operator. Can be either `=` or `!=` for single values, `IN` or `NOT IN` for lists of values.
3. The value. Must be 3-63 characters long and match the regular expression `^[a-z][-a-z0-9]{1,61}[a-z0-9]`.
Example of a filter: `name=my-schedule`. ||
|| order_by | **string**

A sorting expression that sorts snapshot schedules listed in the response.

The expression must specify the field name from [SnapshotSchedule](#yandex.cloud.compute.v1.SnapshotSchedule) and `asc`ending or `desc`ending order,
e.g. `createdAt desc`.

Default value: `id asc`. ||
|#

## ListSnapshotSchedulesResponse {#yandex.cloud.compute.v1.ListSnapshotSchedulesResponse}

```json
{
  "snapshot_schedules": [
    {
      "id": "string",
      "folder_id": "string",
      "created_at": "google.protobuf.Timestamp",
      "name": "string",
      "description": "string",
      "labels": "map<string, string>",
      "status": "Status",
      "schedule_policy": {
        "start_at": "google.protobuf.Timestamp",
        "expression": "string"
      },
      // Includes only one of the fields `retention_period`, `snapshot_count`
      "retention_period": "google.protobuf.Duration",
      "snapshot_count": "int64",
      // end of the list of possible fields
      "snapshot_spec": {
        "description": "string",
        "labels": "map<string, string>"
      }
    }
  ],
  "next_page_token": "string"
}
```

#|
||Field | Description ||
|| snapshot_schedules[] | **[SnapshotSchedule](#yandex.cloud.compute.v1.SnapshotSchedule)**

List of snapshot schedules in the specified folder. ||
|| next_page_token | **string**

Token for getting the next page of the list. If the number of results is greater than
the specified [ListSnapshotSchedulesRequest.page_size](#yandex.cloud.compute.v1.ListSnapshotSchedulesRequest), use `next_page_token` as the value
for the [ListSnapshotSchedulesRequest.page_token](#yandex.cloud.compute.v1.ListSnapshotSchedulesRequest) parameter in the next list request.

Each subsequent page will have its own `next_page_token` to continue paging through the results. ||
|#

## SnapshotSchedule {#yandex.cloud.compute.v1.SnapshotSchedule}

A snapshot schedule. For details about the concept, see [documentation](/docs/compute/concepts/snapshot-schedule).

#|
||Field | Description ||
|| id | **string**

ID of the snapshot schedule. ||
|| folder_id | **string**

ID of the folder that the snapshot schedule belongs to. ||
|| created_at | **[google.protobuf.Timestamp](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#timestamp)**

Creation timestamp. ||
|| name | **string**

Name of the snapshot schedule.

The name is unique within the folder. ||
|| description | **string**

Description of the snapshot schedule. ||
|| labels | **object** (map<**string**, **string**>)

Snapshot schedule labels as `key:value` pairs. ||
|| status | enum **Status**

Status of the snapshot schedule.

- `STATUS_UNSPECIFIED`
- `CREATING`: The snapshot schedule is being created.
- `ACTIVE`: The snapshot schedule is on: new disk snapshots will be created, old ones deleted
(if [SnapshotSchedule.retention_policy](/docs/compute/api-ref/grpc/Disk/listSnapshotSchedules#yandex.cloud.compute.v1.SnapshotSchedule.retention_policy) is specified).
- `INACTIVE`: The schedule is interrupted, snapshots won't be created or deleted.
- `DELETING`: The schedule is being deleted.
- `UPDATING`: Changes are being made to snapshot schedule settings or a list of attached disks. ||
|| schedule_policy | **[SchedulePolicy](#yandex.cloud.compute.v1.SchedulePolicy)**

Frequency settings of the snapshot schedule. ||
|| retention_period | **[google.protobuf.Duration](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/duration)**

Retention period of the snapshot schedule. Once a snapshot created by the schedule reaches this age, it is
automatically deleted.

Includes only one of the fields `retention_period`, `snapshot_count`.

Retention policy of the snapshot schedule. ||
|| snapshot_count | **int64**

Retention count of the snapshot schedule. Once the number of snapshots created by the schedule exceeds this
number, the oldest ones are automatically deleted. E.g. if the number is 5, the first snapshot is deleted
after the sixth one is created, the second is deleted after the seventh one is created, and so on.

Includes only one of the fields `retention_period`, `snapshot_count`.

Retention policy of the snapshot schedule. ||
|| snapshot_spec | **[SnapshotSpec](#yandex.cloud.compute.v1.SnapshotSpec)**

Attributes of snapshots created by the snapshot schedule. ||
|#

## SchedulePolicy {#yandex.cloud.compute.v1.SchedulePolicy}

A resource for frequency settings of a snapshot schedule.

#|
||Field | Description ||
|| start_at | **[google.protobuf.Timestamp](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#timestamp)**

Timestamp for creating the first snapshot. ||
|| expression | **string**

Cron expression for the snapshot schedule (UTC+0).

The expression must consist of five fields (`Minutes Hours Day-of-month Month Day-of-week`) or be one of
nonstandard predefined expressions (e.g. `@hourly`). For details about the format,
see [documentation](/docs/compute/concepts/snapshot-schedule#cron) ||
|#

## SnapshotSpec {#yandex.cloud.compute.v1.SnapshotSpec}

A resource for attributes of snapshots created by the snapshot schedule.

#|
||Field | Description ||
|| description | **string**

Description of the created snapshot. ||
|| labels | **object** (map<**string**, **string**>)

Snapshot labels as `key:value` pairs. ||
|#