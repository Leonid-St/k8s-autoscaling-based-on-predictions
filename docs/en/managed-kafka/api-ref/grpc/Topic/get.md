---
editable: false
sourcePath: en/_api-ref-grpc/mdb/kafka/v1/api-ref/grpc/Topic/get.md
---

# Managed Service for Apache Kafka® API, gRPC: TopicService.Get

Returns the specified Kafka topic.

To get the list of available Kafka topics, make a [List](/docs/managed-kafka/api-ref/grpc/Topic/list#List) request.

## gRPC request

**rpc Get ([GetTopicRequest](#yandex.cloud.mdb.kafka.v1.GetTopicRequest)) returns ([Topic](#yandex.cloud.mdb.kafka.v1.Topic))**

## GetTopicRequest {#yandex.cloud.mdb.kafka.v1.GetTopicRequest}

```json
{
  "cluster_id": "string",
  "topic_name": "string"
}
```

#|
||Field | Description ||
|| cluster_id | **string**

Required field. ID of the Apache Kafka® cluster that the topic belongs to.

To get the cluster ID, make a [ClusterService.List](/docs/managed-kafka/api-ref/grpc/Cluster/list#List) request. ||
|| topic_name | **string**

Required field. Name of the Kafka topic resource to return.

To get the name of the topic, make a [TopicService.List](/docs/managed-kafka/api-ref/grpc/Topic/list#List) request. ||
|#

## Topic {#yandex.cloud.mdb.kafka.v1.Topic}

```json
{
  "name": "string",
  "cluster_id": "string",
  "partitions": "google.protobuf.Int64Value",
  "replication_factor": "google.protobuf.Int64Value",
  // Includes only one of the fields `topic_config_2_8`, `topic_config_3`
  "topic_config_2_8": {
    "cleanup_policy": "CleanupPolicy",
    "compression_type": "CompressionType",
    "delete_retention_ms": "google.protobuf.Int64Value",
    "file_delete_delay_ms": "google.protobuf.Int64Value",
    "flush_messages": "google.protobuf.Int64Value",
    "flush_ms": "google.protobuf.Int64Value",
    "min_compaction_lag_ms": "google.protobuf.Int64Value",
    "retention_bytes": "google.protobuf.Int64Value",
    "retention_ms": "google.protobuf.Int64Value",
    "max_message_bytes": "google.protobuf.Int64Value",
    "min_insync_replicas": "google.protobuf.Int64Value",
    "segment_bytes": "google.protobuf.Int64Value",
    "preallocate": "google.protobuf.BoolValue"
  },
  "topic_config_3": {
    "cleanup_policy": "CleanupPolicy",
    "compression_type": "CompressionType",
    "delete_retention_ms": "google.protobuf.Int64Value",
    "file_delete_delay_ms": "google.protobuf.Int64Value",
    "flush_messages": "google.protobuf.Int64Value",
    "flush_ms": "google.protobuf.Int64Value",
    "min_compaction_lag_ms": "google.protobuf.Int64Value",
    "retention_bytes": "google.protobuf.Int64Value",
    "retention_ms": "google.protobuf.Int64Value",
    "max_message_bytes": "google.protobuf.Int64Value",
    "min_insync_replicas": "google.protobuf.Int64Value",
    "segment_bytes": "google.protobuf.Int64Value",
    "preallocate": "google.protobuf.BoolValue"
  }
  // end of the list of possible fields
}
```

An Kafka topic.
For more information, see the [Concepts -> Topics and partitions](/docs/managed-kafka/concepts/topics) section of the documentation.

#|
||Field | Description ||
|| name | **string**

Name of the topic. ||
|| cluster_id | **string**

ID of an Apache Kafka® cluster that the topic belongs to.

To get the Apache Kafka® cluster ID, make a [ClusterService.List](/docs/managed-kafka/api-ref/grpc/Cluster/list#List) request. ||
|| partitions | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

The number of the topic's partitions. ||
|| replication_factor | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

Amount of data copies (replicas) for the topic in the cluster. ||
|| topic_config_2_8 | **[TopicConfig2_8](#yandex.cloud.mdb.kafka.v1.TopicConfig2_8)**

Includes only one of the fields `topic_config_2_8`, `topic_config_3`.

User-defined settings for the topic. ||
|| topic_config_3 | **[TopicConfig3](#yandex.cloud.mdb.kafka.v1.TopicConfig3)**

Includes only one of the fields `topic_config_2_8`, `topic_config_3`.

User-defined settings for the topic. ||
|#

## TopicConfig2_8 {#yandex.cloud.mdb.kafka.v1.TopicConfig2_8}

A topic settings for 2.8

#|
||Field | Description ||
|| cleanup_policy | enum **CleanupPolicy**

Retention policy to use on old log messages.

- `CLEANUP_POLICY_UNSPECIFIED`
- `CLEANUP_POLICY_DELETE`: This policy discards log segments when either their retention time or log size limit is reached. See also: [KafkaConfig2_8.log_retention_ms](/docs/managed-kafka/api-ref/grpc/Cluster/get#yandex.cloud.mdb.kafka.v1.KafkaConfig2_8) and other similar parameters.
- `CLEANUP_POLICY_COMPACT`: This policy compacts messages in log.
- `CLEANUP_POLICY_COMPACT_AND_DELETE`: This policy use both compaction and deletion for messages and log segments. ||
|| compression_type | enum **CompressionType**

The compression type for a given topic.

- `COMPRESSION_TYPE_UNSPECIFIED`
- `COMPRESSION_TYPE_UNCOMPRESSED`: no codec (uncompressed).
- `COMPRESSION_TYPE_ZSTD`: Zstandard codec.
- `COMPRESSION_TYPE_LZ4`: LZ4 codec.
- `COMPRESSION_TYPE_SNAPPY`: Snappy codec.
- `COMPRESSION_TYPE_GZIP`: GZip codec.
- `COMPRESSION_TYPE_PRODUCER`: the codec to use is set by a producer (can be any of `ZSTD`, `LZ4`, `GZIP` or `SNAPPY` codecs). ||
|| delete_retention_ms | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

The amount of time in milliseconds to retain delete tombstone markers for log compacted topics. ||
|| file_delete_delay_ms | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

The time to wait before deleting a file from the filesystem. ||
|| flush_messages | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

The number of messages accumulated on a log partition before messages are flushed to disk.

This setting overrides the cluster-level [KafkaConfig2_8.log_flush_interval_messages](/docs/managed-kafka/api-ref/grpc/Cluster/get#yandex.cloud.mdb.kafka.v1.KafkaConfig2_8) setting on the topic level. ||
|| flush_ms | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

The maximum time in milliseconds that a message in the topic is kept in memory before flushed to disk.

This setting overrides the cluster-level [KafkaConfig2_8.log_flush_interval_ms](/docs/managed-kafka/api-ref/grpc/Cluster/get#yandex.cloud.mdb.kafka.v1.KafkaConfig2_8) setting on the topic level. ||
|| min_compaction_lag_ms | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

The minimum time in milliseconds a message will remain uncompacted in the log. ||
|| retention_bytes | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

The maximum size a partition can grow to before Kafka will discard old log segments to free up space if the `delete` `cleanup_policy` is in effect.
It is helpful if you need to control the size of log due to limited disk space.

This setting overrides the cluster-level [KafkaConfig2_8.log_retention_bytes](/docs/managed-kafka/api-ref/grpc/Cluster/get#yandex.cloud.mdb.kafka.v1.KafkaConfig2_8) setting on the topic level. ||
|| retention_ms | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

The number of milliseconds to keep a log segment's file before deleting it.

This setting overrides the cluster-level [KafkaConfig2_8.log_retention_ms](/docs/managed-kafka/api-ref/grpc/Cluster/get#yandex.cloud.mdb.kafka.v1.KafkaConfig2_8) setting on the topic level. ||
|| max_message_bytes | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

The largest record batch size allowed in topic. ||
|| min_insync_replicas | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

This configuration specifies the minimum number of replicas that must acknowledge a write to topic for the write
to be considered successful (when a producer sets acks to "all"). ||
|| segment_bytes | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

This configuration controls the segment file size for the log. Retention and cleaning is always done a file
at a time so a larger segment size means fewer files but less granular control over retention.

This setting overrides the cluster-level [KafkaConfig2_8.log_segment_bytes](/docs/managed-kafka/api-ref/grpc/Cluster/get#yandex.cloud.mdb.kafka.v1.KafkaConfig2_8) setting on the topic level. ||
|| preallocate | **[google.protobuf.BoolValue](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/bool-value)**

True if we should preallocate the file on disk when creating a new log segment.

This setting overrides the cluster-level [KafkaConfig2_8.log_preallocate](/docs/managed-kafka/api-ref/grpc/Cluster/get#yandex.cloud.mdb.kafka.v1.KafkaConfig2_8) setting on the topic level. ||
|#

## TopicConfig3 {#yandex.cloud.mdb.kafka.v1.TopicConfig3}

A topic settings for 3.x

#|
||Field | Description ||
|| cleanup_policy | enum **CleanupPolicy**

Retention policy to use on old log messages.

- `CLEANUP_POLICY_UNSPECIFIED`
- `CLEANUP_POLICY_DELETE`: This policy discards log segments when either their retention time or log size limit is reached. See also: [KafkaConfig3.log_retention_ms](/docs/managed-kafka/api-ref/grpc/Cluster/get#yandex.cloud.mdb.kafka.v1.KafkaConfig3) and other similar parameters.
- `CLEANUP_POLICY_COMPACT`: This policy compacts messages in log.
- `CLEANUP_POLICY_COMPACT_AND_DELETE`: This policy use both compaction and deletion for messages and log segments. ||
|| compression_type | enum **CompressionType**

The compression type for a given topic.

- `COMPRESSION_TYPE_UNSPECIFIED`
- `COMPRESSION_TYPE_UNCOMPRESSED`: no codec (uncompressed).
- `COMPRESSION_TYPE_ZSTD`: Zstandard codec.
- `COMPRESSION_TYPE_LZ4`: LZ4 codec.
- `COMPRESSION_TYPE_SNAPPY`: Snappy codec.
- `COMPRESSION_TYPE_GZIP`: GZip codec.
- `COMPRESSION_TYPE_PRODUCER`: the codec to use is set by a producer (can be any of `ZSTD`, `LZ4`, `GZIP` or `SNAPPY` codecs). ||
|| delete_retention_ms | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

The amount of time in milliseconds to retain delete tombstone markers for log compacted topics. ||
|| file_delete_delay_ms | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

The time to wait before deleting a file from the filesystem. ||
|| flush_messages | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

The number of messages accumulated on a log partition before messages are flushed to disk.

This setting overrides the cluster-level [KafkaConfig3.log_flush_interval_messages](/docs/managed-kafka/api-ref/grpc/Cluster/get#yandex.cloud.mdb.kafka.v1.KafkaConfig3) setting on the topic level. ||
|| flush_ms | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

The maximum time in milliseconds that a message in the topic is kept in memory before flushed to disk.

This setting overrides the cluster-level [KafkaConfig3.log_flush_interval_ms](/docs/managed-kafka/api-ref/grpc/Cluster/get#yandex.cloud.mdb.kafka.v1.KafkaConfig3) setting on the topic level. ||
|| min_compaction_lag_ms | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

The minimum time in milliseconds a message will remain uncompacted in the log. ||
|| retention_bytes | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

The maximum size a partition can grow to before Kafka will discard old log segments to free up space if the `delete` `cleanup_policy` is in effect.
It is helpful if you need to control the size of log due to limited disk space.

This setting overrides the cluster-level [KafkaConfig3.log_retention_bytes](/docs/managed-kafka/api-ref/grpc/Cluster/get#yandex.cloud.mdb.kafka.v1.KafkaConfig3) setting on the topic level. ||
|| retention_ms | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

The number of milliseconds to keep a log segment's file before deleting it.

This setting overrides the cluster-level [KafkaConfig3.log_retention_ms](/docs/managed-kafka/api-ref/grpc/Cluster/get#yandex.cloud.mdb.kafka.v1.KafkaConfig3) setting on the topic level. ||
|| max_message_bytes | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

The largest record batch size allowed in topic. ||
|| min_insync_replicas | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

This configuration specifies the minimum number of replicas that must acknowledge a write to topic for the write
to be considered successful (when a producer sets acks to "all"). ||
|| segment_bytes | **[google.protobuf.Int64Value](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/int64-value)**

This configuration controls the segment file size for the log. Retention and cleaning is always done a file
at a time so a larger segment size means fewer files but less granular control over retention.

This setting overrides the cluster-level [KafkaConfig3.log_segment_bytes](/docs/managed-kafka/api-ref/grpc/Cluster/get#yandex.cloud.mdb.kafka.v1.KafkaConfig3) setting on the topic level. ||
|| preallocate | **[google.protobuf.BoolValue](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/bool-value)**

True if we should preallocate the file on disk when creating a new log segment.

This setting overrides the cluster-level [KafkaConfig3.log_preallocate](/docs/managed-kafka/api-ref/grpc/Cluster/get#yandex.cloud.mdb.kafka.v1.KafkaConfig3) setting on the topic level. ||
|#