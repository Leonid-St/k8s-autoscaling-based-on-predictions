---
editable: false
sourcePath: en/_api-ref-grpc/serverless/workflows/v1/workflows/api-ref/grpc/Execution/terminate.md
---

# Workflows Service, gRPC: ExecutionService.Terminate

Terminates specified Workflow execution.

## gRPC request

**rpc Terminate ([TerminateExecutionRequest](#yandex.cloud.serverless.workflows.v1.TerminateExecutionRequest)) returns ([TerminateExecutionResponse](#yandex.cloud.serverless.workflows.v1.TerminateExecutionResponse))**

## TerminateExecutionRequest {#yandex.cloud.serverless.workflows.v1.TerminateExecutionRequest}

```json
{
  "execution_id": "string"
}
```

#|
||Field | Description ||
|| execution_id | **string**

Required field. ID of the Workflow execution. ||
|#

## TerminateExecutionResponse {#yandex.cloud.serverless.workflows.v1.TerminateExecutionResponse}

```json
{
  "execution_id": "string"
}
```

#|
||Field | Description ||
|| execution_id | **string**

ID of the Workflow execution. ||
|#