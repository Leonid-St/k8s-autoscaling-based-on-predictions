---
editable: false
sourcePath: en/_cli-ref/cli-ref/storage/cli-ref/s3api/upload-part.md
---

# yc storage s3api upload-part

Uploads a part in a multipart upload

#### Command Usage

Syntax: 

`yc storage s3api upload-part [Flags...] [Global Flags...]`

#### Flags

| Flag | Description |
|----|----|
|`--bucket`|<b>`string`</b><br/>[REQUIRED] Bucket name|
|`--key`|<b>`string`</b><br/>[REQUIRED] Object key|
|`--upload-id`|<b>`string`</b><br/>Upload ID is used to reference a specific upload.|
|`--part-number`|<b>`int32`</b><br/>[REQUIRED] Part number of the object being uploaded.|
|`--body`|<b>`string`</b><br/>Object data.|
|`--content-md5`|<b>`string`</b><br/>128-bit MD5 hash value of the request body, base64 encoded.|
|`--content-length`|<b>`int`</b><br/>Length of the request body (without headers) in compliance with RFC 2616.|

#### Global Flags

| Flag | Description |
|----|----|
|`--profile`|<b>`string`</b><br/>Set the custom configuration file.|
|`--debug`|Debug logging.|
|`--debug-grpc`|Debug gRPC logging. Very verbose, used for debugging connection problems.|
|`--no-user-output`|Disable printing user intended output to stderr.|
|`--retry`|<b>`int`</b><br/>Enable gRPC retries. By default, retries are enabled with maximum 5 attempts.<br/>Pass 0 to disable retries. Pass any negative value for infinite retries.<br/>Even infinite retries are capped with 2 minutes timeout.|
|`--cloud-id`|<b>`string`</b><br/>Set the ID of the cloud to use.|
|`--folder-id`|<b>`string`</b><br/>Set the ID of the folder to use.|
|`--folder-name`|<b>`string`</b><br/>Set the name of the folder to use (will be resolved to id).|
|`--endpoint`|<b>`string`</b><br/>Set the Cloud API endpoint (host:port).|
|`--token`|<b>`string`</b><br/>Set the OAuth token to use.|
|`--impersonate-service-account-id`|<b>`string`</b><br/>Set the ID of the service account to impersonate.|
|`--no-browser`|Disable opening browser for authentication.|
|`--format`|<b>`string`</b><br/>Set the output format: text (default), yaml, json, json-rest.|
|`--jq`|<b>`string`</b><br/>Query to select values from the response using jq syntax|
|`-h`,`--help`|Display help for the command.|
