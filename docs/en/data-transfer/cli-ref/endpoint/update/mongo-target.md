---
editable: false
sourcePath: en/_cli-ref/cli-ref/datatransfer/cli-ref/endpoint/update/mongo-target.md
---

# yc datatransfer endpoint update mongo-target

Update MongoDB target

#### Command Usage

Syntax: 

`yc datatransfer endpoint update mongo-target [<ENDPOINT-ID>] [Flags...] [Global Flags...]`

#### Flags

| Flag | Description |
|----|----|
|`--file`|<b>`string`</b><br/>Path to a file with grpc request, see proto files in the protobuf API specification: https://github.com/yandex-cloud/cloudapi/tree/master/yandex/cloud/datatransfer/v1|
|`--name`|<b>`string`</b><br/>Endpoint name|
|`--description`|<b>`string`</b><br/>Endpoint description|
|`--host`|<b>`value[,value]`</b><br/>MongoDB server host name or IP address|
|`--port`|<b>`int`</b><br/>MongoDB server host name or IP address|
|`--ca-certificate`|<b>`string`</b><br/>Path to the CA certificate of the MongoDB server. Implies enabled TLS|
|`--replica-set`|<b>`string`</b><br/>Replica set|
|`--cluster-id`|<b>`string`</b><br/>Yandex Managed MongoDB cluster ID|
|`--user`|<b>`string`</b><br/>User name|
|`--raw-password`|<b>`string`</b><br/>Raw password value (unsafe, consider --password-file)|
|`--password-file`|<b>`string`</b><br/>Path to the file containing password|
|`--auth-source`|<b>`string`</b><br/>Database name associated with the credentials|
|`--target-database`|<b>`string`</b><br/>Name of the database where user collections will be created. If not specified, the original database name will be used|
|`--subnet-id`|<b>`string`</b><br/>Yandex Virtual Private Cloud subnet ID to access the MongoDB server through|
|`--security-group`|<b>`value[,value]`</b><br/>Yandex Virtual Private Cloud security group ID to associate with the endpoint|
|`--cleanup-policy`|<b>`string`</b><br/>Policy describes what to do with existing data on target. Options: drop, truncate, disabled (keep the data, which is default)|

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
