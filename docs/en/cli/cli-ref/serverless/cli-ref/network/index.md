---
editable: false
sourcePath: en/_cli-ref/cli-ref/serverless/cli-ref/network/index.md
---

# yc serverless network

Manage networks used in serverless resources

#### Command Usage

Syntax: 

`yc serverless network <command>`

Aliases: 

- `networks`

#### Command Tree

- [yc serverless network get-used](get-used.md) — Show information about the specified network used in serverless resources
- [yc serverless network list-connections](list-connections.md) — List serverless resources connected to any network from specified scope (network, folder or cloud)
- [yc serverless network list-used](list-used.md) — List networks used in serverless resources in specified scope
- [yc serverless network trigger-used-cleanup](trigger-used-cleanup.md) — Force obsolete used network to start cleanup process as soon as possible

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
