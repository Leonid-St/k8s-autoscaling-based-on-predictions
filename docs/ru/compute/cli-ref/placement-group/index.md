---
editable: false
sourcePath: en/_cli-ref/cli-ref/compute/cli-ref/placement-group/index.md
---

# yc compute placement-group

Manage placement groups

#### Command Usage

Syntax: 

`yc compute placement-group <command>`

Aliases: 

- `placement-groups`
- `pg`

#### Command Tree

- [yc compute placement-group add-access-binding](add-access-binding.md) — Add access binding for the specified placement group
- [yc compute placement-group add-labels](add-labels.md) — Add labels to the specified placement group
- [yc compute placement-group create](create.md) — Create a placement group
- [yc compute placement-group delete](delete.md) — Delete the specified placement group
- [yc compute placement-group get](get.md) — Show information about the specified placement group
- [yc compute placement-group list](list.md) — List placement groups
- [yc compute placement-group list-access-bindings](list-access-bindings.md) — List access bindings for the specified placement group
- [yc compute placement-group list-instances](list-instances.md) — List instances of the specified placement group
- [yc compute placement-group list-operations](list-operations.md) — List operations for the specified placement group
- [yc compute placement-group remove-access-binding](remove-access-binding.md) — Remove access binding for the specified placement group
- [yc compute placement-group remove-labels](remove-labels.md) — Remove labels from the specified placement group
- [yc compute placement-group set-access-bindings](set-access-bindings.md) — Set access bindings for the specified placement group and delete all existing access bindings if there were any
- [yc compute placement-group update](update.md) — Update the specified placement group

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
