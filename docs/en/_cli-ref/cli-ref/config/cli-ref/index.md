---
editable: false
---

# yc config

The 'yc config' command group lets you set, view and unset properties used by Yandex Cloud CLI.

#### Command Usage

Syntax: 

`yc config <group|command>`

#### Command Tree

- [yc config get](get.md) — Get value for the specified configuration property
- [yc config list](list.md) — List configuration values
- [yc config set](set.md) — Set value for the specified configuration property
- [yc config unset](unset.md) — Unset value for the specified configuration property
- [yc config profile](profile/index.md) — Manage configuration profiles
	- [yc config profile activate](profile/activate.md) — Activate the specified configuration profile
	- [yc config profile create](profile/create.md) — Create a configuration profile
	- [yc config profile delete](profile/delete.md) — Delete the specified configuration profile
	- [yc config profile get](profile/get.md) — List values for the specified configuration profile
	- [yc config profile list](profile/list.md) — List configuration profiles

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
