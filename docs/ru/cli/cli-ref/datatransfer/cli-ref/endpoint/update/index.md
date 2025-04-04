---
editable: false
sourcePath: en/_cli-ref/cli-ref/datatransfer/cli-ref/endpoint/update/index.md
---

# yc datatransfer endpoint update

Update endpoint

#### Command Usage

Syntax: 

`yc datatransfer endpoint update <command>`

#### Command Tree

- [yc datatransfer endpoint update clickhouse-source](clickhouse-source.md) — Update Clickhouse source
- [yc datatransfer endpoint update clickhouse-target](clickhouse-target.md) — Update Clickhouse target
- [yc datatransfer endpoint update mongo-source](mongo-source.md) — Update MongoDB source
- [yc datatransfer endpoint update mongo-target](mongo-target.md) — Update MongoDB target
- [yc datatransfer endpoint update mysql-source](mysql-source.md) — Update MySQL source
- [yc datatransfer endpoint update mysql-target](mysql-target.md) — Update MySQL target
- [yc datatransfer endpoint update postgres-source](postgres-source.md) — Update PostgreSQL source
- [yc datatransfer endpoint update postgres-target](postgres-target.md) — Update PostgreSQL target
- [yc datatransfer endpoint update yds-source](yds-source.md) — Update YDS source
- [yc datatransfer endpoint update yds-target](yds-target.md) — Update YDS target

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
