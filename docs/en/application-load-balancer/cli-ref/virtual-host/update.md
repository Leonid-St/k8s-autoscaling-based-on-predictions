---
editable: false
sourcePath: en/_cli-ref/cli-ref/application-load-balancer/cli-ref/virtual-host/update.md
---

# yc application-load-balancer virtual-host update

Update the specified virtual host

#### Command Usage

Syntax: 

`yc application-load-balancer virtual-host update <VIRTUAL-HOST-NAME> [Flags...] [Global Flags...]`

#### Flags

| Flag | Description |
|----|----|
|`--http-router-id`|<b>`string`</b><br/>ID of the Http router.|
|`--http-router-name`|<b>`string`</b><br/>Name of the Http router.|
|`--async`|Display information about the operation in progress, without waiting for the operation to complete.|
|`--authority`|<b>`value[,value]`</b><br/>Specifies domains (:authority/Host header values) that will be matched to this virtual host. Wildcard hosts are supported in the form of '*.foo.com' or '*-bar.foo.com'|
|`--clear-authorities`|Clears authority match specifications for the virtual host. Virtual host with no authority specified, matches all :authority/Host headers|
|`--modify-request-header`|<b>`PROPERTY=VALUE[,PROPERTY=VALUE...]`</b><br/>Specifies modifications to be applied to the request headers.<br/><br/>Possible property names:<br/><ul> <li><code>name</code>:     Specifies the name of the header to modify.</li> <li><code>append</code>:     Specifies the value to be appended to the header.</li> <li><code>replace</code>:     Specifies the new value for the header.</li> <li><code>rename</code>:     Specifies the new name fot the header.</li> <li><code>remove</code>:     Remove the specified header</li> </ul>|
|`--clear-request-header-modifications`|Clears all request header modifications for the virtual host.|
|`--modify-response-header`|<b>`PROPERTY=VALUE[,PROPERTY=VALUE...]`</b><br/>Specifies modifications to be applied to the response headers.<br/><br/>Possible property names:<br/><ul> <li><code>name</code>:     Specifies the name of the header to modify.</li> <li><code>append</code>:     Specifies the value to be appended to the header.</li> <li><code>replace</code>:     Specifies the new value for the header.</li> <li><code>rename</code>:     Specifies the new name fot the header.</li> <li><code>remove</code>:     Remove the specified header</li> </ul>|
|`--clear-response-header-modifications`|Clears all response header modifications for the virtual host.|
|`--rate-limit`|<b>`PROPERTY=VALUE[,PROPERTY=VALUE...]`</b><br/>Specifies rate limit configuration to be applied to virtual host.<br/><br/>Possible property names:<br/><ul> <li><code>rps</code>:     Rate per second value.</li> <li><code>rpm</code>:     Rate per minute value.</li> <li><code>all-requests</code>:     Use rate limit for all incoming requests.</li> <li><code>requests-per-ip</code>:     Use rate limit for requests groupped by client ip address.</li> </ul>|
|`--clear-rate-limit`|Clears rate limiter configuration.|
|`--clear-routes`|Clears all routes for the virtual host.|
|`--security-profile-id`|<b>`string`</b><br/>Specifies security profile ID for the virtual host.|

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
