# Processing {{ at-full-name }} events

{{ at-full-name }} is a service for collecting and exporting the audit logs of {{ yandex-cloud }} resources to various target systems, including {{ objstorage-full-name }} and {{ yds-full-name }}. {{ at-name }} and {{ yq-full-name }} are integrated with each other to enable searching through audit logs.

![image](../../_assets/query/audit-trails-query.png)

After audit logs are processed using {{ yq-name }}, you can get such information as:

* Who deleted a cloud folder.
* Who added permissions to access the {{ compute-full-name }} VM serial console.
* Who edited permissions to access an {{ objstorage-name }} bucket.
* Who was granted administrator privileges.

You can find the preset queries for these use cases in [this GitHub repository](https://github.com/yandex-cloud/yc-solution-library-for-security/tree/master/auditlogs/_use_cases_and_searches). You can also write custom [YQL queries](../../query/yql-tutorials/index.md).

In this use case, you will create [trails](../../audit-trails/concepts/trail.md) that will upload audit logs of all [folder](../../resource-manager/concepts/resources-hierarchy.md#folder) resources to an {{ objstorage-name }} [bucket](../../storage/concepts/bucket.md) and send them to a [stream](../../data-streams/concepts/glossary.md#stream-concepts) in {{ yds-name }}. Next, you will run analytical and streaming queries to the log data using {{ yq-name }}.

## Getting started {#before-you-begin}

{% include [before-you-begin](../../_tutorials/_tutorials_includes/before-you-begin.md) %}

## Configure {{ at-name }} {#at-setup}

Create trails:

* [To upload folder audit logs to an {{ objstorage-name }} bucket](../../audit-trails/operations/export-folder-bucket.md).
* [To send folder audit logs to a stream in {{ yds-name }}](../../audit-trails/operations/export-folder-data-streams.md).

## Set up integration between {{ at-name }} and {{ yq-name }} {#integration}

To set up integration:

1. Open the list of trails in the {{ yandex-cloud }} console.
1. Select the trail you previously created for uploading cloud audit logs to a bucket and click **{{ ui-key.yacloud.audit-trails.button_process-in-yq }}**.
1. When switching from {{ at-name }} to {{ yq-name }} for the first time, set up integration:
   1. In the {{ yq-name }} interface, select the service account you want to use to read data from {{ objstorage-name }} in the connection creation dialog and click **{{ ui-key.yql.yq-connection-form.create.button-text }}**.
   1. In the {{ yq-name }} interface, check the preset parameters by clicking **{{ ui-key.yql.yq-binding-form.binding-preview.button-text }}** in the binding creation dialog. Next, click **{{ ui-key.yql.yq-binding-form.binding-create.button-text }}** to complete the integration.

This will automatically redirect you to the **{{ ui-key.yql.yq-audit-trails.audit-trails.title }}** panel of the {{ yq-name }} interface.

Perform similar actions for the previously created trail that sends data to a {{ yds-name }} stream.

## Analytical queries to data in {{ objstorage-name }}

To query {{ at-name }} analytical data stored in {{ objstorage-name }}:

1. Under **{{ ui-key.yql.yq-audit-trails.audit-trails.title }}** in the {{ yq-name }} interface, select **{{ ui-key.yql.yq-audit-trails.audit-trails-type-toggle.option-analytical }}** as the data analysis type. In the list of [data bindings](../../query/concepts/glossary.md#binding), select `audit-trails-test-object_storage`.
1. Select a query to {{ objstorage-name }} data from the list and click **{{ ui-key.yql.yq-query-actions.run-query.button-text }}**.

You can do the following with analytical query results:

* Download them through the {{ yq-name }} UI by clicking **Export**.
* [Save them to an {{ objstorage-name }} bucket](../../query/sources-and-sinks/object-storage-write.md).
* Get and process them via the [{{ yq-name }} HTTP API](../../query/api/index.md).
<!-- * [Visualize them](../../query/tutorials/datalens.md) in {{ datalens-full-name }}. -->

## Streaming queries to data from {{ yds-name }}

To query {{ at-name }} streaming data transferred through {{ yds-name }}:

1. Under **{{ ui-key.yql.yq-audit-trails.audit-trails.title }}** in the **{{ yq-full-name }}** interface, select **{{ ui-key.yql.yq-audit-trails.audit-trails-type-toggle.option-streaming }}** as the data analysis type. In the list of [data bindings](../../query/concepts/glossary.md#binding), select the one you need.
1. Select a query to {{ objstorage-name }} data from the list and click **{{ ui-key.yql.yq-query-actions.run-query.button-text }}**.

You can do the following with streaming query results:

* [Send them to {{ monitoring-full-name }}](../../query/sources-and-sinks/monitoring.md) as metrics.
* [Send them to an output {{ yds-name }}](../../query/sources-and-sinks/data-streams-write.md) stream as data and then process the data using [{{ sf-full-name }}](../../functions/operations/trigger/data-streams-trigger-create.md) triggers.

## See also

* [{{ objstorage-full-name }}](../../storage/)
* [{{ yds-full-name }}](../../data-streams/)
* [{{ datalens-full-name }}](../../datalens/)
