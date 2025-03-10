---
title: Examples of using triggers in {{ tracker-full-name }}
description: In this tutorial, you will learn about using triggers in {{ tracker-name }}.
---

# Trigger use cases in {{ tracker-name }}

Here are some examples of how triggers work in {{ tracker-name }}:

- How to [automatically pick assignees](#assign_ticket) based on their status or component.

- How to [automatically invite the assignee to comment](#summon_ticket) depending on the field status and value.

- How to [automatically change an issue status](#new-link) after adding a certain type of link to it.

- How to [automatically notify a user](#notify_mail) after an issue was created based on a request sent to the support team via email.

- How to [automatically notify a user](#notify_form) after an issue was created based on a request sent to the support team via {{ forms-full-name }}.

- How to [automatically add a form](#insert_form) to the issue comments.

- How to [automatically add issues to your board](#board).

- How to [set up notifications in messengers](#section_vsn_mb2_d3b) via HTTP requests.

- How to [automatically calculate the difference between dates](#tracker_data) in {{ tracker-name }}.


- How to [create a sub-issue and write field values from its parent issue to it](#create-ticket-with-params).


## Picking assignees automatically {#assign_ticket}

It might often be the case that different employees are assigned to perform specific work stages. When an employee has completed their part of the work, they hand over the issue to the next assignee. In {{ tracker-name }}, each issue stage has its own status. When the issue switches over to a certain status, you can use a trigger to automatically set the assignee for the next work stage.

Another way to organize the workflow is to make certain employees responsible for certain work areas. For example, each support employee is responsible for requests relating to their products. To manage this kind of workflow, you can [configure components](components.md) that correspond to specific products. When a certain component is added to an issue, you can use a trigger to automatically set the assignee responsible for the given product.

Let's set up a trigger to automatically assign the issue:


1. Make sure every employee you might want to assign has [full access to {{ tracker-name }}](../access.md).


1. Go to the queue settings, open the **Triggers** section, and click [**Create trigger**](../user/create-trigger.md).

1. Enter a name for the trigger.

1. Set the conditions for the trigger to fire when the **Status** or **Components** parameters of the issue change:

   1. Select **Conditions to be met** → **All**.

   1. Add the condition: **Event** → **{{ ui-key.startrek-backend.messages.trigger.condition.type.eventType.update }}**.

   1. If you want a new assignee to be picked after a status update, add the **{{ ui-key.startrek-backend.fields.issue.fields.system }}** → **Status** → **{{ ui-key.startrek-backend.messages.trigger.condition.type.fieldBecameEqual }}** condition and specify the status. The available statuses depend on the [workflow](workflow.md) set up for the queue.

      ![](../../_assets/tracker/trigger-example-status.png)

      If you want a new assignee to be picked after a component changes, add the **{{ ui-key.startrek-backend.fields.issue.fields.system }}** → **Components** → **{{ ui-key.startrek-backend.messages.trigger.condition.type.fieldBecameEqual }}** condition and specify the components.

      ![](../../_assets/tracker/trigger-example-components.png)

      {% note info %}

      The trigger with this condition will fire only if a single component is specified in the issue.

      {% endnote %}

1. Set the action for the trigger:

   1. Add the action **Update fields**.

   1. Select **{{ ui-key.startrek-backend.fields.issue.fields.system }}** → **{{ ui-key.startrek-backend.fields.issue.assignee-key-value }}** → **Set value** and specify who should be picked as the assignee once the trigger is fired.

      ![](../../_assets/tracker/trigger-example-assignee.png)

1. Save your trigger.
   To check the trigger operation, change the status or components for any issue from the queue that you set up the trigger for.

## Summoning assignees automatically {#summon_ticket}

Having completed the issue, the employee might forget to specify some important information, for example, the time spent. In this case, you can set up a trigger that will automatically summon the user if the issue was closed, but the time spent was not specified.

1. Go to the queue settings, open the **Triggers** section, and click [**Create trigger**](../user/create-trigger.md).

1. Enter a name for the trigger.

1. Set the conditions for the trigger to fire on closing the issue in case the **Time spent** field is empty:

   1. Select **Conditions to be met** → **All**.

   1. Add the condition: **{{ ui-key.startrek-backend.fields.issue.fields.system }}** → **Status** → **{{ ui-key.startrek-backend.messages.trigger.condition.type.fieldBecameEqual }}** → **{{ ui-key.startrek-backend.applinks.samsara.status.closed }}**. The available statuses depend on the [workflow](workflow.md) set up for the queue.

   1. Add the condition: **{{ ui-key.startrek-backend.fields.issue.fields.timetracking }}** → **{{ ui-key.startrek-backend.fields.issue.spent-key-value }}** → **{{ ui-key.startrek-backend.messages.trigger.condition.type.fieldIsEmpty }}**.

1. Set the actions for the trigger:

   1. Add the **Add comment** action.

   1. Click ![](../../_assets/tracker/summon.png) and enter *Assignee* in the **Invite users from field** line.

   1. Enter the comment to be displayed to the issue assignee and choose **Send as robot**.

1. Click **Create** to save the trigger.

   ![](../../_assets/tracker/trigger-example-summon.png)

Whenever an issue is closed without specifying the time spent, the robot will create a comment and summon the assignee.

## Changing an issue status after creating a link {#new-link}

In many projects, issues depend on each other even if different people are working on them. If an issue affects the progress of one or more other issues, it is important to notify other team members if any problems arise. For example, you can link such issues to each other and set the **{{ ui-key.startrek-backend.fields.issue.links.relationship.is.dependent.by }}** [link type](../user/links.md)

Let's set up a trigger that will change an issue status and add a comment for its reporter when the **{{ ui-key.startrek-backend.fields.issue.links.relationship.is.dependent.by }}** link is added:

1. Go to the queue settings, open the **Triggers** section, and click [**Create trigger**](../user/create-trigger.md).

1. Enter a name for the trigger.

1. Set the trigger to fire when the **{{ ui-key.startrek-backend.fields.issue.links.relationship.is.dependent.by }}** link appears:

   1. Select **Conditions to be met** → **All**.

   1. Add the condition: **Action with link** → **{{ ui-key.startrek-backend.fields.trigger.condition.type.links.created }}** → **{{ ui-key.startrek-backend.fields.issue.links.relationship.is.dependent.by }}**.

   ![](../../_assets/tracker/blocker-conditions.png)

1. Set the actions for the trigger:

   1. Add the action **Change issue status**.

   1. In the **Next status** field, select the status to switch the issue to when the condition is met, e.g., **{{ ui-key.startrek-backend.applinks.samsara.status.need.info }}**. The available statuses depend on the [workflow](workflow.md) set up for the queue.

   1. Add the **Add comment** action.

   1. Click ![](../../_assets/tracker/summon.png) and enter **Reporter** in the **Invite users from a field** line.

   1. Enter the comment to be displayed to the issue reporter and choose **Send as robot**. Otherwise, the comment is sent on behalf of the user who initiates the trigger action by adding the link.

   ![](../../_assets/tracker/blocker-actions.png)

1. Click **Create** to save the trigger.

## Sending a notification when an issue is created from an email {#notify_mail}

Let's say the support team is processing user requests in {{ tracker-name }}. Users contact the support team via email, and those emails are used as the basis for issues in {{ tracker-name }}.

Let's set up a trigger that, once the issue is created, sends to the user an email that their request has been registered.

#### Step 1. Set up email integration

You need to set up email integration if you want to send emails right from {{ tracker-name }} and create issues from incoming emails:

1. [Set up an email address for the queue](queue-mail.md#section_gwv_hqb_hgb) to store issues created based on user requests.

   
   If you cannot add an address for the queue, it means that your organization does not have a domain. You need a domain to create mailboxes and newsletters, for example, to add an email address for your queue. You can [add a domain in {{ ya-360 }}]({{ support-business-domain }}) free of charge.


1. [Set up sender names and signatures](queue-mail.md#send_outside) if needed.


1. If the users are not your company employees:

   1. [Allow receiving emails from external addresses](queue-mail.md#mail_tasks).

   1. [Allow sending emails from issue pages to external addresses](queue-mail.md#send_outside).


#### Step 2. Setting up a trigger for sending emails

Set up a trigger that, whenever an issue created based on an email, will notify the user via email:

1. Go to the queue settings, open the **Triggers** section, and click [**Create trigger**](../user/create-trigger.md).

1. Enter a name for the trigger.

1. Set up the conditions to fire the trigger when an issue is created based on an incoming email:

   1. Select **Conditions to be met** → **All**.

   1. Add the condition: **Event** → **{{ ui-key.startrek-backend.messages.trigger.condition.type.eventType.create }}**.

   1. Add the **{{ ui-key.startrek-backend.fields.issue.fields.email }}** → **{{ ui-key.startrek-backend.fields.issue.emailCreatedBy }}** → **{{ ui-key.startrek-backend.fields.trigger.condition.type.text.field.equals.string }}** condition and enter the email address to your queue.

   1. Enable **{{ ui-key.startrek-backend.fields.trigger.condition.property.ignoreCase }}** so that the queue address is not case sensitive.

   ![](../../_assets/tracker/trigger-example-mail-condition.png)

1. As a trigger action, set up sending an email:

   1. Choose the **Add comment** action.

   1. Enable **{{ ui-key.startrek-backend.messages.sla.send.mail.threshold.excess.function.type }}**.

   1. In the **{{ ui-key.startrek-backend.fields.issue.emailTo }}** field, add the variable with the email address of the user who sent the request. To do this, select the **{{ ui-key.startrek-backend.fields.issue.emailTo }}** field, click **Add variable**, and choose **{{ ui-key.startrek-backend.fields.issue.fields.email }}** → **{{ ui-key.startrek-backend.fields.issue.emailFrom }}**.

   1. Write the text of your message. You can add [issue fields](../user/vars.md) to your message by clicking **Add variable**.

   ![](../../_assets/tracker/trigger-example-mail-action.png)

1. Save your trigger.

   To see if the trigger works, send an email to the queue address.

## Sending a notification when an issue is created based on a form {#notify_form}

Let's say the support team is processing user requests in {{ tracker-name }}. Users contact the support team via a feedback form created in [{{ forms-full-name }}]({{ link-forms }}). A {{ tracker-name }} issue is then created based on that form.

Let's set up a trigger that, once the issue is created, sends to the user an email that their request has been registered.

#### Step 1. Set up email integration

You need to set up email integration if you want to send emails from {{ tracker-name }}:

1. [Set up an email address for the queue](queue-mail.md#section_gwv_hqb_hgb) to store issues created based on user requests.

   
   If you cannot add an address for the queue, it means that your organization does not have a domain. You need a domain to create mailboxes and newsletters, for example, to add an email address for your queue. You can [add a domain in {{ ya-360 }}]({{ support-business-domain }}) free of charge.


1. [Set up sender names and signatures](queue-mail.md#send_outside) if needed.

1. If the users are not your company employees, [allow sending emails from issue pages to external addresses](queue-mail.md#send_outside).

#### Step 2. Set up a form to register requests

To create issues based on requests submitted from a form:

1. Go to [{{ forms-full-name }}]({{ link-forms }}) and create a new form.

1. Add questions that allow users to provide relevant information that is required to register their request.

   If you want to know the user email address, add the **Email** question and make it a required field.

   ![](../../_assets/tracker/trigger-example-form-constructor.png)

1. Set up [integration with{{ tracker-name }}](../../forms/create-task.md) for the form.

   1. Specify the queue and other issue parameters.

   1. Use the **Issue description** field to add answers to the questions included in your form.

   1. To save the user's email address in the issue settings, add the **{{ ui-key.startrek-backend.fields.issue.emailFrom }}** field and select **Variables** → **Answer to question** → **Email**.

   1. Save your integration settings.

   ![image](../../_assets/tracker/trigger-example-form-integration.png)

1. [Publish](../../forms/publish.md#section_link) the form.

#### Step 3. Setting up a trigger for sending emails

Set up a trigger that, whenever an issue created from a form, will notify the user via email:

1. Go to the queue settings, open the **Triggers** section, and click [**Create trigger**](../user/create-trigger.md).

1. Enter a name for the trigger.

1. Set up the conditions to fire the trigger when an issue is created based on an incoming email:

   1. Select **Conditions to be met** → **All**.

   1. Add the condition: **Event** → **{{ ui-key.startrek-backend.messages.trigger.condition.type.eventType.create }}**.

   1. Add the condition: **{{ ui-key.startrek-backend.fields.issue.fields.email }}** → **{{ ui-key.startrek-backend.fields.issue.emailFrom }}** → **{{ ui-key.startrek-backend.messages.trigger.condition.type.fieldBecameNotEmpty }}**.

   ![](../../_assets/tracker/trigger-example-form-condition.png)

1. As a trigger action, set up sending an email:

   1. Choose the **Add comment** action.

   1. Enable **{{ ui-key.startrek-backend.messages.sla.send.mail.threshold.excess.function.type }}**.

   1. In the **{{ ui-key.startrek-backend.fields.issue.emailTo }}** field, add the variable with the email address of the user who sent the request. To do this, select the **{{ ui-key.startrek-backend.fields.issue.emailTo }}** field, click **Add variable**, and choose **{{ ui-key.startrek-backend.fields.issue.fields.email }}** → **{{ ui-key.startrek-backend.fields.issue.emailFrom }}**.

   1. Write the text of your message. You can add [issue fields](../user/vars.md) to your message by clicking **Add variable**.

   ![](../../_assets/tracker/trigger-example-mail-action.png)

1. Save your trigger.

   To see if your trigger works, fill out the form you integrated with {{ tracker-name }}.

## Automatically adding a form to the issue comments {#insert_form}

You can choose a trigger to add a form with pre-populated fields to the issue comments. For this, add, as the comment text, a special code including a link to the form. Values can be passed to the form fields via [GET parameters](../../forms/get-params.md). For example, you can pass the issue's parameters using the [variables](../user/vars.md) available in the trigger.

Let's set up a trigger that, whenever an issue is closed, will add a feedback form to the comments and invite the assignee to comment.

#### Step 1. Creating a feedback form

1. Go to [{{ forms-full-name }}]({{ link-forms }}) and create a form. 

1. Add questions so that users could provide the required information.

#### Step 2. Creating a trigger for adding a form

1. Go to the queue settings, open the **Triggers** section, and click [**Create trigger**](../user/create-trigger.md).

1. Set the conditions so that the trigger fires when the issue is closed:

   1. Select **Conditions to be met** → **All**.

   1. Add the condition: **Status** → **{{ ui-key.startrek-backend.messages.trigger.condition.type.fieldBecameEqual }}** → **{{ ui-key.startrek-backend.applinks.samsara.status.closed }}**.

   ![](../../_assets/tracker/trigger-example-add-form-1.png)

1. Add the **Add comment** action.

1. Use the following code as the comment's text:

   
   ```
   {{=<% %>=}}/iframe/(src="https://forms.yandex.ru/surveys/<form_ID>/?iframe=1&<question_ID>=<value>" frameborder=0 width=500)
   ```


   Where:
   - `<form_ID>`: ID of the form to add.

   - `<question_id>`: [Question ID](../../forms/question-id.md#sec_question).

   - `<value>`: Value to use in the form field.

      To transfer issue parameters to the form, use [variables](../user/vars.md) as values: at the bottom of the window, click **Add variable** and choose the issue parameter. Then replace the `not_var{{ }}` characters around the name of the variable with `<% %>`.

      For example, to provide the issue key, use the `<%issue.key%>` value. To provide the assignee login, use `<%issue.assignee.login%>`.

   Here is an example of the code where the issue key is provided to a field of the form.

   
   ```
   {{=<% %>=}}/iframe/(src="https://forms.yandex.ru/surveys/68***/?iframe=1&answer_short_text_584943=<%issue.key%>" frameborder=0 width=100% height=660px scrolling=no)
   ```



1. Click ![](../../_assets/tracker/summon.png) and enter *Assignee* in the **Invite users from field** line.

1. Enable **Send as robot**.


1. Click **Create** to save the trigger.

#### Step 3. Add the yndx-forms-cnt-robot@ robot to the queue

To correctly insert the form, grant the yndx-forms-cnt-robot@ robot access to the queue. To learn more about setting up access, see [Setting access rights for queues](queue-access.md).


When the issue is closed, the robot will create a comment with a form and summon the assignee.

## Automatically adding issues to your board {#board}

The [new board version](agile-new.md) allows you to set up adding issues automatically by a filter or [trigger](trigger-examples.md#board).

Instead of a trigger, you can also [set up an auto action](../user/create-autoaction.md) with a similar condition and action. When using an auto action, the issues meeting the condition will be added to the board at the specified time intervals rather than immediately.

{% note warning %}

Triggers and auto actions only work for issues from the queue where they have been set up.

{% endnote %}


Here is an example of a trigger that will add an issue to the board when it is assigned to a certain user:

1. Go to the queue settings, open the **Triggers** section, and click [**Create trigger**](../user/create-trigger.md).

1. Enter a name for the trigger.

1. Set the condition: **{{ ui-key.startrek-backend.fields.issue.assignee-key-value }}** → **{{ ui-key.startrek-backend.messages.trigger.condition.type.fieldBecameEqual }}** → `<username>`.

   {% note info %}

   The trigger with this condition will also fire when an issue is created with the specified assignee.

   {% endnote %}

1. Set up the action:

   1. Select the action **Update fields**.

   1. Select the **{{ ui-key.startrek-backend.fields.issue.boards }}** field.

   1. Select the action: **Add to list** and specify the board where do you need to add an issue.

   ![](../../_assets/tracker/trigger-example-board.png)

1. Save your trigger.

## Sending notifications to instant messengers {#section_vsn_mb2_d3b}

By using messengers, you can quickly notify your employees about important events. If a messenger has an API, you can use {{ tracker-name }} to set up a trigger that sends HTTP requests to the messenger API when certain events occur. For example, this might be handy when a severe error occurs in your queue.

To view examples for setting up triggers that send notifications to Slack and Telegram, see [{#T}](../messenger.md).

## Automatically calculating the difference between dates {#tracker_data}

Let's set up a trigger to automatically calculate the difference between dates in {{ tracker-name }}:

1. Go to the queue settings, open the **Triggers** section, and click [**Create trigger**](../user/create-trigger.md).

1. Enter a name for the trigger.

1. Select **Conditions to be met** → **All**.

1. Add the condition: **Event** → **{{ ui-key.startrek-backend.messages.trigger.condition.type.calculate.formula.watch }}**.

   ![](../../_assets/tracker/create_trigger.png)

1. Set the actions for the trigger:

   1. Add the value **Calculate value**.

   1. To get the difference of dates, in days, specify the following in the **Formula to calculate the value** field:

      ```
      (not_var{{issue.end.unixEpoch}}-not_var{{issue.start.unixEpoch}})/86400000
      ```

   1. Select **Calculated field** from the [list]({{ link-admin-fields}}).

      You can select one of the standard fields or [create a new](../user/create-param.md) one, such as **Duration**:

      ![](../../_assets/tracker/create_trigger_two.png)

1. Click **Create** to save the trigger.

To test the trigger, change the values in the **{{ ui-key.startrek-backend.fields.issue.start-key-value }}** and **{{ ui-key.startrek-backend.fields.issue.end-key-value }}** fields.


## Creating a sub-issue and writing field values from its parent issue to it {#create-ticket-with-params}

As an example, let's assume we need a trigger that creates a sub-issue and fills out its fields with values from the original issue. You can set up auto creation of issues like this using a trigger and [{{ api-name }}](../about-api.md):

1. Go to the queue settings, open the **Triggers** section, and click [**Create trigger**](../user/create-trigger.md).

1. Select [trigger conditions](../user/set-condition.md).

1. Select [**HTTP request**](../user/set-action.md#create-http) as a target action.

1. Specify the request parameters. In the **Request body** field, set the parameters of a new sub-issue.
    To substitute the values from the original issue, use [variables](../user/vars.md):

   #|
   || **Field** | **Content** ||
   || Method | POST ||
   || Address | `{{ host }}/{{ ver }}/issues` ||
   || Authorization method | OAuth 2.0 ||
   || Token | [How to get a token](../concepts/access.md#section_about_OAuth) ||
   || Authorization header | Authorization ||
   || Token type | OAuth ||
   || Content type | application/json ||
   || Request body |

   > Example: Creating a sub-issue and transmitting to it field values from the original issue, such as description, assignee, followers, and tags.
   >
   > ```
   > {
   >    "summary": "<issue_name>",
   >    "queue": "<queue_key>",
   >    "description": not_var{{issue.description.json}},
   >    "links": [
   >        {
   >            "relationship": "is subtask for",
   >            "issue": "not_var{{issue.key}}"
   >        }
   >    ],
   >    "assignee": "not_var{{issue.assignee.login}}",
   >    "tags": not_var{{issue.tags.json}},
   >    "followers": not_var{{issue.followers.uid.json}}
   > }
   > ```
   For more information about the request, see [{#T}](../concepts/issues/create-issue.md) and [{#T}](../concepts/issues/link-issue.md). ||
   || Headers | Header: `X-Org-ID` or `X-Cloud-Org-ID`.
   Value: Organization ID. If you only have a {{ org-full-name }} organization, use the `X-Cloud-Org-ID` header; if only {{ ya-360 }} or both organization types, use `X-Org-ID`. The ID is shown in the **Organization ID for API** field on the [{{ tracker-name }} settings]({{ link-settings }}) page. ||
   |#

   {% note info %}

   Make sure the parameters you pass in the request body using variables are set in the original issue; otherwise, the trigger will not fire.

   {% endnote %}

1. Click **Create**.

## Updating status, priority and adding a comment in related issues {#update-related-tasks}

For example, we have a trigger that updates the status, priority, and adds a comment in related issues. Using a trigger and [{{ api-name }}](../about-api.md), you can configure automatic update of related issues:

1. Go to the queue settings, open the **Triggers** section, and click [**Create trigger**](../user/create-trigger.md).

1. Select [trigger conditions](../user/set-condition.md).

   {% note warning %}

   When creating a trigger condition, pay attention to the possibility of a cascading call in related issues.

   {% endnote %}

1. Select [**HTTP request**](../user/set-action.md#create-http) as a target action.

1. Specify the request parameters. In the **Request body** field, specify the update parameters in related issues. To substitute the values from the original issue, use [variables](../user/vars.md):

   #|
   || **Field** | **Content** ||
   || Method | POST ||
   || Address | `{{ host }}/{{ ver }}/bulkchange/_transition` ||
   || Authorization method | OAuth 2.0 ||
   || Token | [How to get a token](../concepts/access.md#section_about_OAuth) ||
   || Authorization header | Authorization ||
   || Token type | OAuth ||
   || Content type | application/json ||
   || Request body |

   > Example: Updating status, priority and adding a comment in related issues.
   >
   > ```
   > {
   >    "transition": "need_info",
   >    "issues": "Relates: not_var{{issue.key}}",
   >    "values": {
   >        "comment": "<Comment for related issues>",
   >        "priority": {
   >            "key": "critical"
   >        }
   >    }
   > }
   > ```
   For more information about the request, see [{#T}](../concepts/bulkchange/bulk-transition.md). ||
   || Headers | Header: `X-Org-ID` or `X-Cloud-Org-ID`.
   Value: Organization ID. If you only have a {{ org-full-name }} organization, use the `X-Cloud-Org-ID` header; if only {{ ya-360 }} or both organization types, use `X-Org-ID`. The ID is shown in the **Organization ID for API** field on the [{{ tracker-name }} settings]({{ link-settings }}) page. ||
   |#

   {% note info %}

   If you need to change only field values in the related issues without changing the status, use the request: [{#T}](../concepts/bulkchange/bulk-update-issues.md).

   Make sure the parameters you pass in the request body using variables are set in the original issue; otherwise, the trigger will not fire.

   {% endnote %}

1. Click **Create**.
