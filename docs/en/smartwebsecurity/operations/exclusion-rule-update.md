---
title: Updating a WAF exclusion rule
description: Follow this guide to update a WAF exclusion rule.
---

# Updating a WAF exclusion rule

{% list tabs group=instructions %}

- Management console {#console}

  1. In the [management console]({{ link-console-main }}), select the [folder](../../resource-manager/concepts/resources-hierarchy.md#folder) the [WAF profile](../concepts/waf.md) is in.
  1. In the list of services, select **{{ ui-key.yacloud.iam.folder.dashboard.label_smartwebsecurity }}**.
  1. In the left-hand panel, select ![image](../../_assets/smartwebsecurity/waf.svg) **{{ ui-key.yacloud.smart-web-security.waf.label_profiles }}**.
  1. Select the profile you want to update an [exclusion rule](../concepts/waf.md#exclusion-rules) in.
  1. In the left-hand menu, go to the ![image](../../_assets/console-icons/file-xmark.svg) **{{ ui-key.yacloud.smart-web-security.waf.title_exclusion-rules }}** tab.
  1. In the rule row, click ![options](../../_assets/console-icons/ellipsis.svg) and select **{{ ui-key.yacloud.common.edit }}**. In the window that opens:
      1. Edit the exclusion rule name and description.
      1. (Optional) Enable **{{ ui-key.yacloud.smart-web-security.waf.field_logging }}** to log exception rule triggering.
      1. Under **{{ ui-key.yacloud.smart-web-security.waf.title_exclusion-rule-rules-section }}**, update rules from the basic set for which the exclusion will be valid:
          * `{{ ui-key.yacloud.smart-web-security.waf.value_exclude-all-yes }}`: Exclusion will be valid for all rules.
          * `{{ ui-key.yacloud.smart-web-security.waf.value_exclude-all-no }}`: Exclusion will be valid for the selected rules.

             Click **{{ ui-key.yacloud.smart-web-security.waf.action_exclusion-rule-add-rules }}** to select rules from the basic set.

      1. {% include [waf-rule-traffic-conditions](../../_includes/smartwebsecurity/waf-rule-traffic-conditions.md) %}

      1. Click **{{ ui-key.yacloud.common.save }}**.

- {{ TF }} {#tf}

  {% include [terraform-definition](../../_tutorials/_tutorials_includes/terraform-definition.md) %}

  {% include [terraform-install](../../_includes/terraform-install.md) %}

  1. Open the {{ TF }} configuration file and edit the `exclusion_rule` section with the exclusion rule in `yandex_sws_waf_profile` description.

      ```hcl
      # WAF profile
      resource "yandex_sws_waf_profile" "default" {
        name = "waf-profile-default"
        core_rule_set {
          inbound_anomaly_score = 2
          paranoia_level        = local.waf_paranoia_level
          rule_set {
            name    = "OWASP Core Ruleset"
            version = "4.0.0"
          }
        }

        ...

        # Exclusion rule
        exclusion_rule {
          name = "<exclusion_rule_name>"
          exclude_rules {
            exclude_all = <true_or_false>
            rule_ids    = ["rule_1_ID","rule_2_ID"]
          }
        }
      }
      ```

      For more information about the `sws_waf_profile` resource parameters in {{ TF }}, see the [relevant provider documentation]({{ tf-provider-resources-link }}/sws_waf_profile).

  1. Apply the changes:

      {% include [terraform-validate-plan-apply](../../_tutorials/_tutorials_includes/terraform-validate-plan-apply.md) %}

  You can check the updates of your resources in the [management console]({{ link-console-main }}).

{% endlist %}

### See also {#see-also}

* [{#T}](exclusion-rule-add.md)
* [{#T}](exclusion-rule-delete.md)
* [{#T}](configure-set-rules.md)
