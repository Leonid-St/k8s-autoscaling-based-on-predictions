## Establishing a direct trunk {#trunk-create}

This guide describes the procedure for how to establish a direct trunk.

### Getting started {#trunk-prep}

* See the [documentation](../../interconnect/concepts/index.md) to learn how the service works.
* Contact your {{ yandex-cloud }} manager to agree upon the pricing plan ([connection capacity](../../interconnect/pricing.md)).
* Select a [point of presence](../../interconnect/concepts/pops.md) for connecting a [trunk](../../interconnect/concepts/trunk.md).
  
  {% note warning %}
  
  To ensure fault tolerance, connect trunks in two points of presence.
  
  {% endnote %}
  

* Select the type of an [optical transceiver](../../interconnect/concepts/transceivers.md) to enable a cross connect to the {{ yandex-cloud }} equipment in the [point of presence](../../interconnect/concepts/pops.md). A port of the selected type will be reserved for you on the {{ yandex-cloud }} equipment side. For installation on your equipment, purchase the transceiver of the selected type on your own.
* If a {{ yandex-cloud }} solution architect is working with you, discuss and agree on a {{ interconnect-full-name }} service setup diagram.

### Contacting support for a new trunk {#trunk-ticket}

Create a [new support request]({{ link-console-support }}) to reserve a port on the {{ yandex-cloud }} equipment and set up a trunk.

Follow this request format:


```s
Subject: [CIC] Creating a new trunk.

Request text:
Please set up a new Cloud Interconnect trunk
with the following parameters:

client: My Business LLC 
folder-id: b1g28j9359gic3******
point of presence: M9
Type of transceiver: 10GBASE-LR
connection capacity: 250mbps

Details for a private connection will be provided later in a separate ticket.
```


Where:

* `client`: Company name.
* `folder-id`: [ID of the folder](../../resource-manager/operations/folder/get-id.md) to send {{ interconnect-name }} monitoring metrics to.
* `point of presence`: [Code of the point of presence](../../interconnect/concepts/pops.md).
* `transceiver type`: [Type of transceiver](../../interconnect/concepts/transceivers.md) to enable a cross connect.
* `Connection capacity`: [Bandwidth](../../interconnect/pricing.md) in Mbps.


{% note warning %}

If you need to set up a fault-tolerant {{ interconnect-name }} connection in multiple points of presence, create a separate support request for each connection.

{% endnote %}



### Tech support team's response to the customer's request {#trunk-ticket-resp}

Once the requested trunk is set up, the support team will send you information about the created trunk.


Example of support's response:

```s
trunk_id: euus5dfgchu23b******
pop: M9
rack: 10th floor, room 30-1, row 11, space 10
unit: 36
port: 18
```

Where: 

* `trunk_id`: ID of the created {{ interconnect-name }} trunk.
* `pop`: [Code of the point of presence](../../interconnect/concepts/pops.md) where the {{ interconnect-name }} trunk is created.
* `rack`: Location of the {{ yandex-cloud }} equipment at the point of presence.
* `unit`: Integer from 1 to 45. that indicates the equipment location in the {{ yandex-cloud }} rack.
* `port`: Integer from 1 to 48. that specifies the port number on the equipment to enable the cross connect for.




### How to write an agreement letter for a point of presence {#letter}

{% note warning %}

No agreement letter is required for the `STD` point of presence.

{% endnote %}

* Download the [agreement letter form](../../interconnect/concepts/pops.md#letter) for the relevant point of presence in `.docx` format.
* In the form, specify your company data and equipment location in the point of presence (highlighted in yellow).
* Use the information provided by the support team to set the `unit` and `port` parameter values in the form (highlighted in green).
* Attach the completed form to the open support request.
* Within two business days, you will receive the agreement letter in PDF format signed by {{ yandex-cloud }} as an attachment to the open support request.

### Setting up a cross connect in a point of presence {#trunk-cross}

Request and set up a cross connect between your equipment and {{ yandex-cloud }} in the point of presence you need using the agreement letter you received.

{% note alert %}

A cross connection to a {{ yandex-cloud }} equipment port can only be enabled **by the maintenance crew at the point of presence**, without engaging the {{ yandex-cloud }} engineers.

If required, the maintenance crew at the point of presence may request additional approval from {{ yandex-cloud }} duty engineers for cross-connecting the client equipment to the {{ yandex-cloud }} equipment port.

{% endnote %}



### Monitoring the trunk status {#trunk-check}

Use the [monitoring](../../interconnect/concepts/monitoring.md#trunk-mon) service to track when the trunk's physical port on the {{ yandex-cloud }} equipment switches to the running status. If there are any issues, contact support for diagnostics and troubleshooting.

{% note alert %}

A private or public connection can only be established after the trunk's physical port becomes operational.

{% endnote %}

