| Service | Price of 1 GB per month,<br>without VAT |
|--------------------------------|-----------------------------------------------------------------------:|
| Network HDD storage | {{ sku|USD|mdb.cluster.network-hdd.kafka|month|string }} |
| Non-replicated SSD storage | {{ sku|USD|mdb.cluster.network-ssd-nonreplicated.kafka|month|string }} |
| Network SSD storage | {{ sku|USD|mdb.cluster.network-nvme.kafka|month|string }} |
| High-performance SSD storage | {{ sku|USD|mdb.cluster.network-ssd-io-m3.kafka|month|string }} |
| Local SSD storage^*^ | {{ sku|USD|mdb.cluster.local-nvme.kafka|month|string }} |

^*^ If you use dedicated hosts, this storage class is charged as described in the [Yandex Compute Cloud documentation](../../compute/pricing.md#prices).