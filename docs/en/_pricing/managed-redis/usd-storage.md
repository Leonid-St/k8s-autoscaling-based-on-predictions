| Service                         | Price per 1 GB per month,<br>without VAT                               |
|---------------------------------|-----------------------------------------------------------------------:|
| Non-replicated SSD storage^*^   | {{ sku|USD|mdb.cluster.network-ssd-nonreplicated.redis|month|string }} |
| Network SSD storage             | {{ sku|USD|mdb.cluster.network-nvme.redis|month|string }}              |
| High-performance SSD storage    | {{ sku|USD|mdb.cluster.network-ssd-io-m3.redis|month|string }}         |
| Local SSD storage^*^            | {{ sku|USD|mdb.cluster.local-nvme.redis|month|string }}                |
| Backups beyond the storage size | {{ sku|USD|mdb.cluster.redis.backup|month|string }}                    |
