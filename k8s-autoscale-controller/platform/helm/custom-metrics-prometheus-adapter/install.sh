helm del --purge prometheus-metrics-adapter

helm install stable/prometheus-adapter --name prometheus-metrics-adapter -f values.yaml  --namespace cm --debug --version 2.5.0

