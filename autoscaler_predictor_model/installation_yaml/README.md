### pyenv installation 
```bash
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
```
```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc

echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc

echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi' >> ~/.bashrc

exec "$SHELL"

pyenv install
```

```bash
curl https://pyenv.run | bash
```
### install helm

```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
```

```bash
chmod 700 get_helm.sh
```

```bash
./get_helm.sh
```



### 1. Установите Docker

Сначала установите Docker. Откройте терминал и выполните следующие команды:

```bash
sudo apt update 
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common 
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
sudo apt install -y docker-ce
```
### 2. Добавьте своего пользователя в группу Docker

Чтобы запускать Docker без `sudo`, добавьте своего пользователя в группу Docker:

bash

`sudo usermod -aG docker $USER`

После этого вам нужно выйти из системы и войти снова, чтобы изменения вступили в силу.

### 3. Установите kubectl

Если вы еще не установили `kubectl`, выполните следующие команды:


`curl -LO "https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl" chmod +x ./kubectl sudo mv ./kubectl /usr/local/bin/kubectl`

### 4. Установите Minikube

curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64
/usr/local/bin/minikube

### 5. Запустите Minikube с использованием Docker

Теперь вы можете запустить Minikube, указав драйвер Docker:


`minikube start --driver=docker --extra-config=scheduler.bind-address=0.0.0.0 --extra-config=controller-manager.bind-address=0.0.0.0 --extra-config=etcd.listen-metrics-urls=http://0.0.0.0:2381`

### 6. Проверьте установку

После завершения установки вы можете проверить статус Minikube:


`minikube status`

### 7. Используйте kubectl для управления кластерами

Теперь вы можете использовать `kubectl` для управления вашим кластером Kubernetes:

bash
`kubectl get nodes`

### Примечания

- Убедитесь, что Docker запущен. Вы можете проверить это с помощью команды `sudo systemctl status docker`.
- Если у вас возникли проблемы с правами доступа, убедитесь, что ваш пользователь добавлен в группу Docker и вы вышли и снова вошли в систему.

```bash
minikube start
```

### Installation StackGras (Cluster Postgres with TimescaleDB)


Add the StackGres Helm repository
```bash
helm repo add stackgres-charts https://stackgres.io/downloads/stackgres-k8s/stackgres/helm/
```

Installing the Operator
```bash
helm install stackgres-operator stackgres-charts/stackgres-operator \
  --namespace stackgres --create-namespace \
  --version 1.16.1 \
  --set adminui.service.type=NodePort \
  --set adminui.service.nodePort=30080
```

Waiting for Operator Startup


```bash
kubectl wait -n stackgres deployment -l group=stackgres.io --for=condition=Available
```

forward pod for stackgres operator

```bash
POD_NAME=$(kubectl get pods --namespace stackgres -l "stackgres.io/restapi=true" -o jsonpath="{.items[0].metadata.name}")
```

```bash
kubectl port-forward "$POD_NAME" 8443:9443 --namespace stackgres
```


on local machine

```bash
ssh -L 8443:localhost:8443 ubuntu@ip_adress
```

username and password for stackgras admin ui
```bash
kubectl get secret -n stackgres stackgres-restapi-admin --template '{{ printf "username = %s\npassword = %s\n" (.data.k8sUsername | base64decode) ( .data.clearPassword | base64decode) }}'
```

go to https://localhost:8443/admin/ and create basic cluster with timescaledb

password for postgres cluster - 
```bash
kubectl get secret predicted  -o jsonpath='{.data.superuser-password}' | base64 -d
```
### installing kube-state-metrics
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install kube-state-metrics prometheus-community/kube-state-metrics
```

The exposed metrics can be found here:
https://github.com/kubernetes/kube-state-metrics/blob/master/docs/README.md#exposed-metrics

The metrics are exported on the HTTP endpoint /metrics on the listening port.
In your case, kube-state-metrics.default.svc.cluster.local:8080/metrics

They are served either as plaintext or protobuf depending on the Accept header.
They are designed to be consumed either by Prometheus itself or by a scraper that is compatible with scraping a Prometheus client endpoint.

### installing prometheus-node-exporter

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install node-exporter prometheus-community/prometheus-node-exporter
```

1. Get the application URL by running these commands:
  export POD_NAME=$(kubectl get pods --namespace default -l "app.kubernetes.io/name=prometheus-node-exporter,app.kubernetes.io/instance=node-exporter" -o jsonpath="{.items[0].metadata.name}")
  echo "Visit http://127.0.0.1:9100 to use your application"
  kubectl port-forward --namespace default $POD_NAME 9100

<!-- ### deploy cadvisior:
```bash
kubectl apply -f cadvisior.yaml
``` -->


## deploy vectoria metrics

```bash
kubectl apply -f victoria-metrics.yaml
```


### deploy vectoria metrics agent
```bash
helm repo add victoria-metrics https://victoriametrics.github.io/helm-charts/
helm repo update

helm upgrade --install vmagent victoria-metrics/victoria-metrics-agent \
  --namespace default --create-namespace \
  -f vmagent-true-values.yaml
```


#### port forwarding throw ssh for victoria metrics:
```bash
ssh -L 8428:192.168.49.2:30000 ubuntu@45.147.163.92
```

#### calculate cpu utilization by node

```bash
sum by (instance) (
  rate(node_cpu_seconds_total{mode=~"user|system|iowait|irq|softirq|steal"}[5m])
) * 100
```
#### calculate memory usage  by node

```bash
100 * (
  1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes
)
```

#### calculae count of ready nodes

```bash
count(kube_node_status_condition{condition="Ready", status="true"})
```
<!-- ```bash
kubectl apply -f vmagent-config.yaml
kubectl apply -f vmagent-deployment.yaml
``` -->


#### calculate cpu utilization by pods

CPU использование по подам simple-app
```bash
sum by (pod) (
  rate(container_cpu_usage_seconds_total{
    pod=~"simple-app-.*",
    container!="POD"
  }[5m])
) * 100
```

#### calculate memory usage  by pods
Память использование по подам simple-app

```bash
sum by (pod) (
  container_memory_working_set_bytes{
    pod=~"simple-app-.*",
    container!="POD"
  }
)
```

#### calculae count of ready pods
Количество "Ready" подов simple-app
```bash
count(
  kube_pod_status_ready{
    condition="true",
    pod=~"simple-app-.*"
  }
)
```


Если нужно собрать среднюю загрузку CPU среди всех подов
metricsql

```bash
 avg(
  rate(container_cpu_usage_seconds_total{
    pod=~"simple-app-.*",
    container!="POD"
  }[5m])
) * 100
```


Как теперь считать % использования CPU от лимита ()?
Вот готовая формула на MetricsQL:

Лимит CPU на под	kube_pod_container_resource_limits{resource="cpu", pod=~"simple-app-.*"}
Лимит памяти на под	kube_pod_container_resource_limits{resource="memory", pod=~"simple-app-.*"}
Фактическое использование CPU	rate(container_cpu_usage_seconds_total{pod=~"simple-app-.*", container!="POD"}[5m])

```bash
(
  sum by (pod) ( rate(container_cpu_usage_seconds_total{pod=~"simple-app-.*", container!="POD"}[5m]) )
/
  kube_pod_container_resource_limits{resource="cpu", pod=~"simple-app-.*"}
) * 100
```

Теперь формула для расчёта процентного использования памяти будет выглядеть так:
```bash
(
  sum by (pod) ( container_memory_usage_bytes{pod=~"simple-app-.*", container!="POD"} )
/
  kube_pod_container_resource_limits{resource="memory", pod=~"simple-app-.*"}
) * 100
```

container_memory_usage_bytes — сколько байт реально используется сейчас.

kube_pod_container_resource_limits{resource="memory"} — сколько байт максимум разрешено использовать.

Делим usage на limit → получаем долю → умножаем на 100 для процентов.


Формула расчета средней утилизации на cpu на всех подах simple-app относительно лимитов
```bash
avg(
  rate(container_cpu_usage_seconds_total{pod=~"simple-app-.*", container!="POD"}[5m])
) 
/
avg(kube_pod_container_resource_limits{resource="cpu", pod=~"simple-app-.*"})
* 100
```


Формула расчета средней утилизации ram на всех подах simple-app
```bash
avg(
  container_memory_usage_bytes{pod=~"simple-app-.*", container!="POD"}
) 
/
avg(kube_pod_container_resource_limits{resource="memory", pod=~"simple-app-.*"})
* 100
```
#### UI VictoriaMetrics
```bash
http://<NodeIP>:8428
```
 для проверки В поле "Expression" введи:

nginx_connections_active

И нажми Execute — если видишь граф или цифры, значит метрики приходят 



### deploy nginx
```bash
kubectl apply -f config_map_nginx.yaml 
```

```bash
kubectl apply -f app.yaml
```

### deploy nginx-exporter:
```bash
kubectl apply -f nginx-exporter.yaml
```



#### Убедись, что он тоже поднялся и отвечает на /metrics. Проверить можно из пода через:
```bash
kubectl exec -it <nginx-exporter-pod> -- curl http://simple-app/stub_status
kubectl exec -it <nginx-exporter-pod> -- curl http://localhost:9113/metrics
```



### install keda:

```bash
helm repo add kedacore https://kedacore.github.io/charts
```

```bash
helm repo update

```

```bash

helm install keda kedacore/keda

```

#### uninstall keda:
```bash
kubectl delete $(kubectl get scaledobjects.keda.sh,scaledjobs.keda.sh -A \
  -o jsonpath='{"-n "}{.items[*].metadata.namespace}{" "}{.items[*].kind}{"/"}{.items[*].metadata.name}{"\n"}')
helm uninstall keda -n keda
```


```bash
kubectl apply -f scaled-object-nginx.yaml
```

```bash
kubectl get deployments

kubectl get scaledobjects

kubectl get pods
```

####  Следите за изменениями в Deployment
```bash
kubectl get deployment nginx-deployment -w
```
####  Следите за созданием и удалением Pods
```bash
kubectl get pods -w
```




### build docker and push
```bash
docker build -t leonidusername5432/autoscaler-predictor-model:latest .
```

with platform linux

```bash
docker buildx build --platform linux/amd64 -t leonidusername5432/kubescale-controller:0.0.1 .
```

```bash
docker login
```

```bash
docker push leonidusername5432/autoscaler-predictor-model:latest
```
