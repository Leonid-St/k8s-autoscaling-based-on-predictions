
### minikube installtaion:


### 1. Установите Docker

Сначала установите Docker. Откройте терминал и выполните следующие команды:

bash

Copy Code

`sudo apt update sudo apt install -y apt-transport-https ca-certificates curl software-properties-common curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" sudo apt update sudo apt install -y docker-ce`

### 2. Добавьте своего пользователя в группу Docker

Чтобы запускать Docker без `sudo`, добавьте своего пользователя в группу Docker:

bash

`sudo usermod -aG docker $USER`

После этого вам нужно выйти из системы и войти снова, чтобы изменения вступили в силу.

### 3. Установите kubectl

Если вы еще не установили `kubectl`, выполните следующие команды:


`curl -LO "https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl" chmod +x ./kubectl sudo mv ./kubectl /usr/local/bin/kubectl`

### 4. Установите Minikube

Скачайте и установите Minikube:


`curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 sudo install minikube-linux-amd64 /usr/local/bin/minikube`

### 5. Запустите Minikube с использованием Docker

Теперь вы можете запустить Minikube, указав драйвер Docker:


`minikube start --driver=docker`

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


### for me not working:
```bash
nohup kubectl port-forward pod/victoria-metrics-single-6c8bcd766c-bttgk 8429:8429 > port-forward.log 2>&1 &
```

### deploy nginx
```bash
kubectl apply -f config_map_nginx.yaml 
```

```bash
$ kubectl apply -f app.yaml
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


### installing kube-state-metrics
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install kube-state-metrics prometheus-community/kube-state-metrics
```
### installing prometheus-node-exporter

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install node-exporter prometheus-community/prometheus-node-exporter
```

<!-- ### deploy cadvisior:
```bash
kubectl apply -f cadvisior.yaml
``` -->

## deploy vectoria metrics

```bash
kubectl apply -f victoria-metrics.yaml
```


#### port forwarding throw ssh for victoria metrics:
```bash
ssh -L 8428:192.168.49.2:30000 ubuntu@45.147.163.92
```

### deploy vectoria metrics agent
```bash
helm repo add victoria-metrics https://victoriametrics.github.io/helm-charts/
helm repo update

helm upgrade --install vmagent victoria-metrics/victoria-metrics-agent \
  --namespace default --create-namespace \
  -f vmagent-true-values.yaml
```

#### calculate cpu utilization

```bash
sum by (instance) (
  rate(node_cpu_seconds_total{mode=~"user|system|iowait|irq|softirq|steal"}[5m])
) * 100
```
#### calculate memory usage 

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

#### это UI VictoriaMetrics
```bash
http://<NodeIP>:8428
```
 для проверки В поле "Expression" введи:

nginx_connections_active

И нажми Execute — если видишь граф или цифры, значит метрики приходят 🎉


### install keda:

```bash
helm repo add kedacore https://kedacore.github.io/charts
```

```bash
helm repo update

```

```bash

helm install keda kedacore/keda --namespace keda --create-namespace

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
