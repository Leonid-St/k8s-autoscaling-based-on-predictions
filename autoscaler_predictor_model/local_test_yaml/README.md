```bash
minikube start
```

```bash
kubectl apply -f victoria-metrics.yaml
```

```bash
kubectl apply -f nginx-app.yaml
```

```bash
kubectl apply -f https://github.com/kedacore/keda/releases/latest/download/keda-latest-kubectl.yaml
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
