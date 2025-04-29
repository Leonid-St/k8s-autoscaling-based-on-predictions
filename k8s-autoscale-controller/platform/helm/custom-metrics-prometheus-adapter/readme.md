Guide on the prometheus adapter: https://github.com/DirectXMan12/k8s-prometheus-adapter/blob/master/docs/config-walkthrough.md#configuring-the-adapter

Format: https://github.com/DirectXMan12/k8s-prometheus-adapter/blob/master/docs/config.md

Best guide: https://github.com/DirectXMan12/k8s-prometheus-adapter/issues/116

To check the exported metrics: 
```kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1```

Access the metric with: 
```
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1/namespaces/default/services/pythonwebapp2/requests_per_second_pythonwebapp2
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1/namespaces/default/services/pythonwebapp/requests_per_second_pythonwebapp
```