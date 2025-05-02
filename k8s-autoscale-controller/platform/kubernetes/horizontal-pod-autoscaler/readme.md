Create an autoscaler
```
k apply -f k8s-hpa.yaml
```

Get the explanation of decisions
```
k describe hpa pythonwebapp-hpa 
```

Delete the auto-scaling
```
k delete hpa pythonwebapp-hpa
```
