Create an autoscaler
```
k apply -f k8s-hpa.yaml
```

Get the explanation of decisions
```
k describe hpa headlesswebapp-hpa 
```

Delete the auto-scaling
```
k delete hpa headlesswebapp-hpa
```