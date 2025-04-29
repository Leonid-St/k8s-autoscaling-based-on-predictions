Why use Kube eagle?

The main problem is that Kubernetes resource limits are set as Milli CPUs whilst Prometheus metrics don't use that format. 
More in-depth answer: https://stackoverflow.com/questions/40327062/how-to-calculate-containers-cpu-usage-in-kubernetes-with-prometheus-as-monitori/54968614#54968614
 
The library that also explains how the library works:
https://github.com/cloudworkz/kube-eagle

Helm chart for deployment:
https://github.com/cloudworkz/kube-eagle-helm-chart
