#!/usr/bin/env bash
echo "------------------------------------------------------------------------"
echo "------------------------------------------------------------------------"
echo "------------------------------------------------------------------------"
echo "Webapp type is $1"
echo "Scaling metric is $2 - from cpu/rps"
echo "Workload type $3 - from zigzag,ladder,spiky"
echo "Initial num instances is $4"
echo "starting"

kubectl get hpa
kubectl get deployments
kubectl delete hpa hpa-$1
kubectl delete -n default deployment autoscaler-$1
kubectl describe cronjob gatling-$1-cron

echo "------------------------------------------------------------------------"
echo "------------------------------------------------------------------------"
echo "------------------------------------------------------------------------"
echo "Starting the workloads with no auto-scaling"
date
kubectl get deployment $1
kubectl scale -n default deployment $1 --replicas=$4
sleep 90s
kubectl get deployment $1
# log the gatling simulation
cat platform/kubernetes/gatling/configmap-gatling-custom-simulation-def.yaml
## sleep 479 minutes - in seconds
echo ""
date
echo "sleeping..."
sleep 29480
date
echo "------------------------------------------------------------------------"
echo "------------------------------------------------------------------------"
echo "------------------------------------------------------------------------"
echo "Starting workloads against the Kubescale hybrid autoscaler"
# clean up the old deployments
kubectl delete hpa hpa-$1
kubectl delete -n default deployment autoscaler-$1
kubectl get hpa
kubectl get deployments
cat platform/kubernetes/autoscaler-experiment-configs/kubescale-proactive/$2/configmap-autoscaler-proactive-$2-$3-$1.yaml
kubectl apply -f platform/kubernetes/autoscaler-experiment-configs/kubescale-proactive/$2/configmap-autoscaler-proactive-$2-$3-$1.yaml
# start the auto-scaler
kubectl apply -f deployment-autoscaler-$1.yaml
## sleep 209 minutes - in seconds
echo ""
date
echo "sleeping..."
sleep 12540
echo "done with sleeping"
date
kubectl delete -n default deployment autoscaler-$1
kubectl delete hpa hpa-$1
sleep 15
kubectl get hpa
kubectl get deployments
echo "------------------------------------------------------------------------"
echo "------------------------------------------------------------------------"
echo "------------------------------------------------------------------------"
echo "------------------------------------------------------------------------"
echo "------------------------------------------------------------------------"
echo "Starting workloads against the reactive component of Kubescale autoscaler"
# log the auto-scaler configiguration
cat platform/kubernetes/autoscaler-experiment-configs/kubescale-reactive/$2/configmap-autoscaler-reactive-$2-$3-$1.yaml
kubectl apply -f platform/kubernetes/autoscaler-experiment-configs/kubescale-reactive/$2/configmap-autoscaler-reactive-$2-$3-$1.yaml
# start the reactive auto-scaler
kubectl apply -f deployment-autoscaler-$1.yaml
kubectl rollout -n default restart deployment autoscaler-$1
## sleep 209 minutes - in seconds
echo ""
date
echo "sleeping..."
sleep 12540
echo "done with sleeping"
date
kubectl delete deployment autoscaler-$1
kubectl delete hpa hpa-$1
kubectl delete -n default deployment autoscaler-$1
kubectl delete hpa hpa-$1
sleep 15
kubectl get hpa
kubectl get deployments
echo "------------------------------------------------------------------------"
echo "------------------------------------------------------------------------"
echo "------------------------------------------------------------------------"
echo "------------------------------------------------------------------------"
echo "------------------------------------------------------------------------"
echo "Starting workloads against the Kubernetes Horizontal Pod autoscaler (HPA)"
# log the HPA auto-scaler configiguration
cat hpa/hpa-$2-$3-$1.yaml
# start the HPA auto-scaler
kubectl apply -f platform/kubernetes/horizontal-pod-autoscaler/hpa-$2-$3-$1.yaml
## sleep 209 minutes - in seconds
echo ""
date
echo "sleeping..."
sleep 12540
echo "done with sleeping"
kubectl delete hpa hpa-$1
sleep 15
kubectl delete hpa hpa-$1
kubectl get hpa
kubectl get deployments

date
echo "ended"
