helm repo add kube-eagle https://raw.githubusercontent.com/cloudworkz/kube-eagle-helm-chart/master
helm repo update
helm install --name=kube-eagle kube-eagle/kube-eagle

# deletion
helm del --purge kube-eagle
