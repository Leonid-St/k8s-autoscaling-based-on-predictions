I will be installing a Kubernetes cluster on five Ubuntu 18.04 machines using Kubespray.

### Step 1 - basic requirements

To start off, download (clone) the [Kubespray repository](https://github.com/kubernetes-sigs/kubespray). 

For this project - I forked this version from Jan 2020: https://github.com/kubernetes-sigs/kubespray/tree/5d9986ab5fbaef62420743ed98b562b012117b5b 

Then make sure you have fulfilled their [requirements](https://github.com/kubernetes-sigs/kubespray#requirements) which are pretty basic but you will probably have to install the Python dependencies locally:

`pip install -r  requirements.txt`

Also make sure your SSH keys are on the servers as we are using Ansible for the installation. 
I copied my keys using ssh-copy-id, eg.:

ssh-copy-id dorian@193.61.36.56

Otherwise ansible will fail in the first step as being unable to reach Kubernetes.

Then make sure to copy the `bbk-cluster` cluster into the inventory directory of Kubespray - you don't have to but the commands below expect that structure

### Step 1.1 - Potentially reset the old cluster

ansible-playbook -i inventory/mycluster/hosts.yml reset.yml -b -v   --private-key=~/.ssh/id_rsa -K

eg. 
`[~/code/IdeaProjects/kubespray] => ansible-playbook -i inventory/bbk-cluster/hosts.yml reset.yml -b -v  --private-key=~/.ssh/id_rsa -K`

### Step 2 - Generate the inventory

Download the kubespray project and go to the root directory. 
I suggest using the kubespray script to generate the inventory of hosts and their role but I would then double check the actual output produced in the **inventory/mycluster/hosts.yml** file.

```
cp -r inventory/sample inventory/mycluster
declare -a IPS=(xyz xyz xyz xyz xyz)
CONFIG_FILE=inventory/mycluster/hosts.yml python3 contrib/inventory_builder/inventory.py ${IPS[@]}
```

The inventory files are now in `inventory/mycluster` folder.
Then I had to change the generated hosts.yml file to set the field **ansible_user** to my specific username on the machines for each of the hosts.

### Step 3 - Ubuntu 18.04 specific 

You will have to add a universe repository to the apt package manager as some packages won't install otherwise:

`sudo add-apt-repository universe`

Source of this [trick](https://hub.docker.com/r/coderpews/kubespray-docker).

### Step 4 - Edit variables/specific settings

I edited the file **inventory/mycluster/group_vars/k8s-cluster/addons.yml** and change a few settings for my Kubernetes deployment managed by Kubespray:

```
helm_enabled: true

metrics_server_enabled: true

local_path_provisioner_enabled: true
local_path_provisioner_namespace: "local-path-storage"
local_path_provisioner_storage_class: "local-path"
local_path_provisioner_reclaim_policy: Delete
local_path_provisioner_claim_root: /kube/localstorage/
local_path_provisioner_debug: false
local_path_provisioner_image_repo: "rancher/local-path-provisioner"
local_path_provisioner_image_tag: "v0.0.2"
```

This step also included creating a folder /kube/localstorage/ on each of the hosts for the local path provisioner to be able to construct volumes on (more info [here](https://github.com/rancher/local-path-provisioner)). 
Helm and Kubernetes metrics are generally useful to have.

I don't recommend using Kubespray to install the Kubernetes dashboard. At the time of my installation (Nov 2019) Kubespray installs an old verion of the dashboard not compatible with the latest Kubernetes version (1.16). Though this can be fixed as described in this [issue](https://github.com/kubernetes-sigs/kubespray/issues/5284).

## Part to redo for each installation

### To just add new nodes
- https://kubespray.io/#/docs/nodes?id=addingreplacing-a-worker-node 
`[~/code/IdeaProjects/kubespray] => ansible-playbook -i inventory/bbk-cluster/hosts.yml scale.yml -b -v --private-key=~/.ssh/id_rsa -K`

### Step 5 - Run the Kubernetes installation
Now we can run the actual installation:
`ansible-playbook -i inventory/mycluster/hosts.yml cluster.yml -b -v   --private-key=~/.ssh/id_rsa -K`
eg.  `[~/code/IdeaProjects/kubespray] => ansible-playbook -i inventory/bbk-cluster/hosts.yml cluster.yml -b -v --private-key=~/.ssh/id_rsa -K`

The arguments are:
-i points to my inventory created   
cluster.yml is the playbook  
-b is used because I am running as a non sudo user  
-v is for verbose execution  
--private-key points to my private key (I used ssh-copy-id to setup passwordless ssh)  
-K will ask me for the password when logging in  

The installation took around 30 minutes to complete on my cluster.

### Step 6 - Copy the kubectl config file from the remote to your local server

In order to configure kubectl on your local machine, go to the master node of your Kubernetes cluster and copy the contents of this file:
`sudo cat /etc/kubernetes/admin.conf` 

Carefully paste the contents of that file in your local Kubernetes config at `$HOME/.kube/config`
Kubectl will automatically pick up the config file stored in `$HOME/.kube/config`.

Alternatively, you can also provide a kubeconfig flag for an alternative config file
`kubectl cluster-info --kubeconfig=$HOME/.kube/config_c`

Make sure this is also the case on your cluster nodes. 
So make sure the content of `sudo cat /etc/kubernetes/admin.conf` is in the file `vim ~/.kube/config`.

### Step 7 - Check your cluster status

Run this on one of the hosts: 

```
kubectl get nodes
```
Output: 

```
NAME    STATUS   ROLES    AGE   VERSION
node1   Ready    master   12h   v1.16.2
node2   Ready    master   12h   v1.16.2
node3   Ready    <none>   12h   v1.16.2
node4   Ready    <none>   12h   v1.16.2
node5   Ready    <none>   12h   v1.16.2
```

You can also get the cluster details with:  

```
kubectl cluster-info
```

Output:

```
Kubernetes master is running at https://<xyz>:6443
coredns is running at https://<xyz>:6443/api/v1/namespaces/kube-system/services/coredns:dns/proxy
metrics-server is running at https://<xyz>:6443/api/v1/namespaces/kube-system/services/https:metrics-server:/proxy
```

### Step 8 - Deploy the Kubernetes dashboard

We will use the latest version of Kubernetes dashboard (In Nov 2019 this is [this one](https://github.com/kubernetes/dashboard/releases/tag/v2.0.0-beta5)):  

```
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.0.0-beta5/aio/deploy/recommended.yaml
```

Check the deployment with:

```
kubectl get deployments -o wide --namespace=kubernetes-dashboard
```

Output:

```
NAME                         READY   UP-TO-DATE   AVAILABLE   AGE   CONTAINERS                   IMAGES                                SELECTOR
dashboard-metrics-scraper    1/1     1            1           68s   dashboard-metrics-scraper    kubernetesui/metrics-scraper:v1.0.1   k8s-app=dashboard-metrics-scraper
kubernetes-dashboard         1/1     1            1           51m   kubernetes-dashboard         kubernetesui/dashboard:v2.0.0-beta5   k8s-app=kubernetes-dashboard
kubernetes-metrics-scraper   1/1     1            1           51m   kubernetes-metrics-scraper   kubernetesui/metrics-scraper:v1.0.0   k8s-app=kubernetes-metrics-scraper
```


### Step 9 - Connect to the Kubernetes dashboard

We will need to create a user and get it's credentials in order to login to the dashboard.
I suggest creating a service account from the below snippet.
Just copy it to a file and then deploy to Kubernetes with 
`cd /Users/dbg/code/IdeaProjects/act_project/kubernetes/_initial_kubespray_setup/
kubectl apply -f user_create.yaml`:

The file has the following content:
```
apiVersion: v1
kind: ServiceAccount
metadata:
  name: admin-user
  namespace: kube-system

---
  
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: admin-user
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-admin
subjects:
- kind: ServiceAccount
  name: admin-user
  namespace: kube-system
```

To get the token to login we need to run:

```
kubectl -n kube-system describe secret $(kubectl -n kube-system get secret | grep admin-user | awk '{print $1}')
```

Now we have to run a proxy for accessing the dashboards using the command: `kubectl proxy`

And we can access the dashboard locally at `http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/`

### Step 10 - Label the nodes

Label the kubernetes nodes because all the application helm charts need node selectors.
More details in the file `kubernetes/kubernetes/_initial_kubespray_setup/labels.txt`.
