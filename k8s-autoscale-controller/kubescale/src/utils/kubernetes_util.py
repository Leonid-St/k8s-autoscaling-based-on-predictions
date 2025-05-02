# API - controls the number of instances of a deployment/replica set/stateful set
from kubernetes import client
from typing import List


class PodWrapper:
    def __init__(self, pod: client.V1Pod, container_name: str):
        self._pod: client.V1Pod = pod
        self.prometheus_ip: str = pod.status.pod_ip.replace(".", "-")
        self.prometheus_name: str = pod.metadata.name
        self.id: str = pod.metadata.uid
        self.container_name = container_name
        self.resource_requirements: client.V1ResourceRequirements = self.__get_container_resource_limits_for_pod(
            container_name, pod)

    def get_pod_ip_for_prometheus(self) -> str:
        return self.prometheus_ip

    def get_pod_and_container_names_for_prometheus(self) -> str:
        return self.prometheus_name

    def __get_container_resource_limits_for_pod(self, container_name: str,
                                                pod: client.V1Pod) -> client.V1ResourceRequirements:
        for container in pod._spec._containers:
            if container.name == container_name:
                return container.resources


class KubernetesUtil:
    @staticmethod
    def get_pods_for_deployment(namespace: str, deployment: str, container: str) -> List[PodWrapper]:
        v1 = client.CoreV1Api()
        appsV1 = client.AppsV1Api()
        deployment = appsV1.read_namespaced_deployment(name=deployment, namespace=namespace)
        # replica_set = appsV1.list_namespaced_replica_set(namespace="default", label_selector=','.join([k+"="+v for (k,v) in deployment.spec.selector.match_labels.items()])).items[0]
        # inspired by https://stackoverflow.com/questions/51521252/get-replica-set-of-the-deployment
        # replica_set = [x for x in all_replica_sets.items if all(item in x.spec.selector.match_labels.items() for item in deployment.spec.selector.match_labels.items())][0]
        # or alternatively you can also use
        # [x for x in ret1.items if ret0.spec.selector.match_labels.items() <= x.spec.selector.match_labels.items()]
        running_pods = v1.list_namespaced_pod(namespace="default", label_selector=','.join(
            [k + "=" + v for (k, v) in deployment.spec.selector.match_labels.items()]),
                                              field_selector="status.phase=Running")
        return [PodWrapper(pod, container) for pod in running_pods.items]

    @staticmethod
    def get_number_of_replicas(namespace: str, deployment: str):
        return client.AppsV1Api().read_namespaced_deployment_scale(namespace=namespace, name=deployment).status.replicas

    @staticmethod
    def set_number_of_replicas(namespace: str, deployment: str, replicas: int):
        return client.AppsV1Api().patch_namespaced_deployment_scale(namespace=namespace, name=deployment,
                                                                    body={"spec": {"replicas": replicas}})


    @staticmethod
    def get_service_ip_and_port(namespace_name: str, service_name: str):
        v1 = client.CoreV1Api()
        service = v1.read_namespaced_service(name=service_name, namespace=namespace_name)
        # Получаем clusterIP сервиса
        ip = service.spec.cluster_ip
        # Получаем первый порт сервиса (не nodePort)
        port = service.spec.ports[0].port
        # Проверяем, что IP не "None" (для Headless сервисов)
        if ip is None or ip.lower() == "none":
            # Для Headless сервисов можно попробовать DNS имя: service_name.namespace_name.svc.cluster.local
            # Но для простоты пока вернем ошибку или придумаем другую логику
            raise ValueError(f"Service {service_name} in namespace {namespace_name} is Headless (no clusterIP). Cannot determine IP.")
        return {"ip": ip, "port": port}
