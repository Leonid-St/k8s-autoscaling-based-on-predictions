from prometheus_client import PrometheusDataCollector
import pandas as pd


class NodeMetrics:
    def __init__(self, prometheus_url):
        self.prom = PrometheusDataCollector(
            prometheus_url,
            query_cpu='sum by (node) (node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate)',
            query_memory='node_memory_WorkingSet_bytes{job="node-exporter"}'
        )

        self.nodes = {}

    def _get_node_info(self):
        """Получаем информацию о нодах из kube-state-metrics"""
        query = 'kube_node_info{job="kube-state-metrics"}'
        result = self.prom._query_prometheus(query)
        return {r['metric']['node']: r['metric'] for r in result}

    def collect_cluster_metrics(self):
        """Собираем полную картину кластера"""
        node_info = self._get_node_info()
        resource_usage = self.prom.get_new_data()

        for _, row in resource_usage.iterrows():
            node_name = row['node']
            if node_name in node_info:
                self.nodes[node_name] = {
                    'role': node_info[node_name].get('label_node_role_kubernetes_io/role', 'worker'),
                    'region': node_info[node_name].get('label_topology_kubernetes_io/region', ''),
                    'zone': node_info[node_name].get('label_topology_kubernetes_io/zone', ''),
                    'cpu_usage': row['cpu'],
                    'memory_usage': row['memory']
                }

    def get_cluster_state(self):
        """Текущее состояние кластера для принятия решений"""
        return {
            'total_nodes': len(self.nodes),
            'average_cpu': pd.Series([n['cpu_usage'] for n in self.nodes.values()]).mean(),
            'average_memory': pd.Series([n['memory_usage'] for n in self.nodes.values()]).mean(),
            'nodes': self.nodes
        }
