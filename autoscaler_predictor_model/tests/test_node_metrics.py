import pytest
from utils.cluster_metrics import NodeMetrics
import pandas as pd
from datetime import datetime

@pytest.fixture
def node_metrics():
    class MockPrometheus:
        def _query_prometheus(self, query):
            return [{'metric': {'node': 'node1'}, 'value': [1742939177, 0.5]}]
        
        def get_new_data(self):
            return pd.DataFrame({
                'timestamp': [datetime.now()],
                'node': ['node1'],
                'cpu': [0.5],
                'memory': [0.3]
            })
    
    metrics = NodeMetrics("http://localhost:9090")
    metrics.prom = MockPrometheus()
    return metrics

def test_get_cluster_state(node_metrics):
    state = node_metrics.get_cluster_state()
    assert 'total_nodes' in state
    assert 'average_cpu' in state
    assert 'average_memory' in state
    assert 'nodes' in state

def test_collect_cluster_metrics(node_metrics):
    node_metrics.collect_cluster_metrics()
    assert len(node_metrics.nodes) >= 0 
