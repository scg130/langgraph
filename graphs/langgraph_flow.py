import networkx as nx
from graphviz import Digraph


class FlowBuilder:
    def __init__(self):
        self.G = nx.DiGraph()

    def add_node(self, node_id: str, label: str):
        self.G.add_node(node_id, label=label)

    def add_edge(self, a: str, b: str):
        self.G.add_edge(a, b)

    def to_dot(self) -> str:
        dot = Digraph()
        for n, d in self.G.nodes(data=True):
            dot.node(n, d.get('label', n))
        for a, b in self.G.edges():
            dot.edge(a, b)
        return dot.source


fb = FlowBuilder()
fb.add_node('start', '用户查询')
fb.add_node('cls', 'LLM分类')
fb.add_node('vec', '向量检索')
fb.add_node('graph', '图谱检索')
fb.add_node('web', 'Google搜索')
fb.add_node('fusion', 'LLM融合')
fb.add_node('end', '返回答案')

fb.add_edge('start', 'cls')
fb.add_edge('cls', 'vec')
fb.add_edge('cls', 'graph')
fb.add_edge('cls', 'web')
fb.add_edge('vec', 'fusion')
fb.add_edge('graph', 'fusion')
fb.add_edge('web', 'fusion')
fb.add_edge('fusion', 'end')

dot_text = fb.to_dot()
