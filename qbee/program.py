from .node import Node


class Label(Node):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'<Label {self.name}>'

    @property
    def children(self):
        return []


class Program(Node):
    def __init__(self, nodes):
        self.nodes = nodes

    def __repr__(self):
        return (
            f'<Program with {len(self.nodes)} '
            f'{"node" if len(self.nodes) == 1 else "nodes"}>'
        )

    @property
    def children(self):
        return self.nodes
