from .node import Node
from .exceptions import InternalError


class Label(Node):
    child_fields = []

    def __init__(self, name):
        self.name = self.canonical_name = name

    def __repr__(self):
        return f'<Label {self.name}>'

    @classmethod
    def node_name(cls):
        return 'LABEL'


class LineNo(Node):
    child_fields = []

    def __init__(self, number):
        self.number = number
        self.canonical_name = self.get_canonical_name(number)

    def __repr__(self):
        return f'<LineNo {self.number}>'

    @classmethod
    def node_name(cls):
        return 'LINENO'

    @staticmethod
    def get_canonical_name(line_number: int):
        assert isinstance(line_number, int)
        return f'_lineno_{line_number}'


class Line(Node):
    child_fields = ['nodes']

    def __init__(self, nodes):
        assert isinstance(nodes, list)
        self.nodes = nodes

    def __repr__(self):
        return (
            f'<Line with {len(self.nodes)} '
            f'{"node" if len(self.nodes) == 1 else "nodes"}>'
        )

    @classmethod
    def node_name(cls):
        return 'LINE'


class Program(Node):
    child_fields = ['nodes']

    def __init__(self, nodes):
        self.parent = None
        self.nodes = nodes

    def __repr__(self):
        return (
            f'<Program with {len(self.nodes)} '
            f'{"node" if len(self.nodes) == 1 else "nodes"}>'
        )

    @classmethod
    def node_name(cls):
        return 'PROGRAM'
