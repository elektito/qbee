from .node import Node


class Label(Node):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'<Label {self.name}>'

    def replace_child(self, old_child, new_child):
        pass

    @property
    def children(self):
        return []


class LineNo(Node):
    def __init__(self, number):
        self.number = number

    def __repr__(self):
        return f'<LineNo {self.number}>'

    def replace_child(self, old_child, new_child):
        pass

    @property
    def children(self):
        return []


class Line(Node):
    def __init__(self, nodes):
        assert isinstance(nodes, list)
        self.nodes = nodes

    def __repr__(self):
        return (
            f'<Line with {len(self.nodes)} '
            f'{"node" if len(self.nodes) == 1 else "nodes"}>'
        )

    def replace_child(self, old_child, new_child):
        for i in range(len(self.nodes)):
            if self.nodes[i] == old_child:
                self.nodes[i] = new_child
                break
        else:
            raise InternalError(
                f'No such child to replace: {old_child}')

    @property
    def children(self):
        return self.nodes


class Program(Node):
    def __init__(self, nodes):
        self.nodes = nodes

    def __repr__(self):
        return (
            f'<Program with {len(self.nodes)} '
            f'{"node" if len(self.nodes) == 1 else "nodes"}>'
        )

    def replace_child(self, old_child, new_child):
        for i in range(len(self.nodes)):
            if self.nodes[i] == old_child:
                self.nodes[i] = new_child
                break
        else:
            raise InternalError(
                f'No such child to replace: {old_child}')

    @property
    def children(self):
        return self.nodes
