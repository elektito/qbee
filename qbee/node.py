from abc import ABC, abstractmethod
from .exceptions import InternalError


class Node(ABC):
    "A node in qbee AST"

    def bind(self, compiler):
        self._compiler = compiler
        for child in self.children:
            child.parent = self
            child.bind(compiler)

    @property
    def compiler(self):
        if not hasattr(self, '_compiler') or self._compiler is None:
            raise InternalError(
                'Expression node not bound to a compiler')
        return self._compiler

    @classmethod
    @abstractmethod
    def node_name(cls):
        # An implementation should return the name of the node
        # type. For example, an "exit sub" statement will return "EXIT
        # SUB".
        return 'Node'

    @property
    @abstractmethod
    def children(self):
        # An implementation of this method should return all direct
        # child nodes of this expression. It's important that this is
        # properly implemented in all sub-classes, because the bind
        # method uses it to bind a compiler to the expression which
        # could be needed by some methods or properties.
        pass

    def fold(self):
        for child in self.children:
            folded = child.fold()
            assert isinstance(folded, Node)
            if folded != child:
                self.replace_child(child, folded)
        return self

    @abstractmethod
    def replace_child(self, old_child, new_child):
        pass

    def parents(self):
        if not hasattr(self, 'parent'):
            raise InternalError(
                'Cannot get parents of unbound node')

        results = []
        node = self
        while node.parent:
            results.append(node.parent)
            node = node.parent
        return results
