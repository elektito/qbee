from abc import ABC, abstractmethod


class Node(ABC):
    "A node in qbee AST"

    def bind(self, compiler):
        self._compiler = compiler
        for child in self.children:
            child.bind(compiler)

    @property
    def compiler(self):
        if not hasattr(self, '_compiler') or self._compiler is None:
            raise InternalError(
                'Expression node not bound to a compiler')
        return self._compiler

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
