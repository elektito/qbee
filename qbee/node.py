from abc import ABC, abstractmethod
from .exceptions import InternalError


class Node(ABC):
    "A node in qbee AST"

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj.parent = None
        obj.__init__(*args, **kwargs)
        for child in obj.children:
            child.parent = obj
        return obj

    def bind(self, compilation):
        self._compilation = compilation
        for child in self.children:
            child.parent = self
            child.bind(compilation)

    @property
    def compilation(self):
        if not hasattr(self, '_compilation') or \
           self._compilation is None:
            raise InternalError(
                'Expression node not bound to a compilation unit')
        return self._compilation

    @property
    def parent_routine(self):
        node = self
        while node:
            try:
                return self._parent_routine
            except AttributeError:
                node = node.parent

        return None

    @parent_routine.setter
    def parent_routine(self, node):
        self._parent_routine = node

    @classmethod
    @abstractmethod
    def node_name(cls):
        # An implementation should return the name of the node
        # type. For example, an "exit sub" statement will return "EXIT
        # SUB".
        return 'Node'

    @property
    @abstractmethod
    def child_fields(self):
        # An implementation should provide a list of fields in which
        # either a child node or a list of child nodes are stored
        # (this can be set as a class attribute)
        return []

    @property
    def children(self) -> list['Node']:
        children = []
        for field in self.child_fields:
            child = getattr(self, field)
            if child is None:
                continue
            if isinstance(child, list):
                assert all(
                    n is None or isinstance(n, Node) for n in child)
                children += child
            elif isinstance(child, Node):
                children.append(child)
            else:
                raise InternalError(
                    f'Child {child} is not a Node or a list of Nodes')
        return children

    def fold(self):
        for child in self.children:
            folded = child.fold()
            assert isinstance(folded, Node)
            if folded != child:
                self.replace_child(child, folded)
        return self

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

    def replace_child(self, old_child, new_child):
        if not isinstance(new_child, Node):
            raise InternalError('Replacement child is not a Node')

        found = False
        for field in self.child_fields:
            child = getattr(self, field)
            if child is None:
                continue

            if child == old_child:
                setattr(self, field, new_child)
                found = True
                break
            elif isinstance(child, list):
                for i in range(len(child)):
                    if child[i] == old_child:
                        child[i] = new_child
                        found = True
                        break

        if not found:
            raise InternalError(
                f'No such child to replace: {old_child}')

        new_child.loc_start = old_child.loc_start
        new_child.loc_end = old_child.loc_end
        new_child.parent = self
        new_child.bind(self.compilation)
