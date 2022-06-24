from abc import ABC, abstractmethod
from .exceptions import InternalError


class Node(ABC):
    "A node in qbee AST"

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj.parent = None
        obj._init_args = args
        obj._init_kwargs = kwargs
        obj.__init__(*args, **kwargs)
        for child in obj.children:
            child.parent = obj
        return obj

    def __getnewargs_ex__(self):
        return self._init_args, self._init_kwargs

    def bind(self, context):
        self._context = context
        for child in self.children:
            child.parent = self
            child.bind(context)

    @property
    def context(self):
        if not hasattr(self, '_context') or \
           self._context is None:
            raise InternalError(
                f'Node {self} not bound to an evaluation context')
        return self._context

    @property
    def parent_routine(self):
        return self.context.get_node_routine(self)

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
        new_child.bind(self.context)

    def clone(self):
        def copy_arg(arg):
            if not isinstance(arg, Node):
                return arg
            new_arg = arg.clone()
            new_arg.loc_start = getattr(arg, 'loc_start', None)
            new_arg.loc_end = getattr(arg, 'loc_end', None)
            new_arg.parent = arg.parent
            if hasattr(arg, '_context'):
                new_arg._context = arg._context
            if hasattr(arg, '_parent_routine'):
                new_arg._parent_routine = arg._parent_routine
            return new_arg

        cls = self.__class__
        init_args = [copy_arg(arg) for arg in self._init_args]
        init_kwargs = {
            name: copy_arg(value)
            for name, value in self._init_kwargs.items()
        }
        result = cls.__new__(cls, *init_args, **init_kwargs)
        return result

