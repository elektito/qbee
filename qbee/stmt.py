from abc import ABC, abstractmethod
from .node import Node


class Stmt(Node):
    def bind(self, compiler):
        self._compiler = compiler
        for child in self.children:
            child.bind(compiler)

    @property
    @abstractmethod
    def children(self):
        # An implementation of this method should return all direct
        # child expressions of this statement. It's important that
        # this is properly implemented in all sub-classes, because the
        # bind method (for both statements and expressions) uses it to
        # bind a compiler to the expression which could be needed by
        # some methods or properties.
        pass


class NoChildStmt(Stmt):
    @property
    def children(self):
        return []

    def replace_child(self, old_child, new_child):
        pass


class BeepStmt(NoChildStmt):
    def __repr__(self):
        return '<BeepStmt>'


class CallStmt(Stmt):
    def __init__(self, name, args):
        self.name = name
        self.args = args

    def replace_child(self, old_child, new_child):
        for i in range(len(self.args)):
            if self.args[i] == old_child:
                self.args[i] = new_child
                break
        else:
            raise InternalError(
                f'No such child to replace: {old_child}')

    @property
    def children(self):
        return self.args

    def __repr__(self):
        return f'<CallStmt {self.name} args={self.args}>'


class ClsStmt(NoChildStmt):
    def __repr__(self):
        return '<ClsStmt>'
