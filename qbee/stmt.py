from abc import ABC, abstractmethod


class Stmt(ABC):
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

class BeepStmt(Stmt):
    def __repr__(self):
        return '<BeepStmt>'

    @property
    def children(self):
        return []


class CallStmt(Stmt):
    def __init__(self, name, args):
        self.name = name
        self.args = args

    @property
    def children(self):
        return self.args

    def __repr__(self):
        return f'<CallStmt {self.name} args={self.args}>'


class ClsStmt(Stmt):
    def __repr__(self):
        return '<ClsStmt>'

    @property
    def children(self):
        return []
