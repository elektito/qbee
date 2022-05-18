from abc import ABC, abstractmethod
from .node import Node


class Stmt(Node):
    def bind(self, compiler):
        self._compiler = compiler
        for child in self.children:
            child.bind(compiler)


class NoChildStmt(Stmt):
    @property
    def children(self):
        return []

    def replace_child(self, old_child, new_child):
        pass


class AssignmentStmt(Stmt):
    def __init__(self, lvalue, rvalue):
        self.lvalue = lvalue
        self.rvalue = rvalue

    def __repr__(self):
        return f'<AssignmentStmt {self.lvalue} = {self.rvalue}>'

    @property
    def children(self):
        return [self.lvalue, self.rvalue]

    def replace_child(self, old_child, new_child):
        if self.rvalue == old_child:
            self.rvalue = new_child
        elif self.lvalue == old_child:
            self.lvalue = new_child
        else:
            raise InternalError(
                f'No such child to replace: {old_child}')


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


class DataStmt(NoChildStmt):
    def __init__(self, elements):
        self.elements = elements

    def __repr__(self):
        return '<DataStmt>'


class ExitSubStmt(NoChildStmt):
    def __repr__(self):
        return '<ExitSubStmt>'


class SubStmt(Stmt):
    def __init__(self, name, args):
        self.name = name
        self.args = args

    @property
    def children(self):
        return self.args

    def replace_child(self, old_child, new_child):
        for i in range(len(self.args)):
            if self.args[i] == old_child:
                self.args[i] = new_child
                return

        raise InternalError(
            f'No such child to replace: {old_child}')

# Blocks

class SubBlock(Stmt):
    def __init__(self, name, args, block):
        self.name = name
        self.args = args
        self.block = block

    def __repr__(self):
        return (
            f'<SubBlock "{self.name}" with {len(self.args)} arg(s) '
            f'and {len(self.block)} statement(s)>'
        )

    def replace_child(self, old_child, new_child):
        for i in range(len(self.args)):
            if self.args[i] == old_child:
                self.args[i] = new_child
                return

        for i in range(len(self.block)):
            if self.block[i] == old_child:
                self.block[i] = new_child
                return

        raise InternalError(
            f'No such child to replace: {old_child}')

    @property
    def children(self):
        return self.args + self.block
