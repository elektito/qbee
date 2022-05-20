from abc import ABCMeta, abstractmethod
from .node import Node
from .utils import parse_data


class Stmt(Node):
    def bind(self, compiler):
        self._compiler = compiler
        for child in self.children:
            child.bind(compiler)

    @classmethod
    def type_name(cls):
        if cls.__name__.endswith('Stmt'):
            name = ''
            for c in cls.__name__[:-4]:
                if c.isupper():
                    name += ' ' + c
                else:
                    name += c
            return name.strip().upper()
        raise NameError(
            'Default Stmt.name() implementation only works if class '
            'name ends with "Stmt"')


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


class GotoStmt(NoChildStmt):
    def __init__(self, target):
        self.target = target

    def __repr__(self):
        return '<GotoStmt {self.target}>'


class DataStmt(NoChildStmt):
    def __init__(self, string):
        self.string = string
        self.items = parse_data(self.string)

    def __repr__(self):
        return '<DataStmt>'


class SubStmt(Stmt):
    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __repr__(self):
        return f'<SubStmt {self.name} with {len(self.args)} param(s)>'

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


class EndSubStmt(NoChildStmt):
    def __repr__(self):
        return '<EndSubStmt>'


class ExitSubStmt(NoChildStmt):
    def __repr__(self):
        return '<ExitSubStmt>'


# Blocks


class BlockNodeMetaclass(type):
    def __new__(cls, name, bases, attrs, **kwargs):
        if name == 'Block':
            return super().__new__(cls, name, bases, attrs)

        if 'start' not in kwargs:
            raise TypeError(
                'Block sub-class should have a "start" keyword '
                'argument')
        if 'end' not in kwargs:
            raise TypeError(
                'Block sub-class should have an "end" keyword '
                'argument')

        start_stmt_type = kwargs['start']
        end_stmt_type = kwargs['end']

        if not issubclass(start_stmt_type, Stmt):
            raise TypeError(
                'Block "start" argument should be a Stmt sub-class')
        if not issubclass(end_stmt_type, Stmt):
            raise TypeError(
                'Block "end" argument should be a Stmt sub-class')

        attrs['start_stmt'] = start_stmt_type
        attrs['end_stmt'] = end_stmt_type
        block_class = super().__new__(cls, name, bases, attrs)

        Block.known_blocks[start_stmt_type] = block_class

        return block_class


class BlockMetaclass(BlockNodeMetaclass, ABCMeta):
    # This is needed to allow Block to inherit from Stmt while having
    # BlockNodeMetaclass as its metaclass, otherwise we get an error
    # about conflicting meta-classes, because Stmt (being an ABC) has
    # a different metaclass itself.
    pass


class Block(Stmt, metaclass=BlockMetaclass):
    # A mapping of start block statements (like SubStmt), to their
    # relevant block types (like SubBlock). This is populated by the
    # meta-class.
    known_blocks = {}

    @classmethod
    @abstractmethod
    def create_block(cls, start_stmt, end_stmt, body):
        """Sub-classes should create and return an instance of
themselves in this class method.

        """
        pass

    @classmethod
    def create(cls, start_stmt, end_stmt, body):
        block_type = Block.known_blocks[type(start_stmt)]
        expected_end_stmt = block_type.end_stmt
        if not isinstance(end_stmt, expected_end_stmt):
            raise SyntaxError(*syntax_error_args)

        block = block_type.create_block(start_stmt, end_stmt, body)
        block.loc_start = start_stmt.loc_start
        block.loc_end = end_stmt.loc_end
        return block

    @classmethod
    def start_stmt_from_end(cls, end_stmt):
        if isinstance(end_stmt, Stmt):
            end_stmt = type(end_stmt)
        if not issubclass(end_stmt, Stmt):
            raise ValueError
        for start, block in Block.known_blocks.items():
            if block.end_stmt == end_stmt:
                return block.start_stmt


class SubBlock(Block, start=SubStmt, end=EndSubStmt):
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

    @classmethod
    def create_block(cls, sub_stmt, end_sub_stmt, body):
        return cls(sub_stmt.name, sub_stmt.args, body)
