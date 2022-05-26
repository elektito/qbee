from abc import ABCMeta, abstractmethod
from .node import Node
from .expr import Expr, Type
from .program import LineNo
from .utils import parse_data, split_camel
from .exceptions import SyntaxError, InternalError


class Stmt(Node):
    @classmethod
    def type_name(cls):
        if cls.__name__.endswith('Stmt'):
            name = cls.__name__[:-4]
            name_parts = split_camel(name)
            name = ' '.join(name_parts)
            return name.upper()
        raise NameError(
            'Default Stmt.type_name() implementation only works if '
            'class name ends with "Stmt"')


class NoChildStmt(Stmt):
    @property
    def children(self):
        return []

    def replace_child(self, old_child, new_child):
        pass


class VarDeclClause(NoChildStmt):
    def __init__(self, name, var_type_name):
        self.name = name
        self.var_type_name = var_type_name

    def __repr__(self):
        type_desc = ''
        if self.var_type_name:
            type_desc = f' as {self.var_type_name}'
        return f'<VarDeclClause {self.name}{type_desc}>'

    @property
    def type(self):
        if self.var_type_name:
            return Type.from_name(self.var_type_name)

        # parameter default types are based on the DEF* statements in
        # the module level
        top_level_routine = self.compiler.routines['_main']
        return top_level_routine.get_identifier_type(self.name)

    @classmethod
    def type_name(cls):
        return 'VAR CLAUSE'


class AnyVarDeclClause(NoChildStmt):
    "A 'var AS ANY' clause"

    def __init__(self, name):
        self.name = name

    @classmethod
    def type_name(cls):
        return 'ANY VAR CLAUSE'


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


class ColorStmt(Stmt):
    def __init__(self, foreground, background, border):
        self.foreground = foreground
        self.background = background
        self.border = border

    @property
    def children(self):
        ret = []
        if self.foreground is not None:
            ret.append(self.foreground)
        if self.background is not None:
            ret.append(self.background)
        if self.border is not None:
            ret.append(self.border)
        return ret

    def replace_child(self, old_child, new_child):
        if self.foreground == old_child:
            self.foreground = new_child
        elif self.background == old_child:
            self.background == new_child
        elif self.border == old_child:
            self.border = new_child


class DeclareStmt(Stmt):
    def __init__(self, routine_type, name, params):
        assert routine_type in ('sub,')
        self.routine_type = routine_type
        self.name = name
        self.params = params

    def __repr__(self):
        return f'<DeclareStmt {self.routine_type} {self.name}>'

    def replace_child(self, old_child, new_child):
        for i in range(len(self.params)):
            if self.params[i] == old_child:
                self.params[i] = new_child
                return

        raise InternalError(
            f'No such child to replace: {old_child}')

    @property
    def children(self):
        return self.params


class DimStmt(Stmt):
    def __init__(self, var_decls):
        assert all(
            isinstance(decl, VarDeclClause)
            for decl in var_decls
        )
        self.var_decls = var_decls

    def __repr__(self):
        return f'<DimStmt {self.var_decls}>'

    @property
    def children(self):
        return self.var_decls

    def replace_child(self, old_child, new_child):
        for i in range(len(self.var_decls)):
            if self.var_decls[i] == old_child:
                self.var_decls[i] = new_child
                return

        raise InternalError(
            f'No such child to replace: {old_child}')


class GotoStmt(NoChildStmt):
    def __init__(self, target):
        self.target = target

        if isinstance(target, int):
            self.canonical_target = LineNo.get_canonical_name(target)
        else:
            self.canonical_target = target

    def __repr__(self):
        return f'<GotoStmt {self.target}>'


class IfStmt(Stmt):
    # Denotes an IF statement with at least one statement after THEN,
    # and possibly an ELSE clause.

    def __init__(self, cond, then_stmts, else_clause):
        self.cond = cond
        self.then_stmts = then_stmts
        self.else_clause = else_clause

    def __repr__(self):
        if len(self.then_stmts):
            then_desc = f'then=<{len(self.then_stmts)} stmt(s)>'
        else:
            then_desc = 'then=empty'
        if self.else_clause and len(self.else_clause.stmts):
            else_desc = f'else=<{len(self.else_clause.stmts)} stmt(s)>'
        else:
            else_desc = 'else=empty'
        return f'<IfStmt cond={self.cond} {then_desc} {else_desc}>'

    @property
    def children(self):
        ret = [self.cond]
        ret += self.then_stmts
        if self.else_clause:
            ret += [self.else_clause]
        return ret

    def replace_child(self, old_child, new_child):
        if self.cond == old_child:
            self.cond = new_child
            return

        if self.else_clause == old_child:
            self.else_clause = new_child
            return

        for i in range(len(self.then_stmts)):
            if self.then_stmts[i] == old_child:
                self.then_stmts[i] = new_child
                return

        raise InternalError(
            f'No such child to replace: {old_child}')


class IfBeginStmt(Stmt):
    # An IF statement without anything after THEN, denoting the start of
    # an IF block.

    def __init__(self, cond):
        self.cond = cond

    def __repr__(self):
        return f'<IfBeginStmt cond={self.cond}>'

    @property
    def children(self):
        return [self.cond]

    def replace_child(self, old_child, new_child):
        if old_child == self.cond:
            self.cond = new_child


class ElseClause(Stmt):
    def __init__(self, stmts):
        self.stmts = stmts

    def __repr__(self):
        return f'<ElseClause stmts={len(self.stmts)}>'

    @classmethod
    def type_name(cls):
        return 'ELSE CLAUSE'

    @property
    def children(self):
        return self.stmts

    def replace_child(self, old_child, new_child):
        for i in range(len(self.stmts)):
            if self.stmts[i] == old_child:
                self.stmts[i] = new_child
                return

        raise InternalError(
            f'No such child to replace: {old_child}')


class ElseStmt(NoChildStmt):
    def __repr__(self):
        return '<ElseStmt>'


class ElseIfStmt(Stmt):
    def __init__(self, cond, then_stmts):
        self.cond = cond
        self.then_stmts = then_stmts

    def __repr__(self):
        if len(self.then_stmts):
            then_desc = f'then=<{len(self.then_stmts)} stmt(s)>'
        else:
            then_desc = 'then=empty'
        return f'<ElseIfStmt cond={self.cond} {then_desc}>'

    @property
    def children(self):
        return [self.cond] + self.then_stmts

    def replace_child(self, old_child, new_child):
        if self.cond == old_child:
            self.cond = new_child
            return

        for i in range(len(self.then_stmts)):
            if self.then_stmts[i] == old_child:
                self.then_stmts[i] = new_child
                return

        raise InternalError(
            f'No such child to replace: {old_child}')


class EndIfStmt(NoChildStmt):
    def __repr__(self):
        return '<EndIfStmt>'


class InputStmt(Stmt):
    def __init__(self, same_line: bool, prompt: str,
                 prompt_question: bool, var_list: list):
        self.same_line = same_line
        self.prompt = prompt
        self.prompt_question = prompt_question
        self.var_list = var_list

    def __repr__(self):
        return (
            f'<InputStmt "{self.prompt}" {len(self.var_list)} var(s)>'
        )

    @property
    def children(self):
        return [self.prompt] + self.var_list

    def replace_child(self, old_child, new_child):
        if self.prompt == old_child:
            self.prompt = new_child
            return

        for i in range(len(self.var_list)):
            if self.var_list[i] == old_child:
                self.var_list[i] = new_child
                return

        raise InternalError(
            f'No such child to replace: {old_child}')


class PrintStmt(Stmt):
    def __init__(self, items):
        self.items = items

    def __repr__(self):
        return f'<PrintStmt {" ".join(str(i) for i in self.items)}>'

    @property
    def children(self):
        return [i for i in self.items if isinstance(i, Node)]

    def replace_child(self, old_child, new_child):
        for i in range(len(self.items)):
            if self.items[i] == old_child:
                self.items[i] = new_child
                return

        raise InternalError(
            f'No such child to replace: {old_child}')


class DataStmt(NoChildStmt):
    def __init__(self, string):
        self.string = string
        self.items = parse_data(self.string)

    def __repr__(self):
        return '<DataStmt>'


class TypeStmt(NoChildStmt):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'<TypeStmt {self.name}>'


class EndTypeStmt(NoChildStmt):
    def __repr__(self):
        return '<EndTypeStmt>'


class SubStmt(Stmt):
    def __init__(self, name, params):
        assert all(isinstance(p, VarDeclClause) for p in params)
        self.name = name
        self.params = params

    def __repr__(self):
        return f'<SubStmt {self.name} with {len(self.params)} param(s)>'

    @property
    def children(self):
        return []

    def replace_child(self, old_child, new_child):
        return


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
    def type_name(cls):
        if cls.__name__.endswith('Block'):
            name = cls.__name__[:-5]
            name_parts = split_camel(name)
            name = ' '.join(name_parts)
            name += '_block'
            return name.upper()
        raise NameError(
            'Default Block.name() implementation only works if class '
            'name ends with "Block"')

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
            raise SyntaxError(
                end_stmt.loc_start,
                msg=f'Expected {expected_end_stmt.type_name()}',
            )

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


class IfBlock(Block, start=IfBeginStmt, end=EndIfStmt):
    def __init__(self, if_blocks, else_body):
        # if blocks is a list of tuples, each being a pair in the form
        # of (condition, body) denoting the top-level if block and any
        # elseif blocks.
        assert all(
            isinstance(cond, Expr) and isinstance(body, list)
            for cond, body in if_blocks
        )
        self.if_blocks = if_blocks

        assert else_body is None or isinstance(else_body, list)
        self.else_body = else_body or []

    def __repr__(self):
        then_desc = f'then_blocks={len(self.if_blocks)}'
        if self.else_body:
            else_desc = f'with {len(self.else_body)} stmt(s) in else'
        else:
            else_desc = 'with no else'
        return f'<IfBlock {then_desc} {else_desc}>'

    @classmethod
    def create_block(cls, if_stmt, end_if_stmt, body):
        cur_if_cond = if_stmt.cond
        cur_if_body = []
        if_blocks = []
        else_body = []
        for stmt in body:
            if isinstance(stmt, ElseIfStmt):
                if_blocks.append((cur_if_cond, cur_if_body))
                cur_if_body = stmt.then_stmts
                cur_if_cond = stmt.cond
            elif isinstance(stmt, ElseStmt):
                if_blocks.append((cur_if_cond, cur_if_body))
                cur_if_cond = None
            elif cur_if_cond:
                cur_if_body.append(stmt)
            else:
                else_body.append(stmt)

        if cur_if_cond:
            if_blocks.append((cur_if_cond, cur_if_body))

        return cls(if_blocks, else_body)

    def replace_child(self, old_child, new_child):
        for i in range(len(self.if_blocks)):
            cond, body = self.if_blocks[i]
            if cond == old_child:
                self.if_blocks[i] = (new_child, body)
                return

            for j in range(len(body)):
                if body[j] == old_child:
                    body[j] = new_child
                    return

        raise InternalError(
            f'No such child to replace: {old_child}')

    @property
    def children(self):
        ret = []
        for cond, body in self.if_blocks:
            ret += [cond]
            ret += body
        for stmt in self.else_body:
            ret.append(stmt)

        return ret


class SubBlock(Block, start=SubStmt, end=EndSubStmt):
    def __init__(self, name, params, block):
        self.name = name
        self.params = params
        self.block = block

        # This will be set by the compiler later to a Routine object
        self.routine = None

    def __repr__(self):
        return (
            f'<SubBlock "{self.name}" with {len(self.params)} arg(s) '
            f'and {len(self.block)} statement(s)>'
        )

    def replace_child(self, old_child, new_child):
        for i in range(len(self.params)):
            if self.params[i] == old_child:
                self.params[i] = new_child
                return

        for i in range(len(self.block)):
            if self.block[i] == old_child:
                self.block[i] = new_child
                return

        raise InternalError(
            f'No such child to replace: {old_child}')

    @property
    def children(self):
        return self.params + self.block

    @classmethod
    def create_block(cls, sub_stmt, end_sub_stmt, body):
        return cls(sub_stmt.name, sub_stmt.params, body)


class TypeBlock(Block, start=TypeStmt, end=EndTypeStmt):
    def __init__(self, name: str, fields: list):
        assert isinstance(name, str)
        assert isinstance(fields, list)
        assert all(
            (
                isinstance(f, tuple) and
                len(f) == 2 and
                isinstance(f[0], str) and
                isinstance(f[1], Type)
            )
            for f in fields
        )
        self.name = name
        self.fields = dict(fields)

    def __repr__(self):
        return (
            f'<TypeBlock {self.name} with {len(self.fields)} field(s)>'
        )

    @property
    def children(self):
        return []

    def replace_child(self, old_child, new_child):
        pass

    @classmethod
    def create_block(cls, start_stmt, end_stmt, body):
        fields = []
        field_names = []
        for stmt in body:
            if not isinstance(stmt, VarDeclClause) or \
               not stmt.var_type_name:
                raise SyntaxError(
                    loc=stmt.loc_start,
                    msg='Statement illegal in TYPE block')
            var_type = Type.from_name(stmt.var_type_name)

            if stmt.name in field_names:
                raise SyntaxError(
                    loc=stmt.loc_start,
                    msg='Duplicate definition')

            fields.append((stmt.name, var_type))
            field_names.append(stmt.name)

        return cls(start_stmt.name, fields)
