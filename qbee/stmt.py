from abc import ABCMeta, abstractmethod
from .node import Node
from .expr import Expr, Type, Operator
from .program import LineNo
from .utils import parse_data, split_camel
from .exceptions import ErrorCode as EC, SyntaxError, CompileError


class Stmt(Node):
    @classmethod
    def node_name(cls):
        if cls.__name__.endswith('Stmt'):
            name = cls.__name__[:-4]
            name_parts = split_camel(name)
            name = ' '.join(name_parts)
            return name.upper()
        raise NameError(
            'Default Stmt.node_name() implementation only works if '
            'class name ends with "Stmt"')


class ArrayDimRange(Stmt):
    child_fields = ['lbound', 'ubound']

    def __init__(self, lbound, ubound):
        assert isinstance(lbound, Expr)
        self.lbound = lbound
        self.ubound = ubound

    def __repr__(self):
        return f'<Range {self.lbound} to {self.ubound}>'

    @property
    def static_lbound(self):
        assert self.lbound.is_const
        return int(round(self.lbound.eval()))

    @property
    def static_ubound(self):
        assert self.ubound.is_const
        return int(round(self.ubound.eval()))

    @property
    def is_const(self):
        return self.lbound.is_const and self.ubound.is_const

    @classmethod
    def node_name(cls):
        return 'ARRAY DIM RANGE'


class VarDeclClause(Stmt):
    child_fields = ['array_dims']

    def __init__(self, name, var_type_name, dims=None,
                 is_nodim_array=False):
        self.name = name
        self.var_type_name = var_type_name
        self.array_dims = dims or []
        self.is_nodim_array = is_nodim_array

    def __repr__(self):
        type_desc = ''
        if self.var_type_name:
            type_desc = f' as {self.var_type_name}'
        return f'<VarDeclClause {self.name}{type_desc}>'

    @property
    def var(self):
        return self.parent_routine.get_variable(self.name)

    @property
    def array_dims_are_const(self):
        return all(
            r.lbound.is_const and r.ubound.is_const
            for r in self.array_dims
        )

    @property
    def type(self):
        if self.var_type_name:
            _type = Type.from_name(self.var_type_name)
        else:
            _type = self.parent_routine.get_variable(self.name).type

        if self.array_dims or self.is_nodim_array:
            _type = _type.modified(is_array=True,
                                   array_dims=self.array_dims,
                                   is_nodim_array=self.is_nodim_array)

        return _type

    @classmethod
    def node_name(cls):
        return 'VAR CLAUSE'


class AnyVarDeclClause(Stmt):
    "A 'var AS ANY' clause"

    child_fields = []

    def __init__(self, name):
        self.name = name

    @classmethod
    def node_name(cls):
        return 'ANY VAR CLAUSE'


class AssignmentStmt(Stmt):
    child_fields = ['lvalue', 'rvalue']

    def __init__(self, lvalue, rvalue):
        self.lvalue = lvalue
        self.rvalue = rvalue

    def __repr__(self):
        return f'<AssignmentStmt {self.lvalue} = {self.rvalue}>'


class BeepStmt(Stmt):
    child_fields = []

    def __repr__(self):
        return '<BeepStmt>'


class CallStmt(Stmt):
    child_fields = ['args']

    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __repr__(self):
        return f'<CallStmt {self.name} args={self.args}>'


class ClsStmt(Stmt):
    child_fields = []

    def __repr__(self):
        return '<ClsStmt>'


class ColorStmt(Stmt):
    child_fields = ['foreground', 'background', 'border']

    def __init__(self, foreground, background, border):
        self.foreground = foreground
        self.background = background
        self.border = border


class ConstStmt(Stmt):
    child_fields = ['value']

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return f'<ConstStmt {self.name} = {self.value}>'


class DefSegStmt(Stmt):
    child_fields = ['segment']

    def __init__(self, segment):
        self.segment = segment

    def __repr__(self):
        segment = f' {self.segment}' if self.segment else ''
        return f'<DefSegStmt{segment}>'


class DeclareStmt(Stmt):
    child_fields = ['params']

    def __init__(self, routine_kind, name, params):
        assert routine_kind in ('sub', 'function')
        self.routine_kind = routine_kind
        self.name = name
        self.params = params

    def __repr__(self):
        return f'<DeclareStmt {self.routine_kind} {self.name}>'


class DefTypeStmt(Stmt):
    child_fields = []

    def __init__(self, def_type, letters):
        self.type = def_type
        self.letters = list(letters)

    def __repr__(self):
        letters = ",".join(sorted(self.letters))
        return f'<DefTypeStmt {self.type} {letters}>'


class DimStmt(Stmt):
    child_fields = ['var_decls']

    def __init__(self, var_decls, kind):
        assert all(
            isinstance(decl, VarDeclClause)
            for decl in var_decls
        )
        self.var_decls = var_decls
        self.kind = kind

    def __repr__(self):
        kind = ''
        if self.kind == 'dim_shared':
            kind = 'SHARED '
        elif self.kind == 'static':
            kind = 'STATIC '
        return f'<DimStmt {kind} {self.var_decls}>'


class DoStmt(Stmt):
    child_fields = ['cond']

    def __init__(self, kind, cond):
        assert kind in ('forever', 'while', 'until')
        if kind == 'forever':
            assert cond is None
        if cond is None:
            assert kind == 'forever'

        self.kind = kind
        self.cond = cond

    def __repr__(self):
        cond = f' {self.cond}' if self.cond else ''
        return f'<DoStmt {self.kind}{cond}>'


class LoopStmt(Stmt):
    child_fields = ['cond']

    def __init__(self, kind, cond):
        assert kind in ('forever', 'while', 'until')
        if kind == 'forever':
            assert cond is None
        if cond is None:
            assert kind == 'forever'

        self.kind = kind
        self.cond = cond

    def __repr__(self):
        cond = f' {self.cond}' if self.cond else ''
        return f'<LoopStmt {self.kind}{cond}>'


class EndStmt(Stmt):
    child_fields = []

    def __repr__(self):
        return 'EndStmt'


class ForStmt(Stmt):
    child_fields = ['var', 'from_expr', 'to_expr', 'step_expr']

    def __init__(self, var, from_expr, to_expr, step_expr):
        self.var = var
        self.from_expr = from_expr
        self.to_expr = to_expr
        self.step_expr = step_expr

    def __repr__(self):
        step = ' {self.step_expr}' if self.step_expr else ''
        return (
            f'<ForStmt {self.var} {self.from_expr} to {self.to_expr}'
            f'{step}>'
        )


class NextStmt(Stmt):
    child_fields = ['var']

    def __init__(self, var):
        self.var = var

    def __repr__(self):
        var = f' {self.var}' if self.var else ''
        return f'<NextStmt{var}>'


class GotoStmt(Stmt):
    child_fields = []

    def __init__(self, target):
        self.target = target

        if isinstance(target, int):
            self.canonical_target = LineNo.get_canonical_name(target)
        else:
            self.canonical_target = target

    def __repr__(self):
        return f'<GotoStmt {self.target}>'


class GosubStmt(Stmt):
    child_fields = []

    def __init__(self, target):
        self.target = target

        if isinstance(target, int):
            self.canonical_target = LineNo.get_canonical_name(target)
        else:
            self.canonical_target = target

    def __repr__(self):
        return f'<GosubStmt {self.target}>'


class ReturnStmt(Stmt):
    child_fields = []

    def __init__(self, target):
        self.target = target

        if isinstance(target, int):
            self.canonical_target = LineNo.get_canonical_name(target)
        else:
            self.canonical_target = target

    def __repr__(self):
        if self.target:
            target = f' {self.target}'
        else:
            target = ''
        return f'<ReturnStmt{target}>'


class IfStmt(Stmt):
    # Denotes an IF statement with at least one statement after THEN,
    # and possibly an ELSE clause.

    child_fields = ['cond', 'then_stmts', 'else_clause']

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


class IfBeginStmt(Stmt):
    # An IF statement without anything after THEN, denoting the start of
    # an IF block.

    child_fields = ['cond']

    def __init__(self, cond):
        self.cond = cond

    def __repr__(self):
        return f'<IfBeginStmt cond={self.cond}>'


class ElseClause(Stmt):
    child_fields = ['stmts']

    def __init__(self, stmts):
        self.stmts = stmts

    def __repr__(self):
        return f'<ElseClause stmts={len(self.stmts)}>'

    @classmethod
    def node_name(cls):
        return 'ELSE CLAUSE'


class ElseStmt(Stmt):
    child_fields = []

    def __repr__(self):
        return '<ElseStmt>'


class ElseIfStmt(Stmt):
    child_fields = ['cond', 'then_stmts']

    def __init__(self, cond, then_stmts):
        self.cond = cond
        self.then_stmts = then_stmts

    def __repr__(self):
        if len(self.then_stmts):
            then_desc = f'then=<{len(self.then_stmts)} stmt(s)>'
        else:
            then_desc = 'then=empty'
        return f'<ElseIfStmt cond={self.cond} {then_desc}>'


class EndIfStmt(Stmt):
    child_fields = []

    def __repr__(self):
        return '<EndIfStmt>'


class InputStmt(Stmt):
    child_fields = ['prompt', 'var_list']

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


class LocateStmt(Stmt):
    child_fields = ['row', 'col']

    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __repr__(self):
        return f'<LocateStmt {self.row} {self.col}>'


class PokeStmt(Stmt):
    child_fields = ['address', 'value']

    def __init__(self, address, value):
        self.address = address
        self.value = value

    def __repr__(self):
        return f'<PokeStmt {self.address} {self.value}>'


class PrintSep(Stmt):
    child_fields = []

    def __init__(self, sep):
        assert sep in (';', ',')
        self.sep = sep

    @classmethod
    def node_name(cls):
        return 'PRINT SEP'


class PrintStmt(Stmt):
    child_fields = ['items', 'format_string']

    def __init__(self, items, format_string=None):
        self.items = items
        self.format_string = format_string

    def __repr__(self):
        return f'<PrintStmt {" ".join(str(i) for i in self.items)}>'


class RandomizeStmt(Stmt):
    child_fields = ['seed']

    def __init__(self, seed):
        self.seed = seed

    def __repr__(self):
        return f'<RandomizeStmt {self.seed}>'


class ReadStmt(Stmt):
    child_fields = ['var_list']

    def __init__(self, var_list: list):
        self.var_list = var_list

    def __repr__(self):
        return f'<ReadStmt {len(self.var_list)} var(s)>'


class RestoreStmt(Stmt):
    child_fields = []

    def __init__(self, target):
        self.target = target

        if isinstance(target, int):
            self.canonical_target = LineNo.get_canonical_name(target)
        else:
            self.canonical_target = target

    def __repr__(self):
        return f'<RestoreStmt {self.target}>'


class DataStmt(Stmt):
    child_fields = []

    def __init__(self, string):
        self.string = string
        self.items = parse_data(self.string)

    def __repr__(self):
        return '<DataStmt>'


class ScreenStmt(Stmt):
    child_fields = ['mode', 'color_switch', 'apage', 'vpage']

    def __init__(self, mode, color_switch=None, apage=None, vpage=None):
        self.mode = mode
        self.color_switch = color_switch
        self.apage = apage
        self.vpage = vpage

    def __repr__(self):
        csw = self.color_switch
        csw = f' color_switch={csw}' if csw else ''
        apage = f' apage={self.apage}' if self.apage else ''
        vpage = f' vpage={self.vpage}' if self.vpage else ''
        return f'<ScreenStmt {self.mode}{csw}{apage}{vpage}'


class TypeStmt(Stmt):
    child_fields = []

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'<TypeStmt {self.name}>'


class EndTypeStmt(Stmt):
    child_fields = []

    def __repr__(self):
        return '<EndTypeStmt>'


class ViewPrintStmt(Stmt):
    child_fields = ['top_expr', 'bottom_expr']

    def __init__(self, top_expr, bottom_expr):
        assert (
            (top_expr is not None and bottom_expr is not None) or
            (top_expr is None and bottom_expr is None)
        )
        self.top_expr = top_expr
        self.bottom_expr = bottom_expr

    def __repr__(self):
        lines = ''
        if self.top_expr:
            lines = f' {self.top_expr} to {self.bottom_expr}'
        return f'<ViewPrintStmt{lines}>'


class WhileStmt(Stmt):
    child_fields = ['cond']

    def __init__(self, cond):
        self.cond = cond

    def __repr__(self):
        return '<WhileStmt {self.cond}>'


class WendStmt(Stmt):
    child_fields = []

    def __repr__(self):
        return '<WendStmt>'


class WidthStmt(Stmt):
    child_fields = ['columns', 'lines']

    def __init__(self, columns, lines):
        self.columns = columns
        self.lines = lines

    def __repr__(self):
        columns = f' {self.columns}' if self.columns else ''
        lines = f' {self.lines}' if self.lines else ''
        return f'<WidthStmt{columns}{lines}'


class SubStmt(Stmt):
    child_fields = ['params']

    def __init__(self, name, params, is_static):
        assert all(isinstance(p, VarDeclClause) for p in params)
        self.name = name
        self.params = params
        self.is_static = is_static

    def __repr__(self):
        return f'<SubStmt {self.name} with {len(self.params)} param(s)>'


class EndSubStmt(Stmt):
    child_fields = []

    def __repr__(self):
        return '<EndSubStmt>'


class ExitSubStmt(Stmt):
    child_fields = []

    def __repr__(self):
        return '<ExitSubStmt>'


class FunctionStmt(Stmt):
    child_fields = ['params']

    def __init__(self, name, params, is_static):
        assert all(isinstance(p, VarDeclClause) for p in params)

        if any(name.endswith(c) for c in Type.type_chars):
            self.name = name[:-1]
            self.type_char = name[-1]
        else:
            self.name = name
            self.type_char = None

        self.name = name
        self.params = params
        self.is_static = is_static

    def __repr__(self):
        return (
            f'<FunctionStmt {self.name} with {len(self.params)} '
            f'param(s)>'
        )

    @property
    def type(self):
        if self.type_char:
            return Type.from_type_char(self.type_char)
        else:
            return self.compilation.get_identifier_type(self.name)


class EndFunctionStmt(Stmt):
    child_fields = []

    def __repr__(self):
        return '<EndFunctionStmt>'


class ExitFunctionStmt(Stmt):
    child_fields = []

    def __repr__(self):
        return '<ExitFunctionStmt>'


class ReturnValueSetStmt(Stmt):
    child_fields = ['value']

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'<ReturnValueSetStmt {self.value}>'


class SelectStmt(Stmt):
    child_fields = ['value']

    def __init__(self, value):
        assert isinstance(value, Expr)
        self.value = value

    def __repr__(self):
        return f'<SelectStmt {self.value}>'


class SimpleCaseClause(Stmt):
    child_fields = ['value']

    def __init__(self, value):
        assert isinstance(value, Expr)
        self.value = value

    def __repr__(self):
        return f'Case({self.value})'

    @classmethod
    def node_name(cls):
        return 'SIMPLE CASE CLAUSE'


class RangeCaseClause(Stmt):
    child_fields = ['from_value', 'to_value']

    def __init__(self, from_value, to_value):
        assert isinstance(from_value, Expr)
        assert isinstance(to_value, Expr)
        self.from_value = from_value
        self.to_value = to_value

    def __repr__(self):
        return f'Case({self.from_value} To {self.to_value})'

    @classmethod
    def node_name(cls):
        return 'RANGE CASE CLAUSE'


class CompareCaseClause(Stmt):
    child_fields = ['value']
    node_name = 'COMPARE CASE CLAUSE'

    def __init__(self, op, value):
        assert isinstance(value, Expr)
        self.op = Operator.binary_op_from_token(op)
        self.value = value

        assert self.op.is_comparison

    def __repr__(self):
        return f'Case(Is {self.op} {self.value})'

    @classmethod
    def node_name(cls):
        return 'COMPARE CASE CLAUSE'


class CaseStmt(Stmt):
    child_fields = ['cases']

    def __init__(self, cases):
        self.cases = cases

    def __repr__(self):
        cases = ' '.join(str(c) for c in self.cases)
        return f'<CaseStmt {cases}>'


class CaseElseStmt(Stmt):
    child_fields = []

    def __repr__(self):
        return '<CaseElseStmt>'


class EndSelectStmt(Stmt):
    child_fields = []

    def __repr__(self):
        return '<EndSelectStmt>'


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
    def node_name(cls):
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
                msg=f'Expected {expected_end_stmt.node_name()}',
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
    # child_fields implemented further down as a property

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

    @property
    def child_fields(self):
        # we have a dynamic number of child fields, so we give them
        # fake names here and handle them in __getattr__ and
        # __setattr__
        fields = []
        for i, (cond, body) in enumerate(self.if_blocks):
            fields.append(f'_if_cond_{i}')
            fields.append(f'_if_body_{i}')
        fields.append('else_body')
        return fields

    def __getattr__(self, attr):
        if attr.startswith('_if_cond_'):
            i = int(attr[len('_if_cond_'):])
            return self.if_blocks[i][0]
        elif attr.startswith('_if_body_'):
            i = int(attr[len('_if_body_'):])
            return self.if_blocks[i][1]
        else:
            raise AttributeError

    def __setattr__(self, attr, value):
        if attr.startswith('_if_cond_'):
            i = int(attr[len('_if_cond_'):])
            self.if_blocks[i] = (value, self.if_blocks[i][1])
        elif attr.startswith('_if_body_'):
            i = int(attr[len('_if_body_'):])
            self.if_blocks[i] = (self.if_blocks[i][0], value)
        else:
            super().__setattr__(attr, value)


class SubBlock(Block, start=SubStmt, end=EndSubStmt):
    child_fields = ['params', 'block']

    def __init__(self, name, params, is_static, block):
        self.name = name
        self.params = params
        self.is_static = is_static
        self.block = block

        # This will be set by the compiler later to a Routine object
        self.routine = None

    def __repr__(self):
        return (
            f'<SubBlock "{self.name}" with {len(self.params)} arg(s) '
            f'and {len(self.block)} statement(s)>'
        )

    @classmethod
    def create_block(cls, sub_stmt, end_sub_stmt, body):
        return cls(sub_stmt.name, sub_stmt.params, sub_stmt.is_static,
                   body)


class FunctionBlock(Block, start=FunctionStmt, end=EndFunctionStmt):
    child_fields = ['params', 'block']

    def __init__(self, name, params, is_static, block):
        self._name = name
        self.params = params
        self.is_static = is_static
        self.block = block

        # This will be set by the compiler later to a Routine object
        self.routine = None

    def __repr__(self):
        return (
            f'<FunctionBlock "{self.name}" with {len(self.params)} '
            f'arg(s) and {len(self.block)} statement(s)>'
        )

    @property
    def name(self):
        if Type.is_type_char(self._name[-1]):
            return self._name[:-1]
        else:
            return self._name

    @property
    def type(self):
        return self.parent_routine.get_identifier_type(self._name)

    @classmethod
    def create_block(cls, func_stmt, end_func_stmt, body):
        return cls(func_stmt.name, func_stmt.params,
                   func_stmt.is_static, body)


class TypeBlock(Block, start=TypeStmt, end=EndTypeStmt):
    child_fields = ['decls']

    def __init__(self, name: str, decls: list):
        assert isinstance(name, str)
        assert isinstance(decls, list)
        assert all(isinstance(decl, VarDeclClause)
                   for decl in decls)

        self.name = name
        self.decls = decls
        self.fields = {decl.name: decl.type for decl in decls}

    def __repr__(self):
        return (
            f'<TypeBlock {self.name} with {len(self.fields)} field(s)>'
        )

    @classmethod
    def create_block(cls, start_stmt, end_stmt, body):
        field_names = []
        for stmt in body:
            if not isinstance(stmt, VarDeclClause) or \
               not stmt.var_type_name:
                raise SyntaxError(
                    loc=stmt.loc_start,
                    msg='Statement illegal in TYPE block')

            if stmt.name in field_names:
                raise SyntaxError(
                    loc=stmt.loc_start,
                    msg='Duplicate definition')

            field_names.append(stmt.name)

        return cls(start_stmt.name, body)


class LoopBlock(Block, start=DoStmt, end=LoopStmt):
    child_fields = ['cond', 'body']

    def __init__(self, kind, cond, body):
        self.kind = kind
        self.cond = cond
        self.body = body

    def __repr__(self):
        cond = f' {self.cond}' if self.cond else ''
        return f'<LoopBlock {self.kind}{cond}>'

    @classmethod
    def create_block(cls, do_stmt, loop_stmt, body):
        if do_stmt.cond and loop_stmt.cond:
            raise CompileError(
                EC.BLOCK_MISMATCH,
                'DO and LOOP cannot both have a condition',
                node=loop_stmt
            )

        if do_stmt.cond:
            kind = f'do_{do_stmt.kind}'
            cond = do_stmt.cond
        elif loop_stmt.cond:
            kind = f'loop_{loop_stmt.kind}'
            cond = loop_stmt.cond
        else:
            kind = 'forever'
            cond = None
        return LoopBlock(kind, cond, body)


class ForBlock(Block, start=ForStmt, end=NextStmt):
    child_fields = ['var', 'from_expr', 'to_expr', 'step_expr', 'body']

    def __init__(self, var, from_expr, to_expr, step_expr, body):
        self.var = var
        self.from_expr = from_expr
        self.to_expr = to_expr
        self.step_expr = step_expr
        self.body = body

    def __repr__(self):
        step = ' {self.step_expr}' if self.step_expr else ''
        return (
            f'<ForBlock {self.var} {self.from_expr} to {self.to_expr}'
            f'{step}>'
        )

    @classmethod
    def create_block(cls, for_stmt, next_stmt, body):
        if next_stmt.var and \
           for_stmt.var.base_var != next_stmt.var.base_var:
            raise CompileError(
                EC.BLOCK_MISMATCH,
                'FOR and NEXT variables do not match',
                node=next_stmt.var,
            )

        return ForBlock(
            for_stmt.var,
            for_stmt.from_expr,
            for_stmt.to_expr,
            for_stmt.step_expr,
            body,
        )


class SelectBlock(Block, start=SelectStmt, end=EndSelectStmt):
    # child_fields implemented further down as a property

    def __init__(self, value, case_blocks):
        assert isinstance(value, Expr)
        assert isinstance(case_blocks, list)
        assert all(isinstance(c, tuple) for c in case_blocks)
        assert all(
            isinstance(case, (CaseStmt, CaseElseStmt)) and
            isinstance(body, list) and
            all(isinstance(s, Stmt) for s in body)
            for case, body in case_blocks
        )

        self.value = value
        self.case_blocks = case_blocks

    @property
    def child_fields(self):
        # we have a dynamic number of child fields, so we give them
        # fake names here and handle them in __getattr__ and
        # __setattr__
        fields = ['value']
        for i, (cond, body) in enumerate(self.case_blocks):
            fields.append(f'_case_stmt_{i}')
            fields.append(f'_case_body_{i}')
        return fields

    def __getattr__(self, attr):
        if attr.startswith('_case_stmt_'):
            i = int(attr[len('_case_stmt_'):])
            return self.case_blocks[i][0]
        elif attr.startswith('_case_body_'):
            i = int(attr[len('_case_body_'):])
            return self.case_blocks[i][1]
        else:
            raise AttributeError

    def __setattr__(self, attr, value):
        if attr.startswith('_case_stmt_'):
            i = int(attr[len('_case_stmt_'):])
            self.case_blocks[i] = (value, self.case_blocks[i][1])
        elif attr.startswith('_case_body_'):
            i = int(attr[len('_case_body_'):])
            self.case_blocks[i] = (self.case_blocks[i][0], value)
        else:
            super().__setattr__(attr, value)

    def __repr__(self):
        return (
            f'<SelectBlock {self.value} with {len(self.case_blocks)} '
            'case blocks>'
        )

    @classmethod
    def create_block(cls, start_stmt, end_stmt, body):
        if not body:
            return SelectBlock(start_stmt.value, [])

        if not isinstance(body[0], CaseStmt):
            raise SyntaxError(
                loc=body[0].loc_start,
                msg='Statements illegal between SELECT CASE and CASE')

        cur_case = body[0]
        cur_body = []
        case_blocks = []
        for stmt in body[1:]:
            if isinstance(stmt, (CaseStmt, CaseElseStmt)):
                case_blocks.append((cur_case, cur_body))
                cur_case = stmt
                cur_body = []
            else:
                cur_body.append(stmt)

        case_blocks.append((cur_case, cur_body))

        return SelectBlock(start_stmt.value, case_blocks)


class WhileBlock(Block, start=WhileStmt, end=WendStmt):
    child_fields = ['cond', 'body']

    def __init__(self, cond, body):
        self.cond = cond
        self.body = body

    @classmethod
    def create_block(cls, while_stmt, wend_stmt, body):
        return WhileBlock(while_stmt.cond, body)
