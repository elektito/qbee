from enum import Enum, auto
from .codegen import BaseCodeGen, BaseCode
from .program import Label, LineNo
from .exceptions import InternalError
from . import stmt, expr


class CanonicalOp(Enum):
    "VM ops without scope, type, etc."

    ADD = auto()
    AND = auto()
    CALL = auto()
    CONV = auto()
    DIV = auto()
    EQV = auto()
    EXP = auto()
    IDIV = auto()
    IMP = auto()
    IOREQ = auto()
    MOD = auto()
    MUL = auto()
    NEG = auto()
    NOP = auto()
    NOT = auto()
    OR = auto()
    PUSH = auto()
    READ = auto()
    RET = auto()
    SUB = auto()
    STORE = auto()
    XOR = auto()

    _LABEL = 10000

    def __eq__(self, other):
        if not isinstance(other, CanonicalOp):
            raise InternalError(
                f'Attempting to compare op with: '
                f'{type(other).__name__}')
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.value)


Op = CanonicalOp


class QvmInstr:
    """Represents a QVM instruction. It receives its data from either an
iterable, or a list of arguments it receives.

For certain instructions, like push variants, there's a canonical
form, which is always in the form ('push&', 10), and there's also the
final form which might be in the form ('push1&',).

    """

    def __init__(self, *elements):
        assert elements
        if isinstance(elements[0], tuple):
            assert len(elements) == 1
            elements = elements[0]

        op, *self.args = elements
        op = op.lower()
        assert isinstance(op, str)

        self.type = None
        if expr.Type.is_type_char(op[-1]):
            self.type = expr.Type.from_type_char(op[-1])
            op = op[:-1]

        self.src_type = None
        if expr.Type.is_type_char(op[-1]):
            self.src_type = expr.Type.from_type_char(op[-1])
            op = op[:-1]

        self.scope = None
        if op in ('storel', 'storeg', 'readl', 'readg'):
            self.scope = op[-1]
            op = op[:-1]

        if op in ('push1', 'push0', 'pushm1'):
            intrinsic_value = {
                'push1': 1,
                'push0': 0,
                'pushm1': -1,
            }[op]
            op = 'push'
            self.args = (intrinsic_value,)

        self.op = Op[op.upper()]

    @property
    def op(self):
        return self._op

    @op.setter
    def op(self, value):
        assert isinstance(value, Op)
        self._op = value

    @property
    def canonical(self):
        return (self.op, *self.args)

    @property
    def final(self):
        op = self.op.name.lower()
        if op == 'push' and self.args[0] in (1, 0, -1):
            op = {
                1: 'push1',
                0: 'push0',
                -1: 'pushm1',
            }[self.args[0]]
            args = ()
        else:
            args = self.args

        type_char = ''
        if self.type:
            type_char = self.type.type_char

        src_type_char = ''
        if self.src_type:
            src_type_char = self.src_type.type_char

        scope = ''
        if self.scope:
            scope = self.scope

        op = op + scope + src_type_char + type_char
        return (op, *args)

    def __eq__(self, other):
        if isinstance(other, QvmInstr):
            return self.final == other.final
        elif isinstance(other, tuple):
            return self.final == other
        else:
            raise TypeError('Cannot compare QvmInstr with {other!r}')

    def __str__(self):
        op, *args = self.final
        args_str = ', '.join(str(i) for i in args)
        return f'{op}\t{args_str}'

    def __repr__(self):
        op, *args = self.final
        tup = (op, *args)
        return repr(tup)


class QvmCode(BaseCode):
    def __init__(self):
        self._instrs = []

    def __repr__(self):
        return f'<QvmCode {self._instrs}>'

    def add(self, *instrs):
        if not all(isinstance(i, tuple) for i in instrs):
            raise InternalError('Instruction not a tuple')
        self._instrs.extend([QvmInstr(*i) for i in instrs])

    def add_data(self, data):
        # Ignoring fr now
        pass

    def optimize(self):
        i = 0
        while self._instrs and i < len(self._instrs):
            # in case in one of the iterations we move past the
            # beginning of the list
            if i < 0:
                i = 0

            cur = self._instrs[i]

            if i > 0:
                prev1 = self._instrs[i-1]
            else:
                prev1 = QvmInstr('nop')

            if i > 1:
                prev2 = self._instrs[i-2]
            else:
                prev2 = QvmInstr('nop')

            # when we have a push and a conv instruction, and the
            # conv's source type is the same as the push's type, we
            # can fold them into one push instruction.
            #
            # For example:
            #    push!  1.0
            #    conv!&
            # will be converted to:
            #    push&  1.0
            if (cur.op == Op.CONV and
                prev1.op == Op.PUSH and
                cur.src_type == prev1.type
            ):
                arg, = prev1.args

                # Convert the argument to the dest type
                arg = cur.type.py_type(arg)

                self._instrs[i-1] = QvmInstr(
                    f'push{cur.type.type_char}', arg)

                del self._instrs[i]
                i -= 1

                continue

            # Eliminate pairs of compatible consecutive read/store or
            # store/read.
            #
            # For example this pair:
            #    storel# x
            #    readl#  x
            # or this pair:
            #    readg%  x
            #    storeg% x
            ops = {cur.op, prev1.op}
            if (ops == {Op.READ, Op.STORE} and
               cur.scope == prev1.scope and
               cur.type == prev1.type and
               cur.args == prev1.args
            ):
                # remove both
                del self._instrs[i]
                del self._instrs[i-1]
                i -= 2

                continue

            # Fold push/unary-op
            if (prev1.op == Op.PUSH and
                cur.type == prev1.type and
                cur.op in [Op.NOT, Op.NEG]
            ):
                value = prev1.args[0]
                op = {
                    Op.NOT: expr.Operator.NOT,
                    Op.NEG: expr.Operator.NEG,
                }[cur.op]
                value = expr.NumericLiteral(value, prev1.type)
                unary_expr = expr.UnaryOp(value, op)
                value = unary_expr.eval()

                self._instrs[i-1] = QvmInstr(
                    f'push{cur.type.type_char}', value)
                del self._instrs[i]

                i -= 1

                continue

            # Fold push/push/binary-op
            if (prev1.op == Op.PUSH and prev2.op == Op.PUSH and
                cur.type == prev1.type == prev2.type and
                cur.op in [Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.AND,
                           Op.OR, Op.XOR, Op.EQV, Op.IMP, Op.IDIV,
                           Op.MOD, Op.EXP]
            ):
                left = prev2.args[0]
                right = prev1.args[0]
                op = {
                    Op.ADD: expr.Operator.ADD,
                    Op.SUB: expr.Operator.SUB,
                    Op.MUL: expr.Operator.MUL,
                    Op.DIV: expr.Operator.DIV,
                    Op.AND: expr.Operator.AND,
                    Op.OR: expr.Operator.OR,
                    Op.XOR: expr.Operator.XOR,
                    Op.EQV: expr.Operator.EQV,
                    Op.IMP: expr.Operator.IMP,
                    Op.IDIV: expr.Operator.INTDIV,
                    Op.MOD: expr.Operator.MOD,
                    Op.EXP: expr.Operator.EXP,
                }[cur.op]
                left = expr.NumericLiteral(left, prev2.type)
                right = expr.NumericLiteral(right, prev1.type)
                binary_expr = expr.BinaryOp(left, right, op)
                value = binary_expr.eval()

                self._instrs[i-2] = QvmInstr(
                    f'push{cur.type.type_char}', value)

                # remove the next two instructions
                del self._instrs[i]
                del self._instrs[i-1]
                i -= 2

                continue

            # Eliminate consecutive returns
            if cur.op == prev1.op == Op.RET:
                del self._instrs[i]
                i -= 1

            i += 1

    def __str__(self):
        s = ''
        for instr in self._instrs:
            op, *args = instr.final
            if op == '_label':
                label, = args
                s += f'{label}:\n'
            else:
                s += f'    {op}\t{", ".join(str(i) for i in args)}\n'
        return s

    def __bytes__(self):
        return b'<machine code>'


class QvmCodeGen(BaseCodeGen, cg_name='qvm', code_class=QvmCode):
    def __init__(self, compiler):
        self.compiler = compiler


# Code generators for expressions

@QvmCodeGen.generator_for(expr.StringLiteral)
def gen_str_literal(node, code, codegen):
    code.add((f'push$', f'"{node.value}"'))


@QvmCodeGen.generator_for(expr.NumericLiteral)
def gen_num_literal(node, code, codegen):
    code.add((f'push{node.type.type_char}', node.value))


@QvmCodeGen.generator_for(expr.Identifier)
def gen_identifier(node, code, codegen):
    identifier_type = codegen.compiler.get_identifier_type(node.name)
    type_char = identifier_type.type_char
    if codegen.compiler.is_var_global(node.name):
        scope = 'g' # global
    else:
        scope = 'l' # local
    code.add((f'read{scope}{type_char}', node.name))


@QvmCodeGen.generator_for(expr.BinaryOp)
def gen_binary_op(node, code, codegen):
    Operator = expr.Operator
    Type = expr.Type
    if node.op in [Operator.ADD, Operator.SUB, Operator.MUL,
                   Operator.DIV, Operator.EXP, Operator.NEG]:
       # Convert both operands to the type of the resulting expression
       left_type = node.type
       right_type = node.type
    elif node.op.is_comparison:
        # Convert both operands to the bigger type of both
        if node.left.type == Type.DOUBLE or \
           node.right.type == Type.DOUBLE:
            left_type = right_type = Type.DOUBLE
        elif node.left.type == Type.SINGLE or \
             node.right.type == Type.SINGLE:
            left_type = right_type = Type.SINGLE
        elif node.left.type == Type.LONG or \
             node.right.type == Type.LONG:
            left_type = right_type = Type.LONG
        else:
            left_type = right_type = Type.INTEGER
    elif node.op == Operator.MOD or \
         node.op.is_logical or \
         node.op == Operator.INTDIV:
        if node.type == Type.INTEGER:
            left_type = Type.INTEGER
            right_type = Type.INTEGER
        else:
            left_type = Type.LONG
            right_type = Type.LONG
    else:
        raise InternalError(
            'Unaccounted for binary operator: {node.op}')

    codegen.gen_code_for_node(node.left, code)
    if left_type != node.left.type:
        from_char = node.left.type.type_char
        to_char = left_type.type_char
        code.add((f'conv{from_char}{to_char}',))

    codegen.gen_code_for_node(node.right, code)
    if right_type != node.right.type:
        from_char = node.right.type.type_char
        to_char = right_type.type_char
        code.add((f'conv{from_char}{to_char}',))

    if node.op.is_comparison:
        # both operands are of the same type, so it doesn't matter we
        # use which one here
        op = f'sub{left_type.type_char}'
        code.add((op,))

        op = {
            Operator.CMP_EQ: 'eq',
            Operator.CMP_NE: 'ne',
            Operator.CMP_LT: 'lt',
            Operator.CMP_GT: 'gt',
            Operator.CMP_LE: 'le',
            Operator.CMP_GE: 'ge',
        }[node.op]
        code.add((op,))
    else:
        op = {
            Operator.ADD: 'add',
            Operator.SUB: 'sub',
            Operator.MUL: 'mul',
            Operator.DIV: 'div',
            Operator.MOD: 'mod',
            Operator.INTDIV: 'idiv',
            Operator.EXP: 'exp',
            Operator.CMP_EQ: 'sub',
            Operator.CMP_NE: 'sub',
            Operator.CMP_LT: 'sub',
            Operator.CMP_GT: 'sub',
            Operator.CMP_LE: 'sub',
            Operator.CMP_GE: 'sub',
            Operator.AND: 'and',
            Operator.OR: 'or',
            Operator.XOR: 'xor',
            Operator.EQV: 'eqv',
            Operator.IMP: 'imp',
        }[node.op]

        if op is not None:
            op += node.type.type_char
            code.add((op,))

@QvmCodeGen.generator_for(expr.UnaryOp)
def gen_unary_op(node, code, codegen):
    codegen.gen_code_for_node(node.arg, code)
    if node.op == expr.Operator.NEG:
        type_char = node.arg.type.type_char
        code.add((f'neg{type_char}',))
    elif node.op == expr.Operator.PLUS:
        # no code needed for the unary plus
        pass
    elif node.op == expr.Operator.NOT:
        if node.arg.type == expr.Type.INTEGER:
            result_type = expr.Type.INTEGER
        else:
            result_type = expr.Type.LONG

        result_type_char = result_type.type_char
        if node.arg.type != result_type:
            arg_type_char = node.arg.type.type_char
            code.add((f'conv{arg_type_char}{result_type_char}',))
        code.add((f'not{result_type_char}',))

# Code generators for statements

@QvmCodeGen.generator_for(Label)
def gen_label(node, code, codegen):
    code.add(('_label', node.name))


@QvmCodeGen.generator_for(LineNo)
def gen_lineno(node, code, codegen):
    code.add(('_label', f'_lineno_{node.number}'))


@QvmCodeGen.generator_for(stmt.AssignmentStmt)
def gen_assignment(node, code, codegen):
    codegen.gen_code_for_node(node.rvalue, code)

    dest_type_char = node.lvalue.type.type_char

    if node.rvalue.type != node.lvalue.type:
        src_type_char = node.rvalue.type.type_char
        code.add((f'conv{src_type_char}{dest_type_char}',))

    if codegen.compiler.is_var_global(node.lvalue.name):
        scope = 'g' # global
    else:
        scope = 'l' # local

    code.add((f'store{scope}{dest_type_char}', node.lvalue.name))


@QvmCodeGen.generator_for(stmt.BeepStmt)
def gen_beep(node, code, codegen):
    code.add(('ioreq', 'pcspkr', 'beep'))


@QvmCodeGen.generator_for(stmt.CallStmt)
def gen_call(node, code, codegen):
    for arg in node.args:
        codegen.gen_code_for_node(arg, code)
    code.add(
        ('push%', len(node.args)),
        ('call', node.name),
    )


@QvmCodeGen.generator_for(stmt.ClsStmt)
def gen_cls(node, code, codegen):
    code.add(('ioreq', 'screen', 'cls'))


@QvmCodeGen.generator_for(stmt.DataStmt)
def gen_cls(node, code, codegen):
    code.add_data(node.elements)


@QvmCodeGen.generator_for(stmt.ExitSubStmt)
def gen_sub_block(node, code, codegen):
    code.add(('ret',))


@QvmCodeGen.generator_for(stmt.SubBlock)
def gen_sub_block(node, code, codegen):
    code.add(('_label', '_sub_' + node.name))
    for inner_stmt in node.block:
        codegen.gen_code_for_node(inner_stmt, code)
    code.add(('ret',))
