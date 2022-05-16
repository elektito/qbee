from .codegen import BaseCodeGen, BaseCode
from .program import Label
from .exceptions import InternalError
from . import stmt, expr


class QvmCode(BaseCode):
    def __init__(self):
        self._instrs = []

    def __repr__(self):
        return f'<QvmCode {self._instrs}>'

    def add(self, *instrs):
        if not all(isinstance(i, tuple) for i in instrs):
            raise InternalError('Instruction not a tuple')
        self._instrs.extend(instrs)

    def optimize(self):
        i = 0
        while i < len(self._instrs):
            cur = self._parse_instr(self._instrs[i])

            if i < len(self._instrs) - 1:
                next1 = self._parse_instr(self._instrs[i+1])
            else:
                next1 = self._parse_instr(('nop',))

            # when we have a push and a conv instruction, and the
            # conv's source type is the same as the push's type, we
            # can fold them into one push instruction.
            #
            # For example:
            #    push!  1.0
            #    conv!&
            # will be converted to:
            #    push&  1.0
            if (cur['op'] == 'push' and
                next1['op'] == 'conv' and
                cur['type_char'] == next1['src_type_char']
            ):
                arg, = cur['args']

                # Convert the argument to the dest type
                type_char = next1['type_char']
                dest_type = expr.Type.from_type_char(type_char)
                arg = dest_type.py_type(arg)

                self._instrs[i] = (f'push{next1["type_char"]}',
                                   arg)
                del self._instrs[i+1]
                continue

            # Convert normal push instructions with -1, 0, or 1
            # operands to a single instruction pushing the same value.
            #
            # Notice that it's important that this is after the
            # previous optimization (fold push and conv), because it
            # can then optimize the result of that optimization.
            if cur['op'] == 'push' and \
               cur['args'][0] in [-1, 0, 1]:
                type_char = cur['type_char']
                if cur['args'][0] == 0:
                    self._instrs[i] = (f'push0{type_char}',)
                elif cur['args'][0] == 1:
                    self._instrs[i] = (f'push1{type_char}',)
                else:
                    self._instrs[i] = (f'pushm1{type_char}',)
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
            ops = set([cur['op'], next1['op']])
            if ops == {'read', 'store'} and \
               cur['scope'] == next1['scope'] and \
               cur['type_char'] == next1['type_char'] and \
               cur['args'] == next1['args']:
                # remove both
                del self._instrs[i]
                del self._instrs[i]

                # Go back one instruction, so that if eliminating
                # these two instructions has caused a pair like
                # (push/conv) to become consecutive, we detect it.
                if i > 0:
                    i -= 1

                continue

            # Fold push/unary-op
            if (cur['op'] == 'push' and
                cur['type_char'] == next1['type_char'] and
                next1['op'] in ['not', 'neg']
            ):
                value = cur['args'][0]
                op = {
                    'not': expr.Operator.NOT,
                    'neg': expr.Operator.NEG,
                }[next1['op']]
                value_type = expr.Type.from_type_char(cur['type_char'])
                value = expr.NumericLiteral(value, value_type)
                unary_expr = expr.UnaryOp(value, op)
                value = unary_expr.eval()

                self._instrs[i] = (f'push{cur["type_char"]}', value)
                del self._instrs[i+1]

                continue

            # fold push/push/binary-op

            i += 1

    def __str__(self):
        s = ''
        for instr in self._instrs:
            op, *args = instr
            if op == '_label':
                label, = args
                s += f'{label}:\n'
            else:
                s += f'    {op}\t{", ".join(str(i) for i in args)}\n'
        return s

    def __bytes__(self):
        return b'<machine code>'

    def _parse_instr(self, instr):
        "Helper used by the optimize function"

        op, *args = instr
        type_chars = expr.Type.type_chars()
        if any(op.endswith(c) for c in type_chars):
            type_char = op[-1]
            op = op[:-1]
        else:
            type_char = None

        extras = {}
        if op[:-1] == 'conv':
            src_type_char = op[-1]
            op = op[:-1]
            extras = {'src_type_char': src_type_char}
        if op[:-1] in ['store', 'read']:
            scope = op[-1]
            op = op[:-1]
            extras = {'scope': scope}

        result = {
            'op': op,
            'type_char': type_char,
            'args': args,
        }
        result.update(extras)

        return result


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
    if left_type != left_type:
        from_char = node.type.type_char
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
            Operator.INTDIV: 'intdiv',
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
