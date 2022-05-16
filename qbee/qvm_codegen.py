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
            instr = self._instrs[i]
            op, *args = instr

            cur_instr = self._parse_instr(self._instrs[i])

            if i < len(self._instrs) - 1:
                next_instr = self._parse_instr(self._instrs[i+1])
            else:
                next_instr = self._parse_instr(('nop',))

            # when we have a push and a conv instruction, and the
            # conv's source type is the same as the push's type, we
            # can fold them into one push instruction.
            #
            # For example:
            #    push!  1.0
            #    conv!&
            # will be converted to:
            #    push&  1.0
            if cur_instr['op'] == 'push' and \
               next_instr['op'] == 'conv' and \
               cur_instr['type_char'] == next_instr['src_type_char']:
                self._instrs[i] = (f'push{next_instr["type_char"]}',
                                   *cur_instr['args'])
                del self._instrs[i+1]
                continue

            # Convert normal push instructions with -1, 0, or 1
            # operands to a single instruction pushing the same value.
            #
            # Notice that it's important that this is after the
            # previous optimization (fold push and conv), because it
            # can then optimize the result of that optimization.
            if cur_instr['op'] == 'push' and \
               cur_instr['args'][0] in [-1, 0, 1]:
                type_char = cur_instr['type_char']
                if args[0] == 0:
                    self._instrs[i] = (f'push0{type_char}',)
                elif args[0] == 1:
                    self._instrs[i] = (f'push1{type_char}',)
                else:
                    self._instrs[i] = (f'pushm1{type_char}',)
                continue

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
            Operator.NEG: 'neg',
            Operator.PLUS: None, # noop
            Operator.NOT: 'not',
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
    code.add(
        ('possibly_conv',),
        (node.op, node.type),
        ('possibly_conv',),
    )


@QvmCodeGen.generator_for(expr.Identifier)
def gen_identifier(node, code, codegen):
    code.add(
        ('PUSHID', self.type.type_char, self.name),
    )


# Code generators for statements

@QvmCodeGen.generator_for(Label)
def gen_label(node, code, codegen):
    code.add(('_label', node.name))


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
