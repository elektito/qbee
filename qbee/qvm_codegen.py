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


class QvmCodeGen(BaseCodeGen, cg_name='qvm', code_class=QvmCode):
    def __init__(self, compiler):
        self.compiler = compiler


# Code generators for expressions

@QvmCodeGen.generator_for(expr.NumericLiteral)
def gen_num_literal(node, code, codegen):
    code.add(('PUSH', node.type.type_char, node.value))


@QvmCodeGen.generator_for(expr.BinaryOp)
def gen_binary_op(node, code, codegen):
    code.add(codegen.gen_code_for_node(node.left, code))
    code.add(('possibly_conv',))
    code.add(codegen.gen_code_for_node(node.right, code))
    code.add(
        ('possibly_conv',),
        (node.op, node.type),
        ('possibly_conv',),
    )


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
    code.add(('beep',))


@QvmCodeGen.generator_for(stmt.CallStmt)
def gen_call(node, code, codegen):
    for arg in node.args:
        code.add(codegen.gen_code_for_node(arg, code))
    code.add(
        ('PUSHARGSLEN', len(node.args)),
        ('call', node.name),
    )


@QvmCodeGen.generator_for(stmt.ClsStmt)
def gen_cls(node, code, codegen):
    code.add(('cls',))
