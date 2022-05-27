import struct
from enum import Enum, auto
from collections import defaultdict
from .codegen import BaseCodeGen, BaseCode
from .program import Label, LineNo, Program
from .exceptions import InternalError
from .compiler import Routine
from .qvm_instrs import op_to_instr
from . import stmt, expr


QVM_DEVICES = {
    'screen': {
        'id': 2,
        'ops': {
            'cls': 1,
            'print': 2,
            'color': 3,
        },
    },
    'pcspkr': {
        'id': 3,
        'ops': {
            'beep': 1,
        },
    },
    'keyboard': {
        'id': 4,
        'ops': {
            'input': 1,
        },
    },
}


class CanonicalOp(Enum):
    "VM ops without scope, type, etc."

    ADD = auto()
    AND = auto()
    ARRIDX = auto()
    CALL = auto()
    CMP = auto()
    CONV = auto()
    DEREF = auto()
    DIV = auto()
    EQ = auto()
    EQV = auto()
    EXP = auto()
    FRAME = auto()
    GE = auto()
    GT = auto()
    IDIV = auto()
    IMP = auto()
    IO = auto()
    JMP = auto()
    JZ = auto()
    LE = auto()
    LT = auto()
    MOD = auto()
    MUL = auto()
    NE = auto()
    NEG = auto()
    NOP = auto()
    NOT = auto()
    OR = auto()
    PUSH = auto()
    PUSHREF = auto()
    READ = auto()
    READIDX = auto()
    REFIDX = auto()
    RET = auto()
    SUB = auto()
    STORE = auto()
    STOREIDX = auto()
    STOREREF = auto()
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
        if op in ('storel', 'storeg', 'readl', 'readg',
                  'storeidxl', 'storeidxg', 'readidxl', 'readidxg',
                  'pushrefl', 'pushrefg'):
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

        if self.op == Op.PUSH and self.type == expr.Type.STRING:
            if not self.args[0].startswith('"') or \
               not self.args[0].endswith('"'):
                raise InternalError('push$ argument should be quoted')

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
        if op == 'push' and self.args[0] in (-2, -1, 0, 1, 2):
            op = {
                2: 'push2',
                1: 'push1',
                0: 'push0',
                -1: 'pushm1',
                -2: 'pushm2',
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
        return f'{op: <12}{args_str}'

    def __repr__(self):
        op, *args = self.final
        tup = (op, *args)
        return repr(tup)


class QvmCode(BaseCode):
    def __init__(self):
        self._instrs = []
        self._data = defaultdict(list)
        self._user_types = {}
        self._routines = {}
        self._main_routine = None
        self._consts = []

    def __repr__(self):
        return f'<QvmCode {self._instrs}>'

    def add(self, *instrs):
        if not all(isinstance(i, tuple) for i in instrs):
            raise InternalError('Instruction not a tuple')
        self._instrs.extend([QvmInstr(*i) for i in instrs])

    def add_data(self, data, label):
        for part in data:
            self._data[label].append(part)

    def add_user_type(self, type_block):
        self._user_types[type_block.name] = type_block

    def add_routine(self, routine):
        assert isinstance(routine, Routine)
        self._routines[routine.name] = routine

    def add_const(self, value):
        if value not in self._consts:
            self._consts.append(value)

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
                if cur.type.is_integral and isinstance(arg, float):
                    # perform rounding first if casting from float to
                    # integer
                    arg = round(arg)
                arg = cur.type.py_type(arg)

                self._instrs[i-1] = QvmInstr(
                    f'push{cur.type.type_char}', arg)

                del self._instrs[i]
                i -= 1

                continue

            # Eliminate pairs of compatible consecutive read/store
            # instructions. Notice that we can't eliminate store/read
            # pairs, since they also change the value of the variable.
            #
            # For example this pair:
            #    readl x
            #    storel  x
            # or this pair:
            #    readg  x
            #    storeg x
            if (cur.op == Op.STORE and
                prev1.op == Op.READ and
                cur.scope == prev1.scope and
                cur.args == prev1.args
            ):
                # remove both
                del self._instrs[i]
                del self._instrs[i-1]
                i -= 2

                continue

            # Fold push/unary-op
            if (prev1.op == Op.PUSH and cur.op in [Op.NOT, Op.NEG]):
                value = prev1.args[0]
                op = {
                    Op.NOT: expr.Operator.NOT,
                    Op.NEG: expr.Operator.NEG,
                }[cur.op]
                value = expr.NumericLiteral(value, prev1.type)
                unary_expr = expr.UnaryOp(value, op)
                value = unary_expr.eval()

                self._instrs[i-1] = QvmInstr(
                    f'push{prev1.type.type_char}', value)
                del self._instrs[i]

                i -= 1

                continue

            # Fold push/push/binary-op
            if (prev1.op == prev2.op == Op.PUSH and
                prev1.type == prev2.type and
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
                    f'push{prev1.type.type_char}', value)

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
        def fmt_type(type):
            if not type.is_array:
                return type.name
            base = type.array_base_type.name
            bounds = '()'
            if type.is_static_array:
                bounds = ', '.join(
                    f'{d.static_lbound} to {d.static_ubound}'
                    for d in type.array_dims
                )
                bounds = f'({bounds})'
            return f'{base}{bounds}'

        s = ';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;\n'

        if self._user_types:
            s += '.types\n'
            for user_type in self._user_types.values():
                s += f'\n{user_type.name}:\n'
                for field_name, field_type in user_type.fields.items():
                    s += f'    {field_type.name} {field_name}\n'
            s += '\n;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;\n'

        if self._consts:
            s += '.consts\n'
            for i, const in enumerate(self._consts):
                s += f'    {i} string "{const}"\n'
            s += '\n;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;\n'

        if self._data:
            s += '.data\n\n'
            for label, data in self._data.items():
                s += f'{label}:\n'
                for item in data:
                    s += f'    {item}\n'
            s += '\n;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;\n'

        if self._routines:
            s += '.routines\n\n'
            for routine in self._routines.values():
                s += f'{routine.name}:\n'
                for pname, ptype in routine.params.items():
                    s += f'    {fmt_type(ptype)} {pname}\n'
                for vname, vtype in routine.local_vars.items():
                    s += f'    {fmt_type(vtype)} {vname}\n'
            s += '\n;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;\n'

        s += '.code\n\n'
        for instr in self._instrs:
            op, *args = instr.final
            if op == '_label':
                label, = args
                s += f'{label}:\n'
            else:
                s += f'    {op: <12}{", ".join(str(i) for i in args)}\n'
        return s

    @property
    def assembled(self):
        def bconv(value, type_char):
            value_type = expr.Type.from_type_char(type_char)
            if type_char == '$':
                assert value.startswith('"') and value.endswith('"')
                value = get_const_idx(value[1:-1])
            else:
                value = value_type.py_type(value)
            return {
                '%': lambda v: struct.pack('>h', v),
                '&': lambda v: struct.pack('>i', v),
                '!': lambda v: struct.pack('>f', v),
                '#': lambda v: struct.pack('>d', v),
                '$': lambda v: struct.pack('>H', v)
            }[type_char](value)

        def get_const_idx(value):
            return self._consts.index(value)

        def get_var_idx(routine, var):
            idx = 0
            for pname, ptype in routine.params.items():
                if var == pname:
                    return idx
                idx += expr.Type.get_type_size(ptype, self._user_types)
            return idx

        cur_offset = 0
        cur_routine = self._main_routine
        labels = {}
        patch_positions = {}
        code = bytearray()
        for instr in self._instrs:
            op, *args = instr.final
            if op == '_label':
                assert len(args) == 1
                name = args[0]
                labels[name] = cur_offset
                if name.startswith('_sub_'):
                    cur_routine = name[len('_sub_'):]
                    cur_routine = self._routines[cur_routine]
                continue

            bargs = b''
            if op == 'call':
                assert len(args) == 1
                label = args[0]
                patch_positions[cur_offset + 1] = label
                bargs = struct.pack('>I', 0)  # to be patched
            elif op == 'frame':
                assert len(args) == 2
                params_size, vars_size = args
                bargs += struct.pack('>HH', params_size, vars_size)
            elif op == 'io':
                assert len(args) == 2
                device, device_op = args
                device_id = QVM_DEVICES[device]['id']
                device_op = QVM_DEVICES[device]['ops'][device_op]
                bargs = struct.pack('>BB', device_id, device_op)
            elif op == 'jmp':
                assert len(args) == 1
                label = args[0]
                patch_positions[cur_offset + 1] = label
                bargs = struct.pack('>I', 0)  # to be patched
            elif op == 'jz':
                assert len(args) == 1
                label = args[0]
                patch_positions[cur_offset + 1] = label
                bargs = struct.pack('>I', 0)  # to be patched
            elif op[:4] == 'push' and op[4:] in '%&!#$':
                assert len(args) == 1
                value = args[0]
                type_char = op[4]
                bargs = bconv(value, type_char)
            elif op in ('readg', 'readl', 'storeg', 'storel'):
                assert len(args) == 1
                var = args[0]
                var_idx = get_var_idx(cur_routine, var)
                bargs = struct.pack('>H', var_idx)
            elif op in ('readidxg',
                        'readidxl',
                        'storeidxg',
                        'storeidxl'):
                assert len(args) == 2
                var, idx = args
                var_idx = get_var_idx(cur_routine, var)
                bargs = struct.pack('>HH', var_idx, idx)
            elif op in ('pushrefl', 'pushrefg'):
                assert len(args) == 1
                var = args[0]
                var_idx = get_var_idx(cur_routine, var)
                bargs = struct.pack('>H', var_idx)
            else:
                assert len(args) == 0
                assert op.islower()
            op_code = op_to_instr[op].op_code
            op_code = bytes([op_code])
            code += op_code + bargs
            cur_offset += 1 + len(bargs)

        for pos, label in patch_positions.items():
            addr = labels[label]
            code[pos:pos+4] = struct.pack('>I', addr)

        return bytes(code)

    def __bytes__(self):
        const_section = b'\x01'
        const_section += struct.pack('>H', len(self._consts))
        for const in self._consts:
            const_section += struct.pack('>H', len(const))
            const_section += const.encode('cp437')

        data_section = b'\x02'
        data_section += struct.pack('>H', len(self._data))
        for data_part in self._data.values():
            data_section += struct.pack('>H', len(data_part))
            for data_item in data_part:
                data_section += struct.pack('>H', len(data_item))
                data_section += data_item.encode('cp437')

        code = self.assembled
        code_size = len(code)
        code_size = struct.pack('>I', code_size)
        code_section = b'\x03' + code_size + code

        return const_section + data_section + code_section


class QvmCodeGen(BaseCodeGen, cg_name='qvm', code_class=QvmCode):
    def __init__(self, compiler):
        self.compiler = compiler
        self.last_label = None
        self.label_counter = 1

    def get_label(self, name):
        label = f'_{name}_{self.label_counter}'
        self.label_counter += 1
        return label

    def init_code(self, code):
        for user_type in self.compiler.user_types.values():
            code.add_user_type(user_type)


# Shared code


def get_lvalue_dotted_index(lvalue, codegen):
    assert isinstance(lvalue, expr.Lvalue)

    idx = 0
    if lvalue.dotted_vars:
        base_type = lvalue.base_type
        for var in lvalue.dotted_vars:
            struct_name = base_type.user_type_name
            struct = codegen.compiler.user_types[struct_name]
            field_index = list(struct.fields).index(var)
            idx += field_index
            base_type = struct.fields[var]

    return idx


def gen_lvalue_write(node, code, codegen):
    # this function is called from any code generator that needs to
    # write to an lvalue. the value to write should already be on top
    # of the stack

    assert isinstance(node, expr.Lvalue)

    if node.base_is_ref or node.base_type.is_array:
        gen_lvalue_ref(node, code, codegen)
        code.add(('storeref',))
        return

    idx = get_lvalue_dotted_index(node, codegen)

    if codegen.compiler.is_var_global(node.base_var):
        scope = 'g'  # global
    else:
        scope = 'l'  # local

    if idx == 0:
        code.add((f'store{scope}', node.base_var))
    else:
        code.add((f'storeidx{scope}', node.base_var, idx))


def gen_lvalue_ref(node, code, codegen):
    assert isinstance(node, expr.Lvalue)

    if node.array_indices:
        for i, aidx in enumerate(node.array_indices):
            codegen.gen_code_for_node(aidx, code)
            gen_code_for_conv(expr.Type.LONG, aidx, code, codegen)

    if codegen.compiler.is_var_global(node.base_var):
        scope = 'g'  # global
    else:
        scope = 'l'  # local

    if node.base_is_ref:
        # already a reference; just read it onto stack
        code.add((f'read{scope}', node.base_var))
    else:
        # push a reference to base onto the stack
        code.add((f'pushref{scope}', node.base_var))

    if node.array_indices:
        code.add(('arridx',))

    idx = get_lvalue_dotted_index(node, codegen)
    if idx > 0:
        if idx < 32768:
            code.add(('push%', idx))
        else:
            code.add(('push&', idx))
        code.add(('refidx',))


def gen_code_for_conv(to_type, node, code, codegen):
    assert isinstance(to_type, expr.Type)
    assert not node.type.is_array
    assert not node.type.is_user_defined
    assert not to_type.is_array
    assert not to_type.is_user_defined
    if node.type != to_type:
        from_char = node.type.type_char
        to_char = to_type.type_char
        code.add((f'conv{from_char}{to_char}',))


# Code generators for expressions


@QvmCodeGen.generator_for(Program)
def gen_program(node, code, codegen):
    code._main_routine = node.routine
    code.add_routine(node.routine)

    code.add(('_label', '_sub_' + node.routine.name))
    code.add(('frame',
              node.routine.params_size,
              node.routine.local_vars_size))

    sub_routines = []
    for child in node.children:
        if isinstance(child, (stmt.SubBlock, stmt.FunctionBlock)):
            sub_routines.append(child)
            continue
        codegen.gen_code_for_node(child, code)

    code.add(('ret',))

    for child in sub_routines:
        codegen.gen_code_for_node(child, code)


@QvmCodeGen.generator_for(expr.StringLiteral)
def gen_str_literal(node, code, codegen):
    code.add_const(node.value)
    code.add(('push$', f'"{node.value}"'))


@QvmCodeGen.generator_for(expr.NumericLiteral)
def gen_num_literal(node, code, codegen):
    code.add((f'push{node.type.type_char}', node.value))


@QvmCodeGen.generator_for(expr.ParenthesizedExpr)
def gen_paren(node, code, codegen):
    codegen.gen_code_for_node(node.child, code)


@QvmCodeGen.generator_for(expr.FuncCall)
def gen_func_call(node, code, codegen):
    func = codegen.compiler.get_routine(node.name)
    for arg, param_type in zip(node.args, func.params.values()):
        codegen.gen_code_for_node(arg, code)
        if not isinstance(arg, expr.ArrayPass):
            gen_code_for_conv(param_type, arg, code, codegen)
    code.add(('call', '_func_' + node.name))


@QvmCodeGen.generator_for(expr.Lvalue)
def gen_lvalue(node, code, codegen):
    # notice that this function is only called for reading lvalues;
    # writing is performed in other code generators like those for
    # assignment, input, etc.

    if codegen.compiler.is_var_global(node.base_var):
        scope = 'g'  # global
    else:
        scope = 'l'  # local

    if node.base_is_ref or node.base_type.is_array:
        gen_lvalue_ref(node, code, codegen)
        code.add(('deref',))
    else:
        idx = get_lvalue_dotted_index(node, codegen)
        if idx == 0:
            code.add((f'read{scope}', node.base_var))
        else:
            code.add((f'readidx{scope}', node.base_var, idx))


@QvmCodeGen.generator_for(expr.ArrayPass)
def gen_array_pass(node, code, codegen):
    var_type = node.parent_routine.get_variable_type(node.identifier)
    assert var_type.is_array

    if codegen.compiler.is_var_global(node.identifier):
        scope = 'g'  # global
    else:
        scope = 'l'  # local
    code.add((f'pushref{scope}', node.identifier))


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
        elif (node.left.type == Type.SINGLE or
              node.right.type == Type.SINGLE):
            left_type = right_type = Type.SINGLE
        elif (node.left.type == Type.LONG or
              node.right.type == Type.LONG):
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
    gen_code_for_conv(left_type, node.left, code, codegen)

    codegen.gen_code_for_node(node.right, code)
    gen_code_for_conv(right_type, node.right, code, codegen)

    if node.op.is_comparison:
        # both operands are of the same type, so it doesn't matter we
        # use which one here
        code.add(('cmp',))

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
            Operator.AND: 'and',
            Operator.OR: 'or',
            Operator.XOR: 'xor',
            Operator.EQV: 'eqv',
            Operator.IMP: 'imp',
        }[node.op]

        if op is not None:
            code.add((op,))


@QvmCodeGen.generator_for(expr.UnaryOp)
def gen_unary_op(node, code, codegen):
    codegen.gen_code_for_node(node.arg, code)
    if node.op == expr.Operator.NEG:
        code.add(('neg',))
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
        code.add(('not',))


# Code generators for statements


@QvmCodeGen.generator_for(Label)
@QvmCodeGen.generator_for(LineNo)
def gen_label(node, code, codegen):
    codegen.last_label = node.canonical_name
    code.add(('_label', node.canonical_name))


@QvmCodeGen.generator_for(stmt.AssignmentStmt)
def gen_assignment(node, code, codegen):
    codegen.gen_code_for_node(node.rvalue, code)

    if node.rvalue.type != node.lvalue.type:
        src_type_char = node.rvalue.type.type_char
        dest_type_char = node.lvalue.type.type_char
        code.add((f'conv{src_type_char}{dest_type_char}',))

    gen_lvalue_write(node.lvalue, code, codegen)


@QvmCodeGen.generator_for(stmt.BeepStmt)
def gen_beep(node, code, codegen):
    code.add(('io', 'pcspkr', 'beep'))


@QvmCodeGen.generator_for(stmt.CallStmt)
def gen_call(node, code, codegen):
    routine = codegen.compiler.routines[node.name]
    for arg, param_type in zip(node.args, routine.params.values()):
        if isinstance(arg, expr.Lvalue):
            gen_lvalue_ref(arg, code, codegen)
        else:
            codegen.gen_code_for_node(arg, code)
            if arg.type != param_type:
                from_type_char = arg.type.type_char
                to_type_char = param_type.type_char
                code.add((f'conv{from_type_char}{to_type_char}',))
    code.add(
        ('push%', len(node.args)),
        ('call', '_sub_' + node.name),
    )


@QvmCodeGen.generator_for(stmt.ClsStmt)
def gen_cls(node, code, codegen):
    code.add(('io', 'screen', 'cls'))


@QvmCodeGen.generator_for(stmt.ColorStmt)
def gen_color(node, code, codegen):
    if node.foreground is not None:
        codegen.gen_code_for_node(node.foreground, code)
        if node.foreground.type != expr.Type.INTEGER:
            code.add((f'conv{node.foreground.type.type_char}%',))
    else:
        code.add(('push%', -1))

    if node.background is not None:
        codegen.gen_code_for_node(node.background, code)
        if node.background.type != expr.Type.INTEGER:
            code.add((f'conv{node.background.type.type_char}%',))
    else:
        code.add(('push%', -1))

    if node.border is not None:
        codegen.gen_code_for_node(node.border, code)
        if node.border.type != expr.Type.INTEGER:
            code.add((f'conv{node.border.type.type_char}%',))
    else:
        code.add(('push%', -1))

    code.add(('io', 'screen', 'color'))


@QvmCodeGen.generator_for(stmt.DeclareStmt)
def gen_declare(node, code, codegen):
    # no code for DECLARE statements
    pass


@QvmCodeGen.generator_for(stmt.DimStmt)
def gen_dim(node, code, codegen):
    # no code for DIM statements
    pass


@QvmCodeGen.generator_for(stmt.GotoStmt)
def gen_goto(node, code, codegen):
    code.add(('jmp', node.canonical_target))


@QvmCodeGen.generator_for(stmt.IfBlock)
def gen_if_block(node, code, codegen):
    endif_label = codegen.get_label('endif')

    for cond, body in node.if_blocks:
        else_label = codegen.get_label('else')
        codegen.gen_code_for_node(cond, code)
        code.add(('jz', else_label))
        for inner_stmt in body:
            codegen.gen_code_for_node(inner_stmt, code)
        code.add(('jmp', endif_label))
        code.add(('_label', else_label))

    for inner_stmt in node.else_body:
        codegen.gen_code_for_node(inner_stmt, code)
    code.add(('_label', endif_label))


@QvmCodeGen.generator_for(stmt.IfStmt)
def gen_if_stmt(node, code, codegen):
    else_label = codegen.get_label('else')
    endif_label = codegen.get_label('endif')

    codegen.gen_code_for_node(node.cond, code)
    code.add(('jz', else_label))
    for inner_stmt in node.then_stmts:
        codegen.gen_code_for_node(inner_stmt, code)
    code.add(('jmp', endif_label))
    code.add(('_label', else_label))
    if node.else_clause:
        for inner_stmt in node.else_clause.stmts:
            codegen.gen_code_for_node(inner_stmt, code)
    code.add(('_label', endif_label))


@QvmCodeGen.generator_for(stmt.InputStmt)
def gen_input(node, code, codegen):
    code.add_const(node.prompt.value)

    same_line = -1 if node.same_line else 0
    prompt_question = -1 if node.prompt_question else 0
    code.add(('push%', same_line))
    code.add(('push$', f'"{node.prompt.value}"'))
    code.add(('push%', prompt_question))

    code.add(('push%', len(node.var_list)))
    for var in node.var_list:
        code.add(('push%', var.type.type_id))
    code.add(('io', 'keyboard', 'input'))

    for var in node.var_list:
        # just so we won't forget updating here when arrays are
        # supported.
        assert not var.array_indices

        gen_lvalue_write(var, code, codegen)


@QvmCodeGen.generator_for(stmt.PrintStmt)
def gen_print_stmt(node, code, codegen):
    for item in node.items:
        if isinstance(item, expr.Expr):
            code.add(('push%', 0))
            codegen.gen_code_for_node(item, code)
        elif item == ';':
            code.add(('push%', 1))
        elif item == ',':
            code.add(('push%', 2))
        else:
            assert False
    code.add(('io', 'screen', 'print'))


@QvmCodeGen.generator_for(stmt.DataStmt)
def gen_data(node, code, codegen):
    code.add_data(node.items, codegen.last_label)


@QvmCodeGen.generator_for(stmt.ExitSubStmt)
def gen_exit_sub(node, code, codegen):
    code.add(('ret',))


@QvmCodeGen.generator_for(stmt.ExitFunctionStmt)
def gen_exit_sub(node, code, codegen):
    code.add(('ret',))


@QvmCodeGen.generator_for(stmt.SubBlock)
def gen_sub_block(node, code, codegen):
    code.add_routine(node.routine)

    code.add(('_label', '_sub_' + node.name))

    code.add(('frame',
              node.routine.params_size,
              node.routine.local_vars_size))

    for inner_stmt in node.block:
        codegen.gen_code_for_node(inner_stmt, code)
    code.add(('ret',))


@QvmCodeGen.generator_for(stmt.FunctionBlock)
def gen_func_block(node, code, codegen):
    code.add_routine(node.routine)

    code.add(('_label', '_func_' + node.name))

    code.add(('frame',
              node.routine.params_size,
              node.routine.local_vars_size))

    for inner_stmt in node.block:
        codegen.gen_code_for_node(inner_stmt, code)
    code.add(('ret',))


@QvmCodeGen.generator_for(stmt.TypeBlock)
def gen_type_block(node, code, codegen):
    # no code for type block
    pass
