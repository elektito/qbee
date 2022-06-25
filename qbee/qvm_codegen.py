import struct
from enum import Enum, auto
from collections import defaultdict
from qvm.instrs import op_to_instr
from qvm.cpu import QVM_DEVICES
from qvm.debug_info import DebugInfoCollector
from qvm.memlayout import (
    get_type_size, get_local_var_idx, get_global_var_idx,
    get_params_size, get_local_vars_size, get_dotted_index,
)
from .codegen import BaseCodeGen, BaseCode
from .program import Label, LineNo, Program
from .exceptions import InternalError
from .evalctx import Routine
from . import stmt, expr


class CanonicalOp(Enum):
    "VM ops without scope, type, etc."

    ADD = auto()
    ALLOCARR = auto()
    AND = auto()
    ARRIDX = auto()
    ASC = auto()
    CALL = auto()
    CHR = auto()
    CMP = auto()
    CONV = auto()
    DEREF = auto()
    DIV = auto()
    DUPL = auto()
    EQ = auto()
    EQV = auto()
    EXP = auto()
    FRAME = auto()
    GE = auto()
    GT = auto()
    HALT = auto()
    IDIV = auto()
    IJMP = auto()
    IMP = auto()
    INITARRG = auto()
    INITARRL = auto()
    INT = auto()
    IO = auto()
    JMP = auto()
    JZ = auto()
    LCASE = auto()
    LE = auto()
    LT = auto()
    MOD = auto()
    MUL = auto()
    NE = auto()
    NEG = auto()
    NOP = auto()
    NOT = auto()
    NTOS = auto()
    OR = auto()
    POP = auto()
    PUSH = auto()
    PUSHREF = auto()
    READ = auto()
    READIDX = auto()
    REFIDX = auto()
    RET = auto()
    RETV = auto()
    RND = auto()
    SDBL = auto()
    SIGN = auto()
    SPACE = auto()
    SUB = auto()
    STORE = auto()
    STOREIDX = auto()
    STOREREF = auto()
    STRLEFT = auto()
    STRLEN = auto()
    STRMID = auto()
    STRRIGHT = auto()
    SWAP = auto()
    SWAPPREV = auto()
    UCASE = auto()
    XOR = auto()

    _LABEL = 10000
    _DBG_INFO_START = 10001
    _DBG_INFO_END = 10002
    _EMPTY_BLOCK = 10003

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

        self.type_char = ''
        if expr.Type.is_type_char(op[-1]) or op[-1] == '@':
            self.type_char = op[-1]
            op = op[:-1]

        self.src_type_char = ''
        if expr.Type.is_type_char(op[-1]):
            self.src_type_char = op[-1]
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

        if self.op == Op.PUSH and self.type_char == '$':
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

        args = [
            arg() if callable(arg) else arg
            for arg in args
        ]

        scope = ''
        if self.scope:
            scope = self.scope

        op = op + scope + self.src_type_char + self.type_char
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
        self._string_literals = []
        self._globals = None
        self._consts = {}
        self._debug_info_enabled = False
        self._source_code = None
        self._compilation = None

    def __repr__(self):
        return f'<QvmCode {self._instrs}>'

    def enable_debug_info(self, source_code, compilation):
        self._source_code = source_code
        self._compilation = compilation

    def add(self, *instrs):
        if not all(isinstance(i, tuple) for i in instrs):
            raise InternalError('Instruction not a tuple')
        self._instrs.extend([QvmInstr(*i) for i in instrs])

    def add_data(self, label, data: list):
        self._data[label].extend(data)

    def get_data_label_index(self, label):
        return list(self._data.keys()).index(label)

    def add_user_type(self, type_block):
        self._user_types[type_block.name] = type_block

    def add_routine(self, routine):
        assert isinstance(routine, Routine)
        self._routines[routine.name] = routine

    def add_string_literal(self, value):
        if value not in self._string_literals:
            self._string_literals.append(value)

    def add_const_expr(self, name, value):
        self._consts[name] = value

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
                cur.src_type_char == prev1.type_char
            ):
                arg, = prev1.args

                # Convert the argument to the dest type
                cur_type = expr.Type.from_type_char(cur.type_char)
                if cur_type.is_integral and \
                   isinstance(arg, float):
                    # perform rounding first if casting from float to
                    # integer
                    arg = round(arg)

                # Fold only if the value can fit in target type
                # (otherwise we'll leave it and there will be a
                # conversion error in run time)
                if cur_type.can_hold(arg):
                    cur_type = expr.Type.from_type_char(cur.type_char)
                    arg = cur_type.py_type(arg)

                    self._instrs[i-1] = QvmInstr(
                        f'push{cur.type_char}', arg)

                    del self._instrs[i]
                    i -= 1
                else:
                    i += 1

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
                prev1_type = expr.Type.from_type_char(prev1.type_char)
                value = expr.NumericLiteral(value, prev1_type)
                unary_expr = expr.UnaryOp(value, op)
                value = unary_expr.eval()

                self._instrs[i-1] = QvmInstr(
                    f'push{prev1.type_char}', value)
                del self._instrs[i]

                i -= 1

                continue

            # Fold push/push/binary-op
            if (prev1.op == prev2.op == Op.PUSH and
                prev1.type_char == prev2.type_char and
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
                prev1_type = expr.Type.from_type_char(prev1.type_char)
                prev2_type = expr.Type.from_type_char(prev2.type_char)
                left = expr.NumericLiteral(left, prev2_type)
                right = expr.NumericLiteral(right, prev1_type)
                binary_expr = expr.BinaryOp(left, right, op)
                value = binary_expr.eval()

                self._instrs[i-2] = QvmInstr(
                    f'push{prev1.type_char}', value)

                # remove the next two instructions
                del self._instrs[i]
                del self._instrs[i-1]
                i -= 2

                continue

            # Eliminate consecutive jump-like instructions
            jump_instrs = [
                Op.JMP,
                Op.IJMP,
                Op.RET,
                Op.RETV,
            ]
            if cur.op in jump_instrs and prev1.op in jump_instrs:
                del self._instrs[i]
                i -= 1
                continue

            # push/jz elimination
            if cur.op == Op.JZ and prev1.op == Op.PUSH:
                if prev1.type_char == '%':
                    if prev1.args[0] == 0:
                        # jump will always happen
                        del self._instrs[i-1]
                        self._instrs[i-1] = QvmInstr('jmp', cur.args[0])
                        i -= 1
                    else:
                        # jump will never happen
                        del self._instrs[i]
                        del self._instrs[i-1]
                        i -= 2

            # eliminate any instruction immediately after halt
            if prev1.op == Op.HALT and not cur.op.name.startswith('_'):
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

        if self._string_literals:
            s += '.literals\n'
            for i, string_literal in enumerate(self._string_literals):
                s += f'    {i} string "{string_literal}"\n'
            s += '\n;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;\n'

        if self._data:
            s += '.data\n\n'
            for label, data in self._data.items():
                s += f'{label}:\n'
                for item in data:
                    s += f'    {item}\n'
            s += '\n;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;\n'

        if self._globals:
            s += '.globals\n\n'
            for vname, vtype in self._globals.items():
                s += f'    {fmt_type(vtype)} {vname}\n'
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
            elif op.startswith('_dbg_'):
                pass
            elif op == '_empty_block':
                pass
            else:
                line = f'{op: <12}{", ".join(str(i) for i in args)}'
                s += f'    {line.strip()}\n'
        return s

    @property
    def assembled(self):
        def bconv(value, type_char):
            value_type = expr.Type.from_type_char(type_char)
            if type_char == '$':
                assert value.startswith('"') and value.endswith('"')
                value = get_string_literal_idx(value[1:-1])
            else:
                value = value_type.py_type(value)
            return {
                '%': lambda v: struct.pack('>h', v),
                '&': lambda v: struct.pack('>i', v),
                '!': lambda v: struct.pack('>f', v),
                '#': lambda v: struct.pack('>d', v),
                '$': lambda v: struct.pack('>H', v)
            }[type_char](value)

        def get_string_literal_idx(value):
            return self._string_literals.index(value)

        cur_offset = 0
        cur_routine = self._main_routine
        labels = {}
        patch_positions = {}
        code = bytearray()
        dbg_collector = DebugInfoCollector(self._source_code,
                                           self._compilation,
                                           self._consts)
        for instr in self._instrs:
            op, *args = instr.final
            if op == '_label':
                assert len(args) == 1
                name = args[0]
                labels[name] = cur_offset
                if name.startswith('_sub_'):
                    cur_routine = name[len('_sub_'):]
                    cur_routine = self._routines[cur_routine]
                elif name.startswith('_func_'):
                    cur_routine = name[len('_func_'):]
                    cur_routine = self._routines[cur_routine]
                continue

            if op == '_dbg_info_start':
                dbg_collector.start_node(args[0], cur_offset)
                continue

            if op == '_dbg_info_end':
                dbg_collector.end_node(args[0], cur_offset)
                continue

            if op == '_empty_block':
                dbg_collector.mark_empty_block(cur_offset)
                continue

            bargs = b''
            if op == 'allocarr':
                assert len(args) == 2
                n_indices = args[0]
                element_size = args[1]
                bargs = struct.pack('>Bi', n_indices, element_size)
            elif op == 'arridx':
                assert len(args) == 1
                n_indices = args[0]
                bargs = struct.pack('>B', n_indices)
            elif op == 'call':
                assert len(args) == 1
                label = args[0]
                patch_positions[cur_offset + 1] = label
                bargs = struct.pack('>I', 0)  # to be patched
            elif op == 'frame':
                assert len(args) == 2
                params_size, vars_size = args
                bargs = struct.pack('>HH', params_size, vars_size)
            elif op == 'initarrl':
                assert len(args) == 3
                var, n_dims, element_size = args
                var_idx = get_local_var_idx(cur_routine, var)
                bargs = struct.pack(
                    '>HBi', var_idx, n_dims, element_size)
            elif op == 'initarrg':
                assert len(args) == 3
                var, n_dims, element_size = args
                var = cur_routine.get_variable(var).full_name
                var_idx = get_global_var_idx(self.compilation, var)
                bargs = struct.pack(
                    '>HBi', var_idx, n_dims, element_size)
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
            elif op[:-1] == 'readl' or op == 'storel':
                assert len(args) == 1
                var = args[0]
                var_idx = get_local_var_idx(cur_routine, var)
                bargs = struct.pack('>H', var_idx)
            elif op[:-1] == 'readg' or op == 'storeg':
                assert len(args) == 1
                var = args[0]
                var = cur_routine.get_variable(var).full_name
                var_idx = get_global_var_idx(self.compilation, var)
                bargs = struct.pack('>H', var_idx)
            elif op[:-1] == 'readidxl' or op == 'storeidxl':
                assert len(args) == 2
                var, idx = args
                var_idx = get_local_var_idx(cur_routine, var)
                bargs = struct.pack('>HH', var_idx, idx)
            elif op[:-1] == 'readidxg' or op == 'storeidxg':
                assert len(args) == 2
                var, idx = args
                var = cur_routine.get_variable(var).full_name
                var_idx = get_global_var_idx(self.compilation, var)
                bargs = struct.pack('>HH', var_idx, idx)
            elif op == 'pushrefl':
                assert len(args) == 1
                var = args[0]
                var_idx = get_local_var_idx(cur_routine, var)
                bargs = struct.pack('>H', var_idx)
            elif op == 'pushrefg':
                assert len(args) == 1
                var = args[0]
                var = cur_routine.get_variable(var).full_name
                var_idx = get_global_var_idx(self.compilation, var)
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

        debug_info = None
        if self._debug_info_enabled:
            debug_info = dbg_collector.get_debug_info()

        return bytes(code), debug_info

    def __bytes__(self):
        sections = []

        literals_section = b''
        for literal in self._string_literals:
            literals_section += struct.pack('>H', len(literal))
            literals_section += literal.encode('cp437')
        sections.append((1, literals_section))

        data_section = struct.pack('>H', len(self._data))
        for data_part in self._data.values():
            data_section += struct.pack('>H', len(data_part))
            for data_item in data_part:
                data_section += struct.pack('>H', len(data_item))
                data_section += data_item.encode('cp437')
        sections.append((2, data_section))

        n_global_cells = sum(
            get_type_size(self.compilation, vtype)
            for _, vtype in self._globals.items()
        )
        global_section = struct.pack('>I', n_global_cells)
        sections.append((3, global_section))

        code_section, dbg_info = self.assembled
        sections.append((4, code_section))

        if self._debug_info_enabled:
            sections.append((5, dbg_info.serialize()))

        return b''.join(
            bytes([section_id]) +
            struct.pack('>I', len(section)) +
            section
            for section_id, section in sections
        )


class QvmCodeGen(BaseCodeGen, cg_name='qvm', code_class=QvmCode):
    def __init__(self, compilation, debug_info=False):
        super().__init__(debug_info=debug_info)
        self.compilation = compilation
        self.last_label = None
        self.label_counter = 1

    def set_source_code(self, source_code):
        self.source_code = source_code

    def get_label(self, name):
        label = f'_{name}_{self.label_counter}'
        self.label_counter += 1
        return label

    def init_code(self, code):
        code.compilation = self.compilation

        for user_type in self.compilation.user_types.values():
            code.add_user_type(user_type)

        for data_label, data_items in self.compilation.data.items():
            if data_label is None:
                data_label = '_toplevel_data'
            code.add_data(data_label, data_items)

        code._globals = self.compilation.global_vars
        for routine in self.compilation.routines.values():
            code._globals.update({
                routine.get_variable(svar).full_name: stype
                for svar, stype in routine.static_vars.items()
            })

        code._debug_info_enabled = self.debug_info_enabled
        if self.debug_info_enabled:
            code.enable_debug_info(self.source_code, self.compilation)


# Shared code


def gen_lvalue_write(node, code, codegen):
    # this function is called from any code generator that needs to
    # write to an lvalue. the value to write should already be on top
    # of the stack

    assert isinstance(node, expr.Lvalue)

    base_var = node.get_base_variable()
    if node.base_is_ref or base_var.type.is_array:
        gen_lvalue_ref(node, code, codegen)
        code.add(('storeref',))
        return

    idx = get_dotted_index(
        node.base_type, node.dotted_vars, codegen.compilation)

    if base_var.is_global:
        scope = 'g'  # global
    else:
        scope = 'l'  # local

    if idx == 0:
        code.add((f'store{scope}', base_var.full_name))
    else:
        code.add((f'storeidx{scope}', base_var.full_name, idx))


def gen_lvalue_ref(node, code, codegen):
    assert isinstance(node, expr.Lvalue)

    if node.array_indices:
        for i, aidx in enumerate(node.array_indices):
            codegen.gen_code_for_node(aidx, code)
            gen_code_for_conv(expr.Type.LONG, aidx, code, codegen)

    base_var = node.get_base_variable()
    if base_var.is_global:
        scope = 'g'  # global
    else:
        scope = 'l'  # local

    if node.base_is_ref:
        # already a reference; just read it onto stack
        code.add((f'read{scope}@', base_var.full_name))
    else:
        # push a reference to base onto the stack
        code.add((f'pushref{scope}', base_var.full_name))

    if node.array_indices:
        code.add(('arridx', len(node.array_indices)))

    idx = get_dotted_index(
        node.base_type, node.dotted_vars, codegen.compilation)
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


def gen_code_for_args(args, param_types, code, codegen):
    for arg, param_type in zip(args, param_types):
        if isinstance(arg, expr.Lvalue):
            gen_lvalue_ref(arg, code, codegen)
        elif isinstance(arg, expr.ArrayPass):
            codegen.gen_code_for_node(arg, code)
        else:
            codegen.gen_code_for_node(arg, code)
            gen_code_for_conv(param_type, arg, code, codegen)


def gen_code_for_block(node_list, code, codegen):
    assert isinstance(node_list, list)

    if not node_list:
        # add a dummy statement in the middle so we can differentiate
        # code from the start of block and code from the end of the
        # block when generating debug info.
        code.add(('_empty_block',))
        return

    for inner_stmt in node_list:
        codegen.gen_code_for_node(inner_stmt, code)


# Code generators for expressions


@QvmCodeGen.generator_for(Program)
def gen_program(node, code, codegen):
    main_routine = codegen.compilation.main_routine
    code._main_routine = main_routine
    code.add_routine(main_routine)

    code.add(('call', '_sub__main'),
             ('halt',))
    code.add(('_label', '_sub_' + main_routine.name))
    code.add(('frame',
              get_params_size(main_routine),
              lambda: get_local_vars_size(main_routine)))

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
    code.add_string_literal(node.value)
    code.add(('push$', f'"{node.value}"'))


@QvmCodeGen.generator_for(expr.NumericLiteral)
def gen_num_literal(node, code, codegen):
    code.add((f'push{node.type.type_char}', node.value))


@QvmCodeGen.generator_for(expr.ParenthesizedExpr)
def gen_paren(node, code, codegen):
    codegen.gen_code_for_node(node.child, code)


@QvmCodeGen.generator_for(expr.FuncCall)
def gen_func_call(node, code, codegen):
    routine = codegen.compilation.get_routine(node.name)
    gen_code_for_args(
        node.args, routine.params.values(), code, codegen)
    code.add(('call', '_func_' + node.name))


@QvmCodeGen.generator_for(expr.Lvalue)
def gen_lvalue(node, code, codegen):
    # notice that this function is only called for reading lvalues;
    # writing is performed in other code generators like those for
    # assignment, input, etc.

    if node.is_const:
        # this is a constant declared in a const statement
        if node.type == expr.Type.STRING:
            code.add(('push$', f'"{node.eval()}"'))
        else:
            code.add((f'push{node.type.type_char}', node.eval()))
        return

    base_var = node.get_base_variable()
    if base_var.is_global:
        scope = 'g'  # global
    else:
        scope = 'l'  # local

    if node.base_is_ref or base_var.type.is_array:
        type_char = node.type.type_char
        gen_lvalue_ref(node, code, codegen)
        code.add((f'deref{type_char}',))
    else:
        idx = get_dotted_index(
            node.base_type, node.dotted_vars, codegen.compilation)
        if idx == 0:
            type_char = node.type.type_char
            code.add((f'read{scope}{type_char}', base_var.full_name))
        else:
            type_char = node.type.type_char
            code.add((f'readidx{scope}{type_char}',
                      base_var.full_name,
                      idx))


@QvmCodeGen.generator_for(expr.ArrayPass)
def gen_array_pass(node, code, codegen):
    var = node.parent_routine.get_variable(node.identifier)
    assert var.type.is_array

    if var.is_global:
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
        if node.left.type.is_numeric and node.right.type.is_numeric:
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
        elif node.left.type == node.right.type == Type.STRING:
            left_type = right_type = Type.STRING
        else:
            assert False
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


@QvmCodeGen.generator_for(expr.BuiltinFuncCall)
def gen_builtin_func_call(node, code, codegen):
    if node.name == 'asc':
        codegen.gen_code_for_node(node.args[0], code)
        code.add(('asc',))
    elif node.name == 'chr$':
        codegen.gen_code_for_node(node.args[0], code)
        gen_code_for_conv(
            expr.Type.INTEGER, node.args[0], code, codegen)
        code.add(('chr',))
    elif node.name == 'inkey$':
        code.add(('io', 'terminal', 'inkey'))
    elif node.name == 'int':
        codegen.gen_code_for_node(node.args[0], code)
        code.add(('int',))
    elif node.name == 'lcase$':
        codegen.gen_code_for_node(node.args[0], code)
        code.add(('lcase',))
    elif node.name == 'left$':
        codegen.gen_code_for_node(node.args[0], code)
        codegen.gen_code_for_node(node.args[1], code)
        gen_code_for_conv(
            expr.Type.INTEGER, node.args[1], code, codegen)
        code.add(('strleft',))
    elif node.name == 'len':
        codegen.gen_code_for_node(node.args[0], code)
        code.add(('strlen',))
    elif node.name == 'mid$':
        codegen.gen_code_for_node(node.args[0], code)

        codegen.gen_code_for_node(node.args[1], code)
        gen_code_for_conv(
            expr.Type.INTEGER, node.args[1], code, codegen)

        if len(node.args) == 3:
            codegen.gen_code_for_node(node.args[2], code)
            gen_code_for_conv(
                expr.Type.INTEGER, node.args[2], code, codegen)
        else:
            code.add(('push%', -1))

        code.add(('strmid',))
    elif node.name == 'peek':
        codegen.gen_code_for_node(node.args[0], code)
        gen_code_for_conv(expr.Type.LONG, node.args[0], code, codegen)
        code.add(('io', 'memory', 'peek'))
    elif node.name == 'right$':
        codegen.gen_code_for_node(node.args[0], code)
        codegen.gen_code_for_node(node.args[1], code)
        gen_code_for_conv(
            expr.Type.INTEGER, node.args[1], code, codegen)
        code.add(('strright',))
    elif node.name == 'rnd':
        if len(node.args) == 1:
            codegen.gen_code_for_node(node.args[0], code)
            gen_code_for_conv(
                expr.Type.SINGLE, node.args[0], code, codegen)
        else:
            code.add(('push!', 1))
        code.add(('io', 'rng', 'rnd'))
    elif node.name == 'space$':
        codegen.gen_code_for_node(node.args[0], code)
        gen_code_for_conv(
            expr.Type.INTEGER, node.args[0], code, codegen)
        code.add(('space',))
    elif node.name == 'str$':
        codegen.gen_code_for_node(node.args[0], code)
        code.add(('ntos',))
    elif node.name == 'timer':
        code.add(('io', 'time', 'get_time'))
    elif node.name == 'val':
        codegen.gen_code_for_node(node.args[0], code)
        code.add(('sdbl',))
    elif node.name == 'ucase$':
        codegen.gen_code_for_node(node.args[0], code)
        code.add(('ucase',))
    else:
        assert False, f'Unknown builtin function: {node.name}'


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
    routine = codegen.compilation.routines[node.name]
    gen_code_for_args(
        node.args, routine.params.values(), code, codegen)
    code.add(('call', '_sub_' + node.name))


@QvmCodeGen.generator_for(stmt.ClsStmt)
def gen_cls(node, code, codegen):
    code.add(('io', 'terminal', 'cls'))


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

    code.add(('io', 'terminal', 'color'))


@QvmCodeGen.generator_for(stmt.ConstStmt)
def gen_const_stmt(node, code, codegen):
    # No code for const statements needed
    code.add_const_expr(node.name, node.value)


@QvmCodeGen.generator_for(stmt.DefSegStmt)
def gen_def_seg_stmt(node, code, codegen):
    if node.segment is None:
        code.add(('io', 'memory', 'set_default_segment'))
    else:
        codegen.gen_code_for_node(node.segment, code)
        gen_code_for_conv(expr.Type.LONG, node.segment, code, codegen)
        code.add(('io', 'memory', 'set_segment'))


@QvmCodeGen.generator_for(stmt.DeclareStmt)
def gen_declare(node, code, codegen):
    # no code for DECLARE statements
    pass


@QvmCodeGen.generator_for(stmt.DefTypeStmt)
def gen_def_type(node, code, codegen):
    # no code for DEF* statements
    pass


@QvmCodeGen.generator_for(stmt.DimStmt)
def gen_dim(node, code, codegen):
    for decl in node.children:
        if not decl.type.is_array:
            continue

        element_size = get_type_size(
            codegen.compilation, decl.type.array_base_type)
        if decl.var.is_global:
            scope = 'g'  # global
        else:
            scope = 'l'  # local

        for dim_range in decl.array_dims:
            codegen.gen_code_for_node(dim_range.lbound, code)
            gen_code_for_conv(
                expr.Type.LONG, dim_range.lbound, code, codegen)

            codegen.gen_code_for_node(dim_range.ubound, code)
            gen_code_for_conv(
                expr.Type.LONG, dim_range.ubound, code, codegen)

        if decl.array_dims_are_const:
            # static array
            code.add(
                (f'initarr{scope}',
                 decl.name, len(decl.array_dims), element_size),
            )
        else:
            # dynamic array
            code.add(('allocarr', len(decl.array_dims), element_size))
            code.add((f'store{scope}', decl.name))


@QvmCodeGen.generator_for(stmt.LoopBlock)
def gen_loop(node, code, codegen):
    do_label = codegen.get_label('do')
    loop_label = codegen.get_label('loop')

    code.add(('_label', do_label))
    if node.kind.startswith('do_'):
        codegen.gen_code_for_node(node.cond, code)
        gen_code_for_conv(expr.Type.INTEGER, node.cond, code, codegen)
        if node.kind == 'do_until':
            code.add(('not',))
        code.add(('jz', loop_label))

    gen_code_for_block(node.body, code, codegen)

    if node.kind.startswith('loop_'):
        codegen.gen_code_for_node(node.cond, code)
        if node.kind == 'loop_while':
            code.add(('not',))
        code.add(('jz', do_label))
    else:
        code.add(('jmp', do_label))
    code.add(('_label', loop_label))


@QvmCodeGen.generator_for(stmt.ForBlock)
def gen_for_block(node, code, codegen):
    var_type = node.var.type
    type_char = var_type.type_char
    base_var = node.var.get_base_variable()
    if base_var.is_global:
        scope = 'g'  # global
    else:
        scope = 'l'  # local

    init_label = codegen.get_label('for_init')
    check_label = codegen.get_label('for_check')
    body_label = codegen.get_label('for_body')
    next_label = codegen.get_label('for_next')
    end_label = codegen.get_label('for_end')

    # we use get_label to get unique names for our variables
    step_var = codegen.get_label('for_step')
    step_sign_var = codegen.get_label('for_step_sign')
    to_var = codegen.get_label('for_to')

    node.parent_routine.local_vars[step_var] = var_type
    node.parent_routine.local_vars[step_sign_var] = var_type
    node.parent_routine.local_vars[to_var] = var_type

    var = node.var.get_base_variable()

    code.add(('_label', init_label))
    if node.step_expr:
        codegen.gen_code_for_node(node.step_expr, code)
        gen_code_for_conv(var_type, node.step_expr, code, codegen)
        code.add(
            ('dupl',),
            ('storel', step_var),
            ('sign',),
            ('storel', step_sign_var),
        )
    else:
        code.add(
            (f'push1{type_char}', 1),
            ('storel', step_var),
            (f'push1{type_char}', 1),
            ('storel', step_sign_var),
        )
    codegen.gen_code_for_node(node.from_expr, code)
    gen_code_for_conv(var_type, node.from_expr, code, codegen)
    code.add((f'store{scope}', var.name))
    codegen.gen_code_for_node(node.to_expr, code)
    gen_code_for_conv(var_type, node.to_expr, code, codegen)
    code.add(('storel', to_var))

    # make sure the range is compatible with the step value (by
    # checking if (to - from) has the same sign as step value). if
    # not, skip the loop.
    code.add(
        (f'readl{type_char}', to_var),
        (f'read{scope}{type_char}', var.name),
        ('sub',),
        (f'readl{type_char}', step_sign_var),
        ('mul',),
        (f'push{type_char}', 0),
        ('cmp',),
        ('ge',),
        ('jz', end_label),
    )

    # multiply "to" value with the step sign so that we can always use
    # the same compare instruction
    code.add(
        (f'readl{type_char}', step_sign_var),
        (f'readl{type_char}', to_var),
        ('mul',),
        (f'storel', to_var),
    )

    code.add(('_label', check_label))
    code.add(
        (f'read{scope}{type_char}', var.name),
        (f'readl{type_char}', step_sign_var),
        ('mul',),
        (f'readl{type_char}', to_var),
        ('cmp',),
        ('le',),
        ('jz', end_label),
    )

    code.add(('_label', body_label))
    gen_code_for_block(node.body, code, codegen)

    code.add(
        ('_label', next_label),
        (f'read{scope}{type_char}', var.name),
        (f'readl{type_char}', step_var),
        ('add',),
        (f'store{scope}', var.name),
        ('jmp', check_label),
    )

    code.add(('_label', end_label))


@QvmCodeGen.generator_for(stmt.EndStmt)
def gen_end(node, code, codegen):
    code.add(('halt',))


@QvmCodeGen.generator_for(stmt.GosubStmt)
def gen_gosub(node, code, codegen):
    code.add(('call', node.canonical_target))


@QvmCodeGen.generator_for(stmt.ReturnStmt)
def gen_return(node, code, codegen):
    if node.target:
        code.add(
            # throw away the return address
            ('pop',),

            # jump to target
            ('jmp', node.canonical_target),
        )
    else:
        code.add(('ijmp',))


@QvmCodeGen.generator_for(stmt.GotoStmt)
def gen_goto(node, code, codegen):
    code.add(('jmp', node.canonical_target))


@QvmCodeGen.generator_for(stmt.IfBlock)
def gen_if_block(node, code, codegen):
    endif_label = codegen.get_label('endif')

    for cond, body in node.if_blocks:
        else_label = codegen.get_label('else')

        elseif_stmt = node.get_elseif_for_cond(cond)
        if elseif_stmt:
            code.add(('_dbg_info_start', elseif_stmt))

        codegen.gen_code_for_node(cond, code)
        gen_code_for_conv(expr.Type.INTEGER, cond, code, codegen)
        code.add(('jz', else_label))

        if elseif_stmt:
            code.add(('_dbg_info_end', elseif_stmt))

        gen_code_for_block(body, code, codegen)

        code.add(('jmp', endif_label))
        code.add(('_label', else_label))

    gen_code_for_block(node.else_body, code, codegen)
    code.add(('_label', endif_label))


@QvmCodeGen.generator_for(stmt.IfStmt)
def gen_if_stmt(node, code, codegen):
    else_label = codegen.get_label('else')
    endif_label = codegen.get_label('endif')

    codegen.gen_code_for_node(node.cond, code)
    gen_code_for_conv(expr.Type.INTEGER, node.cond, code, codegen)
    code.add(('jz', else_label))
    gen_code_for_block(node.then_stmts, code, codegen)
    code.add(('jmp', endif_label))
    code.add(('_label', else_label))
    if node.else_clause:
        gen_code_for_block(node.else_clause.stmts, code, codegen)
    code.add(('_label', endif_label))


@QvmCodeGen.generator_for(stmt.InputStmt)
def gen_input(node, code, codegen):
    code.add_string_literal(node.prompt.value)

    same_line = -1 if node.same_line else 0
    prompt_question = -1 if node.prompt_question else 0
    code.add(('push%', same_line))
    code.add(('push$', f'"{node.prompt.value}"'))
    code.add(('push%', prompt_question))

    for var in node.var_list:
        code.add(('push%', var.type.type_id))
    code.add(('push%', len(node.var_list)))
    code.add(('io', 'terminal', 'input'))

    for var in node.var_list:
        gen_lvalue_write(var, code, codegen)


@QvmCodeGen.generator_for(stmt.LocateStmt)
def gen_locate_stmt(node, code, codegen):
    codegen.gen_code_for_node(node.row, code)
    gen_code_for_conv(expr.Type.INTEGER, node.row, code, codegen)

    codegen.gen_code_for_node(node.col, code)
    gen_code_for_conv(expr.Type.INTEGER, node.col, code, codegen)

    code.add(
        ('push%', -1),
        ('push%', -1),
        ('push%', -1),
    )

    code.add(('io', 'terminal', 'locate'))


@QvmCodeGen.generator_for(stmt.PlayStmt)
def gen_play_stmt(node, code, codegen):
    codegen.gen_code_for_node(node.command_string, code)
    code.add(('io', 'pcspkr', 'play'))


@QvmCodeGen.generator_for(stmt.PokeStmt)
def gen_poke_stmt(node, code, codegen):
    codegen.gen_code_for_node(node.address, code)
    gen_code_for_conv(expr.Type.LONG, node.address, code, codegen)

    codegen.gen_code_for_node(node.value, code)
    gen_code_for_conv(expr.Type.INTEGER, node.value, code, codegen)

    code.add(('io', 'memory', 'poke'))


@QvmCodeGen.generator_for(stmt.PrintStmt)
def gen_print_stmt(node, code, codegen):
    nargs = 0
    if node.format_string:
        code.add(('push%', 3))
        codegen.gen_code_for_node(node.format_string, code)
        nargs += 2
    for item in node.items:
        if isinstance(item, expr.Expr):
            code.add(('push%', 0))
            codegen.gen_code_for_node(item, code)
            nargs += 2
        elif isinstance(item, stmt.PrintSep):
            if item.sep == ';':
                code.add(('push%', 1))
            elif item.sep == ',':
                code.add(('push%', 2))
            else:
                assert False

            nargs += 1
        else:
            assert False
    code.add(('push%', nargs))
    code.add(('io', 'terminal', 'print'))


@QvmCodeGen.generator_for(stmt.RandomizeStmt)
def gen_randomize(node, code, codegen):
    codegen.gen_code_for_node(node.seed, code)
    gen_code_for_conv(expr.Type.SINGLE, node.seed, code, codegen)
    code.add(('io', 'rng', 'seed'))


@QvmCodeGen.generator_for(stmt.ReadStmt)
def gen_read_stmt(node, code, codegen):
    for var in node.var_list:
        code.add(('push%', var.type.type_id))
        code.add(('io', 'data', 'read'))
        gen_lvalue_write(var, code, codegen)


@QvmCodeGen.generator_for(stmt.RestoreStmt)
def gen_restore_stmt(node, code, codegen):
    target = node.canonical_target
    if target is None:
        target = ''

    if target:
        label_index = code.get_data_label_index(target)
    else:
        label_index = -1

    code.add(
        ('push%', label_index),
        ('io', 'data', 'restore'),
    )


@QvmCodeGen.generator_for(stmt.ScreenStmt)
def gen_screen_stmt(node, code, codegen):
    codegen.gen_code_for_node(node.mode, code)
    gen_code_for_conv(expr.Type.INTEGER, node.mode, code, codegen)
    if node.color_switch:
        codegen.gen_code_for_node(node.color_switch, code)
        gen_code_for_conv(
            expr.Type.INTEGER, node.color_switch, code, codegen)
    else:
        code.add(('push%', -1))
    if node.apage:
        codegen.gen_code_for_node(node.apage, code)
        gen_code_for_conv(
            expr.Type.INTEGER, node.apage, code, codegen)
    else:
        code.add(('push%', -1))
    if node.vpage:
        codegen.gen_code_for_node(node.vpage, code)
        gen_code_for_conv(
            expr.Type.INTEGER, node.vpage, code, codegen)
    else:
        code.add(('push%', -1))
    code.add(('io', 'terminal', 'set_mode'))


@QvmCodeGen.generator_for(stmt.WidthStmt)
def gen_width_stmt(node, code, codegen):
    if node.columns:
        codegen.gen_code_for_node(node.columns, code)
        gen_code_for_conv(
            expr.Type.INTEGER, node.columns, code, codegen)
    else:
        code.add(('push%', -1))

    if node.lines:
        codegen.gen_code_for_node(node.lines, code)
        gen_code_for_conv(
            expr.Type.INTEGER, node.lines, code, codegen)
    else:
        code.add(('push%', -1))

    code.add(('io', 'terminal', 'width'))


@QvmCodeGen.generator_for(stmt.DataStmt)
def gen_data(node, code, codegen):
    # No code needs to be generated for DATA
    pass


@QvmCodeGen.generator_for(stmt.ExitSubStmt)
def gen_exit_sub(node, code, codegen):
    code.add(('ret',))


@QvmCodeGen.generator_for(stmt.ExitFunctionStmt)
def gen_exit_function(node, code, codegen):
    code.add(('ret',))


@QvmCodeGen.generator_for(stmt.SubBlock)
def gen_sub_block(node, code, codegen):
    code.add_routine(node.routine)

    code.add(('_label', '_sub_' + node.name))

    code.add(('frame',
              get_params_size(node.routine),
              lambda: get_local_vars_size(node.routine)))

    gen_code_for_block(node.block, code, codegen)
    code.add(('ret',))


@QvmCodeGen.generator_for(stmt.FunctionBlock)
def gen_func_block(node, code, codegen):
    code.add_routine(node.routine)

    code.add(('_label', '_func_' + node.name))

    code.add(('frame',
              get_params_size(node.routine),
              lambda: get_local_vars_size(node.routine)))

    gen_code_for_block(node.block, code, codegen)

    type_char = node.routine.return_type.type_char
    code.add((f'readl{type_char}', '_retval'))
    code.add(('retv',))


@QvmCodeGen.generator_for(stmt.ReturnValueSetStmt)
def gen_ret_value(node, code, codegen):
    codegen.gen_code_for_node(node.value, code)
    code.add(('storel', '_retval'))


@QvmCodeGen.generator_for(stmt.TypeBlock)
def gen_type_block(node, code, codegen):
    # no code for type block
    pass


@QvmCodeGen.generator_for(stmt.ViewPrintStmt)
def gen_view_print(node, code, codegen):
    if node.top_expr:
        codegen.gen_code_for_node(node.top_expr, code)
        gen_code_for_conv(
            expr.Type.INTEGER, node.top_expr, code, codegen)

        codegen.gen_code_for_node(node.bottom_expr, code)
        gen_code_for_conv(
            expr.Type.INTEGER, node.bottom_expr, code, codegen)
    else:
        code.add(('pushm1%',))
        code.add(('pushm1%',))

    code.add(('io', 'terminal', 'view_print'))


@QvmCodeGen.generator_for(stmt.WhileBlock)
def gen_while_block(node, code, codegen):
    check_label = codegen.get_label('while_check')
    body_label = codegen.get_label('while_body')
    wend_label = codegen.get_label('wend')

    code.add(('_label', check_label))
    codegen.gen_code_for_node(node.cond, code)
    gen_code_for_conv(expr.Type.INTEGER, node.cond, code, codegen)
    code.add(('jz', wend_label))

    code.add(('_label', body_label))
    for child_stmt in node.body:
        codegen.gen_code_for_node(child_stmt, code)
    code.add(('jmp', check_label))

    code.add(('_label', wend_label))


@QvmCodeGen.generator_for(stmt.SelectBlock)
def gen_select_block(node, code, codegen):
    start_label = codegen.get_label('select_start')
    end_label = codegen.get_label('select_end')

    code.add(('_label', start_label))
    codegen.gen_code_for_node(node.value, code)

    cur_case_label = codegen.get_label('case')
    next_case_label = codegen.get_label('case')
    if node.case_blocks:
        last_case = node.case_blocks[-1][0]
    for case, body in node.case_blocks:
        if case == last_case:
            next_case_label = end_label
        code.add(('_label', cur_case_label))
        codegen.gen_code_for_node(case, code)
        code.add(('jz', next_case_label))

        code.add(('_label', codegen.get_label('case_body')))
        for child_stmt in body:
            codegen.gen_code_for_node(child_stmt, code)
        code.add(('jmp', end_label))

        cur_case_label = next_case_label
        next_case_label = codegen.get_label('case')

    code.add(
        ('_label', end_label),
        ('pop',),
    )


@QvmCodeGen.generator_for(stmt.CaseStmt)
def gen_case_stmt(node, code, codegen):
    codegen.gen_code_for_node(node.cases[0], code)
    for case in node.cases[1:]:
        code.add(('swap',))
        codegen.gen_code_for_node(case, code)
        code.add(('swapprev',))
        code.add(('or',))


@QvmCodeGen.generator_for(stmt.CaseElseStmt)
def gen_case_else_stmt(node, code, codegen):
    code.add(('push%', -1))


@QvmCodeGen.generator_for(stmt.SimpleCaseClause)
def gen_simple_case_clause(node, code, codegen):
    value_type = node.parent.parent.value.type

    code.add(('dupl',))
    codegen.gen_code_for_node(node.value, code)
    gen_code_for_conv(value_type, node.value, code, codegen)
    code.add(('cmp',), ('eq',))


@QvmCodeGen.generator_for(stmt.CompareCaseClause)
def gen_compare_case_clause(node, code, codegen):
    value_type = node.parent.parent.value.type

    code.add(('dupl',))
    codegen.gen_code_for_node(node.value, code)
    gen_code_for_conv(value_type, node.value, code, codegen)

    op = {
        expr.Operator.CMP_EQ: 'eq',
        expr.Operator.CMP_NE: 'ne',
        expr.Operator.CMP_LT: 'lt',
        expr.Operator.CMP_GT: 'gt',
        expr.Operator.CMP_LE: 'le',
        expr.Operator.CMP_GE: 'ge',
    }[node.op]
    code.add((op,))


@QvmCodeGen.generator_for(stmt.RangeCaseClause)
def gen_range_case_clause(node, code, codegen):
    value_type = node.parent.parent.value.type

    code.add(('dupl',))
    code.add(('dupl',))
    codegen.gen_code_for_node(node.from_value, code)
    gen_code_for_conv(value_type, node.from_value, code, codegen)
    code.add(('ge',))
    code.add(('swap',))
    codegen.gen_code_for_node(node.to_value, code)
    gen_code_for_conv(value_type, node.to_value, code, codegen)
    code.add(('le',))
    code.add(('and',))
