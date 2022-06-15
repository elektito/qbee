import logging
import itertools
from enum import Enum
from .instrs import op_code_to_instr


logger = logging.getLogger(__name__)

QVM_DEVICES = {
    'terminal': {
        'id': 2,
        'ops': {
            'cls': 1,
            'print': 2,
            'color': 3,
            'view_print': 4,
            'set_mode': 5,
            'width': 6,
            'locate': 7,
            'input': 8,
            'inkey': 9,
        },
    },
    'pcspkr': {
        'id': 3,
        'ops': {
            'beep': 1,
            'play': 2,
        },
    },
    'time': {
        'id': 5,
        'ops': {
            'get_time': 1,
        }
    },
    'rng': {
        'id': 6,
        'ops': {
            'seed': 1,
            'rnd': 2,
        }
    },
    'memory': {
        'id': 7,
        'ops': {
            'poke': 1,
            'peek': 2,
            'set_segment': 3,
            'set_default_segment': 4,
        }
    },
    'data': {
        'id': 8,
        'ops': {
            'read': 1,
            'restore': 2,
        }
    }
}


def get_device_info_by_id(device_id):
    for device_name, device_info in QVM_DEVICES.items():
        if device_info['id'] == device_id:
            return device_info
    return None


def get_device_info_by_name(device_name):
    return QVM_DEVICES.get(device_name)


def get_device_name_by_id(device_id):
    for device_name, device_info in QVM_DEVICES.items():
        if device_info['id'] == device_id:
            return device_name
    return None


def get_device_op_name_by_id(device_name, op_code):
    for name, code in QVM_DEVICES[device_name]['ops'].items():
        if code == op_code:
            return name
    return None


class Reference:
    def __init__(self, segment, idx=None):
        if isinstance(segment, Reference):
            ref = segment
            self.segment = ref.segment
            self.index = ref.index
            return
        else:
            assert idx is not None
            self.segment = segment
            self.index = idx

    def __repr__(self):
        return f'<REF seg={self.segment} idx={self.index}>'


class CellType(Enum):
    INTEGER = 1
    LONG = 2
    SINGLE = 3
    DOUBLE = 4
    STRING = 5
    FIXED_STRING = 6
    REFERENCE = 7

    @property
    def py_type(self):
        return {
            CellType.INTEGER: int,
            CellType.LONG: int,
            CellType.SINGLE: float,
            CellType.DOUBLE: float,
            CellType.STRING: str,
            CellType.FIXED_STRING: str,
            CellType.REFERENCE: Reference,
        }[self]

    @property
    def is_numeric(self):
        return self in [
            CellType.INTEGER,
            CellType.LONG,
            CellType.SINGLE,
            CellType.DOUBLE,
        ]

    @property
    def is_integral(self):
        return self in [
            CellType.INTEGER,
            CellType.LONG,
        ]


class TrapCode(Enum):
    INVALID_OP_CODE = 1
    DEVICE_NOT_AVAILABLE = 2
    DEVICE_ERROR = 3
    STACK_EMPTY = 4
    INVALID_LOCAL_VAR_IDX = 5
    INVALID_GLOBAL_VAR_IDX = 5
    TYPE_MISMATCH = 7
    NULL_REFERENCE = 8
    INVALID_OPERAND_VALUE = 9


class CallFrame:
    def __init__(self, size, prev_frame):
        self.size = size
        self.local_vars = [None] * size
        self.prev_frame = prev_frame

    def set_local(self, idx, value):
        self.local_vars[idx] = value

    def get_local(self, idx):
        return self.local_vars[idx]

    def set_temp_reference(self, idx, value):
        # get a non reference value, create a temporary cell for it,
        # and then store a reference to it in the given index.
        assert value.type != CellType.REFERENCE
        self.local_vars.append(CellValue(value.type, value.value))
        ref = CellValue(CellType.REFERENCE,
                        Reference(segment=self,
                                  idx=len(self.local_vars) - 1))
        self.set_local(idx, ref)

    def destroy(self):
        logger.info('Destroying dynamic arrays not implemented')

    def __repr__(self):
        ntemps = len(self.local_vars) - self.size
        return f'<CallFrame size={self.size} temps={ntemps}>'


class CellValue:
    def __init__(self, type, value):
        assert isinstance(type, CellType) or type == 'ref'
        assert isinstance(value, type.py_type)
        self.type = type
        self.value = value

    def __repr__(self):
        value = self.value
        if self.type == CellType.STRING:
            value = f'"{self.value}"'
        return (
            f'<CellValue type={self.type.name.upper()} value={value}>'
        )

    def __str__(self):
        return f'{self.type.name.upper()}({self.value})'


class Trapped(Exception):
    def __init__(self, trap_code, trap_kwargs):
        self.trap_code = trap_code
        self.trap_kwargs = trap_kwargs


class QvmCpu:
    def __init__(self, module):
        self.module = module

        self.devices = {}
        self.device_by_id = {}

        self.halted = False
        self.pc = 0
        self.cur_frame = None
        self.stack = []
        self.global_vars = [
            None for _ in range(module.n_global_cells)
        ]

    def connect_device(self, device_name, device):
        logging.info('Connecting device: %s', device_name)

        device_info = get_device_info_by_name(device_name)
        if device_info is None:
            raise RuntimeError(
                f'Unknown device connected: {device_name}')

        self.devices[device_name] = device

        device_id = device_info['id']
        self.device_by_id[device_id] = device

    def read_var(self, scope, idx):
        if scope == 'local':
            try:
                return self.cur_frame.get_local(idx)
            except IndexError:
                self.trap(TrapCode.INVALID_LOCAL_VAR_IDX,
                          idx=idx)
        else:
            try:
                return self.global_vars[idx]
            except IndexError:
                self.trap(TrapCode.INVALID_GLOBAL_VAR_IDX,
                          idx=idx)

    def write_var(self, scope, idx, value):
        assert isinstance(value, CellValue)
        if scope == 'local':
            try:
                self.cur_frame.set_local(idx, value)
            except IndexError:
                self.trap(TrapCode.INVALID_LOCAL_VAR_IDX,
                          idx=idx)
        else:
            try:
                self.global_vars[idx] = value
            except IndexError:
                self.trap(TrapCode.INVALID_GLOBAL_VAR_IDX,
                          idx=idx)

    def run(self):
        while self.pc < len(self.module.code) and not self.halted:
            op_code = self.module.code[self.pc]
            self.pc += 1

            instr = op_code_to_instr.get(op_code)
            if instr is None:
                self._trap(TrapCode.INVALID_OP_CODE, op_code=op_code)
                continue

            operands = []
            for operand in instr.operands:
                operand = operand(
                    self.module.consts, self.module.data, {})
                bvalue = self.module.code[self.pc:self.pc+operand.size]
                value = operand.decode(bvalue)
                self.pc += operand.size
                operands.append(value)

            op_name = instr.op
            op_name = op_name.replace('%', '_integer')
            op_name = op_name.replace('&', '_long')
            op_name = op_name.replace('!', '_single')
            op_name = op_name.replace('#', '_double')
            op_name = op_name.replace('$', '_string')
            op_name = op_name.replace('@', '_reference')
            func = getattr(self, f'_exec_{op_name}', None)
            if func is None:
                print(f'No exec function for op: {instr.op} '
                      f'({instr.op_code})')
                exit(1)

            operands_list = ''
            if operands:
                operands_list = ' ' + \
                    ', '.join(str(i) for i in operands)
            logger.info(f'INSTR: {instr.op}{operands_list}')
            try:
                func(*operands)
            except Trapped as e:
                self._trap(e.trap_code, **e.trap_kwargs)

    def trap(self, code, **kwargs):
        raise Trapped(trap_code=code, trap_kwargs=kwargs)

    def _trap(self, code, **kwargs):
        logger.info('Received trap: %s', code)

        if code == TrapCode.INVALID_OP_CODE:
            op_code = kwargs['op_code']
            print('Invalid op code:', op_code)
        elif code == TrapCode.DEVICE_NOT_AVAILABLE:
            device_id = kwargs['device_id']
            device_name = kwargs['device_name']
            print(f'Device not available: {device_id} ({device_name})')
        elif code == TrapCode.DEVICE_ERROR:
            device_id = kwargs['device_id']
            device_name = get_device_name_by_id(device_id)
            error_code = kwargs['error_code']
            error_msg = kwargs['error_msg']
            print(f'Device Error: device_id={device_id} '
                  f'device_name={device_name} '
                  f'error_code={error_code} '
                  f'error_msg="{error_msg}"')
        elif code == TrapCode.STACK_EMPTY:
            print('Attempting to pop a value from an empty stack')
        elif code == TrapCode.INVALID_LOCAL_VAR_IDX:
            idx = kwargs['idx']
            print(f'Attempting to access invalid local var: {idx}')
        elif code == TrapCode.TYPE_MISMATCH:
            expected_type = kwargs['expected']
            if not isinstance(expected_type, str):
                expected_type = expected_type.name.upper()
            got_type = kwargs['got'].name.upper()
            print(f'Type mismatch: expected {expected_type}, '
                  f'got {got_type}')
        elif code == TrapCode.NULL_REFERENCE:
            scope = kwargs['scope']
            idx = kwargs['idx']
            print(f'Attempting to read a NULL reference at {scope} '
                  f'variable {idx}.')
        else:
            assert False

        self.halted = True

    def push(self, value_type, value):
        logger.info(
            'STACK: Pushing %s value %s to stack.',
            value_type.name.upper(),
            f'"{value}"' if isinstance(value, str) else value)
        boxed_value = CellValue(value_type, value)
        self.stack.append(boxed_value)

    def pop(self, expected_type=None):
        assert expected_type is None or \
            isinstance(expected_type, CellType)

        try:
            value = self.stack.pop()
            logger.info('STACK: Popped value %s from stack.', value)
            if expected_type is not None:
                if value.type != expected_type:
                    self.trap(TrapCode.TYPE_MISMATCH,
                              expected=expected_type,
                              got=value.type)
                value = expected_type.py_type(value.value)
            return value
        except IndexError:
            self.trap(TrapCode.STACK_EMPTY)

    def _exec_add(self):
        b = self.pop()
        a = self.pop()

        if a.type.is_numeric and not b.type.is_numeric:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected=a.type,
                      got=b.type)

        if a.type == CellType.STRING and b.type.is_numeric:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected=a.type,
                      got=b.type)

        if a.type != b.type:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected=a.type,
                      got=b.type)

        result = a.value + b.value
        self.push(a.type, result)

    def _exec_call(self, target):
        self.push(CellType.LONG, self.pc)
        self.pc = target

    def _exec_chr(self):
        char_code = self.pop(CellType.INTEGER)
        if char_code < 0 or char_code > 255:
            self.trap(TrapCode.INVALID_OPERAND_VALUE)

        char = bytes([char_code]).decode('cp437')
        self.push(CellType.STRING, char)

    def _exec_cmp(self):
        b = self.pop()
        a = self.pop()

        if a.type != b.type:
            self.trap(TrapCode.TYPE_MISMATCH,
                      a.type,
                      b.type)

        if a.value == b.value:
            result = 0
        elif a.value < b.value:
            result = -1
        else:
            result = 1
        self.push(CellType.INTEGER, result)

    def _exec_deref(self):
        ref = self.pop(CellType.REFERENCE)
        derefed = ref.segment.get_local(ref.index)
        logger.info(f'Derefed {ref} to {derefed}')
        self.push(derefed.type, derefed.value)

    def _exec_div(self):
        divisor = self.pop()
        dividend = self.pop()

        if not divisor.type.is_numeric:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='numeric',
                      got=divisor.type)

        if not dividend.type.is_numeric:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='numeric',
                      got=dividend.type)

        if divisor.type != dividend.type:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected=dividend.type,
                      got=divisor.type)

        result = dividend.value / divisor.value
        self.push(dividend.type, result)

    def _exec_dupl(self):
        value = self.stack.pop()
        self.push(value.type, value.value)
        self.push(value.type, value.value)

    def _exec_eq(self):
        value = self.pop(CellType.INTEGER)

        if value == 0:
            result = -1
        else:
            result = 0
        self.push(CellType.INTEGER, result)

    def _exec_frame(self, params_size, local_vars_size):
        logger.info(
            'Creating stack frame: params=%d locals=%d',
            params_size, local_vars_size)

        frame = CallFrame(
            size=params_size + local_vars_size,
            prev_frame=self.cur_frame,
        )
        self.cur_frame = frame

        # read return address
        ret_addr = self.pop(CellType.LONG)

        # copy parameters
        for i in range(params_size):
            # popping in reverse order
            idx = params_size - i - 1

            value = self.pop()
            if value.type == CellType.REFERENCE:
                frame.set_local(idx, value)
            else:
                frame.set_temp_reference(idx, value)

        # push back return address
        self.push(CellType.LONG, ret_addr)

    def _exec_ge(self):
        value = self.pop()

        if not value.type.is_numeric:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='numeric',
                      got=a.type)
        result = -1 if value.value >= 0 else 0
        self.push(CellType.INTEGER, result)

    def _exec_halt(self):
        self.halted = True

    def _exec_idiv(self):
        a = self.pop()
        b = self.pop()

        if not a.type.is_integral:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='numeric',
                      got=a.type)

        if not b.type.is_integral:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='numeric',
                      got=b.type)

        if a.type != b.type:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected=a.type,
                      got=b.type)

        result = a.value // b.value
        self.push(a.type, result)

    def _exec_ijmp(self):
        target = self.pop(CellType.LONG)
        self.pc = target

    def _exec_io(self, device_id, operation):
        device_name = get_device_name_by_id(device_id)
        op_name = get_device_op_name_by_id(device_name, operation)
        logger.info(
            'Performing IO: device_id=%d (%s) operation=%d (%s)',
            device_id,
            device_name or 'unknown',
            operation,
            op_name or 'unknown')

        device = self.device_by_id.get(device_id)
        if device is None:
            device_name = get_device_name_by_id(device_id) or 'unknown'
            self.trap(TrapCode.DEVICE_NOT_AVAILABLE,
                      device_id=device_id,
                      device_name=device_name)

        device_name = get_device_name_by_id(device_id)
        op_name = get_device_op_name_by_id(device_name, operation)
        device.execute(op_name)

    def _exec_jmp(self, target):
        self.pc = target

    def _exec_jz(self, target):
        value = self.pop(CellType.INTEGER)
        if value == 0:
            self.pc = target

    def _exec_le(self):
        value = self.pop()

        if not value.type.is_numeric:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='numeric',
                      got=a.type)
        result = -1 if value.value <= 0 else 0
        self.push(CellType.INTEGER, result)

    def _exec_mul(self):
        b = self.pop()
        a = self.pop()

        if not a.type.is_numeric:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='numeric',
                      got=a.type)

        if not b.type.is_numeric:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='numeric',
                      got=b.type)

        if a.type != b.type:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected=a.type,
                      got=b.type)

        result = a.value * b.value
        self.push(a.type, result)

    def _exec_ne(self):
        value = self.pop(CellType.INTEGER)

        if value == 0:
            result = 0
        else:
            result = -1
        self.push(CellType.INTEGER, result)

    def _exec_push_string(self, value):
        self.push(CellType.STRING, value)

    def _exec_ret(self):
        self.cur_frame.destroy()
        self.cur_frame = self.cur_frame.prev_frame
        ret_addr = self.pop(CellType.LONG)
        self.pc = ret_addr

    def _exec_readl_reference(self, idx):
        try:
            value = self.cur_frame.get_local(idx)
        except IndexError:
            self.trap(TrapCode.INVALID_LOCAL_VAR_IDX,
                      idx=idx)

        if value is None:
            self.trap(TrapCode.NULL_REFERENCE, scope='local', idx=idx)
        if value.type != CellType.REFERENCE:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected_type=CellType.REFERENCE,
                      got_type=value.type)
        self.push(CellType.REFERENCE, value.value)

    def _exec_sign(self):
        value = self.pop()
        if not value.type.is_numeric:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='numeric',
                      got=a.type)
        v = value.value
        sign = 1 if v > 0 else -1 if v < 0 else 0
        self.push(value.type, value.type.py_type(sign))

    def _exec_storel(self, idx):
        value = self.pop()
        try:
            self.cur_frame.set_local(idx, value)
        except IndexError:
            self.trap(TrapCode.INVALID_LOCAL_VAR_IDX,
                      idx=idx)

    def _exec_strlen(self):
        value = self.pop(CellType.STRING)
        self.push(CellType.LONG, len(value))

    def _exec_sub(self):
        b = self.pop()
        a = self.pop()

        if not a.type.is_numeric:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='numeric',
                      got=a.type)

        if not b.type.is_numeric:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='numeric',
                      got=b.type)

        if a.type != b.type:
            self.trap(TrapCode.TYPE_MISMATCH,
                      a.type,
                      b.type)

        result = a.value - b.value
        self.push(a.type, result)


# add exec methods for all variants of the push instruction
numeric_types = [
    CellType.INTEGER,
    CellType.LONG,
    CellType.SINGLE,
    CellType.DOUBLE
]
value_types = numeric_types + [CellType.STRING]
for _type in numeric_types:
    type_name = _type.name.lower()
    attr = f'_exec_push_{type_name}'
    def get_method(_type, name):
        def method(self, operand):
            self.push(_type, operand)
        method.__name__ = attr
        return method
    setattr(QvmCpu, attr, get_method(_type, attr))

    for const in [-2, -1, 0, 1, 2]:
        def get_method(_type, const, name):
            def method(self):
                self.push(_type, _type.py_type(const))
            method.__name__ = attr
            return method
        const_name = str(const) if const >=0 else f'm{abs(const)}'
        attr = f'_exec_push{const_name}_{type_name}'
        setattr(QvmCpu, attr, get_method(_type, const, attr))


# add exec methods for all variants of the conv instruction
for src, dst in itertools.product(numeric_types, numeric_types):
    if src == dst:
        continue
    def get_method(src, dst, name):
        conv_func = dst.py_type
        if src.py_type == float and dst.py_type == int:
            conv_func = lambda n: int(round(n))
        def method(self):
            value = self.pop(src)
            new_value = conv_func(value)
            self.push(dst, new_value)
            logger.info(f'Converted {src.name} {value} to '
                        f'{dst.name} {new_value}')
        method.__name__ = attr
        return method

    attr = f'_exec_conv_{src.name.lower()}_{dst.name.lower()}'
    setattr(QvmCpu, attr, get_method(src, dst, attr))


# add exec methods for all variants of the read instruction (except
# for the ones dealing with references)
for scope in ['local', 'global']:
    for _type in value_types:
        type_name = _type.name.lower()
        scope_char = 'l' if scope == 'local' else 'g'
        attr = f'_exec_read{scope_char}_{type_name}'
        def get_method(scope, _type, name):
            default_value = 0 if _type.is_numeric else ''
            def method(self, idx):
                value = self.read_var(scope, idx)
                if value is None:
                    value = CellValue(_type, default_value)
                    self.write_var(scope, idx, value)
                self.push(value.type, value.value)
            method.__name__ = attr
            return method
        setattr(QvmCpu, attr, get_method(scope, _type, attr))
