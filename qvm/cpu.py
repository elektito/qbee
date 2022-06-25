import logging
import itertools
import math
from enum import Enum
from qbee import grammar
from pyparsing.exceptions import ParseException
from .instrs import op_code_to_instr
from .utils import format_number
from .cell import CellType, CellValue, Reference
from .trap import TrapCode, Trapped


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


class MemorySegment:
    def __init__(self, size):
        self.size = size
        self.cells = [None] * size

    def get_cell(self, idx):
        return self.cells[idx]

    def set_cell(self, idx, value):
        assert value is None or isinstance(value, CellValue)
        self.cells[idx] = value

    def get_cell_ref(self, idx):
        return Reference(segment=self, index=idx)

    def append_local(self, value):
        assert isinstance(value, CellValue)
        self.cells.append(value)
        self.size += 1

    def __repr__(self):
        return f'<MemorySegment size={self.size}>'


class CallFrame(MemorySegment):
    def __init__(self, size, prev_frame, code_start, ret_addr):
        super().__init__(size)
        self.original_size = size
        self.prev_frame = prev_frame

        # this should point to the first instruction after the FRAME
        # instruction and can be used by debuggers to identify frames
        self.code_start = code_start

        # also for debugging purposes
        self.ret_addr = ret_addr

    def set_temp_reference(self, idx, value):
        # get a non reference value, create a temporary cell for it,
        # and then store a reference to it in the given index.
        assert value.type != CellType.REFERENCE
        self.append_local(CellValue(value.type, value.value))
        ref = CellValue(CellType.REFERENCE,
                        Reference(segment=self,
                                  index=len(self.cells) - 1))
        self.set_cell(idx, ref)

    def destroy(self):
        # dynamic arrays will be automatically destroyed by python GC
        pass

    def __repr__(self):
        ntemps = self.size - self.original_size
        return f'<CallFrame size={self.size} temps={ntemps}>'

    @property
    def caller_addr(self):
        # we should return the address of the instruction right before
        # the return address. the size of a call instruction is five
        # bytes, so we subtract five.
        return self.ret_addr - 5


class Array(MemorySegment):
    def __init__(self, element_size, bounds):
        assert isinstance(element_size, int)
        assert isinstance(bounds, list)
        assert all(
            isinstance(lb, int) and isinstance(ub, int)
            for lb, ub in bounds
        )

        self.element_size = element_size
        self.bounds = bounds

        header = [
            None,  # reserved
            CellValue(CellType.LONG, len(bounds)),
            CellValue(CellType.LONG, element_size),
        ]

        size = 1
        for lbound, ubound in bounds:
            size *= (ubound - lbound + 1) * element_size
            header.extend([
                CellValue(CellType.LONG, lbound),
                CellValue(CellType.LONG, ubound),
            ])

        super().__init__(len(header) + size)

        for i, value in enumerate(header):
            self.set_cell(i, value)

    def __repr__(self):
        return f'<Array dims={len(self.bounds)} size={self.size}>'


class QvmCpu:
    def __init__(self, module):
        self.module = module

        self.devices = {}
        self.device_by_id = {}

        self.globals_segment = MemorySegment(self.module.n_global_cells)
        self.halted = False
        self.pc = 0
        self.cur_frame = None
        self.stack = []
        self.breakpoints = []
        self.last_breakpoint = None

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
            segment = self.cur_frame
            trap_code = TrapCode.INVALID_LOCAL_VAR_IDX
        else:
            segment = self.globals_segment
            trap_code = TrapCode.INVALID_GLOBAL_VAR_IDX

        try:
            return segment.get_cell(idx)
        except IndexError:
            self.trap(trap_code, idx=idx)

    def write_var(self, scope, idx, value):
        assert isinstance(value, CellValue)

        if scope == 'local':
            segment = self.cur_frame
            trap_code = TrapCode.INVALID_LOCAL_VAR_IDX
        else:
            segment = self.globals_segment
            trap_code = TrapCode.INVALID_GLOBAL_VAR_IDX

        try:
            segment.set_cell(idx, value)
        except IndexError:
            self.trap(trap_code, idx=idx)

    def run(self):
        """Run the cpu until either halted, or a breakpoint is hit.
        Returns a boolean, indicating the reason execution stopped.
        If stopped due to a breakpoint, False is returned, otherwise
        True is returned."""

        self.last_breakpoint = None
        self.halted = False
        while self.pc < len(self.module.code) and not self.halted:
            self.tick()
            for bp in self.breakpoints:
                if bp(self):
                    self.last_breakpoint = bp
                    return False

        return True

    def tick(self):
        instr_addr = self.pc
        instr, operands, size = self.get_current_instruction()
        self.pc += size
        if instr is None:
            return

        op_name = instr.op
        op_name = op_name.replace('%', '_integer')
        op_name = op_name.replace('&', '_long')
        op_name = op_name.replace('!', '_single')
        op_name = op_name.replace('#', '_double')
        op_name = op_name.replace('$', '_string')
        op_name = op_name.replace('@', '_reference')
        func = getattr(self, f'_exec_{op_name}', None)
        assert func is not None, \
            f'No exec function for op: {instr.op} ({instr.op_code})'

        operands_list = ''
        if operands:
            operands_list = ' ' + \
                ', '.join(str(i) for i in operands)
        logger.info(
            f'INSTR: {instr_addr:08x}: {instr.op}{operands_list}')
        try:
            func(*operands)
        except Trapped as e:
            self._trap(e.trap_code, **e.trap_kwargs)

        if self.pc >= len(self.module.code):
            logger.info(
                'Halting because execution reached end of module')
            self.halted = True

    def next(self):
        """Similar to tick, except if current instruction is 'call',
        execution continues until the instruction after 'call', or if
        the machine is halted or a breakpoint is hit.

        Return value has the same meaning as the 'run' method."""

        instr, operands, size = self.get_current_instruction()
        if instr.op == 'call':
            prev_pc = self.pc
            bp = lambda cpu: (cpu.pc == prev_pc + size)
            self.add_breakpoint(bp)
            try:
                ret = self.run()
                if self.last_breakpoint == bp:
                    return False
                else:
                    return ret
            finally:
                self.del_breakpoint(bp)
        else:
            self.tick()
            return True

    def add_breakpoint(self, bp):
        self.breakpoints.append(bp)

    def del_breakpoint(self, bp):
        self.breakpoints.remove(bp)

    def get_current_instruction(self):
        return self.get_instruction_at(self.pc)

    def get_instruction_at(self, addr):
        idx = addr
        op_code = self.module.code[idx]
        idx += 1

        instr = op_code_to_instr.get(op_code)
        if instr is None:
            self._trap(TrapCode.INVALID_OP_CODE, op_code=op_code)
            return None, [], 1

        operands = []
        for operand in instr.operands:
            operand = operand(
                self.module.literals, self.module.data, {})
            bvalue = self.module.code[idx:idx+operand.size]
            value = operand.decode(bvalue)
            idx += operand.size
            operands.append(value)

        return instr, operands, idx - addr

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
        elif code == TrapCode.INVALID_OPERAND_VALUE:
            desc = kwargs.get('desc')
            desc = f': {desc}' if desc else ''
            print(f'Invalid operand value{desc}')
        elif code == TrapCode.INVALID_CELL_VALUE:
            cell_type = kwargs.get('type')
            value = kwargs.get('value')
            print(f'A cell of type {cell_type} cannot hold: {value}')
        elif code == TrapCode.INDEX_OUT_OF_RANGE:
            idx = kwargs.get('idx')
            lbound = kwargs.get('lbound')
            ubound = kwargs.get('ubound')
            print(f'Index {idx} not in range {lbound} to {ubound}')
        elif code == TrapCode.INVALID_DIMENSIONS:
            expected = kwargs.get('expected')
            got = kwargs.get('got')
            print(f'Number of indices ({got}) does not match array '
                  f'dimensions ({expected})')
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

    def _exec_allocarr(self, n_dims, element_size):
        bounds = []
        for i in range(n_dims):
            ubound = self.pop(CellType.LONG)
            lbound = self.pop(CellType.LONG)
            bounds.append((lbound, ubound))

        bounds.reverse()

        array = Array(element_size, bounds)
        ref = Reference(segment=array, index=0)
        self.push(CellType.REFERENCE, ref)

    def _exec_and(self):
        b = self.pop()
        a = self.pop()

        if not a.type.is_integral:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='integral',
                      got=a.type)

        if not b.type.is_integral:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='integral',
                      got=b.type)

        if a.type != b.type:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected=a.type,
                      got=b.type)

        result = a.value & b.value
        self.push(a.type, result)

    def _exec_arridx(self, n_indices):
        array_ref = self.pop(CellType.REFERENCE)
        indices = []
        for i in range(n_indices):
            idx = self.pop(CellType.LONG)
            indices.append(idx)

        base_idx = array_ref.index

        array_n_dims = array_ref.segment.get_cell(base_idx + 1).value
        if array_n_dims != n_indices:
            self.trap(TrapCode.INVALID_DIMENSIONS,
                      expected=array_n_dims,
                      got=n_indices)

        element_size = array_ref.segment.get_cell(base_idx + 2).value

        base_idx += 3
        dim_sizes = []
        bounds = []
        for idx in reversed(indices):
            lbound = array_ref.segment.get_cell(base_idx + 0).value
            ubound = array_ref.segment.get_cell(base_idx + 1).value
            if idx < lbound or idx > ubound:
                self.trap(TrapCode.INDEX_OUT_OF_RANGE,
                          idx=idx,
                          lbound=lbound,
                          ubound=ubound)
            dim_sizes.append((ubound - lbound + 1))
            bounds.append((lbound, ubound))
            base_idx += 2

        idx = base_idx
        for n, cur_idx in enumerate(reversed(indices)):
            lbound, ubound = bounds[n]
            def mul(ls):
                r = 1
                for e in ls:
                    r *= e
                return r
            prev_dims_size = mul(dim_sizes[n+1:]) * element_size
            idx += prev_dims_size * (cur_idx - lbound)

        ref = array_ref.segment.get_cell_ref(idx)
        self.push(CellType.REFERENCE, ref)

    def _exec_asc(self):
        value = self.pop(CellType.STRING)
        if len(value) == 0:
            self.trap(TrapCode.INVALID_OPERAND_VALUE,
                      desc='ASC does not accept empty strings')
        self.push(CellType.INTEGER, ord(value[0]))

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

        # read return address
        ret_addr = self.pop(CellType.LONG)

        frame = CallFrame(
            size=params_size + local_vars_size,
            prev_frame=self.cur_frame,
            code_start=self.pc,
            ret_addr=ret_addr,
        )
        self.cur_frame = frame

        # copy parameters
        for i in range(params_size):
            # popping in reverse order
            idx = params_size - i - 1

            value = self.pop()
            if value.type == CellType.REFERENCE:
                frame.set_cell(idx, value)
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

    def _exec_gt(self):
        value = self.pop()

        if not value.type.is_numeric:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='numeric',
                      got=a.type)
        result = -1 if value.value > 0 else 0
        self.push(CellType.INTEGER, result)

    def _exec_halt(self):
        self.halted = True

    def _exec_idiv(self):
        a = self.pop()
        b = self.pop()

        if not a.type.is_integral:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='integral',
                      got=a.type)

        if not b.type.is_integral:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='integral',
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

    def _exec_initarrl(self, idx, n_dims, element_size):
        bounds = []
        for i in range(n_dims):
            ubound = self.pop(CellType.LONG)
            lbound = self.pop(CellType.LONG)
            bounds.append((lbound, ubound))

        bounds.reverse()

        self.cur_frame.set_cell(
            idx + 1, CellValue(CellType.LONG, n_dims))
        self.cur_frame.set_cell(
            idx + 2, CellValue(CellType.LONG, element_size))
        for i, (lbound, ubound) in enumerate(bounds):
            self.cur_frame.set_cell(idx + 3 + 2 * i + 0,
                                     CellValue(CellType.LONG, lbound))
            self.cur_frame.set_cell(idx + 3 + 2 * i + 1,
                                     CellValue(CellType.LONG, ubound))

    def _exec_initarrg(self, idx, n_dims, element_size):
        bounds = []
        for i in range(n_dims):
            ubound = self.pop(CellType.LONG)
            lbound = self.pop(CellType.LONG)
            bounds.append((lbound, ubound))

        bounds.reverse()

        self.globals_segment.set_cell(
            idx + 1, CellValue(CellType.LONG, n_dims))
        self.globals_segment.set_cell(
            idx + 2, CellValue(CellType.LONG, element_size))
        for i, (lbound, ubound) in enumerate(bounds):
            self.globals_segment.set_cell(
                idx + 3 + 2 * i + 0,
                CellValue(CellType.LONG, lbound))
            self.globals_segment.set_cell(
                idx + 3 + 2 * i + 1,
                CellValue(CellType.LONG, ubound))

    def _exec_int(self):
        value = self.pop()

        if not value.type.is_numeric:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='numeric',
                      got=value.type)

        int_value = math.floor(value.value)
        self.push(CellType.LONG, int_value)

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

    def _exec_lcase(self):
        s = self.pop(CellType.STRING)
        self.push(CellType.STRING, s.lower())

    def _exec_le(self):
        value = self.pop()

        if not value.type.is_numeric:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='numeric',
                      got=a.type)
        result = -1 if value.value <= 0 else 0
        self.push(CellType.INTEGER, result)

    def _exec_lt(self):
        value = self.pop()

        if not value.type.is_numeric:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='numeric',
                      got=a.type)
        result = -1 if value.value < 0 else 0
        self.push(CellType.INTEGER, result)

    def _exec_mod(self):
        b = self.pop()
        a = self.pop()

        if a.type not in (CellType.INTEGER, CellType.LONG):
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='integral',
                      got=a.type)

        if a.type != b.type:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected=a.type,
                      got=b.type)

        result = a.value % b.value
        self.push(a.type, result)

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

    def _exec_neg(self):
        value = self.pop()
        if not value.type.is_numeric:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='numeric',
                      got=value.type)

        self.push(value.type, -value.value)

    def _exec_not(self):
        value = self.pop()
        if not value.type.is_integral:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='integral',
                      got=value.type)

        self.push(value.type, ~value.value)

    def _exec_ntos(self):
        value = self.pop()
        if not value.type.is_numeric:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='numeric',
                      got=value.type)

        self.push(CellType.STRING,
                  format_number(value.value, value.type))

    def _exec_or(self):
        b = self.pop()
        a = self.pop()

        if not a.type.is_integral:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='integral',
                      got=a.type)

        if not b.type.is_integral:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='integral',
                      got=b.type)

        if a.type != b.type:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected=a.type,
                      got=b.type)

        result = a.value | b.value
        self.push(a.type, result)

    def _exec_pop(self):
        self.pop()

    def _exec_push_string(self, value):
        self.push(CellType.STRING, value)

    def _exec_pushrefg(self, idx):
        ref = Reference(segment=self.globals_segment, index=idx)
        self.push(CellType.REFERENCE, ref)

    def _exec_pushrefl(self, idx):
        ref = self.cur_frame.get_cell_ref(idx)
        self.push(CellType.REFERENCE, ref)

    def _exec_ret(self):
        self.cur_frame.destroy()
        self.cur_frame = self.cur_frame.prev_frame
        ret_addr = self.pop(CellType.LONG)
        self.pc = ret_addr

    def _exec_retv(self):
        self.cur_frame.destroy()
        self.cur_frame = self.cur_frame.prev_frame

        retval = self.pop()
        if retval.type == CellType.REFERENCE:
            retval = retval.value.derefed()

        ret_addr = self.pop(CellType.LONG)
        self.pc = ret_addr

        self.push(retval.type, retval.value)

    def _exec_readl_reference(self, idx):
        try:
            value = self.cur_frame.get_cell(idx)
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

    def _exec_refidx(self):
        idx = self.pop()
        ref = self.pop(CellType.REFERENCE)

        if not idx.type.is_integral:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected_type='integral',
                      got_type=idx.type)

        idx = idx.value
        ref.index += idx
        self.push(CellType.REFERENCE, ref)

    def _exec_sdbl(self):
        string = self.pop(CellType.STRING)
        try:
            literal = grammar.numeric_literal.parse_string(string)[0]
            value = float(literal.eval())
        except ParseException:
            value = 0.0
        self.push(CellType.DOUBLE, value)

    def _exec_sign(self):
        value = self.pop()
        if not value.type.is_numeric:
            self.trap(TrapCode.TYPE_MISMATCH,
                      expected='numeric',
                      got=a.type)
        v = value.value
        sign = 1 if v > 0 else -1 if v < 0 else 0
        self.push(value.type, value.type.py_type(sign))

    def _exec_space(self):
        n = self.pop(CellType.INTEGER)
        self.push(CellType.STRING, ' ' * n)

    def _exec_storeg(self, idx):
        value = self.pop()
        try:
            self.globals_segment.set_cell(idx, value)
        except IndexError:
            self.trap(TrapCode.INVALID_GLOBAL_VAR_IDX,
                      idx=idx)

    def _exec_storel(self, idx):
        value = self.pop()
        try:
            self.cur_frame.set_cell(idx, value)
        except IndexError:
            self.trap(TrapCode.INVALID_LOCAL_VAR_IDX,
                      idx=idx)

    def _exec_storeidxg(self, var, idx):
        value = self.pop()
        try:
            self.globals_segment.set_cell(var + idx, value)
        except IndexError:
            self.trap(TrapCode.INVALID_LOCAL_VAR_IDX,
                      idx=var+idx)

    def _exec_storeidxl(self, var, idx):
        value = self.pop()
        try:
            self.cur_frame.set_cell(var + idx, value)
        except IndexError:
            self.trap(TrapCode.INVALID_LOCAL_VAR_IDX,
                      idx=var+idx)

    def _exec_storeref(self):
        ref = self.pop(CellType.REFERENCE)
        value = self.pop()
        ref.segment.set_cell(ref.index, value)

    def _exec_strleft(self):
        n = self.pop(CellType.INTEGER)
        s = self.pop(CellType.STRING)
        self.push(CellType.STRING, s[:n])

    def _exec_strlen(self):
        value = self.pop(CellType.STRING)
        self.push(CellType.LONG, len(value))

    def _exec_strmid(self):
        length = self.pop()
        start = self.pop(CellType.INTEGER)
        string = self.pop(CellType.STRING)

        if length.type == CellType.LONG:
            # a LONG value for length indicates the sub-string should
            # span to the end
            length = None
        else:
            if length.type != CellType.INTEGER:
                self.trap(TrapCode.TYPE_MISMATCH,
                          expected=Type.INTEGER,
                          got=length.type)
            length = length.value

        if start <= 0:
            self.trap(TrapCode.INVALID_OPERAND_VALUE,
                      desc='STRMID start index should be positive')

        if length <= 0:
            self.trap(TrapCode.INVALID_OPERAND_VALUE,
                      desc='STRMID length should be positive')

        start -= 1

        if length is None:
            result = string[start:]
        else:
            result = string[start:start+length]

        self.push(CellType.STRING, result)

    def _exec_strright(self):
        n = self.pop(CellType.INTEGER)
        s = self.pop(CellType.STRING)
        self.push(CellType.STRING, s[-n:])

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

    def _exec_ucase(self):
        s = self.pop(CellType.STRING)
        self.push(CellType.STRING, s.upper())


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
                    value = CellValue(
                        _type, _type.py_type(default_value))
                    self.write_var(scope, idx, value)
                self.push(value.type, value.value)
            method.__name__ = attr
            return method
        setattr(QvmCpu, attr, get_method(scope, _type, attr))


# add exec methods for all variants of the readidx instruction (except
# for the ones dealing with references)
for scope in ['local', 'global']:
    for _type in value_types:
        type_name = _type.name.lower()
        scope_char = 'l' if scope == 'local' else 'g'
        attr = f'_exec_readidx{scope_char}_{type_name}'
        def get_method(scope, _type, name):
            default_value = 0 if _type.is_numeric else ''
            def method(self, var, idx):
                value = self.read_var(scope, var + idx)
                if value is None:
                    value = CellValue(
                        _type, _type.py_type(default_value))
                    self.write_var(scope, idx, value)
                self.push(value.type, value.value)
            method.__name__ = attr
            return method
        setattr(QvmCpu, attr, get_method(scope, _type, attr))


# add exec methods for all variants of the deref instruction
for _type in value_types:
    type_name = _type.name.lower()
    attr = f'_exec_deref_{type_name}'
    def get_method(_type, name):
        default_value = 0 if _type.is_numeric else ''
        def method(self):
            ref = self.pop(CellType.REFERENCE)
            derefed = ref.segment.get_cell(ref.index)
            if derefed is None:
                derefed = CellValue(
                    _type, _type.py_type(default_value))
                ref.segment.set_cell(ref.index, derefed)
            logger.info(f'Derefed {ref} to {derefed}')
            self.push(derefed.type, derefed.value)
        method.__name__ = attr
        return method
    setattr(QvmCpu, attr, get_method(_type, attr))
