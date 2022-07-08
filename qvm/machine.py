import struct
import logging
from datetime import datetime
from enum import Enum
from random import Random
from qbee import expr
from qbee.utils import Empty
from .using import PrintUsingFormatter
from .cpu import QvmCpu, QVM_DEVICES
from .cell import CellType
from .trap import TrapCode
from .subterminal import SubTerminal
from .utils import format_number
from .exceptions import DeviceError


logger = logging.getLogger(__name__)


class Device:
    class Error(Enum):
        UNKNOWN_OP = 1
        BAD_ARG_TYPE = 2
        BAD_ARG_VALUE = 3
        OP_FAILED = 4

    def __init__(self, device_id, cpu, impl):
        assert hasattr(self, 'name')

        self.id = device_id
        self.cpu = cpu
        self.impl = impl

        self.cur_op = None

    def execute(self, op):
        assert isinstance(op, str)
        func_name = f'_exec_{op}'
        func = getattr(self, func_name, None)
        if func is None:
            self.cpu.trap(
                TrapCode.DEVICE_ERROR,
                device_id=self.id,
                error_code=Device.Error.UNKNOWN_OP,
                error_msg=f'Unknown op for {self.name} device: {op}')
        else:
            try:
                self.cur_op = op
                func()
            except DeviceError as e:
                error_code = e.error_code or Device.Error.OP_FAILED
                self.cpu.trap(
                    TrapCode.DEVICE_ERROR,
                    device_id=self.id,
                    error_code=error_code,
                    error_msg=str(e))
            except AttributeError as e:
                if e.obj is self.impl:
                    self.cpu.trap(
                    TrapCode.DEVICE_ERROR,
                    device_id=self.id,
                    error_code=Device.Error.OP_FAILED,
                    error_msg=f'{self.name}::{op} not implemented')
            finally:
                self.cur_op = None

    def _device_error(self, error_code, error_msg):
        self.cpu.trap(
            TrapCode.DEVICE_ERROR,
            device_id=self.id,
            error_code=error_code,
            error_msg=error_msg)

    def _get_arg_from_stack(self, arg_type=None):
        boxed_value = self.cpu.pop()
        if arg_type and boxed_value.type != arg_type:
            cur_op = self.cur_op or ''
            error_msg = (
                f'IO operation {cur_op} expected argument of type '
                f'{arg_type.name.upper()}, got '
                f'{boxed_value.type.name.upper()}.'
            )
            self.cpu.trap(TrapCode.DEVICE_ERROR,
                          device_id=self.id,
                          error_code=Device.Error.BAD_ARG_TYPE,
                          error_msg=error_msg)
        elif arg_type is None:
            return boxed_value
        else:
            return boxed_value.value


class TimeDevice(Device):
    name = 'time'

    def _exec_get_time(self):
        logger.info('DEV TIME: get_time')
        time_since_midnight = self.impl.time_get_time()
        self.cpu.push(CellType.SINGLE, time_since_midnight)


class RngDevice(Device):
    name = 'rng'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_rnd = None

    def _exec_seed(self):
        logger.info('DEV RNG: seed')
        seed = self._get_arg_from_stack(CellType.SINGLE)
        self.impl.rng_seed(seed)

    def _exec_rnd(self):
        arg = self._get_arg_from_stack(CellType.SINGLE)

        if arg == 0:
            if self.last_rnd is None:
                self.last_rnd = self.impl.rng_get_next()
        elif arg < 0:
            self.last_rnd = self.impl.rng_get_with_seed(arg)
        else:
            self.last_rnd = self.impl.rng_get_next()

        self.cpu.push(CellType.SINGLE, self.last_rnd)


class MemoryDevice(Device):
    name = 'memory'

    def _exec_set_segment(self):
        logger.info('DEV MEMORY: set_segment')
        segment = self._get_arg_from_stack(CellType.LONG)
        if segment < 0 or segment > 65535:
            self._device_error(
                error_code=Device.Error.BAD_ARG_VALUE,
                error_msg=f'Invalid segment: 0x{segment:04x}',
            )
            return
        self.impl.memory_set_segment(segment)

    def _exec_set_default_segment(self):
        logger.info('DEV MEMORY: set_default_segment')
        self.impl.memory_set_default_segment()

    def _exec_peek(self):
        logger.info('DEV MEMORY: peek')
        offset = self._get_arg_from_stack(CellType.LONG)
        result = self.impl.memory_peek(offset)
        self.cpu.push(CellType.INTEGER, result)

    def _exec_poke(self):
        logger.info('DEV MEMORY: poke')
        value = self._get_arg_from_stack(CellType.INTEGER)
        if value < 0 or value > 255:
            self._device_error(
                error_code=Device.Error.BAD_ARG_VALUE,
                error_msg=f'Byte value {value} for poke not valid.'
            )
            return
        offset = self._get_arg_from_stack(CellType.LONG)
        self.impl.memory_poke(offset, value)

    def _get_control_keys(self):
        # The original meaning of the bits at 0000:0417, were (from
        # most significant to least significant bits):
        #
        # Insert|Caps Lock|Num Lock|Scroll Lock|Alt|Ctrl|LShift|RShift
        #
        # We do not support reading/setting these values right now
        # however.
        logger.info('Reading control keys not supported; returning 0.')
        self.cpu.push(CellType.INTEGER, 0)

    def _set_control_keys(self):
        # The original meaning of the bits at 0000:0417, were (from
        # most significant to least significant bits):
        #
        # Insert|Caps Lock|Num Lock|Scroll Lock|Alt|Ctrl|LShift|RShift
        #
        # We do not support reading/setting these values right now
        # however.
        logger.info('Setting control keys not supported.')


class TerminalDevice(Device):
    name = 'terminal'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = 0

    def _exec_set_mode(self):
        vpage = self._get_arg_from_stack(CellType.INTEGER)
        apage = self._get_arg_from_stack(CellType.INTEGER)
        color_switch = self._get_arg_from_stack(CellType.INTEGER)
        mode = self._get_arg_from_stack(CellType.INTEGER)

        if vpage != -1 or apage != -1 or color_switch != -1:
            self._device_error(
                error_code=Device.Error.BAD_ARG_VALUE,
                error_msg=(
                    'color_switch, apage, and vpage not supported for '
                    'set_mode operation.'
                )
            )
            return

        self.impl.terminal_set_mode(mode, color_switch, apage, vpage)

    def _exec_width(self):
        lines = self._get_arg_from_stack(CellType.INTEGER)
        columns = self._get_arg_from_stack(CellType.INTEGER)

        self.impl.terminal_width(columns, lines)

    def _exec_color(self):
        border = self._get_arg_from_stack(CellType.INTEGER)
        bg_color = self._get_arg_from_stack(CellType.INTEGER)
        fg_color = self._get_arg_from_stack(CellType.INTEGER)

        self.impl.terminal_color(fg_color, bg_color, border)

    def _exec_cls(self):
        self.impl.terminal_cls()

    def _exec_locate(self):
        stop = self._get_arg_from_stack(CellType.INTEGER)
        start = self._get_arg_from_stack(CellType.INTEGER)
        cursor = self._get_arg_from_stack(CellType.INTEGER)
        column = self._get_arg_from_stack(CellType.INTEGER)
        row = self._get_arg_from_stack(CellType.INTEGER)

        # convert one-based values to zero-based
        if column >= 1:
            column -= 1
        if row >= 1:
            row -= 1

        self.impl.terminal_locate(row, column, cursor, start, stop)

    def _exec_print(self):
        nargs = self._get_arg_from_stack(CellType.INTEGER)
        args = []
        for i in range(nargs):
            arg = self._get_arg_from_stack()
            args.append(arg)
        args.reverse()

        i = 0
        semicolon = object()
        comma = object()
        printables = []
        format_string = None
        while i < len(args):
            arg = args[i].value
            if arg == 0:
                printables.append(args[i + 1])
                i += 2
            elif arg == 1:
                printables.append(semicolon)
                i += 1
            elif arg == 2:
                printables.append(comma)
                i += 1
            elif arg == 3:
                if format_string is not None:
                    self._device_error(
                        error_code=Device.Error.BAD_ARG_VALUE,
                        error_msg=(
                            'Invalid argument type code for PRINT '
                            '(multiple format strings)'
                        )
                    )
                format_string = args[i + 1]
                i += 2
            else:
                self._device_error(
                    error_code=Device.Error.BAD_ARG_VALUE,
                    error_msg=(
                        f'Invalid argument type code for PRINT '
                        f'(unknown code: {args[i]})'
                    )
                )

        if format_string:
            formatter = PrintUsingFormatter(format_string.value)
            new_line = printables[-1] not in [comma, semicolon]
            printables = [a.value for a in printables
                          if a != semicolon and a != comma]
            self.impl.terminal_print(formatter.format(printables))
            if new_line:
                self.impl.terminal_print('\r\n')
        else:
            buf = ''
            def print_number(n):
                nonlocal buf
                nval = n.value
                buf += format_number(nval, n.type) + ' '
            for arg in printables:
                if arg == semicolon:
                    pass
                elif arg == comma:
                    n = 14 - (len(buf) % 14)
                    buf += n * ' '
                elif arg.type.is_numeric:
                    print_number(arg)
                else:
                    buf += arg.value
            if len(printables) == 0 or \
               printables[-1] not in [comma, semicolon]:
                buf += '\r\n'
            self.impl.terminal_print(buf)

    def _exec_inkey(self):
        result = self.impl.terminal_inkey()
        self.cpu.push(CellType.STRING, result)

    def _exec_input(self):
        def push_vars(string, var_types):
            values = string.split(',')
            values = [v.strip() for v in values]
            if len(values) != len(var_types):
                return False

            for v, vtype in reversed(list(zip(values, var_types))):
                if vtype == 1:  # INTEGER
                    try:
                        v = int(v)
                    except ValueError:
                        return False
                    if v < -32768 or v > 32767:
                        return False
                    self.cpu.push(CellType.INTEGER, v)
                elif vtype == 2:  # LONG
                    try:
                        v = int(v)
                    except ValueError:
                        return False
                    if v < -2**31 or v >= 2**31:
                        return False
                    self.cpu.push(CellType.LONG, v)
                elif vtype == 3:  # SINGLE
                    try:
                        v = float(v)
                    except ValueError:
                        return False
                    if not expr.Type.SINGLE.can_hold(v):
                        return False
                    self.cpu.push(CellType.SINGLE, v)
                elif vtype == 4:  # DOUBLE
                    try:
                        v = float(v)
                    except ValueError:
                        return False
                    if not expr.Type.DOUBLE.can_hold(v):
                        return False
                    self.cpu.push(CellType.DOUBLE, v)
                elif vtype == 5:  # STRING
                    self.cpu.push(CellType.STRING, v)
                else:
                    self._device_error(
                        error_code=Device.Error.BAD_ARG_VALUE,
                        error_msg=f'Unknown var type {vtype} for INPUT',
                    )

            return True

        nvars = self.cpu.pop(CellType.INTEGER)
        if nvars <= 0:
            self._device_error(
                error_code=Device.Error.BAD_ARG_VALUE,
                error_msg='INPUT number of variables not positive',
            )

        var_types = []
        for i in range(nvars):
            var_types.append(self.cpu.pop(CellType.INTEGER))
        var_types = list(reversed(var_types))

        prompt_question = self.cpu.pop(CellType.INTEGER)
        prompt = self.cpu.pop(CellType.STRING)
        same_line = self.cpu.pop(CellType.INTEGER)

        while True:
            self.impl.terminal_print(prompt)
            if prompt_question:
                self.impl.terminal_print('? ')

            string = self.impl.terminal_input(same_line)
            success = push_vars(string, var_types)
            if success:
                break
            self.impl.terminal_print('Redo from start\r\n')

    def _exec_view_print(self):
        bottom_line = self.cpu.pop(CellType.INTEGER)
        top_line = self.cpu.pop(CellType.INTEGER)
        self.impl.terminal_view_print(top_line, bottom_line)


class PcSpeakerDevice(Device):
    name = 'pcspkr'

    def _exec_beep(self):
        self.impl.pcspkr_beep()

    def _exec_play(self):
        command = self.cpu.pop(CellType.STRING)
        self.impl.pcspkr_play(command)


class DataDevice(Device):
    name = 'data'

    # NOTE: The DataDevice does not really perform IO, so it won't
    # call 'impl' for its operations.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_part = 0
        self.data_idx = 0

    def _exec_read(self):
        data_type = self.cpu.pop(CellType.INTEGER)
        try:
            s = self.cpu.module.data[self.data_part][self.data_idx]
        except IndexError:
            self._device_error(
                error_code=Device.Error.OP_FAILED,
                error_msg='Out of data',
            )

        try:
            if data_type == 1:
                value = 0 if s == Empty.value else int(s)
                self.cpu.push(CellType.INTEGER, value)
            elif data_type == 2:
                value = 0 if s == Empty.value else int(s)
                self.cpu.push(CellType.LONG, value)
            elif data_type == 3:
                value = 0.0 if s == Empty.value else float(s)
                self.cpu.push(CellType.SINGLE, value)
            elif data_type == 4:
                value = 0.0 if s == Empty.value else float(s)
                self.cpu.push(CellType.DOUBLE, value)
            elif data_type == 5:
                value = '' if s == Empty.value else s
                self.cpu.push(CellType.STRING, value)
            else:
                assert False
        except (ValueError, TypeError) as e:
            self._device_error(
                error_code=Device.Error.BAD_ARG_TYPE,
                error_msg='Cannot READ data as requested type',
            )

        self.data_idx += 1
        if self.data_idx >= len(self.cpu.module.data[self.data_part]):
            self.data_idx = 0
            self.data_part += 1

    def _exec_restore(self):
        part_idx = self.cpu.pop(CellType.INTEGER)
        self.data_part = part_idx
        self.data_idx = 0


class BasePeripheralsImpl:
    def __init__(self):
        super().__init__()

        # memory
        self.cur_segment = None  # default segment

        # rng
        self.rng = Random()
        self.last_rnd = self.rng.random()

    # memory

    def memory_set_segment(self, segment):
        self.cur_segment = segment

    def memory_set_default_segment(self):
        self.cur_segment = None

    def memory_peek(self, offset):
        if self.cur_segment == 0 and offset == 0x417:
            return self.misc_get_control_keys()
        else:
            raise DeviceError(
                error_code=Device.Error.BAD_ARG_VALUE,
                error_msg=(
                    f'Cannot read memory at: {self.cur_segment:04x}:'
                    f'{offset:04x}'
                ),
            )

    def memory_poke(self, offset, value):
        if self.cur_segment == 0 and offset == 0x417:
            self.misc_set_control_keys()
        else:
            raise DeviceError(
                error_code=Device.Error.BAD_ARG_VALUE,
                error_msg=(
                    f'Cannot write to memory at: '
                    f'{self.cur_segment:04x}:{offset:04x}'
                ),
            )

    # pcspkr

    def pcspkr_play(self, command):
        logger.info('PLAY: %s', command)

    # rng

    def rng_seed(self, seed):
        self.rng.seed(seed)

    def rng_get_next(self):
        return self.rng.random()

    def rng_get_with_seed(self, seed):
        state = self.rng.getstate()
        self.rng.seed(seed)
        rnd = self.rng.random()
        self.rng.setstate(state)
        return rnd

    def rng_rnd(self, arg):
        if arg == 0:
            return self.last_rnd
        elif arg < 0:
            state = self.rng.getstate()
            self.rng.seed(arg)
            self.last_rnd = self.rng.random()
            self.rng.setstate(state)
            return self.last_rnd
        else:
            self.last_rnd = self.rng.random()
            return self.last_rnd

    # time

    def time_get_time(self):
        now = datetime.now()
        midnight = now.replace(
            hour=0, minute=0, second=0, microsecond=0)
        time_since_midnight = (now - midnight).total_seconds()
        return time_since_midnight

    # misc

    def misc_get_control_keys(self):
        # The original meaning of the bits at 0000:0417, were (from
        # most significant to least significant bits):
        #
        # Insert|Caps Lock|Num Lock|Scroll Lock|Alt|Ctrl|LShift|RShift
        #
        # We do not support reading/setting these values right now
        # however.
        logger.info('Reading control keys not supported; returning 0.')
        return 0

    def misc_set_control_keys(self):
        # The original meaning of the bits at 0000:0417, were (from
        # most significant to least significant bits):
        #
        # Insert|Caps Lock|Num Lock|Scroll Lock|Alt|Ctrl|LShift|RShift
        #
        # We do not support reading/setting these values right now
        # however.
        logger.info('Setting control keys not supported.')


class SmartTerminalMixin:
    def __init__(self):
        super().__init__()
        self.mode = 0
        self.terminal = SubTerminal()
        self.terminal.launch()

    def terminal_set_mode(self, mode, color_switch, apage, vpage):
        self.mode = mode
        self.terminal.call('set_mode', mode)

    def terminal_width(self, columns, lines):
        print('WIDTH NOT SUPPORTED FOR NOW')

    def terminal_color(self, fg_color, bg_color, border):
        if border >= 0:
            self.terminal.call('set', 'border_color', border)
        if fg_color >= 0:
            self.terminal.call('set', 'fg_color', fg_color)
        if bg_color >= 0:
            self.terminal.call('set', 'bg_color', bg_color)

    def terminal_cls(self):
        self.terminal.call('clear_screen')

    def terminal_locate(self, row, col, cursor, start, stop):
        if row < 0:
            row = None
        if col < 0:
            col = None
        self.terminal.call('locate', row, col)

    def terminal_print(self, text):
        self.terminal.call('put_text', text)

    def terminal_inkey(self):
        k = self.terminal.call_with_result('get_key')
        if k == -1:
            return ''
        elif isinstance(k, int):
            return chr(k)
        elif isinstance(k, tuple):
            return chr(k[0]) + chr(k[1])
        else:
            assert False

    def terminal_input(self, same_line):
        self.terminal.call('set', 'show_cursor', True)
        string = ''
        while True:
            k = self.terminal.call_with_result('get_key')
            if k == -1:
                continue
            if isinstance(k, tuple):
                pass
            elif 32 <= k <= 126:
                self.terminal.call('put_text', chr(k))
                string += chr(k)
            elif k == 13:
                break
            elif k == 8 and string:
                string = string[:-1]
                row, col = self.terminal.call_with_result(
                    'get_cursor_pos')
                col = 0 if col <= 0 else col - 1
                self.terminal.call('locate', row, col)
                self.terminal.call('put_text', ' ')
                self.terminal.call('locate', row, col)

        if not same_line:
            self.terminal.call('put_text', '\r\n')

        self.terminal.call('set', 'show_cursor', False)

        return string

    def terminal_view_print(self, top_line, bottom_line):
        if top_line <= 0:
            # view print with no argument
            top_line = None
            bottom_line = None

        self.terminal.call('view_print', top_line, bottom_line)


class DumbTerminalMixin:
    def __init__(self):
        super().__init__()

    def terminal_set_mode(self, mode, color_switch, apage, vpage):
        if mode != 0:
            self._device_error(
                error_code=Device.Error.BAD_ARG_VALUE,
                error_msg=(
                    'Dumb terminal only supports SCREEN 0.'
                )
            )
            return

    def terminal_width(self, columns, lines):
        if lines != 25 or columns != 80:
            self._device_error(
                error_code=Device.Error.BAD_ARG_VALUE,
                error_msg=(
                    'Only 80x25 text mode is supported in dumb terminal'
                )
            )
            return

    def terminal_color(self, fg_color, bg_color, border):
        if fg_color > 31:
            self._device_error(
                error_code=Device.Error.BAD_ARG_VALUE,
                error_msg=(
                    'Foreground color should be in range [0, 31]'
                )
            )
            return

        if bg_color > 7:
            self._device_error(
                error_code=Device.Error.BAD_ARG_VALUE,
                error_msg=(
                    'Background color should be in range [0, 7]'
                )
            )
            return

        if border > 15:
            self._device_error(
                error_code=Device.Error.BAD_ARG_VALUE,
                error_msg=(
                    'Border color should be in range [0, 15]'
                )
            )
            return

        fg_ansi_code = {
            0: '\033[0;30m',     # black
            1: '\033[0;34m',     # blue
            2: '\033[0;32m',     # green
            3: '\033[0;36m',     # cyan
            4: '\033[0;31m',     # red
            5: '\033[0;35m',     # magenta
            6: '\033[0;33m',     # brown, but actually yellow
            7: '\033[0;37m',     # white
            8: '\033[0;30;1m',   # gray (light black?)
            9: '\033[0;34;1m',   # light blue
            10: '\033[0;32;1m',  # light green
            11: '\033[0;36;1m',  # light cyan
            12: '\033[0;31;1m',  # light red
            13: '\033[0;3f;1m',  # light magenta
            14: '\033[0;33;1m',  # yellow
            15: '\033[0;37;1m',  # bright white
        }

        bg_ansi_code = {
            0: '\033[0;40m',     # black
            1: '\033[0;44m',     # blue
            2: '\033[0;42m',     # green
            3: '\033[0;46m',     # cyan
            4: '\033[0;41m',     # red
            5: '\033[0;45m',     # magenta
            6: '\033[0;43m',     # brown, but actually yellow
            7: '\033[0;47m',     # white
            8: '\033[0;40;1m',   # gray (light black?)
            9: '\033[0;44;1m',   # light blue
            10: '\033[0;42;1m',  # light green
            11: '\033[0;46;1m',  # light cyan
            12: '\033[0;41;1m',  # light red
            13: '\033[0;45;1m',  # light magenta
            14: '\033[0;43;1m',  # yellow
            15: '\033[0;47;1m',  # bright white
        }

        if fg_color > 15:
            fg_color -= 15  # blinking not supported

        if bg_color > 0:
            print(bg_ansi_code[bg_color], end='')
        if fg_color > 0:
            print(fg_ansi_code[fg_color], end='')

    def terminal_cls(self):
        seq =  '\033[2J'    # clear screen
        seq += '\033[1;1H'  # move cursor to screen top-left
        print(seq, end='')

    def terminal_locate(self, row, column, cursor, start, stop):
        print(f'\033[{row};{column}H', end='')

    def terminal_print(self, text):
        text = text.replace('\r\n', '\n')
        print(text, end='')

    def terminal_inkey(self):
        raise DeviceError(
            error_code=Device.Error.BAD_ARG_VALUE,
            error_msg='INKEY not supported on dumb terminal.',
        )

    def terminal_input(self, same_line):
        if same_line:
            raise DeviceError(
                error_code=Device.Error.BAD_ARG_VALUE,
                error_msg='Cannot stay on the same line in dumb terminal',
            )

        return input()

    def terminal_view_print(self, top_line, bottom_line):
        if bottom_line >= 0 or top_line >= 0:
            self._device_error(
                error_code=Device.Error.BAD_ARG_VALUE,
                error_msg='VIEW PRINT with arguments not supported',
            )

        # no need to do anything for view print without arguments


class SmartPeripheralsImpl(BasePeripheralsImpl, SmartTerminalMixin):
    pass


class DumbPeripheralsImpl(BasePeripheralsImpl, DumbTerminalMixin):
    pass


class QvmMachine:
    def __init__(self, module, impl=None, terminal='smart'):
        self.cpu = QvmCpu(module)

        if not impl:
            impl_class = {
                'smart': SmartPeripheralsImpl,
                'dumb': DumbPeripheralsImpl,
            }.get(terminal)
            assert impl_class is not None

            impl = impl_class()

        time_device = TimeDevice(
            QVM_DEVICES['time']['id'], self.cpu, impl)
        self.cpu.connect_device('time', time_device)

        rng_device = RngDevice(
            QVM_DEVICES['rng']['id'], self.cpu, impl)
        self.cpu.connect_device('rng', rng_device)

        memory_device = MemoryDevice(
            QVM_DEVICES['memory']['id'], self.cpu, impl)
        self.cpu.connect_device('memory', memory_device)

        terminal_device = TerminalDevice(
            QVM_DEVICES['terminal']['id'], self.cpu, impl)
        self.cpu.connect_device('terminal', terminal_device)

        speaker_device = PcSpeakerDevice(
            QVM_DEVICES['pcspkr']['id'], self.cpu, impl)
        self.cpu.connect_device('pcspkr', speaker_device)

        data_device = DataDevice(
            QVM_DEVICES['data']['id'], self.cpu, impl)
        self.cpu.connect_device('data', data_device)

    def run(self):
        self.cpu.run()

    def tick(self):
        self.cpu.tick()

    @property
    def halted(self):
        return self.cpu.halted

