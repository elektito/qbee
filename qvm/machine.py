import struct
import logging
from datetime import datetime
from enum import Enum
from qbee import expr
from .using import PrintUsingFormatter
from .cpu import QvmCpu, CellType, TrapCode, QVM_DEVICES
from .subterminal import SubTerminal


logger = logging.getLogger(__name__)


class Device:
    class DeviceError(Enum):
        UNKNOWN_OP = 1
        BAD_ARG_TYPE = 2
        BAD_ARG_VALUE = 3
        OP_FAILED = 4

    def __init__(self, device_id, cpu):
        assert hasattr(self, 'name')

        self.id = device_id
        self.cpu = cpu

        self.cur_op = None

    def execute(self, op):
        assert isinstance(op, str)
        func_name = f'_exec_{op}'
        func = getattr(self, func_name, None)
        if func is None:
            self.cpu.trap(
                TrapCode.DEVICE_ERROR,
                device_id=self.id,
                error_code=Device.DeviceError.UNKNOWN_OP,
                error_msg=f'Unknown op for {self.name} device: {op}')
        else:
            try:
                self.cur_op = op
                func()
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
                          error_code=Device.DeviceError.BAD_ARG_TYPE,
                          error_msg=error_msg)
        elif arg_type is None:
            return boxed_value
        else:
            return boxed_value.value


class TimeDevice(Device):
    name = 'time'

    def _exec_get_time(self):
        logger.info('DEV TIME: get_time')
        now = datetime.now()
        midnight = now.replace(
            hour=0, minute=0, second=0, microsecond=0)
        time_since_midnight = (now - midnight).total_seconds()
        self.cpu.push(CellType.SINGLE, time_since_midnight)


class RngDevice(Device):
    name = 'rng'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = None

    def _exec_seed(self):
        logger.info('DEV RNG: seed')
        seed = self._get_arg_from_stack(CellType.SINGLE)
        self.seed = seed


class MemoryDevice(Device):
    name = 'memory'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cur_segment = None  # default segment

    def _exec_set_segment(self):
        logger.info('DEV MEMORY: set_segment')
        segment = self._get_arg_from_stack(CellType.LONG)
        if segment < 0 or segment > 65535:
            self._device_error(
                error_code=Device.DeviceError.BAD_ARG_VALUE,
                error_msg=f'Invalid segment: 0x{segment:04x}',
            )
            return
        self.cur_segment = segment

    def _exec_set_default_segment(self):
        logger.info('DEV MEMORY: set_default_segment')
        self.cur_segment = None

    def _exec_peek(self):
        logger.info('DEV MEMORY: peek')
        offset = self._get_arg_from_stack(CellType.LONG)
        if self.cur_segment == 0 and offset == 0x417:
            return self._get_control_keys()
        else:
            self._device_error(
                error_code=Device.DeviceError.BAD_ARG_VALUE,
                error_msg=(
                    f'Cannot read memory at: {self.cur_segment:04x}:'
                    f'{offset:04x}'
                ),
            )
            return

    def _exec_poke(self):
        logger.info('DEV MEMORY: poke')
        value = self._get_arg_from_stack(CellType.INTEGER)
        if value < 0 or value > 255:
            self._device_error(
                error_code=Device.DeviceError.BAD_ARG_VALUE,
                error_msg=f'Byte value {value} for poke not valid.'
            )
            return
        offset = self._get_arg_from_stack(CellType.LONG)
        if self.cur_segment == 0 and offset == 0x417:
            return self._set_control_keys()
        else:
            self._device_error(
                error_code=Device.DeviceError.BAD_ARG_VALUE,
                error_msg=(
                    f'Cannot write to memory at: '
                    f'{self.cur_segment:04x}:{offset:04x}'
                ),
            )
            return

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
                error_code=Device.DeviceError.BAD_ARG_VALUE,
                error_msg=(
                    'color_switch, apage, and vpage not supported for '
                    'set_mode operation.'
                )
            )
            return

        self._set_mode(mode, color_switch, apage, vpage)

    def _exec_width(self):
        lines = self._get_arg_from_stack(CellType.INTEGER)
        columns = self._get_arg_from_stack(CellType.INTEGER)

        self._width(columns, lines)

    def _exec_color(self):
        border = self._get_arg_from_stack(CellType.INTEGER)
        bg_color = self._get_arg_from_stack(CellType.INTEGER)
        fg_color = self._get_arg_from_stack(CellType.INTEGER)

        self._color(fg_color, bg_color, border)

    def _exec_cls(self):
        self._cls()

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

        self._locate(row, column, cursor, start, stop)

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
                        error_code=Device.DeviceError.BAD_ARG_VALUE,
                        error_msg=(
                            'Invalid argument type code for PRINT '
                            '(multiple format strings)'
                        )
                    )
                format_string = args[i + 1]
                i += 2
            else:
                self._device_error(
                    error_code=Device.DeviceError.BAD_ARG_VALUE,
                    error_msg=(
                        f'Invalid argument type code for PRINT '
                        f'(unknown code: {args[i]})'
                    )
                )

        if format_string:
            formatter = PrintUsingFormatter(format_string)
            new_line = printables[-1] not in [comma, semicolon]
            printables = [a for a in printables
                          if a != semicolon and a != comma]
            self._print(formatter.format(printables))
            if new_line:
                self._print('\r\n')
        else:
            buf = ''
            def print_number(n):
                nonlocal buf
                if n.value >= 0:
                    buf += ' '
                nval = n.value
                if n.type == CellType.SINGLE:
                    # limit it to a 32 bit float
                    enc = struct.pack('>f', nval)
                    nval, = struct.unpack('>f', enc)
                nval = str(nval)
                if nval.endswith('.0'):
                    nval = nval[:-2]
                if 'e' in nval and n.type == Type.DOUBLE:
                    nval = nval.replace('e', 'D')
                elif 'e' in nval:
                    nval = nval.replace('e', 'E')
                buf += nval
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
            self._print(buf)

    def _exec_inkey(self):
        self._inkey()

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
                        error_code=Device.DeviceError.BAD_ARG_VALUE,
                        error_msg=f'Unknown var type {vtype} for INPUT',
                    )

            return True

        nvars = self.cpu.pop(CellType.INTEGER)
        if nvars <= 0:
            self._device_error(
                error_code=Device.DeviceError.BAD_ARG_VALUE,
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
            self._print(prompt)
            if prompt_question:
                self._print('? ')

            string = self._input(same_line)
            success = push_vars(string, var_types)
            if success:
                break
            self._print('Redo from start\r\n')

    def _exec_view_print(self):
        bottom_line = self.cpu.pop(CellType.INTEGER)
        top_line = self.cpu.pop(CellType.INTEGER)
        self._view_print(top_line, bottom_line)


class DumbTerminalDevice(TerminalDevice):
    def _set_mode(self, mode, color_switch, apage, vpage):
        if mode != 0:
            self._device_error(
                error_code=Device.DeviceError.BAD_ARG_VALUE,
                error_msg=(
                    'Dump terminal only supports SCREEN 0.'
                )
            )
            return

    def _width(self, columns, lines):
        if lines != 25 or columns != 80:
            self._device_error(
                error_code=Device.DeviceError.BAD_ARG_VALUE,
                error_msg=(
                    'Only 80x25 text mode is supported in dumb terminal'
                )
            )
            return

    def _color(self, fg_color, bg_color, border):
        if fg_color > 31:
            self._device_error(
                error_code=Device.DeviceError.BAD_ARG_VALUE,
                error_msg=(
                    'Foreground color should be in range [0, 31]'
                )
            )
            return

        if bg_color > 7:
            self._device_error(
                error_code=Device.DeviceError.BAD_ARG_VALUE,
                error_msg=(
                    'Background color should be in range [0, 7]'
                )
            )
            return

        if border > 15:
            self._device_error(
                error_code=Device.DeviceError.BAD_ARG_VALUE,
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

    def _cls(self):
        seq =  '\033[2J'    # clear screen
        seq += '\033[1;1H'  # move cursor to screen top-left
        print(seq, end='')

    def _locate(self, row, column, cursor, start, stop):
        print(f'\033[{row};{column}H', end='')

    def _print(self, text):
        text = text.replace('\r\n', '\n')
        print(text, end='')

    def _inkey(self):
        self._device_error(
            error_code=Device.DeviceError.BAD_ARG_VALUE,
            error_msg='INKEY not supported on dumb terminal.',
        )

    def _input(self, same_line):
        if same_line:
            self._device_error(
                error_code=Device.DeviceError.BAD_ARG_VALUE,
                error_msg='Cannot stay on the same line in dumb terminal',
            )

        return input()

    def _view_print(self, top_line, bottom_line):
        if bottom_line >= 0 or top_line >= 0:
            self._device_error(
                error_code=Device.DeviceError.BAD_ARG_VALUE,
                error_msg='VIEW PRINT with arguments not supported',
            )

        # no need to do anything for view print without arguments


class SmartTerminalDevice(TerminalDevice):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.terminal = SubTerminal()
        self.terminal.launch()

    def _set_mode(self, mode, color_switch, apage, vpage):
        self.mode = mode

    def _width(self, columns, lines):
        print('WIDTH NOT SUPPORTED FOR NOW')

    def _color(self, fg_color, bg_color, border):
        if border >= 0:
            self.terminal.call('set', 'border_color', border)
        if fg_color >= 0:
            self.terminal.call('set', 'fg_color', fg_color)
        if bg_color >= 0:
            self.terminal.call('set', 'bg_color', bg_color)

    def _cls(self):
        self.terminal.call('clear_screen')

    def _locate(self, row, col, cursor, start, stop):
        if row >= 0:
            self.terminal.call('set', 'cursor_row', row)
        if col >= 0:
            self.terminal.call('set', 'cursor_col', col)

    def _print(self, text):
        self.terminal.call('put_text', text)

    def _inkey(self):
        k = self.terminal.call_with_result('get_key')
        if k == -1:
            self.cpu.push(CellType.STRING, '')
        elif isinstance(k, int):
            self.cpu.push(CellType.STRING, chr(k))
        elif isinstance(k, tuple):
            self.cpu.push(CellType.STRING, chr(k[0]) + chr(k[1]))
        else:
            assert False

    def _input(self, same_line):
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
                col = self.terminal.call_with_result('get', 'cursor_col')
                self.terminal.call('set', 'cursor_col', col - 1)
                self.terminal.call('put_text', ' ')
                self.terminal.call('set', 'cursor_col', col - 1)

        if not same_line:
            self.terminal.call('put_text', '\r\n')

        self.terminal.call('set', 'show_cursor', False)

        return string

    def _view_print(self, top_line, bottom_line):
        if bottom_line >= 0 or top_line >= 0:
            self._device_error(
                error_code=Device.DeviceError.BAD_ARG_VALUE,
                error_msg='VIEW PRINT with arguments not supported',
            )

        # no need to do anything for view print without arguments
        # right now


class PcSpeakerDevice(Device):
    name = 'pcspkr'

    def _exec_play(self):
        command = self.cpu.pop(CellType.STRING)
        logger.info('PLAY: %s', command)


class DataDevice(Device):
    name = 'data'

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
                error_code=Device.DeviceError.OP_FAILED,
                error_msg='Out of data',
            )

        try:
            if data_type == 1:
                self.cpu.push(CellType.INTEGER, int(s))
            elif data_type == 2:
                self.cpu.push(CellType.LONG, int(s))
            elif data_type == 3:
                self.cpu.push(CellType.SINGLE, float(s))
            elif data_type == 4:
                self.cpu.push(CellType.DOUBLE, float(s))
            elif data_type == 5:
                self.cpu.push(CellType.STRING, s)
            else:
                assert False
        except (ValueError, TypeError) as e:
            self._device_error(
                error_code=Device.DeviceError.BAD_ARG_TYPE,
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


class QvmMachine:
    def __init__(self, module, terminal='smart'):
        self.cpu = QvmCpu(module)

        time_device = TimeDevice(QVM_DEVICES['time']['id'], self.cpu)
        self.cpu.connect_device('time', time_device)

        rng_device = RngDevice(QVM_DEVICES['rng']['id'], self.cpu)
        self.cpu.connect_device('rng', rng_device)

        memory_device = MemoryDevice(
            QVM_DEVICES['memory']['id'], self.cpu)
        self.cpu.connect_device('memory', memory_device)

        terminal_class = {
            'smart': SmartTerminalDevice,
            'dumb': DumbTerminalDevice,
        }.get(terminal)
        assert terminal_class is not None
        terminal_device = terminal_class(
            QVM_DEVICES['terminal']['id'], self.cpu)
        self.cpu.connect_device('terminal', terminal_device)

        speaker_device = PcSpeakerDevice(
            QVM_DEVICES['pcspkr']['id'], self.cpu)
        self.cpu.connect_device('pcspkr', speaker_device)

        data_device = DataDevice(QVM_DEVICES['data']['id'], self.cpu)
        self.cpu.connect_device('data', data_device)

    def run(self):
        self.cpu.run()

    def tick(self):
        self.cpu.tick()

    @property
    def halted(self):
        return self.cpu.halted
