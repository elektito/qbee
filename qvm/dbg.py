import argparse
import cmd
from functools import wraps
from qbee.stmt import Block
from .module import QModule
from .machine import QvmMachine
from .debug_info import DebugInfo


class Breakpoint:
    def __init__(self, start_addr, end_addr=None, *, exact=True,
                 line=None, routine=None):
        self.start_addr = start_addr
        self.end_addr = end_addr
        self.exact = exact
        self.line = line
        self.routine = routine

    def __str__(self):
        main_desc = None
        if self.line:
            main_desc = f'line {self.line}'
        elif self.routine:
            main_desc = f'routine {self.routine}'

        exact = ' UNEXACT' if not self.exact else ''
        if self.end_addr:
            if self.end_addr - self.start_addr == 1:
                addr_desc = f'address 0x{self.start_addr:08x}{exact}'
            else:
                addr_desc = (
                    f'range 0x{self.start_addr:08x}-'
                    f'0x{self.end_addr:08x}{exact}'
                )
        else:
            addr_desc = f'address 0x{self.start_addr:08x}{exact}'

        if main_desc:
            return f'{main_desc} ({addr_desc})'
        else:
            return addr_desc

    def __eq__(self, other):
        if not isinstance(other, Breakpoint):
            return False
        return (
            self.start_addr == other.start_addr and
            self.end_addr == other.end_addr and
            self.exact == other.exact
        )

    def __call__(self, cpu):
        if self.end_addr:
            return self.start_addr <= cpu.pc < self.end_addr
        else:
            if self.exact:
                return cpu.pc == self.start_addr
            else:
                return cpu.pc >= self.start_addr


def unhalted(func):
    @wraps(func)
    def wrapped(self, *args, **kwargs):
        if self.machine.halted:
            print('Machine is halted.')
            return
        return func(self, *args, **kwargs)
    return wrapped


class Cmd(cmd.Cmd):
    intro = """
QVM interactive debugger
Type help or ? to list commands.
"""
    prompt = '(qdb) '

    def __init__(self, machine, module):
        if not module.debug_info:
            print('Debug info not available')
            exit(1)

        super().__init__()
        self.module = module
        self.debug_info = self.module.debug_info
        self.machine = machine
        self.cpu = machine.cpu
        self.auto_status_enabled = True

        self.instrs = []
        self.load_instructions()

        self.source_lines = self.debug_info.source_code.split('\n')

        self.set_prompt()

        self.start_debugging()

    def load_instructions(self):
        addr = 0
        while addr < len(self.module.code):
            instr, operands, size = self.cpu.get_instruction_at(addr)
            self.instrs.append((addr, instr, operands))
            addr += size

    def find_stmt(self, addr):
        # if we're at the beginning of the module code, there should
        # be a call instruction telling us where the actual beginning
        # of the module code is. we'll use that to find the first
        # statement.
        if addr == 0:
            instr, operands, _ = self.cpu.get_instruction_at(0)
            if instr.op != 'call':
                return None
            return self.find_stmt(operands[0])

        # if the address falls within the range of some statement,
        # return the one with the smallest range, excluding any
        # blocks.
        matching = []
        for stmt in self.debug_info.stmts:
            if stmt.start_offset <= addr < stmt.end_offset:
                matching.append(stmt)

        blocks = [stmt for stmt in matching if isinstance(stmt, Block)]
        non_blocks = [
            stmt for stmt in matching
            if not isinstance(stmt, Block)]

        non_blocks.sort(key=lambda r: r.end_offset - r.start_offset)
        if non_blocks:
            return non_blocks[0]

        # if not however...

        if blocks:
            blocks.sort(key=lambda r: r.end_offset - r.start_offset)

            # if it falls between the start of a block, and the start
            # of the first statement inside that block, it's
            # considered part of the "state statement" of that block
            first_inner_stmt = first_stmt_in_block(block)
            if first_inner_stmt:
                first_inner_offset = first_inner_stmt.start_offset
            else:
                first_inner_offset = block.end_stmt.start_offset
            if block.start_offset <= addr < first_inner_offset:
                return block.start_stmt

            # if it falls between the end of the last statement inside
            # a block and the end of the block, it's considered part
            # of the "end statement" of that block.
            last_inner_stmt = last_stmt_in_block(block)
            if last_inner_stmt:
                last_inner_offset = last_inner_stmt.end_offset
            else:
                last_inner_offset = block.start_stmt.end_offset
            if last_inner_offset <= addr < block.end_offset:
                return block.end_stmt

        return None

    def find_nonempty_stmt(self, addr):
        stmt = self.find_stmt(addr)
        if stmt is None:
            return None

        idx = self.debug_info.stmts.index(stmt)
        while stmt.end_offset - stmt.start_offset == 0:
            idx += 1
            if idx >= len(self.debug_info.stmts):
                break
            stmt = self.debug_info.stmts[idx]
            if stmt is None:
                break

        return stmt

    def find_routine(self, addr):
        for routine in self.debug_info.routines.values():
            if routine.start_offset <= addr < routine.end_offset:
                return routine
        return None

    def start_debugging(self):
        # run module code until we reach a statement
        assert self.cpu.pc == 0
        while not self.cpu.halted and not self.find_stmt(self.cpu.pc):
            self.cpu.tick()

        if self.cpu.halted:
            print('Empty program finished running already.')

    def set_prompt(self):
        self.prompt = f'(qdb PC={self.cpu.pc:08x}) '

    def parse_breakpoint_spec(self, spec):
        if spec.startswith('0x'):
            try:
                addr = int(spec, base=16)
            except (ValueError, TypeError):
                return None, 'Invalid address'
            return Breakpoint(start_addr=addr), None
        elif spec.isnumeric():
            line_no = int(spec)
            for stmt in self.debug_info.stmts:
                if stmt.source_start_line >= line_no:
                    bp = Breakpoint(
                        start_addr=stmt.start_offset,
                        line=stmt.source_start_line
                    )
                    if stmt.source_start_line > line_no:
                        print(f'Could not set a breakpoint at that '
                              f'precise line; set at line '
                              f'{stmt.source_start_line}')
                    return bp, None
        else:
            routine_name = spec.lower()
            routine = self.debug_info.routines.get(routine_name)
            if routine:
                addr = routine.start_offset
                bp = Breakpoint(
                    start_addr=routine.start_offset,
                    end_addr=routine.end_offset,
                    routine=routine.name,
                )
                return bp, None
            return None, 'no such routine'

    def show_instruction_with_context(self, addr, context_size=4):
        i = None
        for i, (instr_addr, instr, operands) in enumerate(self.instrs):
            if addr == instr_addr:
                break
        if i is None or i == len(self.instrs):
            return
        start = i - context_size
        if start < 0:
            start = 0
        end = i + context_size
        if end >= len(self.instrs):
            end = len(self.instrs) - 1

        cur_idx = i
        for i in range(start, end + 1):
            i_addr, instr, operands = self.instrs[i]
            operands_list = ''
            if operands:
                operands_list = ' ' + \
                    ', '.join(str(i) for i in operands)
            prefix = ' > ' if i == cur_idx else '   '
            print(
                f'{prefix}{i_addr:08x}: {instr.op: <12}{operands_list}')

    def show_statement_with_context(self, stmt, context_size=3):
        start_line = stmt.source_start_line - context_size
        if start_line < 1:
            start_line = 1

        end_line = stmt.source_end_line + context_size
        if end_line > len(self.source_lines):
            end_line = len(self.source_lines)

        for i in range(start_line, end_line + 1):
            line = self.source_lines[i - 1]
            if i == stmt.source_start_line:
                print(f' > {line}')
            else:
                print(f'   {line}')

    def show_auto_status(self):
        if self.auto_status_enabled:
            self.do_cur('')

    def postcmd(self, stop, line):
        if stop:
            print('Goodbye!')
        else:
            self.set_prompt()
        return stop

    def do_autostatus(self, arg):
        """Change/show auto-status. If auto-status is on, the debugger
        would print some status information after each operation that
        progresses the debuggee."""
        arg = arg.strip().upper()
        if arg == '':
            status = 'ON' if self.auto_status_enabled else 'OFF'
            print(f'Auto-status is {status}')
        elif arg == 'ON':
            self.auto_status_enabled = True
        elif arg == 'OFF':
            self.auto_status_enabled = False
        else:
            print('Invalid argument.')

    @unhalted
    def do_continue(self, arg):
        'Continue until the machine is halted or we hit a breakpoint.'
        if not self.cpu.run():
            print('Hit breakpoint')

        self.show_auto_status()

    def do_curi(self, arg):
        'Show current instruction and a few before/after it.'
        self.show_instruction_with_context(self.cpu.pc)

    @unhalted
    def do_stepi(self, arg):
        'Execute one machine instruction.'
        self.machine.tick()

        self.show_auto_status()

    @unhalted
    def do_nexti(self, arg):
        'Execute one machine instruction, skipping over calls.'
        if not self.cpu.next():
            print('Hit breakpoint')

        self.show_auto_status()

    def do_cur(self, arg):
        'Show current statement'
        stmt = self.find_nonempty_stmt(self.cpu.pc)
        if stmt is None:
            assert self.cpu.halted
            print('Program is finished')
            return
        self.show_statement_with_context(stmt)

    @unhalted
    def do_step(self, arg):
        'Execute one source statement'

        stmt = self.find_nonempty_stmt(self.cpu.pc)

        def step_breakpoint(cpu):
            next_stmt = self.find_nonempty_stmt(cpu.pc)
            if next_stmt and next_stmt != stmt:
                return True
            return False

        self.cpu.add_breakpoint(step_breakpoint)
        try:
            if not self.cpu.run():
                print('Hit breakpoint')
        finally:
            self.cpu.del_breakpoint(step_breakpoint)

        self.show_auto_status()

    @unhalted
    def do_next(self, arg):
        stmt = self.find_nonempty_stmt(self.cpu.pc)
        while not self.cpu.halted:
            if not self.cpu.next():
                print('Hit breakpoint')
                break
            next_stmt = self.find_nonempty_stmt(self.cpu.pc)
            if next_stmt and next_stmt != stmt:
                break

        self.show_auto_status()

    def do_break(self, arg):
        """
Add breakpoint. If argument starts with 0x, it is assumed to be an
address to break at. If a decimal number, it is assumed to be a line
number to break at. Otherwise, it is assumed to be a sub or function
name to break at.

        """
        if not arg:
            for bp in self.cpu.breakpoints:
                print('Breakpoint at:', bp)
            if not self.cpu.breakpoints:
                print('No breakpoints')
            return

        bp, err = self.parse_breakpoint_spec(arg)
        if bp is None:
            print(f'Cannot set breakpoint: {err}')
            return

        print(f'Setting a breakpoint at {bp}')
        self.cpu.add_breakpoint(bp)

    def do_delbr(self, arg):
        'Delete breakpoint'
        bp, err = self.parse_breakpoint_spec(arg)
        if bp is None:
            print(f'Error: {err}')
            return

        for existing_bp in self.cpu.breakpoints:
            if bp == existing_bp:
                print(f'Deleting breakpoint at {existing_bp}')
                self.cpu.del_breakpoint(existing_bp)
                break
        else:
            print('No such breakpoint')

    def do_bt(self, arg):
        'Print backtrace'
        frames = []

        frame = self.cpu.cur_frame
        while frame:
            routine_record = self.find_routine(frame.code_start)
            if routine_record is None:
                routine_node = self.debug_info.main_routine
            else:
                routine_node = routine_record.node

            frames.append((frame, routine_record, routine_node))
            frame = frame.prev_frame

        frames.reverse()

        for i, (frame, routine_record, routine_node) in enumerate(frames):
            fidx = len(frames) - i

            if routine_record:
                stmt = self.find_stmt(frame.caller_addr)
                caller_line_no = stmt.source_start_line
                print(f'line {caller_line_no} ')
                print('   ',
                      self.source_lines[caller_line_no - 1].strip())

                print(f'[{fidx}] {routine_record.type.name} '
                      f'{routine_node.name} ', end='')
            else:
                print(f'[{fidx}] {routine_node.name} ', end='')

        stmt = self.find_stmt(self.cpu.pc)
        line_no = stmt.source_start_line
        print(f'line {line_no} ')
        print('   ', self.source_lines[line_no - 1].strip())

    def do_print(self, arg):
        'Print the value of a variable'
        pass

    def do_quit(self, arg):
        'Exit debugger'
        return True

    def do_EOF(self, arg):
        'Exit debugger'
        return True


def main():
    parser = argparse.ArgumentParser(
        description='QVM Debugger')

    parser.add_argument(
        'module_file', help='The module to debug')

    args = parser.parse_args()

    with open(args.module_file, 'rb') as f:
        module = QModule.parse(f.read())

    machine = QvmMachine(module, terminal='smart')
    Cmd(machine, module).cmdloop()


if __name__ == '__main__':
    main()
