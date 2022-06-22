import argparse
import cmd
from functools import wraps
from .module import QModule
from .machine import QvmMachine
from .debug_info import DebugInfo


class Breakpoint:
    def __init__(self, start_addr, end_addr):
        self.start_addr = start_addr
        self.end_addr = end_addr

    def match(self, addr):
        return self.start_addr <= addr < self.end_addr

    def exact_match(self, addr):
        return addr == self.start_addr

    def __str__(self):
        if self.end_addr - self.start_addr == 1:
            return f'address 0x{self.start_addr:08x}'
        else:
            return (
                f'range 0x{self.start_addr:08x}-0x{self.end_addr:08x}'
            )


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
        super().__init__()
        self.module = module
        self.machine = machine
        self.cpu = machine.cpu
        self.breakpoints = []

        self.instrs = []
        self.load_instructions()

        self.set_prompt()

    def load_instructions(self):
        addr = 0
        while addr < len(self.module.code):
            instr, operands, size = self.cpu.get_instruction_at(addr)
            self.instrs.append((addr, instr, operands))
            addr += size

    def print_next(self):
        instr, operands, size = \
            self.cpu.get_current_instruction()
        operands_list = ''
        if operands:
            operands_list = ' ' + \
                ', '.join(str(i) for i in operands)
        print(f'NEXT INSTR: {instr.op}{operands_list}')

    def set_prompt(self):
        self.prompt = f'(qdb PC={self.cpu.pc:08x}) '

    def parse_breakpoint_spec(self, spec):
        if spec.startswith('0x'):
            addr = int(spec, base=16)
            return Breakpoint(start_addr=addr, end_addr=addr+1), None
        elif spec.isnumeric():
            line_no = int(spec)
            print('NOT SUPPORTED YET')
        else:
            routine_name = spec
            routine = self.module.debug_info.routines.get(routine_name)
            if routine:
                return Breakpoint(
                    start_addr=routine.start_offset,
                    end_addr=routine.end_offset
                ), None
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

    def continue_until(self, until_addr):
        while not self.machine.halted:
            self.machine.tick()
            if any(bp.exact_match(self.cpu.pc)
                   for bp in self.breakpoints):
                print(f'Hit breakpoint')
                break
            if until_addr is not None and self.cpu.pc == until_addr:
                break

    def postcmd(self, stop, line):
        if not stop:
            if not self.machine.halted:
                self.print_next()
        else:
            print('Goodbye!')
        self.set_prompt()
        return stop

    def do_continue(self, arg):
        'Continue until the machine is halted or we hit a breakpoint.'
        self.continue_until(None)

    def do_curi(self, arg):
        'Show current instruction and a few before/after it.'
        self.show_instruction_with_context(self.cpu.pc)

    @unhalted
    def do_stepi(self, arg):
        'Execute one machine instruction.'
        self.machine.tick()

    @unhalted
    def do_nexti(self, arg):
        'Execute one machine instruction, skipping over calls.'
        instr, operands, size = \
            self.cpu.get_current_instruction()
        if instr.op == 'call':
            self.continue_until(self.cpu.pc + size)
        else:
            self.machine.tick()

    @unhalted
    def do_step(self, arg):
        pass

    @unhalted
    def do_next(self, arg):
        pass

    def do_break(self, arg):
        """
Add breakpoint. If argument starts with 0x, it is assumed to be an
address to break at. If a decimal number, it is assumed to be a line
number to break at. Otherwise, it is assumed to be a sub or function
name to break at.

        """
        if not arg:
            print('Need an argument')

        bp, err = self.parse_breakpoint_spec(arg)
        if bp is None:
            print(f'Cannot set breakpoint: {err}')
            return

        print(f'Setting a breakpoint at: {bp}')
        self.breakpoints.append(bp)

    def do_delbr(self, arg):
        'Delete breakpoint'
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
