import struct
from .instrs import op_code_to_instr


def parse_consts_section(section):
    consts = []
    idx = 0
    while idx < len(section):
        size, = struct.unpack('>H', section[idx:idx+2])
        idx += 2
        const_value = section[idx:idx+size]
        idx += size
        consts.append(const_value.decode('cp437'))

    if idx != len(section):
        perror('Extra data at the end of consts section')

    return consts


def parse_data_section(section):
    data_parts = []
    idx = 0
    nparts, = struct.unpack('>H', section[idx:idx+2])
    idx += 2
    for i in range(nparts):
        part = []
        nitems, = struct.unpack('>H', section[idx:idx+2])
        idx += 2
        for j in range(nitems):
            size, = struct.unpack('>H', section[idx:idx+2])
            idx += 2
            item = section[idx:idx+size]
            idx += size
            part.append(item.decode('cp437'))
        data_parts.append(part)

    if idx != len(section):
        perror('Extra data at the end of data section')

    return data_parts


def parse_globals_section(section):
    if len(section) != 4:
        perror('consts section is not exactly 4 bytes')

    n_global_cells, = struct.unpack('<I', section)

    return n_global_cells


def parse_code_section(section):
    return section


class QModule:
    def __init__(self, consts, n_global_cells, data, code):
        self.consts = consts
        self.data = data
        self.nglobal_cells = n_global_cells
        self.code = code

    def disassemble(self):
        bcode = self.code

        code = ''
        idx = 0
        while idx < len(bcode):
            op_idx = idx
            args = []
            comments = ''
            op_code = bcode[idx]
            idx += 1
            try:
                instr = op_code_to_instr[op_code]
                op = instr.op
            except KeyError:
                perror(f'Unknown op code: {op_code}')
            if op in ('allocarr', 'arridx'):
                n_indices, = struct.unpack('>B', bcode[idx:idx+1])
                idx += 1
                args = [n_indices]
            elif op in ('call', 'jmp', 'jz'):
                dest, = struct.unpack('>I', bcode[idx:idx+4])
                idx += 4
                args = [f'0x{dest:x}']
            elif op == 'frame':
                psize, vsize = struct.unpack(
                    '>HH', bcode[idx:idx+4])
                idx += 4
                args = [psize, vsize]
            elif op == 'io':
                device, device_op = struct.unpack(
                    '>BB', bcode[idx:idx+2])
                idx += 2
                args = [device, device_op]
            elif op[:4] == 'push' and op[4:] in '%&!#$':
                type_char = op[4]
                if type_char == '%':
                    value, = struct.unpack(
                        '>h', bcode[idx:idx+2])
                    idx += 2
                elif type_char == '&':
                    value, = struct.unpack(
                        '>i', bcode[idx:idx+4])
                    idx += 4
                elif type_char == '!':
                    value, = struct.unpack(
                        '>f', bcode[idx:idx+4])
                    idx += 4
                elif type_char == '#':
                    value, = struct.unpack(
                        '>d', bcode[idx:idx+8])
                    idx += 8
                elif type_char == '$':
                    value, = struct.unpack(
                        '>H', bcode[idx:idx+2])
                    idx += 2
                    comments = f'"{self.consts[value]}"'
                args = [value]
            elif op in ('pushrefg', 'pushrefl'):
                var_idx, = struct.unpack('>H', bcode[idx:idx+2])
                idx += 2
                args = [var_idx]
            elif op[:-1] in ('readg', 'readl') or \
                 op in ('storeg', 'storel'):
                var_idx, = struct.unpack('>H', bcode[idx:idx+2])
                idx += 2
                args = [var_idx]
            elif op[-1] in ('readidxg', 'readidxl') or \
                 op in ('storeidxg', 'storeidxl'):
                var_idx, offset = struct.unpack(
                    '>HH', bcode[idx:idx+4])
                idx += 4
                args = [var_idx, offset]

            args = ', '.join(str(i) for i in args)
            if args:
                line = f'{op_idx:08x}: {op: <12} {args: <12}'
            else:
                line = f'{op_idx:08x}: {op: <12}'
            if comments:
                line += f'; {comments}'
            code += f'{line.strip()}\n'

        if idx != len(bcode):
            remaining = len(bcode) - idx
            perror(
                f'Extra data at the end of code section: '
                f'{remaining} byte(s) remaining')

        return code

    @classmethod
    def parse(cls, bcode: bytes):
        consts = []
        data_parts = []
        code = ''

        encountered_sections = set()

        idx = 0
        while idx < len(bcode):
            section_type = bcode[idx]
            if section_type in encountered_sections:
                perror('Multiple segments of the same type not supported')
            encountered_sections.add(section_type)
            idx += 1

            section_len, = struct.unpack('>I', bcode[idx:idx+4])
            idx += 4

            section = bcode[idx:idx+section_len]
            if len(section) != section_len:
                perror('Invalid input module')
            idx += section_len

            if section_type == 0x01:  # const section
                consts = parse_consts_section(section)
            elif section_type == 0x02:  # data section
                data_parts = parse_data_section(section)
            elif section_type == 0x03:  # globals section
                n_global_cells = parse_globals_section(section)
            elif section_type == 0x04:  # code section
                code = parse_code_section(section)
            else:
                perror(f'Unknown section id: {section_type}')

        if idx != len(bcode):
            remaining = len(bcode) - idx
            perror(f'Extra data at the end: {remaining} byte(s) remaining')

        return cls(consts, n_global_cells, data_parts, code)