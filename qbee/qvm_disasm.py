import argparse
import struct
from .qvm_instrs import op_code_to_instr
from .expr import Type
from .utils import eprint


def perror(msg):
    eprint(msg)
    exit(1)


def disassemble(bcode: bytes):
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
        if section_type == 0x01:  # const section
            nconsts, = struct.unpack('>H', bcode[idx:idx+2])
            idx += 2
            for i in range(nconsts):
                size, = struct.unpack('>H', bcode[idx:idx+2])
                idx += 2
                const_value = bcode[idx:idx+size]
                idx += size
                consts.append(const_value.decode('cp437'))
        elif section_type == 0x02:  # data section
            nparts, = struct.unpack('>H', bcode[idx:idx+2])
            idx += 2
            for i in range(nparts):
                part = []
                nitems, = struct.unpack('>H', bcode[idx:idx+2])
                idx += 2
                for j in range(nitems):
                    size, = struct.unpack('>H', bcode[idx:idx+2])
                    idx += 2
                    item = bcode[idx:idx+size]
                    idx += size
                    part.append(item.decode('cp437'))
                data_parts.append(part)
        elif section_type == 0x03:  # code section
            code_size, = struct.unpack('>I', bcode[idx:idx+4])
            idx += 4
            code= decode_code(bcode[idx:idx+code_size], consts)
            idx += code_size
        else:
            perror(f'Unknown section id: {section_type}')

    if idx != len(bcode):
        remaining = len(bcode) - idx
        perror(f'Extra data at the end: {remaining} byte(s) remaining')

    return code


def decode_code(bcode, consts):
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
        if op in ('call', 'jmp', 'jz'):
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
            _type = Type.from_type_char(type_char)
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
                comments = f'"{consts[value]}"'
            args = [value]
        elif op in ('pushrefg', 'pushrefl'):
            var_idx, = struct.unpack('>H', bcode[idx:idx+2])
            idx += 2
            args = [var_idx]
        elif op in ('readg', 'readl', 'storeg', 'storel'):
            var_idx, = struct.unpack('>H', bcode[idx:idx+2])
            idx += 2
            args = [var_idx]
        elif op in ('readidxg',
                    'readidxl',
                    'storeidxg',
                    'storeidxl'):
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


def main():
    parser = argparse.ArgumentParser(
        description='Disassemble QVM code')

    parser.add_argument('input', help='File to read code from')
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        bcode = f.read()

    result = disassemble(bcode)
    print(result)


if __name__ == '__main__':
    main()
