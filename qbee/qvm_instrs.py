import struct
from dataclasses import dataclass


class DisassemblerError(Exception):
    pass


class Operand:
    def __init__(self, consts, data, labels):
        self.consts = consts
        self.data = data
        self.labels = labels

    def decode(self, data: bytes) -> str:
        assert isinstance(data, bytes)
        assert len(data) == self.size
        return str(self._decode(data))

    def encode(self, value) -> bytes:
        assert isinstance(value, self.py_type)
        return bytes(self._encode(value))


class UInt8(Operand):
    size = 1
    py_type = int

    def _decode(self, data):
        return data[0]

    def _encode(self, value):
        assert 0 <= value <= 255
        return bytes([value])



class Int16(Operand):
    size = 2
    py_type = int

    def _decode(self, data):
        return struct.unpack('>h', data)[0]

    def _encode(self, value):
        assert -32768 <= value <= 32767
        return struct.pack('>h', value)


class UInt16(Operand):
    size = 2
    py_type = int

    def _decode(self, data):
        return struct.unpack('>H', data)[0]

    def _encode(self, value):
        assert 0 <= value <= 65535
        return struct.pack('>H', value)


class Int32(Operand):
    size = 4
    py_type = int

    def _decode(self, data):
        return struct.unpack('>i', data)[0]

    def _encode(self, value):
        assert -2 ** 31 <= value <= 2 ** 31 - 1
        return struct.pack('>i', value)


class Label(Operand):
    size = 4
    py_type = int

    def _decode(self, data):
        return hex(struct.unpack('>I', data)[0])

    def _encode(self, value):
        assert 0 <= value <= 2 ** 32 - 1
        dest = self.labels[value]
        return struct.pack('>I', dest)


class Float32(Operand):
    size = 4
    py_type = float

    def _decode(self, data):
        return struct.unpack('>f', data)[0]

    def _encode(self, value):
        return struct.pack('>f', value)


class Float64(Operand):
    size = 8
    py_type = float

    def _decode(self, data):
        return struct.unpack('>d', data)[0]

    def _encode(self, value):
        return struct.pack('>d', value)


class StringLiteral(Operand):
    size = 2
    py_type = str

    def _decode(self, data):
        idx = struct.unpack('>h', data)[0]
        value = self.consts[idx]
        return value

    def _encode(self, value):
        idx = self.consts.index(value)
        return struct.pack('>H', idx)


@dataclass
class Instr:
    op: str
    op_code: int
    operands: list


instructions: list[Instr] = []
op_code_to_instr: dict[int, Instr] = {}
op_to_instr: dict[str, Instr] = {}


def def_instr(op, op_code, operands=None):
    operands = operands or []
    instruction = Instr(op, op_code, operands)
    instructions.append(instruction)

    assert op_code not in op_code_to_instr, \
        f'Duplicate op code: {op_code}'
    op_code_to_instr[op_code] = instruction

    assert op not in op_to_instr, f'Duplicate op: {op}'
    op_to_instr[op] = instruction


def_instr('add', 2)
def_instr('allocarr', 101, [UInt8])
def_instr('and', 3)
def_instr('arridx', 4, [UInt8])
def_instr('call', 5, [Label])
def_instr('cmp', 105)
def_instr('conv%&', 6)
def_instr('conv%!', 7)
def_instr('conv%#', 8)
def_instr('conv&%', 9)
def_instr('conv&!', 10)
def_instr('conv&#', 11)
def_instr('conv!%', 12)
def_instr('conv!&', 13)
def_instr('conv!#', 14)
def_instr('conv#%', 15)
def_instr('conv#&', 16)
def_instr('conv#!', 17)
def_instr('deref', 18)
def_instr('div', 19)
def_instr('dupl', 103)
def_instr('eq', 20)
def_instr('eqv', 21)
def_instr('exp', 22)
def_instr('frame', 23, [UInt16, UInt16])
def_instr('ge', 24)
def_instr('gt', 102)
def_instr('halt', 100)
def_instr('idiv', 25)
def_instr('ijmp', 109)
def_instr('int', 111)
def_instr('imp', 26)
def_instr('io', 27, [UInt8, UInt8])
def_instr('jmp', 28, [Label])
def_instr('jz', 29, [Label])
def_instr('le', 30)
def_instr('lt', 31)
def_instr('mod', 32)
def_instr('mul', 33)
def_instr('ne', 34)
def_instr('neg', 35)
def_instr('nop', 36)
def_instr('not', 37)
def_instr('or', 38)
def_instr('pop', 104)
def_instr('push%', 39, [Int16])
def_instr('push&', 40, [Int32])
def_instr('push!', 41, [Float32])
def_instr('push#', 42, [Float64])
def_instr('push$', 43, [StringLiteral])
def_instr('pushm2%', 44)
def_instr('pushm2&', 45)
def_instr('pushm2!', 46)
def_instr('pushm2#', 47)
def_instr('pushm1%', 48)
def_instr('pushm1&', 49)
def_instr('pushm1!', 50)
def_instr('pushm1#', 51)
def_instr('push0%', 52)
def_instr('push0&', 53)
def_instr('push0!', 54)
def_instr('push0#', 55)
def_instr('push1%', 56)
def_instr('push1&', 57)
def_instr('push1!', 58)
def_instr('push1#', 59)
def_instr('push2%', 60)
def_instr('push2&', 61)
def_instr('push2!', 62)
def_instr('push2#', 63)
def_instr('pushrefg', 64)
def_instr('pushrefl', 65)
def_instr('readg%', 66, [UInt16])
def_instr('readg&', 67, [UInt16])
def_instr('readg!', 68, [UInt16])
def_instr('readg#', 69, [UInt16])
def_instr('readg$', 70, [UInt16])
def_instr('readg@', 71, [UInt16])
def_instr('readl%', 72, [UInt16])
def_instr('readl&', 73, [UInt16])
def_instr('readl!', 74, [UInt16])
def_instr('readl#', 75, [UInt16])
def_instr('readl$', 76, [UInt16])
def_instr('readl@', 77, [UInt16])
def_instr('readidxg%', 78, [UInt16, UInt16])
def_instr('readidxg&', 79, [UInt16, UInt16])
def_instr('readidxg!', 80, [UInt16, UInt16])
def_instr('readidxg#', 81, [UInt16, UInt16])
def_instr('readidxg$', 82, [UInt16, UInt16])
def_instr('readidxg@', 83, [UInt16, UInt16])
def_instr('readidxl%', 84, [UInt16, UInt16])
def_instr('readidxl&', 85, [UInt16, UInt16])
def_instr('readidxl!', 86, [UInt16, UInt16])
def_instr('readidxl#', 87, [UInt16, UInt16])
def_instr('readidxl$', 88, [UInt16, UInt16])
def_instr('readidxl@', 89, [UInt16, UInt16])
def_instr('refidx', 90)
def_instr('ret', 91)
def_instr('retv', 92)
def_instr('sign', 106)
def_instr('space', 112)
def_instr('sub', 93)
def_instr('storeg', 94, [UInt16])
def_instr('storel', 95, UInt16)
def_instr('storeidxg', 96, [UInt16, UInt16])
def_instr('storeidxl', 97, [UInt16, UInt16])
def_instr('storeref', 98)
def_instr('strlen', 110)
def_instr('swap', 107)
def_instr('swapprev', 108)
def_instr('xor', 99)
