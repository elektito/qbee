from enum import Enum
from qbee import expr
from qbee.numeric import Int16, Int32, Float32, Float64
from .trap import TrapCode, Trapped


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
            CellType.INTEGER: Int16,
            CellType.LONG: Int32,
            CellType.SINGLE: Float32,
            CellType.DOUBLE: Float64,
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


class CellValue:
    def __init__(self, type, value):
        assert isinstance(type, CellType)

        self.type = type

        expr_type = {
            CellType.INTEGER: expr.Type.INTEGER,
            CellType.LONG: expr.Type.LONG,
            CellType.SINGLE: expr.Type.SINGLE,
            CellType.DOUBLE: expr.Type.DOUBLE,
            CellType.STRING: expr.Type.STRING,
        }.get(self.type)
        if expr_type is not None:
            if not expr_type.can_hold(value):
                raise Trapped(
                    trap_code=TrapCode.INVALID_CELL_VALUE,
                    trap_kwargs={
                        'type': self.type,
                        'value': value,
                    }
                )

            self.value = expr_type.coerce(value)
        else:
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


class Reference:
    def __init__(self, segment, index=None):
        if isinstance(segment, Reference):
            ref = segment
            self.segment = ref.segment
            self.index = ref.index
            return
        else:
            assert index is not None
            self.segment = segment
            self.index = index

    def derefed(self):
        return self.segment.get_cell(self.index)

    def __repr__(self):
        return f'<REF seg={self.segment} idx={self.index}>'
