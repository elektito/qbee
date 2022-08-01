from enum import Enum


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
    INVALID_CELL_VALUE = 10
    INDEX_OUT_OF_RANGE = 11
    INVALID_DIMENSIONS = 12
    KEYBOARD_INTERRUPT = 13
    DIVISION_BY_ZERO = 14
    UNINITIALIZED_MEM = 15
    NO_RESUME = 16
    ERRHAND_IN_HANDLER = 17
    CANNOT_RESUME = 18


class Trapped(Exception):
    def __init__(self, trap_code, trap_kwargs):
        self.trap_code = trap_code
        self.trap_kwargs = trap_kwargs
