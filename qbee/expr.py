from enum import Enum
from abc import ABC, abstractmethod
from .exceptions import InternalError
from .node import Node


class Type(Enum):
    INTEGER = 1
    LONG = 2
    SINGLE = 3
    DOUBLE = 4
    STRING = 5

    USER_DEFINED = 100
    UNKNOWN = 200

    @staticmethod
    def type_chars():
        # we have to make this a method, because if we make it a
        # property, it would become a member of the enum
        return '%&!#$'

    @property
    def is_numeric(self):
        return self in (
            Type.INTEGER,
            Type.LONG,
            Type.SINGLE,
            Type.DOUBLE,
        )

    @property
    def py_type(self):
        if self == Type.USER_DEFINED or self == Type.UNKNOWN:
            raise ValueError('Cannot get py_type of {self}')

        return {
            Type.INTEGER: int,
            Type.LONG: int,
            Type.SINGLE: float,
            Type.DOUBLE: float,
            Type.STRING: str,
        }[self]

    @property
    def type_char(self):
        if self == Type.USER_DEFINED or self == Type.UNKNOWN:
            raise ValueError('Cannot get type_char of {self}')

        return {
            Type.INTEGER: '%',
            Type.LONG: '&',
            Type.SINGLE: '!',
            Type.DOUBLE: '#',
            Type.STRING: '$',
        }[self]

    def is_coercible_to(self, other):
        if self == other:
            return True
        if self.is_numeric and other.is_numeric:
            return True
        return False

    @staticmethod
    def from_type_char(type_char):
        return {
            '%': Type.INTEGER,
            '&': Type.LONG,
            '!': Type.SINGLE,
            '#': Type.DOUBLE,
            '$': Type.STRING,
        }[type_char]


class Operator(Enum):
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    MOD = 5
    INTDIV = 6
    EXP = 7
    CMP_EQ = 8
    CMP_NE = 9
    CMP_LT = 10
    CMP_GT = 11
    CMP_LE = 12
    CMP_GE = 13
    NEG = 14
    PLUS = 15 # unary +
    NOT = 16
    AND = 17
    OR = 18
    XOR = 19
    EQV = 20
    IMP = 21

    @property
    def is_unary(self):
        return self in (
            Operator.NEG,
            Operator.PLUS,
            Operator.NOT,
        )

    @property
    def is_logical(self):
        return self in (
            Operator.NOT,
            Operator.AND,
            Operator.OR,
            Operator.XOR,
            Operator.EQV,
            Operator.IMP,
        )

    @property
    def is_comparison(self):
        return self in (
            Operator.CMP_EQ,
            Operator.CMP_NE,
            Operator.CMP_LT,
            Operator.CMP_GT,
            Operator.CMP_LE,
            Operator.CMP_GE,
        )

    @staticmethod
    def binary_op_from_token(token: str):
        return {
            '+': Operator.ADD,
            '-': Operator.SUB,
            '*': Operator.MUL,
            '/': Operator.DIV,
            'mod': Operator.MOD,
            '\\': Operator.INTDIV,
            '^': Operator.EXP,
            '=': Operator.CMP_EQ,
            '<>': Operator.CMP_NE,
            '><': Operator.CMP_NE,
            '<=': Operator.CMP_LT,
            '=<': Operator.CMP_LT,
            '>=': Operator.CMP_GT,
            '=>': Operator.CMP_GT,
            '<': Operator.CMP_LE,
            '>': Operator.CMP_GE,
            'and': Operator.AND,
            'or': Operator.OR,
            'xor': Operator.XOR,
            'eqv': Operator.EQV,
            'imp': Operator.IMP,
        }[token]

    @staticmethod
    def unary_op_from_token(token: str):
        return {
            'not': Operator.NOT,
            '-': Operator.NEG,
            '+': Operator.PLUS,
        }[token]


class Expr(Node):
    # These will be set by the relevant parse action to the start and
    # end indices of the expression in the input string. Since we only
    # wrap the "expr" rule in Located, these will only be available
    # after the expression is parsed, so for example the parse action
    # for NumericLiteral will not have access to these.
    loc_start = loc_end = None

    @property
    @abstractmethod
    def type(self) -> Type:
        pass

    @property
    @abstractmethod
    def is_literal(self) -> Type:
        pass

    @property
    @abstractmethod
    def is_const(self) -> Type:
        pass

    def eval(self):
        if not self.is_const:
            raise ValueError(
                'Attempting to evaluate non-const expression')

        raise InternalError(
            f'eval method not implemented for type '
            f'"{type(self).__name__}".')

    def fold(self):
        if self.is_const:
            value = self.eval()
            if self.type.is_numeric:
                literal = NumericLiteral(value, self.type)
            else:
                literal = StringLiteral(value)
            return literal
        return self


class NumericLiteral(Expr):
    is_literal = True
    is_const = True

    DEFAULT_TYPE = Type.SINGLE

    def __init__(self, value, type:Type=None):
        if type is None:
            type = self.DEFAULT_TYPE
        self._type = type
        self.value = self._type.py_type(value)

    def __repr__(self):
        typechar = self.type.type_char
        return f'<NumericLiteral {self.value}{typechar}>'

    @property
    def type(self):
        return self._type

    def eval(self):
        return self.type.py_type(self.value)

    @classmethod
    def parse(cls, token: str, type_char=None):
        if type_char:
            assert type_char in Type.type_chars()
            literal_type = Type.from_type_char(type_char)
        else:
            literal_type = NumericLiteral.DEFAULT_TYPE

        value = literal_type.py_type(token)
        return cls(value, literal_type)

    @property
    def children(self):
        return []

    def replace_child(self, new_child):
        pass


class BinaryOp(Expr):
    is_literal = False

    def __init__(self, left, right, op: Operator):
        assert not op.is_unary
        self.left = left
        self.right = right
        self.op = op

    def __repr__(self):
        return (
            f'<BinaryOp op={self.op.name} '
            f'left={self.left} '
            f'right={self.right}>'
        )

    def replace_child(self, old_child, new_child):
        if self.left == old_child:
            self.left = new_child
        elif self.right == old_child:
            self.right = new_child
        else:
            raise InternalError('No such child to replace')

    @property
    def children(self):
        return [self.left, self.right]

    @property
    def is_const(self):
        return self.left.is_const and self.right.is_const

    @property
    def type(self):
        if self.op.is_logical:
            if not self.left.type.is_numeric or \
               not self.right.type.is_numeric:
                return Type.UNKNOWN
            if self.left.type == self.right.type == Type.INTEGER:
                return Type.INTEGER
            else:
                return Type.LONG
        if self.op.is_comparison:
            return Type.INTEGER

        ltype = self.left.type
        rtype = self.right.type

        if (ltype == Type.STRING and rtype != Type.STRING) or \
           (rtype == Type.STRING and ltype != Type.STRING):
            return Type.UNKNOWN

        if ltype == rtype == Type.STRING and self.op != Operator.ADD:
            return Type.UNKNOWN

        if self.op == Operator.MOD:
            if not ltype.is_numeric or not rtype.is_numeric:
                return Type.UNKNOWN

            # MOD always coerces its args to an integer and then
            # calculates the result which is always an integral value.
            if ltype == Type.INTEGER and rtype == Type.INTEGER:
                return Type.INTEGER
            else:
                return Type.LONG

        if ltype == Type.UNKNOWN or rtype == Type.UNKNOWN:
            return Type.UNKNOWN
        if ltype == Type.USER_DEFINED or rtype == Type.USER_DEFINED:
            return Type.UNKNOWN

        if ltype == Type.DOUBLE or rtype == Type.DOUBLE:
            return Type.DOUBLE
        if ltype == Type.SINGLE or rtype == Type.SINGLE:
            return Type.SINGLE
        if ltype == Type.LONG or rtype == Type.LONG:
            return Type.LONG
        if ltype == Type.INTEGER or rtype == Type.INTEGER:
            return Type.INTEGER

        return Type.STRING

    def eval(self):
        if not self.is_const:
            raise InternalError(
                'Attempting to evaluate non-const expression')

        if self.left.type.is_numeric and self.right.type.is_numeric:
            return self._eval_numeric()
        elif self.left.type == Type.STRING and \
             self.right.type == Type.STRING:
            return self._eval_string()
        else:
            raise InternalError(
                'Attempting to evaluate binary operation on '
                'non-primitive values')

    def _eval_numeric(self):
        left = self.type.py_type(self.left.eval())
        right = self.type.py_type(self.right.eval())

        if self.left == Type.INTEGER and self.right == Type.INTEGER:
            mask = 0xffff
        else:
            mask = 0xffff_ffff

        result = {
            Operator.ADD: lambda a, b: a + b,
            Operator.SUB: lambda a, b: a - b,
            Operator.MUL: lambda a, b: a * b,
            Operator.DIV: lambda a, b: a / b,
            Operator.MOD: self._qb_mod,
            Operator.INTDIV: lambda a, b: a // b,
            Operator.EXP: lambda a, b: a ** b,
            Operator.CMP_EQ: lambda a, b: a == b,
            Operator.CMP_NE: lambda a, b: a != b,
            Operator.CMP_LT: lambda a, b: a < b,
            Operator.CMP_GT: lambda a, b: a > b,
            Operator.CMP_LE: lambda a, b: a <= b,
            Operator.CMP_GE: lambda a, b: a >= b,
            Operator.AND: lambda a, b: (a & b) & mask,
            Operator.OR: lambda a, b: (a | b) & mask,
            Operator.XOR: lambda a, b: (a ^ b) & mask,
            Operator.EQV: lambda a, b: ~(a ^ b) & mask,
            Operator.IMP: lambda a, b: (~a | b) & mask,
        }[self.op](left, right)

        return result

    def _eval_string(self):
        if self.op != Operator.ADD:
            raise InternalError(
                'Attempting to evaluate invalid operation on strings')

        return self.left.eval() + self.right.eval()

    def _qb_mod(self, a, b):
        a = int(round(a))
        b = int(round(b))
        return a % b


class UnaryOp(Expr):
    is_literal = False

    def __init__(self, arg, op: Operator):
        assert op.is_unary
        self.arg = arg
        self.op = op

    def __repr__(self):
        return f'<UnaryOp op={self.op.name} arg={self.arg}>'

    def replace_child(self, old_child, new_child):
        if self.arg == old_child:
            self.arg = new_child
        else:
            raise InternalError('No such child to replace')

    @property
    def children(self):
        return [self.arg]

    @property
    def is_const(self):
        return self.arg.is_const

    @property
    def type(self):
        if self.op.is_logical:
            if self.arg.type == Type.INTEGER:
                return Type.INTEGER
            else:
                return Type.LONG
        else:
            return self.arg.type

    def eval(self):
        if not self.is_const:
            raise InternalError(
                'Attempting to evaluate non-const expression')

        value = self.arg.eval()
        if self.op == Operator.NOT:
            value = int(round(value))
            value = ~value & 0xffff_ffff
        elif self.op == Operator.NEG:
            value = -value
        elif self.op == Operator.PLUS:
            pass
        else:
            raise InternalError('Unknown unary operator')

        return value


class Identifier(Expr):
    is_literal = False

    def __init__(self, name, type:Type=None):
        self.name = name
        self._type = type

    def __repr__(self):
        return f'<Identifier {self.name}>'

    def replace_child(self, old_child, new_child):
        pass

    @property
    def children(self):
        return []

    @property
    def type(self):
        if self._type:
            return self._type

        return self.compiler.get_identifier_type(self.name)

    @property
    def is_const(self):
        return self.compiler.is_const(self.name)

    def eval(self):
        if not self.is_const:
            raise InternalError(
                'Attempting to evaluate non-const expression')

        return self.compiler.get_const_value(self.name)


class StringLiteral(Expr):
    is_literal = True
    is_const = True

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'<StringLiteral {self.value}>'

    @property
    def type(self):
        return Type.STRING

    def eval(self):
        return self.value

    def replace_child(self, old_child, new_child):
        pass

    @property
    def children(self):
        return []
