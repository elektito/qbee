from enum import Enum
from abc import abstractmethod
from .exceptions import ErrorCode as EC, InternalError, CompileError
from .node import Node
from .utils import split_camel


class BuiltinType(Enum):
    INTEGER = 1
    LONG = 2
    SINGLE = 3
    DOUBLE = 4
    STRING = 5

    USER_DEFINED = 100
    UNKNOWN = 200


class Type:
    type_chars = '%&!#$'

    def __init__(self, builtin_type, user_type_name=None):
        self._type = builtin_type
        self.user_type_name = user_type_name
        self.is_array = False
        self.array_dims = None
        self.is_nodim_array = False  # e.g.: x() as integer

    def __eq__(self, other):
        assert isinstance(other, Type)
        if (self._type == BuiltinType.UNKNOWN or
                other._type == BuiltinType.UNKNOWN):
            # unknown types are not equal to anything; even to
            # themselves.
            return False

        if (self.is_user_defined and self.user_type_name is None) or \
           (other.is_user_defined and other.user_type_name is None):
            # Two user-defined types cannot be equal if the name of
            # the user type is not set (not even equal to self)
            return False

        return (
            self._type == other._type and
            self.user_type_name == other.user_type_name and
            self.is_array == other.is_array
        )

    def __hash__(self):
        return hash((self._type, self.user_type_name))

    def __repr__(self):
        if self.is_builtin:
            s = f'Type.{self.name.upper()}'
        else:
            s = f'Type.USER_DEFINED({self.name})'
        if self.is_array:
            s += '()'
        return s

    @property
    def name(self):
        if self.is_builtin:
            ret = self._type.name.lower()
        else:
            ret = self.user_type_name
        if self.is_array:
            ret += '()'
        return ret

    @property
    def is_user_defined(self):
        return (self._type == BuiltinType.USER_DEFINED)

    @property
    def is_static_array(self):
        return (
            self.is_array and
            not self.is_nodim_array and
            all(
                d.lbound.is_const and d.ubound.is_const
                for d in self.array_dims
            )
        )

    @property
    def is_dynamic_array(self):
        return self.is_array and not self.is_static_array

    @property
    def is_numeric(self):
        if self.is_array:
            return False
        return self._type in (
            BuiltinType.INTEGER,
            BuiltinType.LONG,
            BuiltinType.SINGLE,
            BuiltinType.DOUBLE,
        )

    @property
    def is_builtin(self):
        return self._type in (
            BuiltinType.INTEGER,
            BuiltinType.LONG,
            BuiltinType.SINGLE,
            BuiltinType.DOUBLE,
            BuiltinType.STRING,
        )

    @property
    def py_type(self):
        if self._type in (BuiltinType.USER_DEFINED,
                          BuiltinType.UNKNOWN):
            raise ValueError('Cannot get py_type of {self}')

        return {
            BuiltinType.INTEGER: int,
            BuiltinType.LONG: int,
            BuiltinType.SINGLE: float,
            BuiltinType.DOUBLE: float,
            BuiltinType.STRING: str,
        }[self._type]

    @property
    def type_char(self):
        if self._type == BuiltinType.USER_DEFINED or \
           self._type == BuiltinType.UNKNOWN:
            raise ValueError(f'Cannot get type_char of {self}')

        return {
            BuiltinType.INTEGER: '%',
            BuiltinType.LONG: '&',
            BuiltinType.SINGLE: '!',
            BuiltinType.DOUBLE: '#',
            BuiltinType.STRING: '$',
        }[self._type]

    @property
    def array_base_type(self):
        assert self.is_array
        return Type(self._type, self.user_type_name)

    @property
    def type_id(self):
        return self._type.value

    def is_coercible_to(self, other):
        if self == other:
            return True
        if self.is_numeric and other.is_numeric:
            return True
        return False

    @property
    def is_integral(self):
        return self._type in (BuiltinType.INTEGER, BuiltinType.LONG)

    @property
    def is_float(self):
        return self._type in (BuiltinType.SINGLE, BuiltinType.DOUBLE)

    @property
    def default_value(self):
        assert not self.is_array
        assert not self.is_user_defined
        if self._type == BuiltinType.STRING:
            return ''
        else:
            return 0

    @classmethod
    @property
    def INTEGER(cls):
        return cls(BuiltinType.INTEGER)

    @classmethod
    @property
    def LONG(cls):
        return cls(BuiltinType.LONG)

    @classmethod
    @property
    def SINGLE(cls):
        return cls(BuiltinType.SINGLE)

    @classmethod
    @property
    def DOUBLE(cls):
        return cls(BuiltinType.DOUBLE)

    @classmethod
    @property
    def STRING(cls):
        return cls(BuiltinType.STRING)

    @classmethod
    @property
    def UNKNOWN(cls):
        return cls(BuiltinType.UNKNOWN)

    @classmethod
    @property
    def builtin_types(cls):
        return (
            cls.INTEGER,
            cls.LONG,
            cls.SINGLE,
            cls.DOUBLE,
            cls.STRING,
        )

    @classmethod
    def user_defined(cls, type_name):
        assert type_name not in cls.builtin_types
        return cls(BuiltinType.USER_DEFINED, type_name)

    @classmethod
    def from_name(cls, type_name):
        builtins = {
            'integer': Type.INTEGER,
            'long': Type.LONG,
            'single': Type.SINGLE,
            'double': Type.DOUBLE,
            'string': Type.STRING,
        }
        if type_name in builtins:
            return builtins[type_name]
        else:
            return cls(BuiltinType.USER_DEFINED, type_name)

    @staticmethod
    def name_ends_with_type_char(name: str):
        assert isinstance(name, str)
        return any(name.endswith(c) for c in '%&!#$')

    @staticmethod
    def is_type_char(char):
        return any(char == c for c in Type.type_chars)

    @staticmethod
    def from_type_char(type_char):
        return {
            '%': Type.INTEGER,
            '&': Type.LONG,
            '!': Type.SINGLE,
            '#': Type.DOUBLE,
            '$': Type.STRING,
        }[type_char]

    @staticmethod
    def get_type_size(type, user_types):
        if type.is_array:
            return 1

        from .stmt import TypeBlock
        assert all(
            isinstance(k, str) and isinstance(v, TypeBlock)
            for k, v in user_types.items()
        )

        builtin_types = [t.name for t in Type.builtin_types]
        if type.name in builtin_types:
            return 1
        else:
            struct = user_types[type.name]
            return sum(
                Type.get_type_size(ftype, user_types)
                for ftype in struct.fields.values()
            )


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
    PLUS = 15  # unary +
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

    @classmethod
    def type_name(cls):
        name = cls.__name__
        parts = split_camel(name)
        name = ' '.join(parts)
        return name.upper()

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


class ParenthesizedExpr(Expr):
    is_literal = False

    def __init__(self, child_expr):
        self.child = child_expr

    def __repr__(self):
        return f'<Paren {self.child}>'

    @property
    def type(self):
        return self.child.type

    @property
    def is_const(self):
        return self.child.is_const

    def eval(self):
        return self.child.eval()

    @property
    def children(self):
        return [self.child]

    def replace_child(self, old_child, new_child):
        if self.child == old_child:
            self.child = new_child
            return

        raise InternalError('No such child to replace')


class NumericLiteral(Expr):
    is_literal = True
    is_const = True

    DEFAULT_TYPE = Type.SINGLE

    def __init__(self, value, type: Type = None):
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
            assert type_char in Type.type_chars
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
        if ltype.is_user_defined or rtype.is_user_defined:
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
        elif (self.left.type == Type.STRING and
              self.right.type == Type.STRING):
            return self._eval_string()
        else:
            raise InternalError(
                'Attempting to evaluate binary operation on '
                'non-primitive values')

    def _eval_numeric(self):
        left = self.type.py_type(self.left.eval())
        right = self.type.py_type(self.right.eval())

        if self.left.type == Type.INTEGER and \
           self.right.type == Type.INTEGER:
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
            value = ~value
        elif self.op == Operator.NEG:
            value = -value
        elif self.op == Operator.PLUS:
            pass
        else:
            raise InternalError('Unknown unary operator')

        if self.arg.type == Type.INTEGER:
            max_positive_int = 2**15 - 1
            max_negative_int = -2**15
        else:
            max_positive_int = 2**31 - 1
            max_negative_int = -2**31

        if value > max_positive_int or value < max_negative_int:
            value = max_negative_int

        return value


class Lvalue(Expr):
    is_literal = False

    def __init__(self, base_var, array_indices, dotted_vars):
        # x
        # Lvalue(base_var='x', array_indices=[], dotted_vars=[])
        #
        # x$
        # Lvalue(base_var='x$', array_indices=[], dotted_vars=[])
        #
        # x.y.z
        # Lvalue(base_var='x', array_indices=[], dotted_vars=['y', 'z'])
        #
        # x(1, 2)
        # Lvalue(base_var='x',
        #        array_indices=[NumericLiteral(1), NumericLiteral(2)],
        #        dotted_vars=['y', 'z'])
        #
        # x(5).y
        # Lvalue(base_var='x',
        #        array_indices=[NumericLiteral(5)],
        #        dotted_vars=['y'])

        assert isinstance(base_var, str)
        assert isinstance(array_indices, list)
        assert isinstance(dotted_vars, list)

        self.base_var = base_var
        self.array_indices = array_indices
        self.dotted_vars = dotted_vars

    def __repr__(self):
        ret = '<Lvalue ' + self.base_var
        if self.array_indices:
            ret += f' arridx={self.array_indices}'
        if self.dotted_vars:
            ret += f' dots={".".join(self.dotted_vars)}'
        ret += '>'
        return ret

    @property
    def is_const(self):
        if self.array_indices or self.dotted_vars:
            return False

        return self.compiler.is_const(self.base_var)

    @property
    def type(self):
        var_type = self.base_type

        if var_type.is_array and self.array_indices:
            var_type = var_type.array_base_type

        # should we do this?
        #
        # if self.dotted_vars and not var_type.is_user_defined:
        #     # special case: QBASIC allows a name like x.y.z as a
        #     # single variable name, as long as x is not defined as a
        #     # user-defined type.
        #     var_name = self.base_var + '.'.join(self.dotted_vars)
        #     return self.parent_routine.get_variable_type(var_name)

        for var in self.dotted_vars:
            if not var_type.is_user_defined:
                raise CompileError(
                    EC.INVALID_IDENTIFIER,
                    'Identifier cannot include period',
                    node=self)

            struct = self.compiler.user_types.get(
                var_type.user_type_name)
            if struct is None:
                raise CompileError(
                    EC.INVALID_IDENTIFIER,
                    'Identifier cannot include period',
                    node=self)

            var_type = struct.fields.get(var)
            if var_type is None:
                raise CompileError(
                    EC.ELEMENT_NOT_DEFINED,
                    node=self)

        return var_type

    @property
    def base_type(self):
        return self.parent_routine.get_variable_type(self.base_var)

    @property
    def base_is_ref(self):
        return (
            self.base_var in self.parent_routine.params or
            self.type.is_dynamic_array
        )

    @property
    def children(self):
        return self.array_indices

    def replace_child(self, old_child, new_child):
        for i in range(len(self.array_indices)):
            if self.array_indices[i] == old_child:
                self.array_indices[i] = new_child
                return

        raise InternalError('No such child to replace')

    def eval(self):
        if not self.is_const:
            raise InternalError(
                'Attempting to evaluate non-const expression')

        return self.compiler.consts[self.base_var].eval()


class Variable(Expr):
    is_literal = False
    is_const = False


class StringLiteral(Expr):
    is_literal = True
    is_const = True

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'<StringLiteral "{self.value}">'

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


class ArrayPass(Expr):
    """This class is only used for passing an array to a sub or
function. For example: CALL foo(x())"""

    is_literal = False
    is_const = False

    def __init__(self, identifier):
        self.identifier = identifier

    def __repr__(self):
        return f'<ArrayPass {self.identifier}()>'

    @property
    def type(self):
        return self.parent_routine.get_variable_type(self.identifier)

    @property
    def children(self):
        return []

    def replace_child(self, old_child, new_child):
        pass


class FuncCall(Expr):
    is_literal = False
    is_const = False

    def __init__(self, name: str, type: Type, args: list[Expr]):
        assert isinstance(name, str)
        assert isinstance(type, Type)
        assert isinstance(args, list)
        assert all(isinstance(arg, Expr) for arg in args)

        self.name = name
        self._type = type
        self.args = args

    def __repr__(self):
        return (
            f'<FuncCall {self.name}{self._type.type_char} with '
            f'{len(self.args)} arg(s)>'
        )

    @property
    def type(self):
        return self._type

    @property
    def children(self):
        return self.args

    def replace_child(self, old_child, new_child):
        for i in len(self.args):
            if self.args[i] == old_child:
                self.args[i] = new_child
                return

        raise InternalError('No such child to replace')
