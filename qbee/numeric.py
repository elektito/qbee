import ctypes
from math import inf


def _get_int_binary_method(method_name):
    def method(self, other):
        cls = type(self)
        if not isinstance(other, cls):
            other = cls(other)
        super_method = getattr(super(cls, self), method_name)
        result = super_method(other)
        if result < self.minval or result > self.maxval:
            raise OverflowError
        return cls(self.c_type(result).value)
    return method


def _get_int_unary_method(method_name):
    def method(self):
        cls = type(self)
        super_method = getattr(super(cls, self), method_name)
        result = super_method()
        print('jjjjjj', cls, method_name)
        return cls(self.c_type(result).value)
    return method


class Int16(int):
    qname = 'INTEGER'
    c_type = ctypes.c_short
    bits = 16
    minval = -(2 ** (bits - 1))
    maxval = 2 ** (bits - 1) - 1

    def __new__(cls, value):
        result = super().__new__(cls, value)
        if result < cls.minval or result > cls.maxval:
            raise OverflowError
        return result

    __add__ = _get_int_binary_method('__add__')
    __radd__ = _get_int_binary_method('__radd__')
    __sub__ = _get_int_binary_method('__sub__')
    __rsub__ = _get_int_binary_method('__rsub__')
    __mul__ = _get_int_binary_method('__mul__')
    __rmul__ = _get_int_binary_method('__rmul__')
    __truediv__ = _get_int_binary_method('__floordiv__')
    __rtruediv__ = _get_int_binary_method('__rfloordiv__')
    __floordiv__ = _get_int_binary_method('__floordiv__')
    __rfloordiv__ = _get_int_binary_method('__rfloordiv__')
    __mod__ = _get_int_binary_method('__mod__')
    __rmod__ = _get_int_binary_method('__rmod__')
    __divmod__ = _get_int_binary_method('__divmod__')
    __rdivmod__ = _get_int_binary_method('__rdivmod__')
    __pow__ = _get_int_binary_method('__pow__')
    __rpow__ = _get_int_binary_method('__rpow__')
    __lshift__ = _get_int_binary_method('__lshift__')
    __rlshift__ = _get_int_binary_method('__rlshift__')
    __rshift__ = _get_int_binary_method('__rshift__')
    __rrshift__ = _get_int_binary_method('__rrshift__')
    __and__ = _get_int_binary_method('__and__')
    __rand__ = _get_int_binary_method('__rand__')
    __or__ = _get_int_binary_method('__or__')
    __ror__ = _get_int_binary_method('__ror__')
    __xor__ = _get_int_binary_method('__xor__')
    __rxor__ = _get_int_binary_method('__rxor__')

    __neg__ = _get_int_unary_method('__neg__')
    __pos__ = _get_int_unary_method('__pos__')
    __abs__ = _get_int_unary_method('__abs__')
    __invert__ = _get_int_unary_method('__invert__')

    __round__ = _get_int_unary_method('__round__')
    __trunc__ = _get_int_unary_method('__trunc__')
    __floor__ = _get_int_unary_method('__floor__')
    __ceil__ = _get_int_unary_method('__ceil__')

    def __int__(self):
        return super().__int__()

    def __complex__(self):
        return super().__complex__()

    def __float__(self):
        return super().__float__()

    def __str__(self):
        return str(int(self))

    def __repr__(self):
        return f'{self.qname}({str(int(self))})'


class Int32(int):
    qname = 'LONG'
    c_type = ctypes.c_int
    bits = 32
    minval = -(2 ** (bits - 1))
    maxval = 2 ** (bits - 1) - 1

    def __new__(cls, value):
        result = super().__new__(cls, value)
        if result < cls.minval or result > cls.maxval:
            raise OverflowError
        return result

    __add__ = _get_int_binary_method('__add__')
    __radd__ = _get_int_binary_method('__radd__')
    __sub__ = _get_int_binary_method('__sub__')
    __rsub__ = _get_int_binary_method('__rsub__')
    __mul__ = _get_int_binary_method('__mul__')
    __rmul__ = _get_int_binary_method('__rmul__')
    __truediv__ = _get_int_binary_method('__floordiv__')
    __rtruediv__ = _get_int_binary_method('__rfloordiv__')
    __floordiv__ = _get_int_binary_method('__floordiv__')
    __rfloordiv__ = _get_int_binary_method('__rfloordiv__')
    __mod__ = _get_int_binary_method('__mod__')
    __rmod__ = _get_int_binary_method('__rmod__')
    __divmod__ = _get_int_binary_method('__divmod__')
    __rdivmod__ = _get_int_binary_method('__rdivmod__')
    __pow__ = _get_int_binary_method('__pow__')
    __rpow__ = _get_int_binary_method('__rpow__')
    __lshift__ = _get_int_binary_method('__lshift__')
    __rlshift__ = _get_int_binary_method('__rlshift__')
    __rshift__ = _get_int_binary_method('__rshift__')
    __rrshift__ = _get_int_binary_method('__rrshift__')
    __and__ = _get_int_binary_method('__and__')
    __rand__ = _get_int_binary_method('__rand__')
    __or__ = _get_int_binary_method('__or__')
    __ror__ = _get_int_binary_method('__ror__')
    __xor__ = _get_int_binary_method('__xor__')
    __rxor__ = _get_int_binary_method('__rxor__')

    __neg__ = _get_int_unary_method('__neg__')
    __pos__ = _get_int_unary_method('__pos__')
    __abs__ = _get_int_unary_method('__abs__')
    __invert__ = _get_int_unary_method('__invert__')

    __round__ = _get_int_unary_method('__round__')
    __trunc__ = _get_int_unary_method('__trunc__')
    __floor__ = _get_int_unary_method('__floor__')
    __ceil__ = _get_int_unary_method('__ceil__')

    def __int__(self):
        return super().__int__()

    def __complex__(self):
        return super().__complex__()

    def __float__(self):
        return super().__float__()

    def __str__(self):
        return str(int(self))

    def __repr__(self):
        return f'{self.qname}({str(int(self))})'


def _get_float_binary_method(method_name):
    def method(self, other):
        cls = type(self)
        if not isinstance(other, cls):
            other = cls(other)
        super_method = getattr(super(cls, self), method_name)
        result = super_method(other)
        return cls(self.c_type(result).value)
    return method


def _get_float_unary_method(method_name):
    def method(self):
        cls = type(self)
        super_method = getattr(super(cls, self), method_name)
        result = super_method()
        return cls(self.c_type(result).value)
    return method


class Float32(float):
    qname = 'SINGLE'
    c_type = ctypes.c_float

    def __new__(cls, value):
        if isinstance(value, str):
            value = float(value)
        value = ctypes.c_float(value).value
        return super().__new__(cls, value)

    __add__ = _get_float_binary_method('__add__')
    __radd__ = _get_float_binary_method('__radd__')
    __sub__ = _get_float_binary_method('__sub__')
    __rsub__ = _get_float_binary_method('__rsub__')
    __mul__ = _get_float_binary_method('__mul__')
    __rmul__ = _get_float_binary_method('__rmul__')
    __truediv__ = _get_float_binary_method('__floordiv__')
    __rtruediv__ = _get_float_binary_method('__rfloordiv__')
    __floordiv__ = _get_float_binary_method('__floordiv__')
    __rfloordiv__ = _get_float_binary_method('__rfloordiv__')
    __mod__ = _get_float_binary_method('__mod__')
    __rmod__ = _get_float_binary_method('__rmod__')
    __divmod__ = _get_float_binary_method('__divmod__')
    __rdivmod__ = _get_float_binary_method('__rdivmod__')
    __pow__ = _get_float_binary_method('__pow__')
    __rpow__ = _get_float_binary_method('__rpow__')
    __lshift__ = _get_float_binary_method('__lshift__')
    __rlshift__ = _get_float_binary_method('__rlshift__')
    __rshift__ = _get_float_binary_method('__rshift__')
    __rrshift__ = _get_float_binary_method('__rrshift__')
    __and__ = _get_float_binary_method('__and__')
    __rand__ = _get_float_binary_method('__rand__')
    __or__ = _get_float_binary_method('__or__')
    __ror__ = _get_float_binary_method('__ror__')
    __xor__ = _get_float_binary_method('__xor__')
    __rxor__ = _get_float_binary_method('__rxor__')

    __neg__ = _get_float_unary_method('__neg__')
    __pos__ = _get_float_unary_method('__pos__')
    __abs__ = _get_float_unary_method('__abs__')
    __invert__ = _get_float_unary_method('__invert__')

    __round__ = _get_float_unary_method('__round__')
    __trunc__ = _get_float_unary_method('__trunc__')
    __floor__ = _get_float_unary_method('__floor__')
    __ceil__ = _get_float_unary_method('__ceil__')

    def __int__(self):
        return super().__int__()

    def __complex__(self):
        return super().__complex__()

    def __float__(self):
        return super().__float__()

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return f'{self.qname}({super().__repr__()})'


class Float64(float):
    qname = 'DOUBLE'

    def __repr__(self):
        return f'{self.qname}({super().__repr__()})'
