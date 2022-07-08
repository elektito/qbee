import ctypes
from .cell import CellType


def format_number(n, n_type):
    'Convert the given number to a string, the way QB used to do.'
    if n_type == CellType.SINGLE:
        n = ctypes.c_float(n).value
        sn = str(n)
        if '.' in sn and 'e' not in sn:
            digits = len(sn) - 1
            before_decimal = sn.index('.')
            desired_total_digits = 7
            n = round(n, ndigits=desired_total_digits-before_decimal)
    s = str(n)
    if s.endswith('.0'):
        s = s[:-2]
    if 'e' in s and n_type == CellType.DOUBLE:
        s = s.replace('e', 'D')
    elif 'e' in s:
        s = s.replace('e', 'E')

    if n >= 0:
        s = ' ' + s

    return s
