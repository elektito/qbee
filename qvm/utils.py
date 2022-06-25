from .cell import CellType


def format_number(n, n_type):
    'Convert the given number to a string, the way QB used to do.'
    s = str(n)
    if s.endswith('.0'):
        s = s[:-2]
    if 'e' in s and n_type == CellType.DOUBLE:
        s = s.replace('e', 'D')
    elif 'e' in s:
        s = s.replace('e', 'E')

    if n > 0:
        s = ' ' + s

    return s
