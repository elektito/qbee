from enum import Enum


class InternalError(Exception):
    """Errors of this type are considered bugs in the compiler. Ideally,
this should never be raised. If it is, there is a bug that needs to be
fixed.

    """
    pass


class SyntaxError(Exception):
    def __init__(self, loc, msg):
        self.loc_start = loc
        self.loc_end = None
        self.msg = msg

    def __repr__(self):
        return f'{self.msg}'

    def __str__(self):
        return repr(self)


class CodeGenError(Exception):
    pass


class ErrorCode(Enum):
    TYPE_MISMATCH = 'Type mismatch'


class CompileError(Exception):
    def __init__(self, err_code: ErrorCode, msg=None, *, node=None,
                 loc_start=None, loc_end=None):
        if not msg:
            msg = err_code.value
        elif not isinstance(msg, str):
            raise InternalError(
                'Invalid exception message: must be a string')
        self.msg = msg
        self.node = node
        self.loc_start = loc_start
        self.loc_end = loc_end

        if self.loc_start is None:
            self.loc_start = getattr(self.node, 'loc_start', None)
        if self.loc_end is None:
            self.loc_end = getattr(self.node, 'loc_end', None)

    def __repr__(self):
        return self.msg

    def __str__(self):
        return repr(self)
