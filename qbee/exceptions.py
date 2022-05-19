from enum import Enum


class InternalError(Exception):
    """Errors of this type are considered bugs in the compiler. Ideally,
this should never be raised. If it is, there is a bug that needs to be
fixed.

    """
    pass


class SyntaxError(Exception):
    def __init__(self, loc, msg=None):
        self.loc_start = loc
        self.loc_end = None
        if msg is None:
            self.msg = 'Syntax Error'

    def __repr__(self):
        return self.msg

    def __str__(self):
        return repr(self)


class CodeGenError(Exception):
    pass


class ErrorCode(Enum):
    TYPE_MISMATCH = 'Type mismatch'
    DUPLICATE_LABEL = 'Duplicate label'
    DUPLICATE_DEFINITION = 'Duplicate definition'
    ILLEGAL_IN_SUB = 'Illegal in sub-routine'


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
