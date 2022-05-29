from enum import Enum


class InternalError(Exception):
    """Errors of this type are considered bugs in the compiler. Ideally,
this should never be raised. If it is, there is a bug that needs to be
fixed.

    """
    pass


class SyntaxError(Exception):
    def __init__(self, loc, msg=None):
        assert isinstance(loc, int)

        self.loc_start = loc
        self.loc_end = None
        if msg is None:
            self.msg = 'Syntax Error'
        else:
            self.msg = msg

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
    INVALID_EXIT = 'EXIT statement in invalid context'
    LABEL_NOT_DEFINED = 'Label not defined'
    ELSE_WITHOUT_IF = 'ELSE without IF'
    SUBPROGRAM_NOT_FOUND = 'Sub-program not found'
    INVALID_IDENTIFIER = 'Invalid identifier'
    ILLEGAL_IN_TYPE_BLOCK = 'Statement illegal in type block'
    ARGUMENT_COUNT_MISMATCH = 'Argument count mismatch'
    ELEMENT_NOT_DEFINED = 'Element not defined'
    WRONG_NUMBER_OF_DIMENSIONS = 'Wrong number of dimensions'
    INVALID_USE_OF_FUNCTION = 'Invalid use of function'
    INVALID_CONSTANT = 'Invalid constant'
    TYPE_NOT_DEFINED = 'Type not defined'


class CompileError(Exception):
    def __init__(self, err_code: ErrorCode, msg=None, *, node=None,
                 loc_start=None, loc_end=None):
        if not msg:
            msg = err_code.value
        elif not isinstance(msg, str):
            raise InternalError(
                'Invalid exception message: must be a string')
        self.code = err_code
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
