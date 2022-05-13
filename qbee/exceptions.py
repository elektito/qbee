class InternalError(Exception):
    """Errors of this type are considered bugs in the compiler. Ideally,
this should never be raised. If it is, there is a bug that needs to be
fixed.

    """
    pass


class SyntaxError(Exception):
    def __init__(self, loc, msg):
        self.loc = loc
        self.msg = msg

    def __repr__(self):
        return f'Syntax error at {self.loc}: {self.msg}'

    def __str__(self):
        return repr(self)
