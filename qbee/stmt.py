from abc import ABC, abstractmethod


class Stmt(ABC):
    compiler = None

    # These will be set by the relevant parse action to the start and
    # end indices of the statement in the input string. Since we only
    # wrap the "stmt" rule in Located, these will only be available
    # after the expression is parsed, so for example the parse action
    # for CallStmt will not have access to these.
    loc_start = loc_end = None

    @abstractmethod
    def compile(self):
        pass

class BeepStmt(Stmt):
    def __repr__(self):
        return '<BeepStmt>'

    def compile(self):
        return [('BEEP',)]


class CallStmt(Stmt):
    def __init__(self, name, args):
        self.name = name
        self.args = args

    def compile(self):
        ret = []
        for arg in self.args:
            ret += arg.compile()
        ret.append(('PUSHARGSLEN', len(self.args)))
        ret.append(('CALL',))
        return ret

    def __repr__(self):
        return f'<CallStmt {self.name} args={self.args}>'


class ClsStmt(Stmt):
    def __repr__(self):
        return '<ClsStmt>'

    def compile(self):
        return [('CLS',)]
