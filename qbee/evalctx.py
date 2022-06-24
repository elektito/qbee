from abc import ABC, abstractmethod
from .expr import Type
from .stmt import SubBlock, FunctionBlock
from .exceptions import EvalError


class EvaluationContext(ABC):
    def __init__(self):
        self.consts = {}  # maps const names to (const) Expr objects
        self.global_vars = {}  # maps global var names to types
        self.user_types = {}  # maps user type names to type blocks
        self.def_letter_types = {}  # maps a single letter to a type
        self.routines = {}  # maps routine names to Routine objects
        self.main_routine = None

    def is_const(self, name):
        "Return whether the given name is a const or not"
        return name in self.consts

    def get_node_routine(self, node):
        parent = node.parent
        while parent and \
              not isinstance(parent, (SubBlock, FunctionBlock)):
            parent = parent.parent
        if parent is None:
            return self.main_routine
        else:
            routine_name = parent.name
            routine = self.routines.get(routine_name)
            if routine:
                return routine
            raise EvalError(f'No such routine: {routine_name}')

    @abstractmethod
    def eval_lvalue(self, lvalue):
        pass


class Variable:
    def __init__(self, name, type, scope, routine):
        assert isinstance(name, str)
        assert isinstance(type, Type)
        assert scope in ['param', 'local', 'global', 'static']
        assert isinstance(routine, Routine)

        self.name = name
        self.type = type
        self.scope = scope
        self.routine = routine

    def __repr__(self):
        return f'<Var {self.name} {self.scope}>'

    @property
    def full_name(self):
        if self.scope == 'static':
            return f'_static_{self.routine.name}_{self.name}'
        else:
            return self.name

    @property
    def is_global(self):
        return self.scope in ('global', 'static')

    @property
    def is_local(self):
        return self.scope in ('local', 'param')


class Routine:
    "Represents a SUB or a FUNCTION"

    def __init__(self, name, kind, context, params, is_static=False,
                 return_type=None):
        kind = kind.lower()
        assert kind in ('sub', 'function', 'toplevel')

        assert all(
            isinstance(pname, str) and isinstance(ptype, Type)
            for pname, ptype in params
        )
        assert return_type is None or isinstance(return_type, Type)
        assert isinstance(context, EvaluationContext)

        self.context = context
        self.name = name
        self.kind = kind
        self.is_static = is_static
        self.return_type = return_type
        self.params: dict[str, Type] = dict(params)
        self.local_vars: dict[str, Type] = {}
        self.static_vars: dict[str, Type] = {}
        self.labels = set()

    def __repr__(self):
        static = ' STATIC' if self.is_static else ''
        return f'<Routine {self.kind} {self.name}{static}>'

    def get_identifier_type(self, identifier: str):
        if Type.is_type_char(identifier[-1]):
            return Type.from_type_char(identifier[-1])
        else:
            if identifier in self.context.global_vars:
                return self.context.global_vars[identifier]

            def_type = self.context.def_letter_types.get(
                identifier[0].lower())
            if def_type:
                return def_type

            # No matching DEF* statement, so use the default type
            return Type.SINGLE

    def get_variable(self, name: str):
        assert isinstance(name, str)

        if name in self.params:
            var_type = self.params[name]
            scope = 'param'
        elif name in self.local_vars:
            var_type = self.local_vars[name]
            scope = 'local'
        elif name in self.static_vars:
            var_type = self.static_vars[name]
            scope = 'static'
        elif name in self.context.global_vars:
            var_type = self.context.global_vars[name]
            scope = 'global'
        else:
            var_type = self.get_identifier_type(name)
            scope = 'local'

        return Variable(name,
                        type=var_type,
                        scope=scope,
                        routine=self)

    def has_variable(self, name: str):
        return (
            name in self.local_vars or
            name in self.params or
            name in self.static_vars or
            name in self.context.global_vars
        )
