from .grammar import program
from .stmt import Stmt, AssignmentStmt, SubBlock
from .expr import Type, Expr, BinaryOp, UnaryOp, Identifier
from .program import Label, LineNo
from .codegen import CodeGen
from .exceptions import ErrorCode as EC, InternalError, CompileError

# Import codegen implementations to enable them (this is needed even
# though we don't directly use the imported module)
from . import qvm_codegen


class Routine:
    "Represents a SUB or a FUNCTION"

    def __init__(self, name):
        self.name = name
        self.labels = set()
        self.variables = set()


class Compiler:
    def __init__(self, optimization_level=0):
        self.optimization_level = optimization_level

        self.cur_routine = Routine('_main')
        self.routines = {'_main': self.cur_routine}

        self._codegen = CodeGen('qvm', self)

    def compile(self, input_string):
        tree = program.parse_string(input_string, parse_all=True)
        tree = tree[0]
        tree.bind(self)

        self._compile_tree(tree)

        if self.optimization_level > 0:
            tree.fold()

        code = self._codegen.gen_code(tree)

        if self.optimization_level > 1:
            code.optimize()

        return code

    def is_const(self, name):
        "Return whether the given name is a const or not"
        return False

    def get_identifier_type(self, name: str) -> Type:
        """Return the type of the given name, taking into account the DEF*
        statements encountered so far.

        """

        if Type.is_type_char(name[-1]):
            return Type.from_type_char(name[-1])
        else:
            # for now DEF* statements are not supported, so always the
            # default type
            return Type.SINGLE

    def is_var_global(self, name):
        return False

    def _compile_tree(self, tree):
        for node in tree.children:
            if isinstance(node, AssignmentStmt):
                if not node.lvalue.type.is_coercible_to(node.rvalue.type):
                    raise CompileError(EC.TYPE_MISMATCH, node=node)
            elif isinstance(node, SubBlock):
                if self.cur_routine.name != '_main':
                    raise CompileError(
                        EC.ILLEGAL_IN_SUB,
                        'Sub-routine only allowed in the top-level',
                        node=node)
                if node.name in self.routines:
                    raise CompileError(
                        EC.DUPLICATE_DEFINITION,
                        f'Duplicate sub-routine definition: {node.name}',
                        node=node)
                routine = Routine(node.name)
                self.routines[node.name] = routine
                self.cur_routine = routine

            if isinstance(node, Stmt):
                self._compile_tree(node)
            elif isinstance(node, Expr):
                self._compile_expr(node)
            elif isinstance(node, (LineNo, Label)):
                if isinstance(node, LineNo):
                    name = f'_label_{node.number}'
                else:
                    name = node.name
                if name in self.cur_routine.labels:
                    raise CompileError(EC.DUPLICATE_LABEL,
                                       f'Duplicate label: {name}',
                                       node=node)
                self.cur_routine.labels.add(name)
            else:
                raise InternalError(
                    f'Do not know how to compile node: {node}')

            if isinstance(node, SubBlock):
                self.cur_routine = self.routines['_main']

    def _compile_expr(self, expr):
        for child in expr.children:
            self._compile_expr(child)

        if isinstance(expr, Identifier):
            if expr.name not in self.cur_routine.variables:
                self.cur_routine.variables.add(expr.name)
        elif isinstance(expr, BinaryOp):
            if expr.type == Type.UNKNOWN:
                raise CompileError(EC.TYPE_MISMATCH, node=expr)
        elif isinstance(expr, UnaryOp):
            # unary operators (NOT, +, -) are only valid on numeric
            # expressions.
            if not expr.arg.type.is_numeric:
                raise CompileError(EC.TYPE_MISMATCH, node=expr)
