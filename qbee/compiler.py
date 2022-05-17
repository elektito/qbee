from .grammar import program
from .stmt import Stmt, AssignmentStmt
from .expr import Type, Expr, BinaryOp, UnaryOp
from .program import Label
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


class Compiler:
    def __init__(self, optimization_level=0):
        self.optimization_level = optimization_level

        self.cur_routine = Routine('_main')
        self.routines = {'': self.cur_routine}

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
            if isinstance(node, Stmt):
                self._compile_stmt(node)
            elif isinstance(node, Expr):
                self._compile_expr(node)
            elif isinstance(node, Label):
                if node.name in self.cur_routine.labels:
                    raise CompileError(EC.DUPLICATE_LABEL,
                                       f'Duplicate label: {node.name}',
                                       node=node)
                self.cur_routine.labels.add(node.name)
            else:
                raise InternalError(
                    f'Do not know how to compile node: {node}')

    def _compile_stmt(self, stmt):
        if isinstance(stmt, AssignmentStmt):
            if not stmt.lvalue.type.is_coercible_to(stmt.rvalue.type):
                raise CompileError(EC.TYPE_MISMATCH, node=stmt)

        for child in stmt.children:
            if isinstance(child, Expr):
                self._compile_expr(child)
            else:
                self._compile_tree(child)

    def _compile_expr(self, expr):
        for child in expr.children:
            self._compile_expr(child)

        if isinstance(expr, BinaryOp):
            if expr.type == Type.UNKNOWN:
                raise CompileError(EC.TYPE_MISMATCH, node=expr)
        elif isinstance(expr, UnaryOp):
            # unary operators (NOT, +, -) are only valid on numeric
            # expressions.
            if not expr.arg.type.is_numeric:
                raise CompileError(EC.TYPE_MISMATCH, node=expr)
