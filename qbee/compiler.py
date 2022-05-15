from .grammar import program
from .stmt import Stmt
from .expr import Type, Expr, BinaryOp, UnaryOp
from .program import Label
from .codegen import CodeGen
from .asm import Assembler
from .exceptions import ErrorCode as EC, InternalError, CompileError

# Import codegen implementations to enable them (this is needed even
# though we don't directly use the imported module)
from . import qvm_codegen


class Compiler:
    def __init__(self, optimization_level=0):
        self.optimization_level = optimization_level

        self._codegen = CodeGen('qvm', self)
        self._asm = Assembler(self)

    def compile(self, input_string):
        tree = program.parse_string(input_string, parse_all=True)
        tree = tree[0]
        tree.bind(self)

        self._compile_tree(tree)

        if self.optimization_level > 0:
            tree.fold()

        code = self._codegen.gen_code(tree)

        print(code)

        if self.optimization_level > 1:
            code.optimize()

        module = self._asm.assemble(code)
        return module

    def _compile_tree(self, tree):
        for node in tree.children:
            if isinstance(node, Stmt):
                for child in node.children:
                    if isinstance(child, Expr):
                        self._compile_expr(child)
                    else:
                        self._compile_tree(child)
            elif isinstance(node, Expr):
                self._compile_expr(node)
            elif isinstance(node, Label):
                pass
            else:
                raise InternalError(
                    f'Do not know how to compile node: {node}')

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