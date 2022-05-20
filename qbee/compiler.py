from .stmt import Stmt, AssignmentStmt, SubBlock, ExitSubStmt
from .expr import Type, Expr, BinaryOp, UnaryOp, Identifier
from .program import Label, LineNo
from .codegen import CodeGen
from .exceptions import ErrorCode as EC, InternalError, CompileError
from .parser import parse_string

# Import codegen implementations to enable them (this is needed even
# though we don't directly use the imported module)
from . import qvm_codegen


class Routine:
    "Represents a SUB or a FUNCTION"

    def __init__(self, name, type):
        type = type.lower()
        assert type in ('sub', 'function', 'toplevel')

        self.name = name
        self.type = type
        self.labels = set()
        self.variables = set()

    def __repr__(self):
        return f'<Routine {self.type} {self.name}>'


class Compiler:
    def __init__(self, optimization_level=0):
        self.optimization_level = optimization_level

        self.cur_routine = Routine('_main', 'toplevel')
        self.routines = {'_main': self.cur_routine}

        self.all_labels = set()
        self._codegen = CodeGen('qvm', self)

    def compile(self, input_string):
        tree = parse_string(input_string)
        tree.bind(self)

        self._compile_tree(tree, _pass=1)

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

    def _compile_tree(self, tree, _pass):
        # This function recursively processes the given node. Before
        # recursing the children, a "pre" function for the node is
        # called, and afterwards, a "post" function. Function names
        # are deduced from node type names. For example, on pass1, for
        # the ExitSubStmt nodes we call the
        # `_compile_exit_sub_pass1_pre` and
        # `_compile_exit_sub_pass1_post` functions.

        # call pre-children compile functions for this pass
        func = self._get_node_compile_func(tree, 'pre', _pass)
        if func is not None:
            func(tree)

        for node in tree.children:
            # Recurse the children
            if isinstance(node, (Stmt, Expr)):
                self._compile_tree(node, _pass)
            elif not isinstance(node, (Label, LineNo)):
                raise InternalError(
                    f'Do not know how to compile node: {node}')

        # call post-children compile functions for this pass
        func = self._get_node_compile_func(tree, 'post', _pass)
        if func is not None:
            func(tree)

    def _get_node_compile_func(self, node, pre_or_pos, _pass):
        assert pre_or_pos in ['pre', 'post']

        type_name = node.type_name().lower().replace(' ', '_')
        func_name = f'_compile_{type_name}_pass{_pass}_{pre_or_pos}'
        func = getattr(self, func_name, None)
        return func

    def _compile_label_pass1_pre(self, node):
        if node.name in self.all_labels:
            raise CompileError(
                EC.DUPLICATE_LABEL,
                f'Duplicate label: {node.name}',
                node=node)
        self.cur_routine.labels.add(node.name)
        self.all_labels.add(node.name)

    def _compile_lineno_pass1_pre(self, node):
        canonical_name = f'_label_{node.number}'
        if canonical_name in self.all_labels:
            raise CompileError(
                EC.DUPLICATE_LABEL,
                f'Duplicate line number: {node.number}',
                node=node)
        self.cur_routine.labels.add(canonical_name)
        self.all_labels.add(canonical_name)

    def _compile_identifier_pass1_pre(self, node):
        if node.name not in self.cur_routine.variables:
            self.cur_routine.variables.add(node.name)

    def _compile_binary_op_pass1_pre(self, node):
        if node.type == Type.UNKNOWN:
            raise CompileError(EC.TYPE_MISMATCH, node=node)

    def _compile_unary_op_pass1_pre(self, node):
        # unary operators (NOT, +, -) are only valid on numeric
        # expressions.
        if not node.arg.type.is_numeric:
            raise CompileError(EC.TYPE_MISMATCH, node=node)

    def _compile_assignment_pass1_pre(self, node):
        if not node.lvalue.type.is_coercible_to(node.rvalue.type):
            raise CompileError(EC.TYPE_MISMATCH, node=node)

    def _compile_sub_block_pass1_pre(self, node):
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
        routine = Routine(node.name, 'sub')
        self.routines[node.name] = routine
        self.cur_routine = routine

    def _compile_sub_block_pass1_post(self, node):
        if isinstance(node, SubBlock):
            self.cur_routine = self.routines['_main']

    def _compile_exit_sub_pass1_pre(self, node):
        if self.cur_routine.name == '_main' or \
           self.cur_routine.type != 'sub':
            raise CompileError(
                EC.INVALID_EXIT,
                'EXIT SUB can only be used inside a SUB',
                node=node)
