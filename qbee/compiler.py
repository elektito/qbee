from .stmt import Stmt, IfBlock, VarDeclClause
from .expr import Type, Expr, Lvalue
from .program import Label, LineNo
from .codegen import CodeGen
from .exceptions import ErrorCode as EC, InternalError, CompileError
from .parser import parse_string


class Routine:
    "Represents a SUB or a FUNCTION"

    def __init__(self, name, type, params):
        type = type.lower()
        assert type in ('sub', 'function', 'toplevel')

        self.name = name
        self.type = type
        self.params = params
        self.local_vars: dict[str, VarDeclClause] = {}
        self.labels = set()

    def __repr__(self):
        return f'<Routine {self.type} {self.name}>'


class Compiler:
    def __init__(self, codegen_name, optimization_level=0):
        self.optimization_level = optimization_level

        self.cur_routine = Routine('_main', 'toplevel', params=[])
        self.routines = {'_main': self.cur_routine}

        self.all_labels = set()
        self._codegen = CodeGen(codegen_name, self)

        self._check_compile_methods()

    def compile(self, input_string):
        tree = parse_string(input_string)
        tree.bind(self)

        self._compile_tree(tree, _pass=1)
        self._compile_tree(tree, _pass=2)

        if self.optimization_level > 0:
            tree.fold()

        code = self._codegen.gen_code(tree)

        if self.optimization_level > 1:
            code.optimize()

        return code

    def is_const(self, name):
        "Return whether the given name is a const or not"
        return False

    def get_identifier_type(self, identifier: str, routine: Routine):
        # DEF* statements not supported yet. when support is added,
        # the "routine" parameter needs to be used, because DEF*
        # statements are local to the current routine.
        return Type.SINGLE

    def get_variable_type(self, var: str, routine: Routine) -> Type:
        """
Return the type of the given variable name, in the given routine, taking
into account the DEF* statements and the DIM statements in the routine.
        """

        assert isinstance(var, str)
        assert isinstance(routine, Routine)

        if Type.is_type_char(var[-1]):
            return Type.from_type_char(var[-1])
        else:
            # remove this when we add support for global variables
            assert not self.is_var_global(var)

            if var in routine.local_vars:
                return routine.local_vars[var].type

            # for now DEF* statements are not supported, so always the
            # default type
            return Type.SINGLE

    def is_var_global(self, name):
        return False

    def get_variable(self, node, variable_name: str):
        assert False

    def _check_compile_methods(self):
        # Check if all the _compile_* functions match a node. This is
        # a sanity check to make sure we haven't misspelled anything.

        import inspect, re
        from . import expr, stmt, program
        from .node import Node

        all_nodes = [
            getattr(module, member)
            for module in [expr, stmt, program]
            for member in dir(module)
            if inspect.isclass(getattr(module, member))
            if issubclass(getattr(module, member), Node)
        ]
        all_nodes = [
            node.type_name().lower().replace(' ', '_')
            for node in all_nodes
        ]

        compile_method_re = re.compile(
            r'_compile_(?P<node>.+)_pass\d+_(?:pre|post)')
        for member in dir(self):
            if not callable(getattr(self, member)):
                continue
            m = compile_method_re.match(member)
            if m is None:
                continue
            node_name = m.group('node')
            if node_name not in all_nodes:
                raise NameError(
                    f'Cannot find a node for compile function: '
                    f'{member}')

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
            if _pass == 1:
                self._set_parent_routine(node, self.cur_routine)

            # Recurse the children
            if isinstance(node, (Stmt, Expr, Label, LineNo, Lvalue)):
                self._compile_tree(node, _pass)
            else:
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

    def _set_parent_routine(self, node, routine):
        node.parent_routine = routine
        for child in node.children:
            self._set_parent_routine(child, routine)

    def _compile_label_pass1_pre(self, node):
        if node.name in self.all_labels:
            raise CompileError(
                EC.DUPLICATE_LABEL,
                f'Duplicate label: {node.name}',
                node=node)
        self.cur_routine.labels.add(node.name)
        self.all_labels.add(node.name)

    def _compile_lineno_pass1_pre(self, node):
        if node.canonical_name in self.all_labels:
            raise CompileError(
                EC.DUPLICATE_LABEL,
                f'Duplicate line number: {node.number}',
                node=node)
        self.cur_routine.labels.add(node.canonical_name)
        self.all_labels.add(node.canonical_name)

    def _compile_lvalue_pass1_pre(self, node):
        if node.base_var not in self.cur_routine.local_vars:
            # Implicitly defined variable
            decl = VarDeclClause(node.base_var, None)
            self.cur_routine.local_vars[node.base_var] = decl

            # These are here to make sure we won't forget to check
            # indices and dotted variables when they're implemented.
            assert not node.array_indices
            assert not node.dotted_vars

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

    def _compile_call_pass2_pre(self, node):
        routine = self.routines.get(node.name)
        if routine is None:
            raise CompileError(EC.SUBPROGRAM_NOT_FOUND, node=node)
        if routine.type != 'sub':
            raise CompileError(EC.SUBPROGRAM_NOT_FOUND,
                               msg='Routine is a FUNCTION not a SUB',
                               node=node)
        if len(node.args) != len(routine.params):
            raise CompileError(
                EC.ARGUMENT_COUNT_MISMATCH,
                node=node)

        for param, arg in zip(routine.params, node.args):
            if not arg.type.is_coercible_to(param.type):
                expected_type_name = param.type.name
                if param.type == Type.USER_DEFINED:
                    expected_type_name = 'user-defined type '
                    expected_type_name += param.type.user_type_name
                error_msg = (
                    f'Argument type mismatch: '
                    f'expected {expected_type_name}, '
                    f'got {arg.type.name}'
                )
                raise CompileError(EC.TYPE_MISMATCH,
                                   msg=error_msg,
                                   node=arg)

    def _compile_dim_pass1_pre(self, node):
        for decl in node.var_decls:
            if decl.name in self.cur_routine.local_vars:
                raise CompileError(
                    EC.DUPLICATE_DEFINITION,
                    node=decl)
            self.cur_routine.local_vars[decl.name] = decl

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
        routine = Routine(node.name, 'sub', node.params)
        self.routines[node.name] = routine
        self.cur_routine = routine

    def _compile_sub_block_pass1_post(self, node):
        self.cur_routine = self.routines['_main']

    def _compile_exit_sub_pass1_pre(self, node):
        if self.cur_routine.name == '_main' or \
           self.cur_routine.type != 'sub':
            raise CompileError(
                EC.INVALID_EXIT,
                'EXIT SUB can only be used inside a SUB',
                node=node)

    def _compile_goto_pass2_pre(self, node):
        if isinstance(node.target, int):
            label_type = 'Line number'
        else:
            label_type = 'Label'

        if node.canonical_target not in self.all_labels:
            raise CompileError(
                EC.LABEL_NOT_DEFINED,
                f'{label_type} not defined: {node.target}',
                node=node)
        if node.canonical_target not in node.parent_routine.labels:
            raise CompileError(
                EC.LABEL_NOT_DEFINED,
                (f'{label_type} not in the same routine as GOTO: '
                 f'{node.target}'),
                node=node)

    def _compile_else_if_pass1_pre(self, node):
        if not any(isinstance(p, IfBlock) for p in node.parents()):
            raise CompileError(
                EC.ELSE_WITHOUT_IF,
                'ELSEIF outside IF block',
                node=node)

    def _compile_else_pass1_pre(self, node):
        if not any(isinstance(p, IfBlock) for p in node.parents()):
            raise CompileError(
                EC.ELSE_WITHOUT_IF,
                'ELSE outside IF block',
                node=node)

    def _compile_print_pass1_pre(self, node):
        for item in node.items:
            if isinstance(item, Expr):
                if item.type not in Type.builtin_types():
                    raise CompileError(
                        EC.TYPE_MISMATCH,
                        f'Cannot print value: {item}',
                        node=node)
