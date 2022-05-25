from .stmt import Stmt, IfBlock, VarDeclClause
from .expr import Type, Expr, Lvalue
from .program import Label, LineNo
from .codegen import CodeGen
from .exceptions import ErrorCode as EC, InternalError, CompileError
from .parser import parse_string


class Routine:
    "Represents a SUB or a FUNCTION"

    def __init__(self, name, type, compiler, params):
        type = type.lower()
        assert type in ('sub', 'function', 'toplevel')

        assert all(
            isinstance(pname, str) and isinstance(ptype, Type)
            for pname, ptype in params
        )

        self.compiler = compiler
        self.name = name
        self.type = type
        self.params: dict[str, Type] = dict(params)
        self.local_vars: dict[str, Type] = {}
        self.labels = set()

    def __repr__(self):
        return f'<Routine {self.type} {self.name}>'

    def get_variable_type(self, var_name):
        if var_name in self.params:
            var_type = self.params[var_name]
            return var_type
        if var_name in self.local_vars:
            var_type = self.local_vars[var_name]
            return var_type

        # global variables not implemented yet. when they are, query
        # the compiler for it and remove this assert.
        assert not self.compiler.is_var_global(var_name)

        return self.get_identifier_type(var_name)

    def get_identifier_type(self, identifier: str):
        if Type.is_type_char(identifier[-1]):
            return Type.from_type_char(identifier[-1])
        else:
            # remove this when we add support for global variables
            assert not self.compiler.is_var_global(identifier)

            # for now DEF* statements are not supported, so always the
            # default type
            return Type.SINGLE

    @property
    def params_size(self):
        # the number of cells in a call frame the parameters to this
        # routine need
        return sum(
            self.compiler.get_type_size(ptype)
            for ptype in self.params.values()
        )

    @property
    def local_vars_size(self):
        # the number of cells in a call frame the local variables of
        # this routine need
        return sum(
            self.compiler.get_type_size(vtype)
            for vtype in self.local_vars.values()
        )

    @property
    def frame_size(self):
        # the total number of cells a call frame for this routine needs
        return self.params_size + self.local_vars_size


class Compiler:
    def __init__(self, codegen_name, optimization_level=0):
        self.optimization_level = optimization_level

        self.cur_routine = Routine('_main', 'toplevel', self, params=[])
        self.routines = {'_main': self.cur_routine}

        self.user_types = {}

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

    def is_var_global(self, name):
        return False

    def get_variable(self, node, variable_name: str):
        assert False

    def get_type_size(self, type):
        return Type.get_type_size(type, self.user_types)

    def _check_compile_methods(self):
        # Check if all the _compile_* functions match a node. This is
        # a sanity check to make sure we haven't misspelled anything.

        import inspect
        import re
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

    def _compile_program_pass1_pre(self, node):
        node.routine = self.routines['_main']

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
            decl.bind(self)
            self.cur_routine.local_vars[node.base_var] = decl.type

        # This is here to make sure we won't forget to check
        # indices when it's implemented.
        assert not node.array_indices

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

        for param_type, arg in zip(routine.params.values(), node.args):
            if not arg.type.is_coercible_to(param_type):
                expected_type_name = param_type.name
                if param_type.is_user_defined:
                    expected_type_name = 'user-defined type '
                    expected_type_name += param_type.user_type_name
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
            self.cur_routine.local_vars[decl.name] = decl.type

    def _compile_input_pass1_pre(self, node):
        for lvalue in node.var_list:
            if not lvalue.type.is_builtin:
                raise CompileError(
                    EC.TYPE_MISMATCH,
                    'Input can only have builtin types',
                    node=lvalue)

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
        params = [(decl.name, decl.type) for decl in node.params]
        routine = Routine(node.name, 'sub', self, params)
        self.routines[node.name] = routine
        self.cur_routine = routine
        node.routine = routine

    def _compile_sub_block_pass1_post(self, node):
        self.cur_routine = self.routines['_main']

    def _compile_exit_sub_pass1_pre(self, node):
        if self.cur_routine.name == '_main' or \
           self.cur_routine.type != 'sub':
            raise CompileError(
                EC.INVALID_EXIT,
                'EXIT SUB can only be used inside a SUB',
                node=node)

    def _compile_type_block_pass1_pre(self, node):
        if node.name in self.user_types:
            raise CompileError(
                EC.DUPLICATE_DEFINITION,
                'Duplicate type name',
                node=node)
        if len(node.fields) == 0:
            raise CompileError(
                EC.ELEMENT_NOT_DEFINED,
                'Type definition has no elements',
                node=node)
        self.user_types[node.name] = node

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
                if not item.type.is_builtin:
                    raise CompileError(
                        EC.TYPE_MISMATCH,
                        f'Cannot print value: {item}',
                        node=node)
