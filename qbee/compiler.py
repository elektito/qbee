from .stmt import (
    Stmt, IfBlock, VarDeclClause, ArrayDimRange, CallStmt,
    ReturnValueSetStmt, FunctionBlock, SubBlock,
)
from .expr import Type, Expr, Lvalue, NumericLiteral, FuncCall
from .program import Label, LineNo
from .codegen import CodeGen
from .exceptions import ErrorCode as EC, InternalError, CompileError
from .parser import parse_string


class Routine:
    "Represents a SUB or a FUNCTION"

    def __init__(self, name, kind, compiler, params, return_type=None):
        kind = kind.lower()
        assert kind in ('sub', 'function', 'toplevel')

        assert all(
            isinstance(pname, str) and isinstance(ptype, Type)
            for pname, ptype in params
        )
        assert return_type is None or isinstance(return_type, Type)

        self.compiler = compiler
        self.name = name
        self.kind = kind
        self.return_type = return_type
        self.params: dict[str, Type] = dict(params)
        self.local_vars: dict[str, Type] = {}
        self.labels = set()

    def __repr__(self):
        return f'<Routine {self.kind} {self.name}>'

    def get_variable_type(self, var_name):
        if var_name in self.params:
            var_type = self.params[var_name]
            return var_type
        if var_name in self.local_vars:
            var_type = self.local_vars[var_name]
            return var_type

        if var_name in self.compiler.global_vars:
            return self.compiler.global_vars[var_name]

        return self.get_identifier_type(var_name)

    def get_identifier_type(self, identifier: str):
        if Type.is_type_char(identifier[-1]):
            return Type.from_type_char(identifier[-1])
        else:
            if identifier in self.compiler.global_vars:
                return self.compiler.global_vars[identifier]

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

        self.cur_routine = Routine(
            '_main', 'toplevel', self, params=[])

        self.routines = {'_main': self.cur_routine}
        self.user_types = {}
        self.consts = {}
        self.global_vars = {}

        self.all_labels = set()
        self._codegen = CodeGen(codegen_name, self)

        self._check_compile_methods()

    def compile(self, input_string):
        tree = parse_string(input_string)
        tree.bind(self)

        self._compile_tree(tree, _pass=1)
        self._compile_tree(tree, _pass=2)
        self._compile_tree(tree, _pass=3)

        if self.optimization_level > 0:
            tree.fold()

        code = self._codegen.gen_code(tree)

        if self.optimization_level > 1:
            code.optimize()

        return code

    def is_const(self, name):
        "Return whether the given name is a const or not"
        return name in self.consts

    def is_var_global(self, name):
        return name in self.global_vars

    def get_type_size(self, type):
        return Type.get_type_size(type, self.user_types)

    def get_routine(self, name: str, kind=None):
        assert kind is None or kind in ('sub', 'function')

        if any(name.endswith(c) for c in Type.type_chars):
            type_char = name[-1]
            name = name[:-1]
        else:
            type_char = None

        routine = self.routines.get(name)
        if routine is None:
            return None

        if kind is None or routine.kind == kind:
            if type_char is None:
                return routine

            return_type_char = routine.return_type.type_char
            if routine.return_type and return_type_char == type_char:
                return routine

        return None

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
            node.node_name().lower().replace(' ', '_')
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

        if isinstance(tree, (SubBlock, FunctionBlock)):
            self.cur_routine = tree.routine

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

        if isinstance(tree, (SubBlock, FunctionBlock)):
            self.cur_routine = self.routines['_main']

    def _get_node_compile_func(self, node, pre_or_pos, _pass):
        assert pre_or_pos in ['pre', 'post']

        type_name = node.node_name().lower().replace(' ', '_')
        func_name = f'_compile_{type_name}_pass{_pass}_{pre_or_pos}'
        func = getattr(self, func_name, None)
        return func

    def _set_parent_routine(self, node, routine):
        node.parent_routine = routine
        for child in node.children:
            self._set_parent_routine(child, routine)

    def _validate_decl(self, decl: VarDeclClause):
        if decl.type.is_builtin:
            return
        if decl.type.is_array:
            decl.type.array_base_type
        if not decl.type.is_user_defined:
            return
        if decl.type.user_type_name not in self.user_types:
            raise CompileError(EC.TYPE_NOT_DEFINED,
                               node=decl)

    def _perform_argument_matching(self, node, kind):
        routine = self.routines.get(node.name)
        if routine is None:
            raise CompileError(EC.SUBPROGRAM_NOT_FOUND, node=node)
        if routine.kind != kind:
            if kind == 'sub':
                msg = 'Routine is a FUNCTION not a SUB'
            else:
                msg = 'Routine is a SUB not a FUNCTION'
            raise CompileError(EC.SUBPROGRAM_NOT_FOUND,
                               msg=msg,
                               node=node)
        if len(node.args) != len(routine.params):
            raise CompileError(
                EC.ARGUMENT_COUNT_MISMATCH,
                node=node)

        for param_type, arg in zip(routine.params.values(), node.args):
            if isinstance(arg, Lvalue):
                # argument type must match exactly for lvalues,
                # because pass is by reference
                if arg.type != param_type:
                    raise CompileError(
                        EC.TYPE_MISMATCH,
                        'Parameter type mismatch',
                        node=arg)
            elif not arg.type.is_coercible_to(param_type):
                error_msg = (
                    f'Argument type mismatch: '
                    f'expected {param_type.name}, '
                    f'got {arg.type.name}'
                )
                raise CompileError(EC.TYPE_MISMATCH,
                                   msg=error_msg,
                                   node=arg)

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

    def _compile_const_pass1_pre(self, node):
        if not node.value.is_const:
            raise CompileError(
                EC.INVALID_CONSTANT,
                node=node.value)
        self.consts[node.name] = node.value

    def _compile_lvalue_pass2_pre(self, node):
        func = self.get_routine(node.base_var, 'function')
        if func is not None:
            if node.dotted_vars:
                raise CompileError(
                    EC.INVALID_USE_OF_FUNCTION,
                    node=node)
            new_node = FuncCall(func.name,
                                func.return_type,
                                node.array_indices)
            node.parent.replace_child(node, new_node)
            return

        if Type.name_ends_with_type_char(node.base_var):
            base_name = node.base_var[:-1]
        else:
            base_name = node.base_var
        if base_name in self.routines:
            raise CompileError(
                EC.DUPLICATE_DEFINITION,
                'A sub-routine with the same name exists',
                node=node)

        if node.base_var not in self.cur_routine.local_vars and \
           node.base_var not in self.cur_routine.params and \
           not node.base_var in self.consts and \
           not node.base_var in self.global_vars:
            # Implicitly defined variable
            decl = VarDeclClause(node.base_var, None)
            if node.array_indices:
                # It's an array; implicit arrays have a range of 0 to
                # 10 for all their dimensions.
                decl.array_dims = [
                    ArrayDimRange(NumericLiteral(0),
                                  NumericLiteral(10))
                    for _ in node.array_indices
                ]
                for d in decl.array_dims:
                    d.bind(self)
            decl.bind(self)
            self.cur_routine.local_vars[node.base_var] = decl.type

        if node.array_indices:
            if not node.base_type.is_array:
                raise CompileError(
                    EC.TYPE_MISMATCH,
                    'Cannot index non-array (note that this is '
                    'valid in original QB and causes an implicit '
                    'array with the same name be created)',
                    node=node)

            expected_dims = len(node.base_type.array_dims)
            if len(node.array_indices) != expected_dims and \
               not node.base_type.is_nodim_array:
                raise CompileError(
                    EC.WRONG_NUMBER_OF_DIMENSIONS,
                    node=node)

    def _compile_array_pass_pass1_pre(self, node):
        var_type = node.parent_routine.get_variable_type(
            node.identifier)
        if not var_type.is_array:
            raise CompileError(
                EC.TYPE_MISMATCH,
                'Not an array',
                node=node)

    def _compile_array_pass_pass3_pre(self, node):
        if not isinstance(node.parent, (FuncCall, CallStmt)):
            raise CompileError(
                EC.TYPE_MISMATCH,
                'Array-pass expression is only valid when calling '
                'functions or sub-routines',
                node=node,
            )

    def _compile_func_call_pass3_pre(self, node):
        for arg in node.args:
            if isinstance(arg, Lvalue) and arg.type.is_array:
                raise CompileError(
                    EC.TYPE_MISMATCH,
                    f'Parameter type mismatch. Did you mean '
                    f'{arg.base_var}()?',
                    node=arg,
                )
        self._perform_argument_matching(node, 'function')

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

        if Type.name_ends_with_type_char(node.lvalue.base_var):
            base_name = node.lvalue.base_var[:-1]
        else:
            base_name = node.lvalue.base_var
        if base_name == self.cur_routine.name and \
           self.cur_routine.kind == 'function' and \
           not node.lvalue.dotted_vars and \
           not node.lvalue.array_indices:
            # assigning to function name (return value)
            new_node = ReturnValueSetStmt(node.rvalue)
            node.parent.replace_child(node, new_node)
            return

    def _compile_assignment_pass3_pre(self, node):
        # Checking if there's an assignment to a FuncCall (converted
        # from an Lvalue node). We have to do this in pass 3, because
        # on pass 2 where the Lvalue is replaced with a FuncCall,
        # we're already traversing the tree and we'll still see the
        # old Lvalue in the tree.
        if not isinstance(node.lvalue, Lvalue):
            raise CompileError(
                EC.DUPLICATE_DEFINITION,
                'A function with the same name exists',
                node=node.lvalue)

    def _compile_call_pass1_pre(self, node):
        for arg in node.args:
            if isinstance(arg, Lvalue) and arg.type.is_array:
                raise CompileError(
                    EC.TYPE_MISMATCH,
                    f'Parameter type mismatch. Did you mean '
                    f'{arg.base_var}()?',
                    node=arg,
                )

    def _compile_call_pass2_pre(self, node):
        self._perform_argument_matching(node, 'sub')

    def _compile_dim_pass1_pre(self, node):
        for decl in node.var_decls:
            self._validate_decl(decl)

            if decl.name in self.cur_routine.local_vars or \
               decl.name in self.global_vars or \
               decl.name in self.routines:
                raise CompileError(
                    EC.DUPLICATE_DEFINITION,
                    node=decl)

            if node.shared:
                self.global_vars[decl.name] = decl.type
            else:
                self.cur_routine.local_vars[decl.name] = decl.type

    def _compile_for_block_pass1_pre(self, node):
        var_type = node.parent_routine.get_identifier_type(
            node.var.base_var)
        if not var_type.is_numeric:
            raise CompileError(
                EC.TYPE_MISMATCH,
                'FOR variable must be numeric',
                node=node.var,
            )

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
        for decl in node.params:
            self._validate_decl(decl)
        params = [(decl.name, decl.type) for decl in node.params]
        routine = Routine(node.name, 'sub', self, params)
        self.routines[node.name] = routine
        self.cur_routine = routine
        node.routine = routine

    def _compile_function_block_pass1_pre(self, node):
        if self.cur_routine.name != '_main':
            raise CompileError(
                EC.ILLEGAL_IN_SUB,
                'Function only allowed in the top-level',
                node=node)
        if node.name in self.routines:
            raise CompileError(
                EC.DUPLICATE_DEFINITION,
                f'Duplicate routine definition: {node.name}',
                node=node)
        for decl in node.params:
            self._validate_decl(decl)
        params = [(decl.name, decl.type) for decl in node.params]
        routine = Routine(node.name, 'function', self, params,
                          return_type=node.type)
        self.routines[node.name] = routine
        self.cur_routine = routine
        node.routine = routine

        routine.local_vars['_retval'] = node.type

    def _compile_exit_sub_pass1_pre(self, node):
        if self.cur_routine.name == '_main' or \
           self.cur_routine.kind != 'sub':
            raise CompileError(
                EC.INVALID_EXIT,
                'EXIT SUB can only be used inside a SUB',
                node=node)

    def _compile_exit_function_pass1_pre(self, node):
        if self.cur_routine.kind != 'function':
            raise CompileError(
                EC.INVALID_EXIT,
                'EXIT FUNCTION can only be used inside a FUNCTION',
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
        for decl in node.decls:
            self._validate_decl(decl)
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
                        f'Cannot print value',
                        node=item)

    def _compile_view_print_pass1_pre(self, node):
        if node.top_expr and not node.top_expr.type.is_numeric:
            raise CompileError(
                EC.TYPE_MISMATCH,
                'Expected numeric expression',
                node=node.top_expr
            )

        if node.bottom_expr and not node.bottom_expr.type.is_numeric:
            raise CompileError(
                EC.TYPE_MISMATCH,
                'Expected numeric expression',
                node=node.bottom_expr
            )
