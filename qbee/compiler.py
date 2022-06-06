from .stmt import (
    Stmt, IfBlock, VarDeclClause, ArrayDimRange, CallStmt,
    ReturnValueSetStmt, FunctionBlock, SubBlock, SimpleCaseClause,
    RangeCaseClause, CompareCaseClause, CaseElseStmt,
)
from .expr import Type, Expr, Lvalue, NumericLiteral, FuncCall
from .program import Label, LineNo
from .codegen import CodeGen
from .exceptions import ErrorCode as EC, InternalError, CompileError
from .parser import parse_string


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

    def __init__(self, name, kind, compilation, params, is_static=False,
                 return_type=None):
        kind = kind.lower()
        assert kind in ('sub', 'function', 'toplevel')

        assert all(
            isinstance(pname, str) and isinstance(ptype, Type)
            for pname, ptype in params
        )
        assert return_type is None or isinstance(return_type, Type)
        assert isinstance(compilation, CompilationUnit)

        self.compilation = compilation
        self.name = name
        self.kind = kind
        self.is_static = is_static
        self.return_type = return_type
        self.params: dict[str, Type] = dict(params)
        self.local_vars: dict[str, Type] = {}
        self.static_vars: dict[str, Type] = {}
        self.labels = set()
        self.def_letter_types = {}  # maps a single letter to a type

    def __repr__(self):
        static = ' STATIC' if self.is_static else ''
        return f'<Routine {self.kind} {self.name}{static}>'

    def get_identifier_type(self, identifier: str):
        if Type.is_type_char(identifier[-1]):
            return Type.from_type_char(identifier[-1])
        else:
            if identifier in self.compilation.global_vars:
                return self.compilation.global_vars[identifier]

            def_type = self.def_letter_types.get(identifier[0])
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
        elif name in self.compilation.global_vars:
            var_type = self.compilation.global_vars[name]
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
            name in self.compilation.global_vars
        )

    @property
    def params_size(self):
        # the number of cells in a call frame the parameters to this
        # routine need
        return sum(
            self.compilation.get_type_size(ptype)
            for ptype in self.params.values()
        )

    @property
    def local_vars_size(self):
        # the number of cells in a call frame the local variables of
        # this routine need
        return sum(
            self.compilation.get_type_size(vtype)
            for vtype in self.local_vars.values()
        )

    @property
    def frame_size(self):
        # the total number of cells a call frame for this routine needs
        return self.params_size + self.local_vars_size


class CompilationUnit:
    def __init__(self):
        self.main_routine = Routine(
            '_main', 'toplevel', self, params=[])

        self.all_labels = set()
        self.routines = {'_main': self.main_routine}
        self.user_types = {}
        self.consts = {}
        self.global_vars = {}

    def is_const(self, name):
        "Return whether the given name is a const or not"
        return name in self.consts

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

    def validate_decl(self, decl: VarDeclClause):
        if decl.type.is_builtin:
            return
        if decl.type.is_array:
            decl.type.array_base_type
        if not decl.type.is_user_defined:
            return
        if decl.type.user_type_name not in self.user_types:
            raise CompileError(EC.TYPE_NOT_DEFINED,
                               node=decl)

    def perform_argument_matching(self, node, kind):
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


class CompilePass:
    def __init__(self, compilation):
        self.compilation = compilation

        self._check_compile_methods()

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
            r'_compile_(?P<node>.+)_(?:pre|post)')
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

    def _set_parent_routine(self, node, routine):
        node.parent_routine = routine
        for child in node.children:
            self._set_parent_routine(child, routine)

    def process_tree(self, tree):
        # This function recursively processes the given node. Before
        # recursing the children, a "pre" function for the node is
        # called, and afterwards, a "post" function. Function names
        # are deduced from node type names. For example, for the
        # ExitSubStmt nodes we call the `process_exit_sub_pre` and
        # `process_exit_sub_post` functions.

        # call pre-children compile functions for this pass
        func = self.get_node_compile_func(tree, 'pre')
        if func is not None:
            func(tree)

        for node in tree.children:
            # Recurse the children
            if isinstance(node, (Stmt, Expr, Label, LineNo, Lvalue)):
                self.process_tree(node)
            else:
                raise InternalError(
                    f'Do not know how to compile node: {node}')

        # call post-children compile functions for this pass
        func = self.get_node_compile_func(tree, 'post')
        if func is not None:
            func(tree)

        if isinstance(tree, (SubBlock, FunctionBlock)):
            self.cur_routine = self.compilation.routines['_main']

    def get_node_compile_func(self, node, pre_or_pos):
        assert pre_or_pos in ['pre', 'post']

        type_name = node.node_name().lower().replace(' ', '_')
        func_name = f'process_{type_name}_{pre_or_pos}'
        func = getattr(self, func_name, None)
        return func


class Pass1(CompilePass):
    # This pass does the following:
    #
    # 1. Gather labels/line numbers
    # 2. Gather subs and functions, create routines, set parent_routine
    # 3. Gather variable declarations (both implicit and explicit)
    # 4. Gather user-defined types.
    # 5. Convert some assignments to return value statements in
    #    functions
    # 6. Perform checks on some statements and expressions
    # 7. Gather DEF* statements

    def process_program_pre(self, node):
        node.routine = self.compilation.routines['_main']
        self._set_parent_routine(
            node, self.compilation.routines['_main'])

    def process_label_pre(self, node):
        if node.name in self.compilation.all_labels:
            raise CompileError(
                EC.DUPLICATE_LABEL,
                f'Duplicate label: {node.name}',
                node=node)
        node.parent_routine.labels.add(node.name)
        self.compilation.all_labels.add(node.name)

    def process_lineno_pre(self, node):
        if node.canonical_name in self.compilation.all_labels:
            raise CompileError(
                EC.DUPLICATE_LABEL,
                f'Duplicate line number: {node.number}',
                node=node)
        node.parent_routine.labels.add(node.canonical_name)
        self.compilation.all_labels.add(node.canonical_name)

    def process_def_type_pre(self, node):
        for letter in node.letters:
            node.parent_routine.def_letter_types[letter] = node.type

    def process_const_pre(self, node):
        if not node.value.is_const:
            raise CompileError(
                EC.INVALID_CONSTANT,
                node=node.value)
        self.compilation.consts[node.name] = node.value

    def process_array_pass_pre(self, node):
        var = node.parent_routine.get_variable(node.identifier)
        if not var.type.is_array:
            raise CompileError(
                EC.TYPE_MISMATCH,
                'Not an array',
                node=node)

    def process_binary_op_pre(self, node):
        if node.op.is_comparison:
            if node.left.type.is_numeric and \
               not node.right.type.is_numeric:
                raise CompileError(EC.TYPE_MISMATCH, node=node)
            elif node.right.type.is_numeric and \
                 not node.left.type.is_numeric:
                raise CompileError(EC.TYPE_MISMATCH, node=node)
            elif node.left.type == Type.STRING and \
                 node.right.type != Type.STRING:
                raise CompileError(EC.TYPE_MISMATCH, node=node)
            elif node.right.type == Type.STRING and \
                 node.left.type != Type.STRING:
                raise CompileError(EC.TYPE_MISMATCH, node=node)
            elif not node.left.type.is_builtin or \
                 not node.right.type.is_builtin:
                raise CompileError(EC.TYPE_MISMATCH, node=node)

        if node.type == Type.UNKNOWN:
            raise CompileError(EC.TYPE_MISMATCH, node=node)

    def process_unary_op_pre(self, node):
        # unary operators (NOT, +, -) are only valid on numeric
        # expressions.
        if not node.arg.type.is_numeric:
            raise CompileError(EC.TYPE_MISMATCH, node=node)

    def process_assignment_pre(self, node):
        if not node.lvalue.type.is_coercible_to(node.rvalue.type):
            raise CompileError(EC.TYPE_MISMATCH, node=node)

        if Type.name_ends_with_type_char(node.lvalue.base_var):
            base_name = node.lvalue.base_var[:-1]
        else:
            base_name = node.lvalue.base_var
        if base_name == node.parent_routine.name and \
           node.parent_routine.kind == 'function' and \
           not node.lvalue.dotted_vars and \
           not node.lvalue.array_indices:
            # assigning to function name (return value)
            new_node = ReturnValueSetStmt(node.rvalue)
            node.parent.replace_child(node, new_node)
            return

    def process_call_pre(self, node):
        for arg in node.args:
            if isinstance(arg, Lvalue) and arg.type.is_array:
                raise CompileError(
                    EC.TYPE_MISMATCH,
                    f'Parameter type mismatch. Did you mean '
                    f'{arg.base_var}()?',
                    node=arg,
                )

    def process_dim_pre(self, node):
        if node.kind == 'dim_shared' \
           and node.parent_routine.kind != 'toplevel':
            raise CompileError(
                EC.ILLEGAL_IN_SUB,
                'SHARED is only allowed in top-level',
                node=node,
            )
        elif node.kind == 'static' and \
             node.parent_routine.kind == 'toplevel':
            raise CompileError(
                EC.ILLEGAL_OUTSIDE_SUB,
                'STATIC is only allowed in SUB/FUNCTION',
                node=node,
            )

        for decl in node.var_decls:
            self.compilation.validate_decl(decl)

            if node.parent_routine.has_variable(decl.name) or \
               decl.name in self.compilation.routines:
                raise CompileError(
                    EC.DUPLICATE_DEFINITION,
                    node=decl)

            if node.kind == 'dim_shared':
                self.compilation.global_vars[decl.name] = decl.type
            elif node.kind == 'static' or \
                 node.parent_routine.is_static:
                node.parent_routine.static_vars[decl.name] = decl.type
            else:
                node.parent_routine.local_vars[decl.name] = decl.type

    def process_for_block_pre(self, node):
        if not node.var.base_type.is_numeric:
            raise CompileError(
                EC.TYPE_MISMATCH,
                'FOR variable must be numeric',
                node=node.var,
            )

    def process_input_pre(self, node):
        for lvalue in node.var_list:
            if not lvalue.type.is_builtin:
                raise CompileError(
                    EC.TYPE_MISMATCH,
                    'Input can only have builtin types',
                    node=lvalue)

    def process_sub_block_pre(self, node):
        if node.parent_routine.name != '_main':
            raise CompileError(
                EC.ILLEGAL_IN_SUB,
                'Sub-routine only allowed in the top-level',
                node=node)
        if node.name in self.compilation.routines:
            raise CompileError(
                EC.DUPLICATE_DEFINITION,
                f'Duplicate sub-routine definition: {node.name}',
                node=node)
        for decl in node.params:
            self.compilation.validate_decl(decl)
        params = [(decl.name, decl.type) for decl in node.params]
        routine = Routine(node.name, 'sub', self.compilation, params,
                          is_static=node.is_static)
        self.compilation.routines[node.name] = routine
        node.parent_routine = routine
        node.routine = routine

        node.parent_routine = self.compilation.main_routine
        for inner_stmt in node.block:
            self._set_parent_routine(inner_stmt, routine)

    def process_function_block_pre(self, node):
        if node.parent_routine.name != '_main':
            raise CompileError(
                EC.ILLEGAL_IN_SUB,
                'Function only allowed in the top-level',
                node=node)
        if node.name in self.compilation.routines:
            raise CompileError(
                EC.DUPLICATE_DEFINITION,
                f'Duplicate routine definition: {node.name}',
                node=node)
        for decl in node.params:
            self.compilation.validate_decl(decl)
        params = [(decl.name, decl.type) for decl in node.params]
        routine = Routine(node.name, 'function', self.compilation,
                          params,
                          is_static=node.is_static,
                          return_type=node.type)
        self.compilation.routines[node.name] = routine
        node.parent_routine = routine
        node.routine = routine

        routine.local_vars['_retval'] = node.type

        node.parent_routine = self.compilation.main_routine
        for inner_stmt in node.block:
            self._set_parent_routine(inner_stmt, routine)

    def process_exit_sub_pre(self, node):
        if node.parent_routine.name == '_main' or \
           node.parent_routine.kind != 'sub':
            raise CompileError(
                EC.INVALID_EXIT,
                'EXIT SUB can only be used inside a SUB',
                node=node)

    def process_exit_function_pre(self, node):
        if node.parent_routine.kind != 'function':
            raise CompileError(
                EC.INVALID_EXIT,
                'EXIT FUNCTION can only be used inside a FUNCTION',
                node=node)

    def process_type_block_pre(self, node):
        if node.name in self.compilation.user_types:
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
            self.compilation.validate_decl(decl)
        self.compilation.user_types[node.name] = node

    def process_else_if_pre(self, node):
        if not any(isinstance(p, IfBlock) for p in node.parents()):
            raise CompileError(
                EC.ELSE_WITHOUT_IF,
                'ELSEIF outside IF block',
                node=node)

    def process_else_pre(self, node):
        if not any(isinstance(p, IfBlock) for p in node.parents()):
            raise CompileError(
                EC.ELSE_WITHOUT_IF,
                'ELSE outside IF block',
                node=node)

    def process_view_print_pre(self, node):
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


class Pass2(CompilePass):
    # This pass does the following:
    #
    # 1. Convert lvalues to function calls where the base_var is a
    #    function.
    # 2. Perform argument count/type checking on sub calls (but not on
    #    functions, since
    #    we're just finding function calls in this pass)
    # 3. Perform target checking in GOTO statements.

    def process_lvalue_pre(self, node):
        func = self.compilation.get_routine(node.base_var, 'function')
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
        if base_name in self.compilation.routines:
            raise CompileError(
                EC.DUPLICATE_DEFINITION,
                'A sub-routine with the same name exists',
                node=node)

        if not node.parent_routine.has_variable(node.base_var) and \
           not node.base_var in self.compilation.consts:
            # Implicitly defined variable
            decl = VarDeclClause(node.base_var, None)
            decl.parent_routine = node.parent_routine
            if node.array_indices:
                # It's an array; implicit arrays have a range of 0 to
                # 10 for all their dimensions.
                decl.array_dims = [
                    ArrayDimRange(NumericLiteral(0),
                                  NumericLiteral(10))
                    for _ in node.array_indices
                ]
                for d in decl.array_dims:
                    d.bind(self.compilation)
            decl.bind(self.compilation)

            if node.parent_routine.is_static:
                node.parent_routine.static_vars[node.base_var] = decl.type
            else:
                node.parent_routine.local_vars[node.base_var] = decl.type

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

    def process_call_pre(self, node):
        self.compilation.perform_argument_matching(node, 'sub')

    def process_goto_pre(self, node):
        if isinstance(node.target, int):
            label_type = 'Line number'
        else:
            label_type = 'Label'

        if node.canonical_target not in self.compilation.all_labels:
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


class Pass3(CompilePass):
    # This pass does the following:
    #
    # 1. Perform argument count/type checking for function calls
    # 2. Check for assignment to function calls (which is an error)

    def process_array_pass_pre(self, node):
        if not isinstance(node.parent, (FuncCall, CallStmt)):
            raise CompileError(
                EC.TYPE_MISMATCH,
                'Array-pass expression is only valid when calling '
                'functions or sub-routines',
                node=node,
            )

    def process_func_call_pre(self, node):
        for arg in node.args:
            if isinstance(arg, Lvalue) and arg.type.is_array:
                raise CompileError(
                    EC.TYPE_MISMATCH,
                    f'Parameter type mismatch. Did you mean '
                    f'{arg.base_var}()?',
                    node=arg,
                )
        self.compilation.perform_argument_matching(node, 'function')

    def process_assignment_pre(self, node):
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

    def process_select_block_pre(self, node):
        vtype = node.value.type
        for case, body in node.case_blocks:
            if isinstance(case, CaseElseStmt):
                continue
            for clause in case.cases:
                if isinstance(clause, SimpleCaseClause):
                    if not clause.value.type.is_coercible_to(vtype):
                        raise CompileError(
                            EC.TYPE_MISMATCH,
                            node=clause.value,
                        )
                elif isinstance(clause, RangeCaseClause):
                    if not clause.from_value.type.is_coercible_to(vtype):
                        raise CompileError(
                            EC.TYPE_MISMATCH,
                            node=clause.from_value,
                        )
                    elif not clause.to_value.type.is_coercible_to(vtype):
                        raise CompileError(
                            EC.TYPE_MISMATCH,
                            node=clause.to_value,
                        )
                elif isinstance(clause, CompareCaseClause):
                    if not clause.value.type.is_coercible_to(vtype):
                        raise CompileError(
                            EC.TYPE_MISMATCH,
                            node=clause.value,
                        )

    def process_print_pre(self, node):
        if node.format_string and \
           node.format_string.type != Type.STRING:
            raise CompileError(
                EC.TYPE_MISMATCH,
                f'Format string must be a STRING',
                node=node.format_string)

        for item in node.items:
            if isinstance(item, Expr):
                if not item.type.is_builtin:
                    raise CompileError(
                        EC.TYPE_MISMATCH,
                        f'Cannot print value',
                        node=item)

    def process_while_block_pre(self, node):
        if not node.cond.type.is_numeric:
            raise CompileError(
                EC.TYPE_MISMATCH,
                'WHILE condition should be a numeric expression',
                node=node.cond)


class Compiler:
    def __init__(self, codegen_name, optimization_level=0):
        self.optimization_level = optimization_level

        self._compilation = CompilationUnit()
        self._codegen = CodeGen(codegen_name, self._compilation)

    def compile(self, input_string):
        tree = parse_string(input_string)
        tree.bind(self._compilation)

        pass1 = Pass1(self._compilation)
        pass1.process_tree(tree)

        pass2 = Pass2(self._compilation)
        pass2.process_tree(tree)

        pass3 = Pass3(self._compilation)
        pass3.process_tree(tree)

        if self.optimization_level > 0:
            tree.fold()

        code = self._codegen.gen_code(tree)

        if self.optimization_level > 1:
            code.optimize()

        return code
