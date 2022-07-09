import logging
from collections import defaultdict
from dataclasses import dataclass
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
from .evalctx import EvaluationContext, Routine


logger = logging.getLogger(__name__)


@dataclass
class BlockContext:
    kind: str


class CompilationUnit(EvaluationContext):
    def __init__(self):
        super().__init__()

        self.main_routine = Routine(
            '_main', 'toplevel', self, params=[])

        self.all_labels = set()
        self.routines = {'_main': self.main_routine}
        self.global_vars = {}
        self.def_letter_types = {}  # maps a single letter to a type

        # Maps labels to all data values after it. The DATA statements
        # before which no label is found are collected under a "None"
        # key.
        self.data = defaultdict(list)

    def eval_lvalue(self, lvalue):
        if lvalue.array_indices or lvalue.dotted_vars:
            raise InternalError(
                'Attempting to evaluate non-const expression')
        if lvalue.base_var in lvalue.parent_routine.local_consts:
            return lvalue.parent_routine.local_consts[lvalue.base_var].eval()
        if lvalue.base_var in self.global_consts:
            return self.global_consts[lvalue.base_var].eval()

        raise InternalError(
            'Attempting to evaluate non-const expression')

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
                        f'Parameter type mismatch; expected a '
                        f'{param_type.name.upper()}, got '
                        f'{arg.type.name.upper()}',
                        node=arg)
            elif not arg.type.is_coercible_to(param_type):
                error_msg = (
                    f'Argument type mismatch: '
                    f'expected {param_type.name.upper()}, '
                    f'got {arg.type.name.upper()}'
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
    # 7. Gather values in DATA statements and their labels

    def __init__(self, compilation):
        super().__init__(compilation)
        self._last_label = None
        self._cur_blocks = []

    def process_label_pre(self, node):
        if node.name in self.compilation.all_labels:
            raise CompileError(
                EC.DUPLICATE_LABEL,
                f'Duplicate label: {node.name}',
                node=node)
        node.parent_routine.labels.add(node.name)
        self.compilation.all_labels.add(node.name)
        self._last_label = node.canonical_name

    def process_lineno_pre(self, node):
        if node.canonical_name in self.compilation.all_labels:
            raise CompileError(
                EC.DUPLICATE_LABEL,
                f'Duplicate line number: {node.number}',
                node=node)
        node.parent_routine.labels.add(node.canonical_name)
        self.compilation.all_labels.add(node.canonical_name)
        self._last_label = node.canonical_name

    def process_def_type_pre(self, node):
        for letter in node.letters:
            letter = letter.lower()
            self.compilation.def_letter_types[letter] = node.type

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

        params = []
        for decl in node.params:
            self.compilation.validate_decl(decl)

            # set type based on existing top-level DEF* statements (if
            # any) when no explicit type is specified
            param_type = decl.type
            if decl.var_type_name is None and \
               not Type.name_ends_with_type_char(decl.name):
                letter = decl.name[0].lower()
                if letter in self.compilation.def_letter_types:
                    letter = decl.name[0].lower()
                    param_type = self.compilation.def_letter_types[letter]
            params.append((decl.name, param_type))

        routine = Routine(node.name, 'sub', self.compilation, params,
                          is_static=node.is_static)
        self.compilation.routines[node.name] = routine

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

        params = []
        for decl in node.params:
            self.compilation.validate_decl(decl)

            # set type based on existing top-level DEF* statements (if
            # any) when no explicit type is specified
            param_type = decl.type
            if decl.var_type_name is None and \
               not Type.name_ends_with_type_char(decl.name):
                letter = decl.name[0].lower()
                if letter in self.compilation.def_letter_types:
                    letter = decl.name[0].lower()
                    param_type = self.compilation.def_letter_types[letter]
            params.append((decl.name, param_type))

        routine = Routine(node.name, 'function', self.compilation,
                          params,
                          is_static=node.is_static,
                          return_type=node.type)
        self.compilation.routines[node.name] = routine

        routine.local_vars['_retval'] = node.type

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

    def process_loop_block_pre(self, node):
        self._cur_blocks.append(BlockContext('do'))

    def process_loop_block_post(self, node):
        block = self._cur_blocks.pop()
        assert block.kind == 'do'

    def process_for_block_pre(self, node):
        self._cur_blocks.append(BlockContext('for'))

    def process_for_block_post(self, node):
        block = self._cur_blocks.pop()
        assert block.kind == 'for'

    def process_exit_do_pre(self, node):
        do_block = None
        for block in reversed(self._cur_blocks):
            if block.kind == 'do':
                do_block = block
                break
        if do_block is None:
            raise CompileError(
                EC.INVALID_EXIT,
                'EXIT DO outside DO...LOOP block',
                node=node,
            )

    def process_exit_for_pre(self, node):
        for_block = None
        for block in reversed(self._cur_blocks):
            if block.kind == 'for':
                for_block = block
                break
        if for_block is None:
            raise CompileError(
                EC.INVALID_EXIT,
                'EXIT FOR outside FOR...NEXT block',
                node=node,
            )

    def process_type_block_pre(self, node):
        if node.parent_routine != self.compilation.main_routine:
            raise CompileError(
                EC.ILLEGAL_IN_SUB,
                'TYPE block is illegal in SUB/FUNCTION',
                node=node)
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

    def process_data_pre(self, node):
        if node.parent_routine != self.compilation.main_routine:
            raise CompileError(
                EC.ILLEGAL_IN_SUB,
                'DATA is illegal in SUB/FUNCTION',
                node=node)
        self.compilation.data[self._last_label].extend(node.items)


class Pass2(CompilePass):
    # This pass does the following:
    #
    # 1. Convert lvalues to function calls where the base_var is a
    #    function.
    # 2. Perform argument count/type checking on sub calls (but not on
    #    functions, since
    #    we're just finding function calls in this pass)
    # 3. Perform target checking in GOTO statements.

    def _check_function_args(self, func_node, nargs, arg_types):
        if isinstance(nargs, tuple):
            nfrom, nto = nargs
            if not (nfrom <= len(func_node.args) <= nto):
                raise CompileError(
                    EC.ARGUMENT_COUNT_MISMATCH,
                    node=func_node)
        else:
            assert nargs == len(arg_types)
            if len(func_node.args) != len(arg_types):
                raise CompileError(
                        EC.ARGUMENT_COUNT_MISMATCH,
                        node=func_node)
        for arg, arg_type in zip(func_node.args, arg_types):
            if isinstance(arg_type, Type):
                if not arg.type.is_coercible_to(arg_type):
                    raise CompileError(
                        EC.TYPE_MISMATCH,
                        f'Type mismatch; expected '
                        f'{arg_type.name.upper()}; got '
                        f'{arg.type.name.upper()}',
                        node=arg)
            elif isinstance(arg_type, tuple):
                assert all(isinstance(t, Type) for t in arg_type)
                assert len(arg_type) > 1
                if not any(arg.type.is_coercible_to(t)
                           for t in arg_type):
                    expected_types = ', '.join(
                        t.name.upper() for t in arg_type)
                    raise CompileError(
                        EC.TYPE_MISMATCH,
                        f'Type mismatch; expected '
                        f'any of [{expected_types}]; got '
                        f'{arg.type.name.upper()}',
                        node=arg)
            elif arg_type == 'numeric':
                if not arg.type.is_numeric:
                    raise CompileError(
                        EC.TYPE_MISMATCH,
                        f'Type mismatch; expected a numeric value; got '
                        f'{func_node.args[0].type.name.upper()}',
                        node=arg)
            else:
                assert False

    def process_builtin_func_call_pre(self, node):
        nargs, *arg_types = {
            'asc': (1, Type.STRING),
            'chr$': (1, Type.INTEGER),
            'inkey$': (0,),
            'int': (1, 'numeric'),
            'lcase$': (1, Type.STRING),
            'left$': (2, Type.STRING, Type.INTEGER),
            'len': (1, Type.STRING),
            'mid$': ((2, 3), Type.STRING, Type.INTEGER, Type.INTEGER),
            'peek': (1, Type.INTEGER),
            'right$': (2, Type.STRING, Type.INTEGER),
            'rnd': ((0, 1), Type.SINGLE),
            'space$': (1, Type.INTEGER),
            'str$': (1, 'numeric'),
            'string$': (2, Type.INTEGER, (Type.INTEGER, Type.STRING)),
            'timer': (0,),
            'ucase$': (1, Type.STRING),
            'val': (1, Type.STRING),
        }.get(node.name, (None,))
        if nargs is None:
            raise InternalError(
                f'Unknown built-in function: {node.name}')
        self._check_function_args(node, nargs, arg_types)

    def process_const_pre(self, node):
        if node.parent_routine.has_variable(node.name):
            raise CompileError(
                EC.DUPLICATE_DEFINITION,
                node=node)

        if not node.value.is_const:
            raise CompileError(
                EC.INVALID_CONSTANT,
                node=node.value)

        if node.parent_routine == self.compilation.main_routine:
            if node.name in self.compilation.global_consts:
                raise CompileError(
                    EC.DUPLICATE_DEFINITION,
                    node=node)

            self.compilation.global_consts[node.name] = node.value
        else:
            if node.name in node.parent_routine.local_consts:
                raise CompileError(
                    EC.DUPLICATE_DEFINITION,
                    node=node)

            node.parent_routine.local_consts[node.name] = node.value

    def process_lvalue_pre(self, node):
        const = node.parent_routine.local_consts.get(node.base_var)
        if const is None:
            const = self.compilation.global_consts.get(node.base_var)
        if const is not None:
            if node.array_indices or node.dotted_vars:
                raise CompileError(
                    EC.INVALID_IDENTIFIER,
                    'Invalid use of CONST value (this might be valid '
                    'QBASIC, but we are not accepting it for now)',
                    node=node,
                )

            const = const.clone()
            const.loc_start = node.loc_start
            const.loc_end = node.loc_end
            node.parent.replace_child(node, const)

            return

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
           node.base_var not in node.parent_routine.local_consts and \
           node.base_var not in self.compilation.global_consts:
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
                    d.bind(self.compilation)
            decl.bind(self.compilation)

            if node.parent_routine.is_static:
                node.parent_routine.static_vars[node.base_var] = decl.type
            else:
                node.parent_routine.local_vars[node.base_var] = decl.type

            node.implicit_decl = decl

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

            for idx in node.array_indices:
                if not idx.type.is_numeric:
                    raise CompileError(
                        EC.TYPE_MISMATCH,
                        'Array indices must be numeric',
                        node=idx)

    def process_gosub_pre(self, node):
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

    def process_locate_pre(self, node):
        if not node.row.type.is_coercible_to(Type.LONG):
            raise CompileError(EC.TYPE_MISMATCH, node=node.row)
        if not node.col.type.is_coercible_to(Type.INTEGER):
            raise CompileError(EC.TYPE_MISMATCH, node=node.col)

    def process_poke_pre(self, node):
        if not node.address.type.is_coercible_to(Type.LONG):
            raise CompileError(EC.TYPE_MISMATCH, node=node.address)
        if not node.value.type.is_coercible_to(Type.INTEGER):
            raise CompileError(EC.TYPE_MISMATCH, node=node.value)

    def process_restore_pre(self, node):
        if node.target is None:
            return

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

    def process_return_pre(self, node):
        if node.target is None:
            return

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

            if decl.array_dims:
                for dim_range in decl.array_dims:
                    if not dim_range.is_const:
                        continue
                    lbound = dim_range.static_lbound
                    ubound = dim_range.static_ubound
                    if lbound > ubound:
                        raise CompileError(
                            EC.INVALID_DIMENSIONS,
                            f'Array LBOUND ({lbound}) is greater than '
                            f'array UBOUND ({ubound})',
                            node=dim_range,
                        )

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

    def process_assignment_pre(self, node):
        if not node.lvalue.type.is_coercible_to(node.rvalue.type):
            raise CompileError(EC.TYPE_MISMATCH, node=node)

        if node.lvalue.base_var in node.parent_routine.local_consts or \
           node.lvalue.base_var in self.compilation.global_consts:
            raise CompileError(EC.DUPLICATE_DEFINITION, node=node)

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

    def process_call_pre(self, node):
        for arg in node.args:
            if isinstance(arg, Lvalue) and arg.type.is_array:
                raise CompileError(
                    EC.TYPE_MISMATCH,
                    f'Parameter type mismatch. Did you mean '
                    f'{arg.base_var}()?',
                    node=arg,
                )
        self.compilation.perform_argument_matching(node, 'sub')

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

    def process_screen_pre(self, node):
        if not node.mode.type.is_numeric:
            raise CompileError(EC.TYPE_MISMATCH, node=node.mode)
        if node.color_switch and not node.color_switch.type.is_numeric:
            raise CompileError(EC.TYPE_MISMATCH, node=node.color_switch)
        if node.apage and not node.apage.type.is_numeric:
            raise CompileError(EC.TYPE_MISMATCH, node=node.apage)
        if node.vpage and not node.vpage.type.is_numeric:
            raise CompileError(EC.TYPE_MISMATCH, node=node.vpage)

    def process_width_pre(self, node):
        if node.columns and not node.columns.type.is_numeric:
            raise CompileError(EC.TYPE_MISMATCH, node=node.columns)
        if node.lines and not node.lines.type.is_numeric:
            raise CompileError(EC.TYPE_MISMATCH, node=node.lines)

    def process_play_pre(self, node):
        if node.command_string.type != Type.STRING:
            raise CompileError(EC.TYPE_MISMATCH,
                               node=node.command_string)


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
                'Format string must be a STRING',
                node=node.format_string)

        for item in node.items:
            if isinstance(item, Expr):
                if not item.type.is_builtin:
                    raise CompileError(
                        EC.TYPE_MISMATCH,
                        'Cannot print value',
                        node=item)

    def process_while_block_pre(self, node):
        if not node.cond.type.is_numeric:
            raise CompileError(
                EC.TYPE_MISMATCH,
                'WHILE condition should be a numeric expression',
                node=node.cond)


class Compiler:
    def __init__(self, codegen_name, optimization_level=0,
                 debug_info=False):
        self.optimization_level = optimization_level

        self._compilation = CompilationUnit()
        self._debug_info_enabled = debug_info
        self._codegen = CodeGen(
            codegen_name, self._compilation,
            debug_info=debug_info)

    def compile(self, input_string):
        if self._debug_info_enabled:
            self._codegen.set_source_code(input_string)

        logger.info('Parsing...')
        tree = parse_string(input_string)
        tree.bind(self._compilation)

        logger.info('Pass 1...')
        pass1 = Pass1(self._compilation)
        pass1.process_tree(tree)

        logger.info('Pass 2...')
        pass2 = Pass2(self._compilation)
        pass2.process_tree(tree)

        logger.info('Pass 3...')
        pass3 = Pass3(self._compilation)
        pass3.process_tree(tree)

        if self.optimization_level > 0:
            logger.info('Optimizing AST...')
            tree.fold()

        logger.info('Generating code...')
        code = self._codegen.gen_code(tree)

        if self.optimization_level > 1:
            logger.info('Optimizing code...')
            code.optimize()

        return code
