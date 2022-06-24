from numbers import Number
from itertools import product
from qbee import grammar
from qbee.evalctx import EvaluationContext, Routine, EvalError
from .memlayout import get_global_var_idx, get_local_var_idx


class QArray:
    def __init__(self, array_data, bounds, element_type):
        self.array_data = array_data
        self.bounds = bounds
        self.element_type = element_type

        if self.element_type.is_numeric:
            self.default_value = self.element_type.py_type(0)
        else:
            self.default_value = ''

    def at(self, *indices):
        def arridx(data, indices, bounds):
            idx = indices[0]
            lb, ub = bounds[0]
            if idx < lb or idx > ub:
                raise EvalError(
                    f'Index {idx} out of range ({lb}, {ub})')
            if len(indices) == 1:
                return data[idx - lb]
            else:
                return arridx(data[idx - lb],
                              indices[1:],
                              bounds[1:])
        if len(indices) != len(self.bounds):
            raise EvalError(
                f'Incorrect number of array indices (expected '
                f'{len(self.bounds)})')

        value = arridx(self.array_data, indices, self.bounds)
        if value is None:
            value = self.default_value
        else:
            value = value.value
        return value

    def __str__(self):
        all_indices = product(
            *(range(lb, ub+1) for lb, ub in self.bounds))
        lines = []
        lines.append(f'Array of type: {self.element_type.name}')
        lines.append(f'Bounds: {self.bounds}')
        for indices in all_indices:
            if len(indices) == 1:
                lines.append(f'{indices[0]}: {self.at(*indices)}')
            else:
                lines.append(f'{indices}: {self.at(*indices)}')
        return '\n'.join(lines)


class QvmEval(EvaluationContext):
    def __init__(self, cpu, main_routine, find_routine_func):
        super().__init__()

        self.cpu = cpu
        self.main_routine = main_routine
        self.find_routine_func = find_routine_func

    def eval_lvalue(self, lvalue):
        if lvalue.base_var in self.consts and \
           (lvalue.array_indices or lvalue.dotted_vars):
            raise ValueError(
                'Indices and dotted vars not valid with conts')
        elif lvalue.base_var in self.consts:
            return self.consts[lvalue.base_var].eval()

        base_type = lvalue.base_type
        segment, base_idx = self.eval_var(lvalue.base_var)

        if not base_type.is_array and not base_type.is_user_defined:
            result = segment.get_cell(base_idx)
            if result is None:
                raise EvalError(
                    f'{lvalue.base_var} does not have a value yet')
            return result.value

        if base_type.is_array:
            value = self.read_array(segment, base_idx,
                                    base_type.array_base_type)

        if base_type.is_user_defined:
            value = self.read_struct(segment, base_idx, base_type)

        if lvalue.array_indices:
            if not isinstance(value, QArray):
                raise EvalError(
                    'Cannot read an array element from a struct')

            indices = [i.eval() for i in lvalue.array_indices]
            for i in range(len(indices)):
                if isinstance(indices[i], Number):
                    indices[i] = int(round(indices[i]))
                else:
                    raise EvalError('Invalid array index type')

            value = value.at(*indices)

        if lvalue.dotted_vars:
            if not isinstance(value, QStruct):
                raise EvalError(
                    'Cannot read a field from a non-struct value')
            value = value.get_field(*lvalue.dotted_vars)
            if value is not None:
                value = value.value

        if value is None:
            raise EvalError(
                f'{lvalue.base_var} does not have a value yet')

        return value

    def eval_var(self, var):
        var = var.lower()
        if var in self.global_vars:
            try:
                return (self.cpu.globals_segment,
                        get_global_var_idx(self, var))
            except KeyError:
                pass

        frame = self.cpu.cur_frame
        routine = self.find_routine_func(frame.code_start)

        try:
            var_idx = get_local_var_idx(routine, var)
        except KeyError:
            raise EvalError('Unknown variable')
        return self.cpu.cur_frame, var_idx

    def read_array(self, segment, base_idx, element_type):
        def mul(ls):
            r = 1
            for e in ls:
                r *= e
            return r

        def read_sub_array(base_idx, rest_bounds, rest_dim_sizes):
            if rest_bounds == []:
                return segment.get_cell(base_idx)
            else:
                first_lbound, first_ubound = rest_bounds[0]
                array = []
                for _ in range(first_lbound, first_ubound + 1):
                    sub_array = read_sub_array(
                        base_idx, rest_bounds[1:], rest_dim_sizes[1:])
                    array.append(sub_array)
                    base_idx += mul(rest_dim_sizes[1:])
                return array

        n_dims = segment.get_cell(base_idx + 1)
        if n_dims is None:
            raise EvalError('Array not initialized')
        n_dims = n_dims.value

        element_size = segment.get_cell(base_idx + 2).value
        base_idx += 3

        dim_sizes = []
        bounds = []
        for idx in range(n_dims):
            lbound = segment.get_cell(base_idx + 0).value
            ubound = segment.get_cell(base_idx + 1).value
            dim_sizes.append((ubound - lbound + 1) * element_size)
            bounds.append((lbound, ubound))
            base_idx += 2

        first_lbound, first_ubound = bounds[0]
        array = []
        for _ in range(first_lbound, first_ubound + 1):
            sub_array = read_sub_array(
                base_idx, bounds[1:], dim_sizes[1:])
            array.append(sub_array)
            base_idx += mul(dim_sizes[1:])

        return QArray(array, bounds, element_type)
