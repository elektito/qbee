from qbee.stmt import TypeBlock
from qbee.expr import Type


def get_type_size(context, type):
    if type.is_array:
        if not type.is_static_array:
            return 1
        element_size = get_type_size(context, type.array_base_type)
        size = 1
        for dim in type.array_dims:
            nrange = (dim.static_ubound - dim.static_lbound + 1)
            size *= nrange
        size *= element_size
        header_size = 3 + len(type.array_dims) * 2
        return size + header_size

    assert all(
        isinstance(k, str) and isinstance(v, TypeBlock)
        for k, v in context.user_types.items()
    )

    builtin_types = [t.name for t in Type.builtin_types]
    if type.name in builtin_types:
        return 1
    else:
        struct = context.user_types[type.name]
        return sum(
            get_type_size(context, ftype)
            for ftype in struct.fields.values()
        )

def get_local_var_idx(routine, var):
    context = routine.context
    idx = 0
    for pname, ptype in routine.params.items():
        if var == pname:
            return idx
        idx += get_type_size(context, ptype)
    for vname, vtype in routine.local_vars.items():
        if var == vname:
            return idx
        idx += get_type_size(context, vtype)
    raise KeyError(
        f'Local variable not found in routine "{routine.name}": {var}'
    )

def get_global_var_idx(context, var):
    idx = 0
    for vname, vtype in context.global_vars.items():
        if var == vname:
            return idx
        idx += get_type_size(context, vtype)
    raise KeyError(f'Global variable not found: {var}')


def get_params_size(routine):
    # the number of cells in a call frame the parameters to a routine
    # need
    return sum(
        get_type_size(routine.context, ptype)
        for ptype in routine.params.values()
    )

def get_local_vars_size(routine):
    # the number of cells in a call frame the local variables of
    # a routine need
    return sum(
        get_type_size(routine.context, vtype)
        for vtype in routine.local_vars.values()
    )

def get_frame_size(roiutine):
    # the total number of cells a call frame for a routine needs
    return get_params_size(routine) + get_local_vars_size(routine)
