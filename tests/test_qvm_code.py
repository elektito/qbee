from qbee.qvm_codegen import QvmCode


def test_fold_push_and_conv():
    code = QvmCode()
    code.add(
        ('push!', 10.0),
        ('conv!&',),
    )
    code.optimize()
    assert code._instrs == [('push&', 10)]


def test_fold_push0_and_conv():
    code = QvmCode()
    code.add(
        ('push0!',),
        ('conv!%',),
    )
    code.optimize()
    assert code._instrs == [('push0%',)]


def test_push_one():
    code = QvmCode()
    code.add(
        ('push#', 1.0),
    )
    code.optimize()
    assert code._instrs == [('push1#',)]


def test_push_zero():
    code = QvmCode()
    code.add(
        ('push%', 0),
    )
    code.optimize()
    assert code._instrs == [('push0%',)]


def test_push_minus_one():
    code = QvmCode()
    code.add(
        ('push!', -1),
    )
    code.optimize()
    assert code._instrs == [('pushm1!',)]


def test_fold_push_and_conv_one():
    code = QvmCode()
    code.add(
        ('push&', 1),
        ('conv&!',),
    )
    code.optimize()
    assert code._instrs == [('push1!',)]


def test_fold_push_and_conv_invalid():
    code = QvmCode()
    code.add(
        ('push&', 100000),
        ('conv&%',),
    )
    code.optimize()
    assert code._instrs == [('push&', 100000), ('conv&%',)]


def test_eliminate_read_store():
    code = QvmCode()
    code.add(
        ('readg', 'x'),
        ('storeg', 'x'),
    )
    code.optimize()
    assert code._instrs == []


def test_eliminate_read_store_incompatible_scope():
    code = QvmCode()
    code.add(
        ('readl', 'x'),
        ('storeg', 'x'),
    )
    code.optimize()
    assert code._instrs == [
        ('readl', 'x'),
        ('storeg', 'x'),
    ]


def test_eliminate_read_store_incompatible_arg():
    code = QvmCode()
    code.add(
        ('readl', 'x'),
        ('storel', 'y'),
    )
    code.optimize()
    assert code._instrs == [
        ('readl', 'x'),
        ('storel', 'y'),
    ]


def test_eliminate_extra_jump1():
    code = QvmCode()
    code.add(
        ('ret',),
        ('ret',),
    )
    code.optimize()
    assert code._instrs == [
        ('ret',),
    ]


def test_eliminate_extra_jump2():
    code = QvmCode()
    code.add(
        ('jmp', 1000),
        ('ret',),
    )
    code.optimize()
    assert code._instrs == [
        ('jmp', 1000),
    ]


def test_eliminate_extra_jump3():
    code = QvmCode()
    code.add(
        ('ret',),
        ('jmp', 1000),
    )
    code.optimize()
    assert code._instrs == [
        ('ret',),
    ]


def test_eliminate_extra_jump4():
    code = QvmCode()
    code.add(
        ('jmp', 2000),
        ('jmp', 1000),
    )
    code.optimize()
    assert code._instrs == [
        ('jmp', 2000),
    ]


def test_eliminate_extra_jump5():
    code = QvmCode()
    code.add(
        ('ijmp',),
        ('retv',),
    )
    code.optimize()
    assert code._instrs == [
        ('ijmp',),
    ]
