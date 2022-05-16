from qbee.qvm_codegen import QvmCode


def test_fold_push_and_conv():
    code = QvmCode()
    code.add(
        ('push!', 10.0),
        ('conv!&',),
    )
    code.optimize()
    assert code._instrs == [('push&', 10)]


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


def test_eliminate_store_read():
    code = QvmCode()
    code.add(
        ('storel!', 'x'),
        ('readl!', 'x'),
    )
    code.optimize()
    assert code._instrs == []


def test_eliminate_read_store():
    code = QvmCode()
    code.add(
        ('readg!', 'x'),
        ('storeg!', 'x'),
    )
    code.optimize()
    assert code._instrs == []


def test_eliminate_read_store_incompatible_scope():
    code = QvmCode()
    code.add(
        ('readl!', 'x'),
        ('storeg!', 'x'),
    )
    code.optimize()
    assert code._instrs == [
        ('readl!', 'x'),
        ('storeg!', 'x'),
    ]


def test_eliminate_read_store_incompatible_type():
    code = QvmCode()
    code.add(
        ('readl!', 'x'),
        ('storel#', 'x'),
    )
    code.optimize()
    assert code._instrs == [
        ('readl!', 'x'),
        ('storel#', 'x'),
    ]


def test_eliminate_read_store_incompatible_arg():
    code = QvmCode()
    code.add(
        ('readl!', 'x'),
        ('storel!', 'y'),
    )
    code.optimize()
    assert code._instrs == [
        ('readl!', 'x'),
        ('storel!', 'y'),
    ]


def test_fold_push_and_conv_after_store_read():
    code = QvmCode()
    code.add(
        ('push!', 10.0),
        ('storel!', 'x'),
        ('readl!', 'x'),
        ('conv!&',),
    )
    code.optimize()
    assert code._instrs == [('push&', 10)]
