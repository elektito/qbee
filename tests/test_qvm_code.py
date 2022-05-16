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
